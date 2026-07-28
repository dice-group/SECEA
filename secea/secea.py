from typing import Literal
import heapq
from collections import defaultdict, Counter
import math

import torch

from secea.alignment import Alignment
from secea.dataset import Dataset, iri_label_uri
from secea.iterative_matcher import EqualTripleAndEqualLabelMatcher, \
    EqualTripleMatcher, \
    UniqueEqualTripleMatcher, IterativeMatcherMatchAllFromInput, \
    IterativeMatcherClassBasedOnCommonInstances, SameClassAndEquivalentLabelMatcher
from secea.matcher_combinations import DynamicIterativeMatcherCombiner
from decorators import timed


class SECEA:
    def __init__(self, source: Dataset, target: Dataset, matcher_scoring_function = None):
        self.source = source
        self.target = target
        self.alignment = Alignment(source, target)
        # The following part must be at the top to have things including literals matched if their IRI is equivalent
        self.alignment.propose_matches(compute_pairs_with_equivalent_iri(source, target), "INITIAL IRI")
        iterative_matcher_groups = []

        source_label_relations, target_label_relations = self.find_potential_label_relations()
        for element_type in ['relation', 'class']:
            label_equality_pair_counter = self.compute_pairs_with_at_least_k_unique_matching_labels(element_type,
                                                                                                       source_label_relations,
                                                                                                       target_label_relations)
            iterative_matcher_groups.append(
                [IterativeMatcherMatchAllFromInput(label_equality_pair_counter[i], element_type + str(i)) for i
                 in
                 sorted(label_equality_pair_counter.keys(), reverse=True)])


        entity_label_equality_pair_counter = self.compute_pairs_with_at_least_k_unique_matching_labels('entity',
                                                                                                       source_label_relations,
                                                                                                       target_label_relations)

        iterative_matcher_groups.append([EqualTripleAndEqualLabelMatcher(self.alignment,
                                                                         entity_label_equality_pair_counter[
                                                                                                           i], i) for i
                                         in sorted(entity_label_equality_pair_counter.keys(), reverse=True)])

        iterative_matcher_groups.append(
            [UniqueEqualTripleMatcher(i) for i in reversed(range(1, 20))])
        iterative_matcher_groups.append([EqualTripleMatcher(i) for i in reversed(range(1, 20))])
        iterative_matcher_groups.append(
            [IterativeMatcherMatchAllFromInput(entity_label_equality_pair_counter[i], "entity " + str(i)) for i in
             sorted(entity_label_equality_pair_counter.keys(), reverse=True)])

        iterative_matcher_groups.append(
            [SameClassAndEquivalentLabelMatcher(self.alignment, entity_label_equality_pair_counter[i], i) for i in
             sorted(entity_label_equality_pair_counter.keys(), reverse=True)])

        iterative_matcher_groups.append([IterativeMatcherClassBasedOnCommonInstances(f) for f in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]])

        tf_idf_fractions = [i / 100 for i in range(0, 1000, 20) if i > 0]
        tfidf_sets = compute_pairs_with_tf_idf_word_literals(source, target, tf_idf_fractions)
        if tfidf_sets:
            iterative_matcher_groups.append(
                [IterativeMatcherMatchAllFromInput(tfidf_sets[f], f"tfidf literals top {f}*|NumEntities|")
                 for f in sorted(tfidf_sets.keys())]
            )

        DynamicIterativeMatcherCombiner(iterative_matcher_groups, matcher_scoring_function).compute_matches(self.alignment)

    def get_alignment(self) -> Alignment:
        return self.alignment

    @timed
    def find_potential_label_relations(self):
        print("Source label relations:")
        source_label_relations = find_potential_label_relations(self.source)
        print("Target label relations:")
        target_label_relations = find_potential_label_relations(self.target)
        return source_label_relations, target_label_relations

    @timed
    def compute_pairs_with_at_least_k_unique_matching_labels(self, match_type: Literal['class', 'relation', 'entity'],
                                                             source_label_relations, target_label_relations):
        match match_type:
            case 'class':
                source_valid_entity_mask = self.alignment.source.class_id_mask
                target_valid_entity_mask = self.alignment.target.class_id_mask
            case 'relation':
                source_valid_entity_mask = self.alignment.source.rel_id_mask
                target_valid_entity_mask = self.alignment.target.rel_id_mask
            case 'entity':
                source_valid_entity_mask = self.alignment.source.entity_id_mask
                target_valid_entity_mask = self.alignment.target.entity_id_mask
            case _:
                raise ValueError(
                    "Invalid match type " + match_type + ". Valid types are 'class', 'relation', 'entity'.")
        literal_relation_iri_pairs = torch.cartesian_prod(source_label_relations, target_label_relations)
        source_triples = self.alignment.source.triples_tensor
        target_triples = self.alignment.target.triples_tensor
        source_triples = source_triples[source_valid_entity_mask[source_triples[:, 0]]]
        target_triples = target_triples[target_valid_entity_mask[target_triples[:, 0]]]
        literal_relation_index_to_matched = list()
        for literal_relation_index, (source_literal_relation_id, target_literal_relation_id) in enumerate(
                literal_relation_iri_pairs.tolist()):
            pairs_for_this_relation = set()
            source_literal_triples = source_triples[source_triples[:, 1] == source_literal_relation_id]
            target_literal_triples = target_triples[target_triples[:, 1] == target_literal_relation_id]
            source_labels_to_things = defaultdict(set)
            target_labels_to_things = defaultdict(set)
            for head, literal_attribute_id in source_literal_triples[:,(0,2)].tolist():
                label = self.alignment.source.id_to_anything[literal_attribute_id]
                label = label.lower()
                source_labels_to_things[label].add(head)

            for head, literal_attribute_id in target_literal_triples[:,(0,2)].tolist():
                label = self.alignment.target.id_to_anything[literal_attribute_id]
                label = label.lower()
                target_labels_to_things[label].add(head)

            for literal, source_things in source_labels_to_things.items():
                if len(source_things) == 1:
                    if literal in target_labels_to_things:
                        target_things = target_labels_to_things[literal]
                        if len(target_things) == 1:
                            matched_source = list(source_things)[0]
                            matched_target = list(target_things)[0]
                            pairs_for_this_relation.add((matched_source, matched_target))

            #Do the same with plural s removed
            for head, literal_attribute_id in source_literal_triples[:,(0,2)].tolist():
                label = self.alignment.source.id_to_anything[literal_attribute_id]
                if label.endswith('s'):
                    label = label[:-1]
                label = label.lower()
                source_labels_to_things[label].add(head)

            for head, literal_attribute_id in target_literal_triples[:,(0,2)].tolist():
                label = self.alignment.target.id_to_anything[literal_attribute_id]
                if label.endswith('s'):
                    label = label[:-1]
                label = label.lower()
                target_labels_to_things[label].add(head)

            for literal, source_things in source_labels_to_things.items():
                if len(source_things) == 1:
                    if literal in target_labels_to_things:
                        target_things = target_labels_to_things[literal]
                        if len(target_things) == 1:
                            matched_source = list(source_things)[0]
                            matched_target = list(target_things)[0]
                            pairs_for_this_relation.add((matched_source, matched_target))

            literal_relation_index_to_matched.extend(pairs_for_this_relation)
        pair_counter = Counter()
        for matched_source, matched_target in literal_relation_index_to_matched:
            pair_counter[(matched_source, matched_target)] += 1

        occurences = dict()
        if len(pair_counter) > 0:
            max_count = max(pair_counter.values())
            for i in range(1, max_count + 1):
                pairs_with_i_common_labels = [key for key, value in pair_counter.items() if value >= i]
                occurences[i] = torch.tensor(pairs_with_i_common_labels, dtype=torch.long)
        return occurences


@timed
def find_potential_label_relations(dataset: Dataset):
    triples = dataset.triples_tensor
    rels = triples[:, 1]
    tails = triples[:, 2]
    rel_ids, rel_counts = torch.unique(rels, return_counts=True)
    tail_is_literal = dataset.literal_id_mask[tails]

    heads_with_literal = triples[tail_is_literal, 0].unique()
    head_amount = heads_with_literal.numel()
    # Literal triples per rel
    literal_rel_ids, literal_counts = torch.unique(rels[tail_is_literal], return_counts=True)
    literal_count_map = dict(zip(literal_rel_ids.tolist(), literal_counts.tolist()))
    possible_labels = set()
    for rel_id, count in zip(rel_ids.tolist(), rel_counts.tolist()):
        literal_count = literal_count_map.get(rel_id, 0)
        # must be entirely literal tails
        if literal_count != count:
            continue

        ratio = count / head_amount
        if ratio < 0.10:  # Must be present for at least 10%
            continue
        # injectivity on literal values for this relation
        rel_mask = (rels == rel_id) & tail_is_literal
        rel_literals = tails[rel_mask]
        n = rel_literals.numel()

        num_unique_literals = torch.unique(rel_literals).numel()
        injectivity = (num_unique_literals - 1) / (n - 1)

        if injectivity < 0.95: #Must have at least 95% unique tails
            continue

        rel_name = dataset.id_to_rel.get(rel_id)
        print(
            f"Potential label relation: {rel_name} {rel_id} ratio={ratio:.4f} injectivity={injectivity:.4f}"
        )
        possible_labels.add(rel_id)

    if iri_label_uri in dataset.anything_to_id:
        possible_labels.add(dataset.anything_to_id[iri_label_uri])

    return torch.tensor(list(possible_labels), dtype=torch.long)


@timed
def compute_pairs_with_tf_idf_word_literals(
        source: Dataset,
        target: Dataset,
        fractions_of_results: list[float],
        max_df: float = 0.01,
        min_df_count: int = 2,
) -> dict[float, torch.Tensor]:
    # --- literal triples ---
    source_literal_triples = source.triples_tensor[source.literal_id_mask[source.triples_tensor[:, 2]]]
    target_literal_triples = target.triples_tensor[target.literal_id_mask[target.triples_tensor[:, 2]]]

    # --- per-entity word sets / TF ---
    src_entity_words = defaultdict(set)
    tgt_entity_words = defaultdict(set)
    tgt_tf = defaultdict(Counter)

    # --- Source ---
    for head_id, _, literal_id in source_literal_triples.tolist():
        words = source.id_to_literal[literal_id].lower().split()
        if words:
            src_entity_words[head_id].update(words)

    # --- Target ---
    for head_id, _, literal_id in target_literal_triples.tolist():
        words = target.id_to_literal[literal_id].lower().split()
        if not words:
            continue
        tgt_tf[head_id].update(words)
        tgt_entity_words[head_id].update(words)

    # --- DF ---
    document_count = len(src_entity_words) + len(tgt_entity_words)
    if document_count == 0:
        return {}

    df = Counter()
    for wset in src_entity_words.values():
        df.update(wset)
    for wset in tgt_entity_words.values():
        df.update(wset)

    if not df:
        return {}

    max_df_count = max(1, int(max_df * document_count))
    allowed_words = {w for w, c in df.items() if min_df_count <= c <= max_df_count}
    if not allowed_words:
        return {}

    idf = {w: (math.log((document_count + 1.0) / (df[w] + 1.0)) + 1.0) for w in allowed_words}

    # --- Inverted index (filtered) ---
    literal_word_dict_source = defaultdict(list)
    for s_ent, wset in src_entity_words.items():
        for w in wset:
            if w in allowed_words:
                literal_word_dict_source[w].append(s_ent)

    if not literal_word_dict_source:
        return {}

    # --- Determine required sizes ---
    min_num_entities_with_literals = min(len(src_entity_words), len(tgt_entity_words))
    fractions_sorted = sorted(fractions_of_results, reverse=True)

    n_by_frac: dict[float, int] = {}
    max_n = 0
    for frac in fractions_sorted:
        n = int(frac * min_num_entities_with_literals)
        if n < 0:
            n = 0
        n_by_frac[frac] = n
        if n > max_n:
            max_n = n

    if max_n <= 0:
        return {frac: torch.empty((0, 2), dtype=torch.long) for frac in fractions_sorted}

    # --- Bounded heap instead of global pair_scores ---
    heap = []  # min-heap of (score, (s_ent, t_ent))

    for t_ent, tf_counter in tgt_tf.items():

        local_scores = defaultdict(float)

        for w, tf_w in tf_counter.items():
            if w not in allowed_words:
                continue
            src_ents = literal_word_dict_source.get(w)
            if not src_ents:
                continue
            weight = tf_w * idf[w]
            for s_ent in src_ents:
                local_scores[s_ent] += weight

        for s_ent, score in local_scores.items():
            if len(heap) < max_n:
                heapq.heappush(heap, (score, (s_ent, t_ent)))
            else:
                if score > heap[0][0]:
                    heapq.heapreplace(heap, (score, (s_ent, t_ent)))

    if not heap:
        return {}

    top_items = sorted(heap, key=lambda x: x[0], reverse=True)
    biggest = torch.tensor([pair for (_score, pair) in top_items], dtype=torch.long)

    out: dict[float, torch.LongTensor] = {}
    for frac in fractions_sorted:
        n = min(n_by_frac[frac], biggest.shape[0])
        out[frac] = biggest[:n]

    return out

def compute_pairs_with_equivalent_iri(source: Dataset, target: Dataset):
    source_thing_with_id = source.anything_to_id
    target_thing_with_id = target.anything_to_id
    matches = []
    for target_thing, target_id in target_thing_with_id.items():
        if target_thing in source_thing_with_id.keys():
            matches.append((source_thing_with_id[target_thing], target_id))
    return torch.tensor(matches, dtype=torch.long)
