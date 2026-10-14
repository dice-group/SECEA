from secea.dataset import Dataset

import torch
from secea.dataset import rdf_type_uri, owl_class_uri, iri_label_uri, wiki_page_link_uri
from collections import defaultdict


class Alignment:

    def __init__(self, source: Dataset, target: Dataset):
        self.source = source
        self.target = target
        self.source_to_target_mappings = -torch.ones(len(self.source.anything_to_id), dtype=torch.long)
        self.target_to_source_mappings = -torch.ones(len(self.target.anything_to_id), dtype=torch.long)
        self.matches_source_to_target = torch.empty((0, 2), dtype=torch.long)
        self.matches_by_which_matcher = []

        self.triple_representation_counter = TripleEquivalenceCounter(self.source, self.target)
        self.matched_instance_to_class_counter = MatchedInstanceToClassCounter(self.source, self.target)
        self.empty_tensor = torch.empty((0, 2), dtype=torch.long)

    def propose_already_checked_matches(self, final_matches: torch.Tensor, from_which_mather: str):
        self.source_to_target_mappings[final_matches[:, 0]] = final_matches[:, 1]
        self.target_to_source_mappings[final_matches[:, 1]] = final_matches[:, 0]

        self.matches_source_to_target = torch.cat((self.matches_source_to_target, final_matches))
        for i in range(final_matches.shape[0]):
            self.matches_by_which_matcher.append(from_which_mather)

        self.triple_representation_counter.on_new_matches(self, final_matches)
        self.matched_instance_to_class_counter.on_new_matches(self, final_matches)

    def propose_matches(self, matches: torch.Tensor, from_which_mather: str):
        final_matches = self.compute_valid_matches(matches)
        if final_matches.numel() > 0:
            self.propose_already_checked_matches(final_matches, from_which_mather)

    def compute_valid_matches(self, matches):
        matches = matches[self.check_does_not_violate_type_matches(matches)]
        matches = matches[self.check_does_not_violate_existing_matches(matches)]
        matches = matches[self.check_does_not_violate_one_to_one_matches(matches)]
        return matches

    def check_does_not_violate_type_matches(self, matches):
        return (
                (self.source.entity_id_mask[matches[:, 0]] == self.target.entity_id_mask[matches[:, 1]]) &
                (self.source.rel_id_mask[matches[:, 0]] == self.target.rel_id_mask[matches[:, 1]]) &
                (self.source.class_id_mask[matches[:, 0]] == self.target.class_id_mask[matches[:, 1]])
        )  # Literal mask not needed as if these three are equal, the fourth one is implied

    def check_does_not_violate_existing_matches(self, matches: torch.Tensor,
                                                allow_matches_inside_current_matches=False):
        valid_mask = (
            ((self.source_to_target_mappings[matches[:, 0]] == -1) & (
                    self.target_to_source_mappings[matches[:, 1]] == -1))
        )
        if allow_matches_inside_current_matches:
            valid_mask |= self.source_to_target_mappings[matches[:, 0]] == matches[:, 1]
        return valid_mask

    def check_does_not_violate_one_to_one_matches(self, matches: torch.Tensor):
        source_ids = matches[:, 0]
        target_ids = matches[:, 1]

        _, source_inv, source_counts = torch.unique(source_ids, return_inverse=True, return_counts=True)
        _, target_inv, target_counts = torch.unique(target_ids, return_inverse=True, return_counts=True)

        return (source_counts[source_inv] == 1) & (target_counts[target_inv] == 1)

    def get_non_literal_matches(self) -> torch.Tensor:
        return self.matches_source_to_target[~self.source.literal_id_mask[self.matches_source_to_target[:, 0]]]

    def get_entity_matches(self) -> torch.Tensor:
        return self.matches_source_to_target[self.source.entity_id_mask[self.matches_source_to_target[:, 0]]]


class TripleEquivalenceCounter:
    def __init__(self, source, target):
        self.source_triples = source.triples_tensor
        self.target_triples = target.triples_tensor

        self.excluded_relations_uris = [rdf_type_uri, owl_class_uri, iri_label_uri, wiki_page_link_uri]
        source_excluded_relations = torch.tensor(
            [source.anything_to_id[rel_id] for rel_id in self.excluded_relations_uris
             if rel_id in source.anything_to_id],
            dtype=torch.long
        )
        target_excluded_relations = torch.tensor(
            [target.anything_to_id[rel_id] for rel_id in self.excluded_relations_uris
             if rel_id in target.anything_to_id],
            dtype=torch.long
        )

        # Filter triples
        self.source_triples = self.source_triples[~torch.isin(self.source_triples[:, 1], source_excluded_relations)]
        self.target_triples = self.target_triples[~torch.isin(self.target_triples[:, 1], target_excluded_relations)]

        # Multiplier
        self.key_multiplier = max(len(source.anything_to_id), len(target.anything_to_id)) + 1  # >10M as assumed

        target_heads = self.target_triples[:, 0]
        target_rels = self.target_triples[:, 1]
        target_tails = self.target_triples[:, 2]

        target_key_rt = target_rels * self.key_multiplier + target_tails
        target_key_ht = target_heads * self.key_multiplier + target_tails
        target_key_hr = target_heads * self.key_multiplier + target_rels

        self.key_to_target_ids_dict_rt = defaultdict(set)
        self.key_to_target_ids_dict_ht = defaultdict(set)
        self.key_to_target_ids_dict_hr = defaultdict(set)

        for (head_id, rel_id, tail_id), key in zip(self.target_triples.tolist(), target_key_rt.tolist()):
            self.key_to_target_ids_dict_rt[key].add(head_id)
        for (head_id, rel_id, tail_id), key in zip(self.target_triples.tolist(), target_key_hr.tolist()):
            self.key_to_target_ids_dict_hr[key].add(tail_id)
        for (head_id, rel_id, tail_id), key in zip(self.target_triples.tolist(), target_key_ht.tolist()):
            self.key_to_target_ids_dict_ht[key].add(rel_id)

        self.source_index = defaultdict(list)
        for idx, (h, r, t) in enumerate(self.source_triples.tolist()):
            self.source_index[h].append(idx)
            self.source_index[r].append(idx)
            self.source_index[t].append(idx)

        self.processed_triple_mask_per_component = torch.zeros((len(self.source_triples), 3), dtype=torch.bool)

        self.pair_to_index_common = {}
        self.pair_to_index_unique = {}

        self.pairs_common = torch.empty((0, 2), dtype=torch.long)
        self.counts_common = torch.empty((0,), dtype=torch.int32)

        self.pairs_unique = torch.empty((0, 2), dtype=torch.long)
        self.counts_unique = torch.empty((0,), dtype=torch.int32)

        self.new_pairs_common = []
        self.new_counts_common = []
        self.new_pairs_unique = []
        self.new_counts_unique = []

        self.common_dirty = False
        self.unique_dirty = False

    def update_common_tensor(self):
        if not self.new_pairs_common:
            self.common_dirty = False
            return

        new_pairs_t = torch.tensor(self.new_pairs_common, dtype=torch.long)
        new_counts_t = torch.tensor(self.new_counts_common, dtype=torch.int32)

        if self.pairs_common.numel() == 0:
            self.pairs_common = new_pairs_t
            self.counts_common = new_counts_t
        else:
            self.pairs_common = torch.cat([self.pairs_common, new_pairs_t], dim=0)
            self.counts_common = torch.cat([self.counts_common, new_counts_t], dim=0)

        self.new_pairs_common.clear()
        self.new_counts_common.clear()
        self.common_dirty = False

    def update_common_unique_tensor(self):
        if not self.new_pairs_unique:
            self.unique_dirty = False
            return

        new_pairs_t = torch.tensor(self.new_pairs_unique, dtype=torch.long)
        new_counts_t = torch.tensor(self.new_counts_unique, dtype=torch.int32)

        if self.pairs_unique.numel() == 0:
            self.pairs_unique = new_pairs_t
            self.counts_unique = new_counts_t
        else:
            self.pairs_unique = torch.cat([self.pairs_unique, new_pairs_t], dim=0)
            self.counts_unique = torch.cat([self.counts_unique, new_counts_t], dim=0)

        self.new_pairs_unique.clear()
        self.new_counts_unique.clear()
        self.unique_dirty = False

    # This gets called by the alignment. It computes the triples which need to be checked for equivalences and then calls upadte
    def on_new_matches(self, alignment, matches):
        if matches.numel() == 0:
            return
        newly_matched = matches[:, 0].tolist()
        lists = [self.source_index.get(ent, []) for ent in newly_matched]
        flat = torch.tensor([i for sub in lists for i in sub], dtype=torch.long)
        updated_triples_mask = torch.unique(flat)
        subset = self.source_triples[updated_triples_mask]
        self.update(alignment, subset, updated_triples_mask)

    def increment_common(self, pair):
        self.common_dirty = True
        idx = self.pair_to_index_common.get(pair)
        if idx is None:
            idx = len(self.pair_to_index_common)
            self.pair_to_index_common[pair] = idx
            self.new_pairs_common.append(pair)
            self.new_counts_common.append(1)
            return

        base = self.counts_common.numel()
        if idx < base:
            self.counts_common[idx] += 1
        else:
            self.new_counts_common[idx - base] += 1

    def increment_unique(self, pair):
        self.unique_dirty = True
        idx = self.pair_to_index_unique.get(pair)
        if idx is None:
            idx = len(self.pair_to_index_unique)
            self.pair_to_index_unique[pair] = idx
            self.new_pairs_unique.append(pair)
            self.new_counts_unique.append(1)
            return

        base = self.counts_unique.numel()
        if idx < base:
            self.counts_unique[idx] += 1
        else:
            self.new_counts_unique[idx - base] += 1

    def update(self, alignment, source_triples_with_newly_matched_source, updated_triples_mask):
        mapped_heads = alignment.source_to_target_mappings[source_triples_with_newly_matched_source[:, 0]]
        mapped_rels = alignment.source_to_target_mappings[source_triples_with_newly_matched_source[:, 1]]
        mapped_tails = alignment.source_to_target_mappings[source_triples_with_newly_matched_source[:, 2]]

        valid_hr = (mapped_heads != -1) & (mapped_rels != -1)
        valid_ht = (mapped_heads != -1) & (mapped_tails != -1)
        valid_rt = (mapped_rels != -1) & (mapped_tails != -1)

        global_indices = updated_triples_mask

        already_processed_rt = self.processed_triple_mask_per_component[global_indices, 0]
        already_processed_ht = self.processed_triple_mask_per_component[global_indices, 1]
        already_processed_hr = self.processed_triple_mask_per_component[global_indices, 2]

        valid_rt = valid_rt & (~already_processed_rt)
        valid_ht = valid_ht & (~already_processed_ht)
        valid_hr = valid_hr & (~already_processed_hr)

        if not valid_hr.any() and not valid_rt.any() and not valid_ht.any():
            return

        source_heads = source_triples_with_newly_matched_source[valid_rt, 0]
        source_rels = source_triples_with_newly_matched_source[valid_ht, 1]
        source_tails = source_triples_with_newly_matched_source[valid_hr, 2]

        mapped_heads_hr = mapped_heads[valid_hr]
        mapped_rels_hr = mapped_rels[valid_hr]

        mapped_heads_ht = mapped_heads[valid_ht]
        mapped_tails_ht = mapped_tails[valid_ht]

        mapped_rels_rt = mapped_rels[valid_rt]
        mapped_tails_rt = mapped_tails[valid_rt]

        source_key_rt = mapped_rels_rt * self.key_multiplier + mapped_tails_rt
        source_key_ht = mapped_heads_ht * self.key_multiplier + mapped_tails_ht
        source_key_hr = mapped_heads_hr * self.key_multiplier + mapped_rels_hr

        self.handle_component(source_heads, source_key_rt, self.key_to_target_ids_dict_rt)
        self.handle_component(source_rels, source_key_ht, self.key_to_target_ids_dict_ht)
        self.handle_component(source_tails, source_key_hr, self.key_to_target_ids_dict_hr)

        self.processed_triple_mask_per_component[global_indices[valid_rt], 0] = True
        self.processed_triple_mask_per_component[global_indices[valid_ht], 1] = True
        self.processed_triple_mask_per_component[global_indices[valid_hr], 2] = True

    def handle_component(self, source_ids, key_tensor, target_dict):
        _, inv, counts = torch.unique(key_tensor, return_inverse=True, return_counts=True)
        is_unique_occ = counts[inv].eq(1)

        for head_id, key, source_is_unique in zip(source_ids.tolist(), key_tensor.tolist(), is_unique_occ.tolist()):
            target_candidates = target_dict.get(key)
            if target_candidates is None:
                continue

            for target_id in target_candidates:
                self.increment_common((head_id, target_id))

            if source_is_unique and len(target_candidates) == 1:
                (target_id,) = target_candidates
                self.increment_unique((head_id, target_id))

    def get_pairs_with_at_least_k_common_tensor(self, k):
        if self.common_dirty:
            self.update_common_tensor()
        if self.pairs_common.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long)
        mask = self.counts_common >= k
        return self.pairs_common[mask]

    def get_pairs_with_at_least_k_common_unique_tensor(self, k):
        if self.unique_dirty:
            self.update_common_unique_tensor()
        if self.pairs_unique.numel() == 0:
            return torch.empty((0, 2), dtype=torch.long)
        mask = self.counts_unique >= k
        return self.pairs_unique[mask]

    def common_counts_for_pairs(self, pairs):
        # Only use this is there are not so many pairs
        if self.common_dirty:
            self.update_common_tensor()

        if pairs.numel() == 0:
            return torch.empty((0,), dtype=self.counts_common.dtype, )

        base = self.pairs_common.shape[0]

        indices = []
        for source_id, target_id in pairs.tolist():
            index = self.pair_to_index_common.get((source_id, target_id))
            indices.append(index if index is not None and index < base else -1)

        indices = torch.tensor(indices, dtype=torch.long)

        counts = torch.zeros(pairs.shape[0], dtype=self.counts_common.dtype)

        valid = indices >= 0
        if valid.any():
            counts[valid] = self.counts_common[indices[valid]]

        return counts


class MatchedInstanceToClassCounter:
    def __init__(self, source, target):
        self.source_count = defaultdict(int)
        self.target_count = defaultdict(int)
        self.common_count = defaultdict(int)
        self.pairs = None
        self.pair_scores = None
        self.no_types = False
        if source.anything_to_type_id is None or target.anything_to_type_id is None:
            self.no_types = True

    def on_new_matches(self, alignment, matches):
        if self.no_types:
            return
        if matches.numel() == 0:
            return
        # Filter down to entitiy matches
        matches = matches[alignment.source.entity_id_mask[matches[:, 0]]]
        matches = matches[alignment.target.entity_id_mask[matches[:, 1]]]
        # Filter down to things that have a type
        source_types = alignment.source.anything_to_type_id[matches[:, 0]]
        target_types = alignment.target.anything_to_type_id[matches[:, 1]]
        both_have_type = (source_types != -1) & (target_types != -1)
        if not both_have_type.any():
            return
        source_types = source_types[both_have_type]
        target_types = target_types[both_have_type]
        for source_type_id, target_type_id in zip(source_types.tolist(), target_types.tolist()):
            self.source_count[source_type_id] += 1
            self.target_count[target_type_id] += 1
            self.common_count[(source_type_id, target_type_id)] += 1

        class_pairs = []
        scores = []
        for (source_type_id, target_type_id), common_count_for_this_pair in self.common_count.items():
            if common_count_for_this_pair < 10:
                continue
            source_count_for_this_type = self.source_count[source_type_id]
            target_count_for_this_type = self.target_count[target_type_id]
            percentage = (2 * common_count_for_this_pair) / (source_count_for_this_type + target_count_for_this_type)
            class_pairs.append((source_type_id, target_type_id))
            scores.append(percentage)

        if class_pairs:
            self.pairs = torch.tensor(class_pairs, dtype=torch.long)
            self.pair_scores = torch.tensor(scores, dtype=torch.float)

    def get_pairs_with_at_least_f_common_percentage(self, f):
        if self.pairs is None:
            return torch.empty((0, 2), dtype=torch.long)
        mask = self.pair_scores >= f
        return self.pairs[mask]
