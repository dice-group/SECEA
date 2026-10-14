from abc import ABC

import torch

from secea.alignment import Alignment
from secea.dataset import rdf_type_uri


class IterativeMatcher(ABC):
    def compute_matches(self, alignment: Alignment):
        # print("Run IterativeMatcher ", self.__class__.__name__)
        pass

    def get_name(self):
        return self.__class__.__name__


class EqualTripleAndEqualLabelMatcher(IterativeMatcher):
    def __init__(self, alignment, entities_with_enough_common_labels, min_common_triple_representation):
        self.entities_with_enough_common_labels = entities_with_enough_common_labels
        self.min_common_triple_representation = min_common_triple_representation
        self.num_entities_with_enough_common_labels = (entities_with_enough_common_labels.shape[0])

    def get_name(self):
        return super().get_name() + " " + str(self.min_common_triple_representation) + " " + str(self.entities_with_enough_common_labels.shape[0])

    def compute_matches(self, alignment: Alignment):
        super().compute_matches(alignment)

        if self.num_entities_with_enough_common_labels == 0:
            return alignment.empty_tensor

        counts = alignment.triple_representation_counter.common_counts_for_pairs(self.entities_with_enough_common_labels)

        mask = counts >= self.min_common_triple_representation
        if not mask.any():
            return alignment.empty_tensor

        return self.entities_with_enough_common_labels[mask]


class EqualTripleMatcher(IterativeMatcher):

    def __init__(self, min_common_triple_representation):
        self.min_common_triple_representation = min_common_triple_representation

    def get_name(self):
        return super().get_name() + " " + str(self.min_common_triple_representation)

    def compute_matches(self, alignment: Alignment):
        super().compute_matches(alignment)
        return alignment.triple_representation_counter.get_pairs_with_at_least_k_common_tensor(
            self.min_common_triple_representation)


class UniqueEqualTripleMatcher(IterativeMatcher):

    def __init__(self, min_common_triple_representation):
        self.min_common_triple_representation = min_common_triple_representation

    def get_name(self):
        return super().get_name() + " " + str(self.min_common_triple_representation)

    def compute_matches(self, alignment: Alignment):
        super().compute_matches(alignment)
        return alignment.triple_representation_counter.get_pairs_with_at_least_k_common_unique_tensor(
            self.min_common_triple_representation)


class IterativeMatcherMatchAllFromInput(IterativeMatcher):
    def __init__(self, input_tensor: torch.Tensor, name_addition):
        super().__init__()
        self.input_tensor = input_tensor
        self.name_addition = name_addition

    def get_name(self):
        return super().get_name() + " " + self.name_addition

    def compute_matches(self, alignment: Alignment):
        super().compute_matches(alignment)
        return self.input_tensor


class IterativeMatcherSameClassAndEqualLabel(IterativeMatcher):
    def __init__(self, alignment, entities_with_enough_common_labels, number):
        self.number = number
        self.entities_with_enough_common_labels = entities_with_enough_common_labels

        if rdf_type_uri not in alignment.source.anything_to_id or rdf_type_uri not in alignment.target.anything_to_id:
            self.has_type_uri = False
            return
        self.has_type_uri = True
        source_type_id = alignment.source.anything_to_id[rdf_type_uri]
        target_type_id = alignment.target.anything_to_id[rdf_type_uri]

        source_triples = alignment.source.triples_tensor
        source_type_triples = source_triples[source_triples[:, 1] == source_type_id]
        source_type_map = dict(zip(source_type_triples[:, 0].tolist(), source_type_triples[:, 2].tolist()))

        target_triples = alignment.target.triples_tensor
        target_type_triples = target_triples[target_triples[:, 1] == target_type_id]
        target_type_map = dict(zip(target_type_triples[:, 0].tolist(), target_type_triples[:, 2].tolist()))

        source_entity_ids = entities_with_enough_common_labels[:, 0].tolist()
        target_entity_ids = entities_with_enough_common_labels[:, 1].tolist()

        source_valid_mask = torch.tensor([e in source_type_map for e in source_entity_ids],
                                         device=entities_with_enough_common_labels.device)
        target_valid_mask = torch.tensor([e in target_type_map for e in target_entity_ids],
                                         device=entities_with_enough_common_labels.device)

        valid_mask = source_valid_mask & target_valid_mask

        filtered_entities = entities_with_enough_common_labels[valid_mask]
        source_types = torch.tensor([source_type_map[e] for e in filtered_entities[:, 0].tolist()],
                                    device=entities_with_enough_common_labels.device)

        target_types = torch.tensor([target_type_map[e] for e in filtered_entities[:, 1].tolist()],
                                    device=entities_with_enough_common_labels.device)

        self.source_entities_with_enough_common_labels_with_types = torch.stack([filtered_entities[:, 0], source_types],
                                                                                dim=1)  # X,2 : id->type
        self.target_entities_with_enough_common_labels_with_types = torch.stack([filtered_entities[:, 1], target_types],
                                                                                dim=1)  # X,2 : id->type

    def get_name(self):
        return super().get_name() + " " + str(self.number)

    def compute_matches(self, alignment: Alignment):
        super().compute_matches(alignment)
        if not self.has_type_uri:
            return alignment.empty_tensor
        mapped_source_types = alignment.source_to_target_mappings[
            self.source_entities_with_enough_common_labels_with_types[:, 1]]
        mapped_to_anything_mask = mapped_source_types != -1
        if not mapped_to_anything_mask.any():
            return alignment.empty_tensor
        source_entities_with_mapped_type = self.source_entities_with_enough_common_labels_with_types[
            mapped_to_anything_mask, 0]
        mapped_source_types_if_mapped_to_anything = mapped_source_types[mapped_to_anything_mask]
        target_entities_with_type = self.target_entities_with_enough_common_labels_with_types[mapped_to_anything_mask]
        same_type = mapped_source_types_if_mapped_to_anything == target_entities_with_type[:, 1]
        if not same_type.any():
            return alignment.empty_tensor
        return torch.stack([source_entities_with_mapped_type[same_type], target_entities_with_type[same_type, 0]],
                           dim=1)


class IterativeMatcherClassBasedOnCommonInstances(IterativeMatcher):
    def __init__(self, min_percentage):
        self.min_percentage = min_percentage

    def get_name(self):
        return super().get_name() + " " + str(self.min_percentage)

    def compute_matches(self, alignment: Alignment):
        return alignment.matched_instance_to_class_counter.get_pairs_with_at_least_f_common_percentage(
            self.min_percentage)


class SameClassAndEquivalentLabelMatcher(IterativeMatcher):

    def __init__(self, alignment, entities_with_enough_common_labels, number):
        self.number = number
        self.entities_with_enough_common_labels = entities_with_enough_common_labels
        if rdf_type_uri not in alignment.source.anything_to_id or rdf_type_uri not in alignment.target.anything_to_id:
            self.has_type_uri = False
            return
        self.has_type_uri = True
        source_type_id = alignment.source.anything_to_id[rdf_type_uri]
        target_type_id = alignment.target.anything_to_id[rdf_type_uri]
        source_triples = alignment.source.triples_tensor
        source_type_triples = source_triples[source_triples[:, 1] == source_type_id]
        source_type_map = dict(zip(source_type_triples[:, 0].tolist(), source_type_triples[:, 2].tolist()))
        target_triples = alignment.target.triples_tensor
        target_type_triples = target_triples[target_triples[:, 1] == target_type_id]
        target_type_map = dict(zip(target_type_triples[:, 0].tolist(), target_type_triples[:, 2].tolist()))
        source_entity_ids = entities_with_enough_common_labels[:, 0].tolist()
        target_entity_ids = entities_with_enough_common_labels[:, 1].tolist()
        source_valid_mask = torch.tensor([e in source_type_map for e in source_entity_ids],
                                         device=entities_with_enough_common_labels.device)
        target_valid_mask = torch.tensor([e in target_type_map for e in target_entity_ids],
                                         device=entities_with_enough_common_labels.device)
        valid_mask = source_valid_mask & target_valid_mask
        filtered_entities = entities_with_enough_common_labels[valid_mask]
        source_types = torch.tensor([source_type_map[e] for e in filtered_entities[:, 0].tolist()],
                                    device=entities_with_enough_common_labels.device)
        target_types = torch.tensor([target_type_map[e] for e in filtered_entities[:, 1].tolist()],
                                    device=entities_with_enough_common_labels.device)
        self.source_entities_with_enough_common_labels_with_types = torch.stack([filtered_entities[:, 0], source_types], dim=1)  # X,2 : id->type
        self.target_entities_with_enough_common_labels_with_types = torch.stack([filtered_entities[:, 1], target_types], dim=1)  # X,2 : id->type

    def get_name(self):
        return super().get_name() + " " + str(self.number)

    def compute_matches(self, alignment: Alignment):
        super().compute_matches(alignment)
        if not self.has_type_uri:
            return alignment.empty_tensor

        src = self.source_entities_with_enough_common_labels_with_types
        tgt = self.target_entities_with_enough_common_labels_with_types

        mapped_source_types = alignment.source_to_target_mappings[src[:, 1]]
        mapped_mask = mapped_source_types != -1
        if not mapped_mask.any():
            return alignment.empty_tensor

        src_entities = src[mapped_mask, 0]
        mapped_types = mapped_source_types[mapped_mask]
        tgt_entities = tgt[mapped_mask, 0]
        tgt_types = tgt[mapped_mask, 1]

        same_type = mapped_types == tgt_types
        if not same_type.any():
            return alignment.empty_tensor

        return torch.stack([src_entities[same_type], tgt_entities[same_type]], dim=1)
