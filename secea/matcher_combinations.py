from abc import ABC

from secea.alignment import Alignment
from secea.iterative_matcher import IterativeMatcher, IterativeMatcherMatchAllFromInput
from secea.matcher_scoring_functions import ratio_matches_that_violate_nothing


class MatcherCombiner(IterativeMatcher, ABC):
    def compute_matches(self, alignment: Alignment):
        super().compute_matches(alignment)

class TieredIterativeMatcherCombiner(MatcherCombiner):
    def __init__(self, iterative_matchers: list[IterativeMatcher]):
        self.iterative_matchers = iterative_matchers

    def compute_matches(self, alignment: Alignment):
        super().compute_matches(alignment)
        i = 0
        while i < len(self.iterative_matchers):
            #print("Run iterative matcher ", i, " of ", len(self.iterative_matchers), "")
            matches = self.iterative_matchers[i].compute_matches(alignment)
            previous_matches = alignment.matches_source_to_target.shape[0]
            alignment.propose_matches(matches, self.iterative_matchers[i].get_name())
            current_matches = alignment.matches_source_to_target.shape[0]
            #print("Found ", current_matches - previous_matches, " new matches", "Current:", current_matches, "Previous:", previous_matches,)
            if current_matches > previous_matches:
                i=-1
                #print("Restart")
            i+=1


class DynamicIterativeMatcherCombiner(MatcherCombiner):
    def __init__(self, iterative_matchers: list[list[IterativeMatcher]], matcher_scoring_function = None):
        self.iterative_matchers = iterative_matchers
        self.next_matcher_index_per_group = [0 for i in range(len(self.iterative_matchers))]
        self.max_index_per_group = [len(self.iterative_matchers[i]) for i in range(len(self.iterative_matchers))]
        if matcher_scoring_function is None:
            self.matcher_scoring_function = ratio_matches_that_violate_nothing
        else:
            self.matcher_scoring_function = matcher_scoring_function
        print("USING MATCHER SCORING METHOD: ", self.matcher_scoring_function.__name__)
    def compute_matches(self, alignment: Alignment):
        ignored_matchers = set()
        while True:
            matches_found_by_group = []
            for iterative_matcher_group_index in range(0, len(self.iterative_matchers)):
                for matcher_in_this_group_index in range(0, len(self.iterative_matchers[iterative_matcher_group_index])):
                    matcher = self.iterative_matchers[iterative_matcher_group_index][matcher_in_this_group_index]
                    name = matcher.get_name()
                    if name in ignored_matchers:
                        continue
                    matches = matcher.compute_matches(alignment)
                    type_ok = alignment.check_does_not_violate_type_matches(matches)
                    if not type_ok.any():
                        if type(matcher) == IterativeMatcherMatchAllFromInput:
                            print("Ignoring matcher ", name, " as it does not match any type.")
                            ignored_matchers.add(name)
                        continue

                    type_filtered = matches[type_ok]
                    existing_ok = alignment.check_does_not_violate_existing_matches(type_filtered)

                    if not existing_ok.any():
                        if type(matcher) == IterativeMatcherMatchAllFromInput:
                            print("Ignoring matcher ", name, " all matches violate existing matches.")
                            ignored_matchers.add(name)
                        continue

                    valid = alignment.check_does_not_violate_one_to_one_matches(type_filtered[existing_ok])
                    if valid.any():
                        matches_that_could_be_applied = type_filtered[existing_ok][valid]
                        matches_found_by_group.append((matcher, matches, matches_that_could_be_applied))
                        break
            if len(matches_found_by_group) == 0:
                return

            best_id = -1
            best_score = -alignment.triple_representation_counter.key_multiplier
            #Selection mechanism based on least matches found
            if len(matches_found_by_group) == 1:
                best_id = 0
            else:
                for i in range(len(matches_found_by_group)):
                    matches = matches_found_by_group[i][1]
                    score = self.matcher_scoring_function(matches, alignment)
                    if score > best_score:
                        best_id = i
                        best_score = score

            chosen = matches_found_by_group[best_id]
            previous_matches = alignment.matches_source_to_target.shape[0]
            alignment.propose_already_checked_matches(chosen[2], chosen[0].get_name())
            current_matches = alignment.matches_source_to_target.shape[0]
            matched_entities = alignment.source.entity_id_mask[alignment.matches_source_to_target[:, 0]].sum().item()
            print("Chose", matches_found_by_group[best_id][0].get_name(), "Resulted in ", current_matches - previous_matches, " new matches. Total entity matches ", matched_entities)
