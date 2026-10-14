def ratio_does_not_violate_existing_matches(matches, alignment):
    violate_existing_matches = alignment.check_does_not_violate_existing_matches(matches,
                                                                                 allow_matches_inside_current_matches=True)
    return violate_existing_matches.sum().item() / matches.shape[0]

def ratio_does_not_violate_type_matches(matches, alignment):
    violate_type_matches = alignment.check_does_not_violate_type_matches(matches)
    return violate_type_matches.sum().item() / matches.shape[0]

def ratio_does_not_violate_one_to_one_matches(matches, alignment):
    violate_one_to_one_matches = alignment.check_does_not_violate_one_to_one_matches(matches)
    return violate_one_to_one_matches.sum().item() / matches.shape[0]

def ratio_matches_that_are_neither_type_nor_validation_of_existing(matches, alignment):
    violate_existing_matches = alignment.check_does_not_violate_existing_matches(matches,
                                                                                 allow_matches_inside_current_matches=True)
    violate_type_matches = alignment.check_does_not_violate_type_matches(matches)
    return 1/  max((~(violate_existing_matches & violate_type_matches)).sum().item(), 1)

def ratio_matches_that_violate_nothing(matches, alignment):
    violate_existing_matches = alignment.check_does_not_violate_existing_matches(matches,
                                                                                 allow_matches_inside_current_matches=True)
    violate_type_matches = alignment.check_does_not_violate_type_matches(matches)
    violate_1_to_1_matches = alignment.check_does_not_violate_one_to_one_matches(matches)
    return 1/  max((~(violate_existing_matches & violate_type_matches & violate_1_to_1_matches)).sum().item(), 1)

def ratio_of_matches_that_will_actually_be_created(matches, alignment):
    return alignment.compute_valid_matches(matches).shape[0] / matches.shape[0]

def highest_number_matches(matches, alignment):
    return matches.shape[0]

def lowest_number_matches(matches, alignment):
    return  -matches.shape[0]

def select_matcher_scoring_function_by_id(i):
    if i is None:
        i = 4
    funcs =  {
        0: ratio_of_matches_that_will_actually_be_created,
        1: ratio_does_not_violate_existing_matches,
        2: ratio_does_not_violate_type_matches,
        3: ratio_does_not_violate_one_to_one_matches,
        4: ratio_matches_that_violate_nothing, # This one is the one we choose
        5: highest_number_matches,
        6: lowest_number_matches
    }
    return funcs[i]
