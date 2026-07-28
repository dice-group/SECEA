import os
import sys
import time
import torch

from secea.matcher_scoring_functions import select_matcher_scoring_function_by_id
from secea.secea import SECEA
from utils import load_datasets_from_path, load_evaluation_alignment_from_path

if __name__ == "__main__":
    matcher_selection_function_id = int(sys.argv[1]) if len(sys.argv) > 1 else None
    base_dir = "datasets/OpenEA_dataset_v1.1/"
    results = []

    for pair_dir in os.listdir(base_dir):
        dataset_dir = os.path.join(base_dir, pair_dir)
        if not os.path.isdir(dataset_dir):
            continue

        print("RUN DATASET:", pair_dir)

        source_path = os.path.join(dataset_dir, "source.nt")
        target_path = os.path.join(dataset_dir, "target.nt")
        reference_path = os.path.join(dataset_dir, "alignment")

        source_dataset, target_dataset = load_datasets_from_path(
            source_path, target_path, add_iri_label=False, multithread=False
        )

        dataset_start = time.perf_counter()
        secea = SECEA(source_dataset, target_dataset, select_matcher_scoring_function_by_id(matcher_selection_function_id))
        alignment = secea.get_alignment()
        matches = alignment.get_entity_matches()
        dataset_runtime = time.perf_counter() - dataset_start

        reference_alignment = load_evaluation_alignment_from_path(
            reference_path, source_dataset, target_dataset
        )

        multiplier = max(
            len(alignment.source.id_to_anything),
            len(alignment.target.id_to_anything)
        ) + 1

        computed_matches_keys = matches[:, 0] * multiplier + matches[:, 1]
        reference_matches_keys = reference_alignment[:, 0] * multiplier + reference_alignment[:, 1]

        true_positive_mask = torch.isin(computed_matches_keys, reference_matches_keys)

        tp = true_positive_mask.sum().item()
        fp = (~true_positive_mask).sum().item()
        fn = reference_alignment.shape[0] - tp
        hits1 = tp / (reference_alignment.shape[0] or 1)

        print("True positive:", tp, "False positive:", fp, "False negative:", fn)
        print("Hits@1 =", hits1)
        print(f"Runtime: {dataset_runtime:.2f} seconds")
        print("")

        results.append((pair_dir, tp, fp, fn, hits1, dataset_runtime))

    print("")
    print("")
    print("FINAL RESULTS:")
    header = f"{'Dataset':15} {'TP':>8} {'FP':>8} {'FN':>8} {'Hits@1':>10} {'Time(s)':>10}"
    print(header)
    print("-" * len(header))

    for pair_dir, tp, fp, fn, hits1, runtime in results:
        print(f"{pair_dir:15} {tp:8d} {fp:8d} {fn:8d} {hits1:10.4f} {runtime:10.2f}")
