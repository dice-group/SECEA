import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from secea.matcher_scoring_functions import select_matcher_scoring_function_by_id
from secea.secea import SECEA
from utils import load_datasets_from_path


def get_element_type_by_iri(iri: str) -> str | None:
    if iri is None:
        return None
    if "/resource/" in iri:
        return "resource"
    if "/class/" in iri:
        return "class"
    if "/property/" in iri:
        return "property"
    return None


def compute_tp_fp_fn_by_element_type(source_dataset, target_dataset, ref_matches_id, matches: torch.Tensor):
    reference_matches = torch.tensor(ref_matches_id, dtype=torch.long)

    reference_pairs = set(map(tuple, reference_matches.tolist()))
    computed_pairs = set(map(tuple, matches.tolist()))

    reference_sources = set(reference_matches[:, 0].tolist())
    reference_targets = set(reference_matches[:, 1].tolist())

    computed_sources = set(matches[:, 0].tolist()) if matches.numel() else set()
    computed_targets = set(matches[:, 1].tolist()) if matches.numel() else set()

    source_element_type = {i: get_element_type_by_iri(source_dataset.id_to_anything.get(i)) for i in
                           (reference_sources | computed_sources)}
    target_element_type = {i: get_element_type_by_iri(target_dataset.id_to_anything.get(i)) for i in
                           (reference_targets | computed_targets)}

    counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in ("resource", "class", "property")}
    fp_pairs_by_kind = {k: [] for k in ("resource", "class", "property")}
    fn_pairs_by_kind = {k: [] for k in ("resource", "class", "property")}

    for (s, t) in (reference_pairs & computed_pairs):
        element_type = source_element_type.get(s)
        if element_type is not None:
            counts[element_type]["tp"] += 1

    for (s, t) in (reference_pairs - computed_pairs):
        element_type = source_element_type.get(s)
        if element_type is not None:
            counts[element_type]["fn"] += 1
            fn_pairs_by_kind[element_type].append((s, t))

    for (s, t) in (computed_pairs - reference_pairs):
        if (s not in reference_sources) and (t not in reference_targets):
            continue
        element_type = source_element_type.get(s)
        if element_type is not None:
            counts[element_type]["fp"] += 1
            fp_pairs_by_kind[element_type].append((s, t))

    return counts, fp_pairs_by_kind, fn_pairs_by_kind


# For the possibility that no matches get found -> no tp and fp -> precision and recall would divide by zero
def divide_if_possible_else_zero(a: float, b: float) -> float:
    if b != 0:
        return a / b
    else:
        return 0.0


def print_tp_fp_fn_matrix(counts: dict):
    element_types = ["class", "property", "resource"]
    labels = {"resource": "instance", "class": "class", "property": "property"}
    header = ["type", "tp", "fp", "fn", "precision", "recall", "f1"]
    rows = []

    for k in element_types:
        c = counts.get(k, {"tp": 0, "fp": 0, "fn": 0})
        tp, fp, fn = c["tp"], c["fp"], c["fn"]
        p = divide_if_possible_else_zero(tp, tp + fp)
        r = divide_if_possible_else_zero(tp, tp + fn)
        f1 = divide_if_possible_else_zero(2 * p * r, p + r)
        rows.append([labels[k], tp, fp, fn, f"{p:.2f}", f"{r:.2f}", f"{f1:.2f}"])

    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]

    # Print some separation between columns
    def format_row(r):
        return "   ".join(str(v).ljust(w) for v, w in zip(r, widths))

    print(format_row(header))
    print(format_row(["-" * w for w in widths]))
    for r in rows:
        print(format_row(r))


def process(source, target, reference, matcher_selection_function_id):
    source_dataset, target_dataset = load_datasets_from_path(source, target, add_iri_label=True)

    # Load reference mapping from XML
    ref_matches = []
    with open(reference, "r", encoding="utf-8") as f:
        e1 = None
        for line in f:
            line = line.strip().replace("&amp;", "&")
            if line.startswith("<entity1 rdf:resource="):
                e1 = line.split('"')[1]
            elif line.startswith("<entity2 rdf:resource="):
                e2 = line.split('"')[1]
                ref_matches.append(("<" + e1 + ">", "<" + e2 + ">"))

    ref_matches_id = [(source_dataset.anything_to_id[e1], target_dataset.anything_to_id[e2]) for e1, e2 in ref_matches]

    secea = SECEA(source_dataset, target_dataset, select_matcher_scoring_function_by_id(matcher_selection_function_id))
    matches = secea.get_alignment().get_non_literal_matches()

    counts, fp_pairs_by_kind, fn_pairs_by_kind = compute_tp_fp_fn_by_element_type(
        source_dataset, target_dataset, ref_matches_id, matches
    )
    print_tp_fp_fn_matrix(counts)
    return counts


def run_task(args, matcher_selection_function_id):
    t = time.time()
    ontology_path, reference_path, src, tgt = args
    ref = f"{src.replace('.rdf', '')}-{tgt}"

    counts = process(
        ontology_path + src,
        ontology_path + tgt,
        reference_path + ref,
        matcher_selection_function_id
    )

    return f"{src}  vs  {tgt}", counts, time.time() - t


if __name__ == "__main__":
    matcher_selection_function_id = int(sys.argv[1]) if len(sys.argv) > 1 else None

    ontology_path = r"datasets/OAEI_KG_v4/ontologies/"
    reference_path = r"datasets/OAEI_KG_v4/references/"

    tasks = [
        (ontology_path, reference_path, "marvelcinematicuniverse.rdf", "marvel.rdf"),
        (ontology_path, reference_path, "memoryalpha.rdf", "memorybeta.rdf"),
        (ontology_path, reference_path, "memoryalpha.rdf", "stexpanded.rdf"),
        (ontology_path, reference_path, "starwars.rdf", "swg.rdf"),
        (ontology_path, reference_path, "starwars.rdf", "swtor.rdf"),
    ]

    results = []

    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_task, task, matcher_selection_function_id) for task in tasks]

        for future in as_completed(futures):
            results.append(future.result())

    # Order from within MELT
    desired_order = [
        "memoryalpha.rdf  vs  memorybeta.rdf",
        "starwars.rdf  vs  swtor.rdf",
        "marvelcinematicuniverse.rdf  vs  marvel.rdf",
        "starwars.rdf  vs  swg.rdf",
        "memoryalpha.rdf  vs  stexpanded.rdf",
    ]

    order_index = {name: i for i, name in enumerate(desired_order)}

    results.sort(key=lambda x: order_index.get(x[0], 999))

    print("")
    print("")
    print("FINAL RESULTS:")

    for name, counts, time in results:
        print("")
        print("-" * 80)
        print(name, " time in seconds: ", time)
        print("-" * 80)
        print_tp_fp_fn_matrix(counts)
