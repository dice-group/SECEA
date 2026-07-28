import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from secea.dataset import Dataset
from decorators import timed


def load_datasets_from_path(source_path, target_path, multithread=True, add_iri_label : bool = True) -> tuple[Dataset, Dataset]:
    if multithread:
        with ProcessPoolExecutor(max_workers=2) as executor:
            future_source = executor.submit(load_dataset_from_path, source_path, add_iri_label)
            future_target = executor.submit(load_dataset_from_path, target_path, add_iri_label)

            source_dataset = future_source.result()
            target_dataset = future_target.result()
    else:
        source_dataset = load_dataset_from_path(source_path, add_iri_label)
        target_dataset = load_dataset_from_path(target_path, add_iri_label)
    return source_dataset, target_dataset


def load_dataset_from_path(path, add_iri_label : bool = True) -> Dataset:
    with open(path, "rb") as f:
        raw_bytes = f.read()
    return load_dataset_from_bytes(raw_bytes, Path(path).suffix.lower(), add_iri_label)

def load_evaluation_alignment_from_path(path, source: Dataset, target: Dataset) -> torch.Tensor:
    source_ids = []
    target_ids = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s_uri, t_uri = line.split()
            source_ids.append(source.entity_to_id[s_uri])
            target_ids.append(target.entity_to_id[t_uri])
    return torch.stack((
        torch.tensor(source_ids, dtype=torch.long),
        torch.tensor(target_ids, dtype=torch.long)
    ), dim=1)

@timed
def load_datasets_from_bytes(source_bytes, target_bytes, source_suffix, target_suffix, multithread = True, add_iri_label : bool = True) -> tuple[Dataset, Dataset]:
    if multithread:
        with ProcessPoolExecutor(max_workers=2) as executor:
            future_source = executor.submit(load_dataset_from_bytes, source_bytes, source_suffix, add_iri_label)
            future_target = executor.submit(load_dataset_from_bytes, target_bytes, target_suffix, add_iri_label)

            source_dataset = future_source.result()
            target_dataset = future_target.result()
    else:
        source_dataset = load_dataset_from_bytes(source_bytes, source_suffix, add_iri_label)
        target_dataset = load_dataset_from_bytes(target_bytes, target_suffix, add_iri_label)
    return source_dataset, target_dataset


@timed
def load_dataset_from_bytes(bytes, suffix, add_iri_label : bool = True) -> Dataset:
    text = bytes.decode("utf-8", errors="ignore")
    clean_text = sanitize_rdf_content(text)
    return Dataset(clean_text, suffix, add_iri_label)


@timed
def sanitize_rdf_content(text: str) -> str:
    #This regex part gave errors when parsing the OAEI datasets, so we filter it out
    text = text.replace("\uFFFD", "")
    text = re.sub(
        r'(http://[^/\s"]+):([A-Za-z_])',
        r'\1/\2',
        text
    )
    return text
