# SECEA: Self Configuring Matcher Framework for Entity and Knowledge Graph Alignment

This repository contains a Python implementation of the paper:
 **SECEA: Self Configuring Matcher Framework for Entity and Knowledge Graph Alignment**.

## Requirements

- Python **3.10** or **3.11**

## Installation

Clone this repository and install the dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Setup

### 1. OAEI Knowledge Graph Datasets

Download the OAEI Knowledge Graph datasets from:
<https://oaei.ontologymatching.org/2025/knowledgegraph/index.html>.
Place the downloaded files in the following directory:
`
datasets/OAEI_KG_v4/
`.
For example, the SWG knowledge graph should be located at:
`
datasets/OAEI_KG_v4/ontologies/swg.rdf
`

### 2. OpenEA Entity Alignment Dataset

Download the OpenEA dataset from:
<https://github.com/nju-websoft/OpenEA>.
Place the dataset in the following directory:
`
datasets/OpenEA_dataset_v1.1/
`.
For example, the ent_links file for the `EN_FR_100K_V1` dataset should be located at:
`
datasets/OpenEA_dataset_v1.1/EN_FR_100K_V1/ent_links
`.
Then, convert the OpenEA dataset into the format required by SECEA:

```bash
python datasets/OpenEA_dataset_v1.1/OpenEAConverter.py
```

### Expected Dataset Structure

```text
datasets/
├── OAEI_KG_v4/
│   └── ontologies/
│       └── swg.rdf
└── OpenEA_dataset_v1.1/
    ├── OpenEAConverter.py
    └── EN_FR_100K_V1/
        └── ent_links
```

## Running Experiments

### Knowledge Graph Alignment Experiments

To run the OAEI Knowledge Graph Alignment experiments:

```bash
python main_oaei_local.py
```

### Entity Alignment Experiments

To run the OpenEA Entity Alignment experiments:

```bash
python main_openea.py
```

To allow running experiments with other datasets as well, we provide a MELT (<https://dwslab.github.io/melt/http-definition>) compatible web interface on port 8080:
```bash
python main_web.py
```
