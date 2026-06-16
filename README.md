# Logical Constraints for One-Class Graph Neural Networks

Bachelor's thesis research on incorporating logical constraints into One-Class Graph Neural Networks (OCGNN) for anomaly detection in graph data.

**Author:** Anastasiia Romanenko

## Overview

This project extends OCGNNs (based on the Deep SVDD one-class classification framework) with logical constraints derived from decision trees. The constraints are injected as differentiable soft logic into the training or inference of graph neural networks.

### Key Concepts

- **OCGNN** — One-Class GNN that learns a hypersphere around normal node/graph embeddings. Anomalies are detected by distance from the center.
- **Logical Constraints** — Decision rules extracted from Gini-based decision trees trained on node/graph features (degree, clustering coefficient, raw features).
- **Two Levels:**
  - **Node-Level** — Predict anomalies per node in a single graph.
  - **Graph-Level** — Predict anomalies per graph in a dataset of graphs.
- **Two Constraint Handlers:**
  - **Fuzzy Handler** — Soft sigmoid-based logic: `a <= b` as `sigmoid(b - a)`, AND as product, OR via De Morgan.
  - **Distance Handler** — Distance to decision boundaries with sigmoid normalization.
- **Two Integration Methods:**
  - **Loss Integration** — Constraints modulate the Deep SVDD loss (add, weight, or suppress).
  - **Inference Integration** — Constraints are used during anomaly score computation at test time.

### Model Variants

| Model | Description |
|---|---|---|
| `SimpleNodeOCGIN` / `SimpleGraphOCGIN` | Baseline OCGIN without constraints | 
| `LOGIC_ADD_*` | Logical constraints added to loss | 
| `LOGIC_WEIGHT_*` | Loss weighted by constraint satisfaction |
| `LOGIC_IGNORE_SUS_*` | Suspicious (high constraint) samples suppressed in loss |
| `LOGIC_INFERENCE_ADD_*` | Constraints used at inference (additive score) | 
| `LOGIC_INFERENCE_WEIGHT_*` | Constraints used at inference (weighted score) | 

## Project Structure

```
thesis/
├── constraints/                          # Logical constraint generation & handling
│   ├── constraints_handlers/             # Fuzzy & Distance constraint handlers
│   │   ├── fuzzy_handler.py              # Soft sigmoid-based logic
│   │   └── distance_handler.py           # Distance-to-boundary scoring
│   ├── graph_decision_trees/             # Decision tree training for constraint extraction
│   │   ├── config.py                     # Node-level feature extractor
│   │   ├── model.py                      # Custom Gini-based decision tree
│   │   ├── filename_utils.py             # Constraint file naming
│   │   └── train_and_print.py            # Train trees and save as JSON
│   └── data/                             # Generated constraint JSON files (gitignored)
├── models/                               # OCGIN model implementations
│   ├── model_registry.py                 # Enum registry of all model variants
│   ├── utils.py                          # Anomaly scoring, evaluation metrics, decision boundaries
│   ├── simple_ocgin/                     # Baseline OCGIN (no constraints)
│   │   ├── node_level_ocgin.py           # NodeOCGIN + training loop
│   │   └── graph_level_ocgin.py          # GraphOCGIN + training loop
│   ├── constraints_in_loss/              # Constraints integrated into loss function
│   │   ├── nl_ocgin.py                   # Node-level constraint-augmented training
│   │   └── gl_ocgin.py                   # Graph-level constraint-augmented training
│   └── constraints_inference/            # Constraints used at inference time
│       ├── nl_ocgin.py                   # Node-level inference constraint models
│       └── gl_ocgin.py                   # Graph-level inference constraint models
├── experiments/                          # Experiment runners & pipelines
│   ├── logging_utils.py                  # Logging setup, file naming
│   ├── the_pipeline.py                   # Master pipeline (end-to-end experiment run)
│   ├── node_level/                       # Node-level experiment code
│   │   ├── data_loader_nl.py             # Dataset loading + train/val/test/tree splits
│   │   ├── run_experiment_nl.py          # Single experiment runner
│   │   └── run_bunch_experiments.py      # Hyperparameter grid search
│   └── graph_level/                      # Graph-level experiment code
│       ├── data_loader_gl.py             # TUDataset loading + splits
│       ├── run_experiment_gl.py          # Single experiment runner
│       └── run_bunch_experiments.py      # Hyperparameter grid search
├── data_analysis/                        # Results analysis & visualization
│   ├── generate_csv.py                   # Flatten JSON results to CSV
│   ├── analysis_plots.py                 # Boxplots, heatmaps, statistical tests
│   └── constraints_from_trees/           # Results & plots from main experiments
│       ├── constraints_from_trees_final.csv  # 27k+ experiment results
│       ├── results/                      # Raw JSON experiment results
│       └── plots/                        # Generated analysis PNGs
├── data/                                 # Benchmark datasets (see Setup)
├── requirements.txt                      # Python dependencies
└── LICENSE                               # MIT License
```

## Datasets

### Node-Level

| Dataset | Source | Domain |
|---|---|---|
| Cora | Planetoid | Citation network |
| CiteSeer | Planetoid | Citation network |
| PubMed | Planetoid | Citation network |
| CS | Coauthor | Co-authorship |
| Photo | Amazon | Product co-purchase |
| Texas | WebKB | Web page |
| Cornell | WebKB | Web page |
| Wisconsin | WebKB | Web page |

### Graph-Level

| Dataset | Source | Domain |
|---|---|---|
| MUTAG | TUDataset | Small molecules |
| AIDS | TUDataset | Small molecules |
| DHFR | TUDataset | Small molecules |
| ENZYMES | TUDataset | Bioinformatics |
| PROTEINS | TUDataset | Bioinformatics |
| COIL-RAG | TUDataset | Computer vision |
| MSRC_21 | TUDataset | Computer vision |

## Setup

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Train a decision tree and generate logical constraints

```bash
python constraints/graph_decision_trees/train_and_print.py
```

Configure the dataset, feature list (`attribute_list`), and tree depth inside the script.

### Run a single experiment

```bash
python experiments/node_level/run_experiment_nl.py
```

Edit the `config` dict at the bottom of the file to change model, dataset, hyperparameters, etc.

### Run a full hyperparameter grid

```bash
python experiments/node_level/run_bunch_experiments.py
```

### Run the end-to-end pipeline

```bash
python experiments/the_pipeline.py
```

This iterates over all datasets, model variants, hyperparameters, and saves results as JSON.

### Analyze results

```bash
# Flatten JSON results to CSV
python data_analysis/generate_csv.py

# Generate plots and statistical analysis
python data_analysis/analysis_plots.py
```

### Constraint Handlers

The constraints are stored as JSON:

```json
{
  "constraints": [
    {
      "conditions": [
        { "feature_index": 0, "op": "<=", "threshold": 0.5 },
        { "feature_index": 205, "op": "<=", "threshold": 0.0 }
      ],
      "predicted_class": 1,
      "meta": { "depth": 2, "y_label_number": "98 v.s. 225" }
    }
  ],
  "additional_attributes": {
    "1433": "node_degree",
    "1434": "clustering_coefficient"
  }
}
```

All conditions in a rule are AND-ed together. 

## Results Summary

The main experiments (`constraints_from_trees`) compare all model variants across 15 datasets with ~1000 hyperparameter combinations each (27k+ total runs). Analysis includes:
- ROC-AUC boxplots per dataset and model variant
- Effect of the constraint scaling factor λ
- Loss-only vs inference-time constraint comparison
- Fuzzy vs distance handler comparison
- Architecture (hidden_dim × layers) heatmaps
- Best-performing configuration per dataset

## License

MIT — see [LICENSE](LICENSE).
