Here's a full critique of your thesis project directory:

## Project Identity

A bachelor's thesis on **One-Class Graph Neural Networks with Logical Constraints for Graph Anomaly Detection**. The core idea: OCGIN (One-Class GIN) with Deep SVDD objective + decision-tree-extracted logical constraints injected as soft loss terms. Node-level and graph-level variants across many datasets. Serious work, recognizable architecture.

## Criticisms

### Naming & Consistency

1. **"constrains" is everywhere** — directory names, filenames, class names, variables. If it's a joke, it's committed deeply enough to confuse anyone reading your code or thesis. Decide: fix it or own it.

2. **Inconsistent directory naming style**: `info_n_literature`, `sweat_n_tears` (informal/incomplete), vs. `constrains`, `experiments`, `wrapping_data` (descriptive but not self-explanatory). `wrapping_data` doesn't clearly communicate "experiment results + analysis."

3. **Mixed German/English** in comments and documentation. Pick one for a thesis codebase.

### Project Structure

4. **No `__init__.py` files** in `constrains/`, `experiments/`, `models/`, `wrapping_data/` — these aren't proper Python packages. Imports probably rely on `PYTHONPATH` or running from root, which is fragile.

5. **No `setup.py` / `pyproject.toml`** — the project isn't installable. Anyone reproducing this needs to replicate your exact working directory setup.

6. **Hardcoded absolute paths** (e.g., `/home/ana/stuff/1uni/thesis/...` in `logging_utils.py`). Zero portability.

7. **Redundant nesting**: `data/AIDS/AIDS/raw/`, `data/Cora/Cora/processed/` — the dataset name appears twice. This comes from TUDataset conventions but you could symlink or restructure.

### Code Duplication

8. **`models/constrains_in_loss/` vs `models/constrains_inference/`** — near-identical model files with minor differences in where constraints apply. This is textbook copy-paste inheritance. Should be a base class with a flag or strategy pattern.

9. **`constrains/graph_decision_trees/node_level/` and `graph_level/`** — highly duplicated configs, filename utils, template configs. The `config.py` and `filename_utils.py` in each differ only slightly. Refactor into shared code.

### Dependency Management

10. **Two `requirements.txt` files** — root one (pinned, modern) and `GraphLevel-AnomalyDetection/requiresments.txt` (typo intentional? "requiresments"). Old fork deps may conflict with modern ones.

11. **`GraphLevel-AnomalyDetection/` is a full fork** of Bosch GLAD (AGPL-3.0) checked into your repo. It's 50+ files and its own `RESULTS/` directory. Consider whether you need the full fork or just the models you adapted.

### Documentation

12. **`what_do_i_do.md`** is 710 lines of unstructured stream-of-consciousness. Valuable for you, impenetrable for anyone else. Contains rants ("ALRIGHT MOTHERFUCKER AHDHSADJA"), debugging notes, TODOs, and meeting notes. This should be split: a proper technical notebook (maybe a Jupyter notebook or organized markdown) and a separate personal log.

13. **`README.md`** is a single line: `# ocgnn_logical_constrains`. For a thesis project that will be evaluated, this is unacceptable. Should explain what the project is, how to set it up, how to run experiments, and how results are organized.

### Testing & Reproducibility

14. **Zero tests.** No `tests/` directory despite `pytest` being in requirements. For a research project this is somewhat excusable, but at minimum the constraint handlers and model forward passes should have smoke tests.

15. **`structure.png`** is a static image documenting the directory tree — a maintenance nightmare since it will inevitably become out of date.

### Version Control

16. **7 git branches** with names like `fixiny_fix`, `experimental_results`, `experiments_analyse` suggests unplanned development. The `main` branch may not represent the final coherent state.

### The Good

- Clear separation of concerns: models, constraints, experiments, results, writing — the high-level organization is actually decent.
- The constraint generation pipeline (decision trees → JSON → fuzzy handlers → loss) is well-thought-out architecturally.
- Comprehensive literature collection in `info_n_literature/literature/`.
- The LaTeX thesis in `sweat_n_tears/ba/main.tex` is substantial (1637 lines) — the actual writing is happening.
- Result JSONs and analysis plots are organized by experiment group.

### Priority Fixes

1. `README.md` — make it useful
2. Hardcoded paths → relative paths / config
3. `__init__.py` files for proper packaging
4. Remove or shrink `GraphLevel-AnomalyDetection/` fork dependency
5. Consolidate duplicated model code with base classes
6. `what_do_i_do.md` → split into organized docs
