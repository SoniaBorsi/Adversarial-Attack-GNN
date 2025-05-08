
# Adversarial-Attack-GNN-nettack

This repository provides an implementation of a targeted adversarial attack on Graph Neural Networks (GNNs) using **Nettack**. Nettack perturbs the local neighborhood of a chosen node to degrade the model's classification performance on that node.

The codebase includes scripts to:

* Train a clean GCN on standard graph datasets (e.g., Cora).
* Apply Nettack to a specific target node.
* Retrain the GCN on the perturbed graph.
* Evaluate performance before and after the attack.
* Visualize the original and perturbed graphs.

---

## Prerequisites

* **Operating System**: Linux, macOS, or Windows with WSL
* **Python**: 3.8
* **Conda**: Miniconda or Anaconda

---

## Installation

1. **Clone or unzip** this repository into a folder, e.g.: `~/projects/Adversarial-Attack-GNN-nettack/`

2. **Open a terminal** and navigate to the project directory:

   ```bash
   cd ~/projects/Adversarial-Attack-GNN-nettack
   ```

3. **Create and activate** a Conda environment:

   ```bash
   conda create -n gnn_attack python=3.8 -y
   conda activate gnn_attack
   ```

4. **Install dependencies**:

   ```bash
   pip install torch torchvision torchaudio
   pip install torch-scatter torch-sparse torch-spline-conv -f \
     https://data.pyg.org/whl/torch-$(python -c "import torch;print(torch.__version__) ")+cpu.html
   pip install torch-geometric deeprobust networkx ogb matplotlib seaborn
   ```

---

## Usage

### 1. Train a clean GCN

By default, the script trains on the **Cora** dataset. To select a different dataset, edit the `datasets` list in `main.py`.

```bash
python main.py
```

This will:

1. Load and preprocess the dataset.
2. Train a clean GCN and save its weights under `clean_models/`.
3. Evaluate and log clean performance.
4. Visualize the original graph (if ≤ 5 000 nodes) in `visuals/` as `<dataset>_clean.png`.

### 2. Run Globalized Nettack (8% budget)

The code now:

1. Samples **8 % of the test nodes** as targets (minimum 1).
2. Splits the total budget (8 % of all edges) evenly across these targets.
3. Sequentially applies Nettack (structure + feature flips) to each target on the **same** graph, accumulating perturbations.
4. Dumps each per‐target perturbed graph and features to `perturbed_data/` as `<dataset>_perturbed<budget>_<target>.pt`.
5. Retrains and evaluates one final poisoned GCN on the fully perturbed graph:

   * Saves the final poisoned model checkpoint to `poisoned_models/<dataset>_poisoned<budget>_final.pt`.
   * Logs **global metrics** (ΔACC, Precision, Recall, F1) to `results/eval_results.txt` using the existing `compare_results` / `avg_std`.
   * Visualizes the final poisoned graph (if ≤ 5 000 nodes) in `visuals/` as `<dataset>_poisoned_final.png`.
6. Computes **targeted metrics** for Nettack:

   * **Targeted Success Rate**: fraction of attacked test nodes whose predicted label flipped.
   * **Average Confidence Drop**: average drop in the model’s probability on the true class for each target.
     These are appended to `results/eval_results.txt` at the end of each experiment run.

To execute:

```bash
python main.py
```

After completion, inspect:

* `clean_models/` for the clean GCN weights.
* `perturbed_data/` for per‐target perturbed graph dumps.
* `poisoned_models/` for the final poisoned GCN checkpoint.
* `visuals/` for original and final graph images.
* `acc_boxplots/` for global ΔACC plots.
* `results/eval_results.txt` for both global and targeted metrics.

## Repository Structure

```
Adversarial-Attack-GNN-nettack/
├── clean_models/            # Saved clean GCN weights
├── poisoned_models/         # Saved poisoned GCN weights after Nettack
├── visuals/                 # Original & perturbed graph images
├── acc_boxplots/            # Accuracy comparison plots
├── results/                 # Evaluation logs and metrics
├── datasets.py              # Data loading, masking, and patching utilities
├── model.py                 # GCN definition and training code
├── nettack.py               # Nettack implementation wrapper
├── evaluation.py            # Evaluation metrics & logging
├── visualization.py         # Plotting utilities
├── constants.py             # Dataset constants and paths
├── main.py                  # Orchestrates training, Nettack attack, retraining, and eval
├── README.md                # This file
└── environment.yml          # (optional) Conda environment specification
```

---

## Results & Visualizations

* **Original Graph**: `visuals/<dataset>_clean.png`
* **Perturbed Graph**: `visuals/<dataset>_poisoned_run<i>.png`
* **Accuracy Plots**: `acc_boxplots/<dataset>_acc.png`
* **Metrics Log**: `results/eval_results.txt`

---

## References

* Zügner, D., Akbarnejad, A., & Günnemann, S. (2018). Adversarial Attacks on Neural Networks for Graph Data. In Proceedings of the 24th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD ’18), 2847–2856. DOI: https://doi.org/10.1145/3219819.3220078. (Also available as arXiv:1805.07984 [stat.ML], v4, December 9, 2021)