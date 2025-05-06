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

### 2. Run Nettack (Targeted Attack)

1. **Ensure** `nettack.py` is present in the repository root.
2. **Import** in `main.py`:

   ```python
   from nettack import apply_nettack
   ```
3. **Configure** the attack inside the `run_experiment` loop in `main.py`:

   ```python
   # Instantiate and train the surrogate model
   attack_model = train_model(GCN(data.num_features, 16, num_classes), data)
   attack_model.eval()

   # Select a target node (e.g., the first test node)
   test_nodes = data.test_mask.nonzero(as_tuple=False).view(-1)
   target = int(test_nodes[0])

   # Apply Nettack to the target node
   poisoned_data = apply_nettack(
       model=attack_model,
       data=data,
       target_node=target,
       n_perturbations=num_perturbations,
       attack_structure=True,
       attack_features=False
   )
   ```
4. **Run**:

   ```bash
   python main.py
   ```

Outputs (models, metrics, and visualizations) will be saved in:

* `poisoned_models/`
* `results/eval_results.txt`
* `visuals/` (graph images)
* `acc_boxplots/`

---

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
