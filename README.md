# Adversarial Attacks on GNNs – Novel Attack Strategy

This repository contains a novel structural attack on Graph Neural Networks (GNNs), focused on evaluating vulnerabilities in node classification tasks using datasets such as Cora and Citeseer.

> **Note:** This `main` branch contains **only our novel attack**. For the implementation of **Metattack**, switch to the [`mettack`](https://github.com/SoniaBorsi/Adversarial-Attack-GNN/tree/mettack) branch, which includes a dedicated README and setup instructions.

---

## Project Structure

```src/
├── init.py
├── attack.py # Novel attack implementation
├── config.py # Configuration handling
├── data.py # Dataset loading and preprocessing
├── layers.py # Custom GNN layers
├── main.py # Script entry point
├── metrics.py # Evaluation metrics
├── model.py # GCN model definition
└── utils.py # Helper utilities

```

```

```

---

## Novel Attack

Our custom attack operates in the following stages:

1. **Hub Selection**: Uses PageRank to rank nodes and select top-𝑘 high-centrality nodes.
2. **Edge Swap**: Rewires the graph by replacing same-class edges with cross-class edges.
3. **Surrogate Loss Scoring**: A one-layer GCN evaluates whether a swap increases classification loss.
4. **Feature Noise**: Adds minor binary feature perturbations for enhanced misclassification.

### Attack Constraints

- Preserves graph degree sequence, connectivity, and total edge count.
- Only targets structurally important nodes for maximum impact.

---

## How to Run

To execute the attack, use the following command:

```
python3 -m src.attack \
  --dataset cora \
  --budget 2000 \
  --p_surrogate 0.2

```

### Parameters

* `--dataset`: Choose between** **`cora` or** **`citeseer`.
* `--budget`: Number of edge swaps (1 swap = 1 delete + 1 add).
* `--p_surrogate`: Probability to accept a swap based on surrogate GCN loss.

Other optional flags:

* `--flip_features`: Add minor feature noise to targeted hubs.
* `--topk_hubs`: Number of top-ranked PageRank nodes used as swap centers.

---

## Evaluation Metrics

* **F1 Score**
* **Attack Success Rate (ASR)**
* **Classification Accuracy Drop (CAD)**
* **Computational Efficiency**

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/SoniaBorsi/Adversarial-Attack-GNN.git
   cd Adversarial-Attack-GNN

   ```

Install dependencies:

```
 pip install -r requirements.txt

```

## Data:

* **Cora** : 2,708 scientific papers, 5,429 citation edges, 7 classes.
* **Citeseer** : 3,312 papers, 4,732 citations, 6 classes.

These are automatically downloaded and cached using** **`torch_geometric`.

## 📂 Other Branches

* [`mettack`](https://github.com/SoniaBorsi/Adversarial-Attack-GNN/tree/mettack): Implements Metattack strategy. Refer to its own README for setup and usage.

## Authors

* Zoe Kenick (Virginia Tech)
* Jiyoon Paik (Virginia Tech)
* Samuel Scalzo (Virginia Tech)
* Sonia Borsi (University of Trento)
