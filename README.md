# FedBaC: Federated Bias-and-Consensus Aware Learning (PyTorch)

This repository began as a fork fro a PyTorch implementation of **Federated Averaging (FedAvg)** from *Communication-Efficient Learning of Deep Networks from Decentralized Data* (McMahan et al.), and has since been extended with my own research contributions under the **FedBaC** framework. This began as a course project but has been slightly extended. The project report detailing the mathematical formulation, metrics, ablation study, and preliminary results can be viewed here: [Project Report](report.pdf)


FedBaC augments standard federated learning with **directional consensus**, **reliability-aware aggregation**, and an optional **client-side alignment regularizer**, targeting improved stability and performance under heterogeneous (non-IID) data. On non-IID MNIST, FedBaC improves final accuracy by ~6% over FedAvg.

The codebase supports reproducible experiments on **MNIST** and **CIFAR-10** using **Dirichlet-based non-IID partitioning**.

---

## Requirements

Install dependencies from `requirements.txt`:

* Python 3.8+
* PyTorch
* Torchvision
* NumPy
* Matplotlib
* TensorBoardX

The listed versions in `requirements.txt` reflect the original environment; newer PyTorch versions should work but have not been exhaustively tested. (I ended up running my experiments on Python 3.11.7)

---

## Data

* Datasets are automatically downloaded via `torchvision` if not present.
* Supported datasets:

  * MNIST
  * CIFAR-10

### Non-IID Partitioning

This repository **uses Dirichlet label-skew partitioning** for non-IID experiments. Legacy IID / equal-split non-IID logic has been superseded.

Key controls:

* `--dirichlet 1` enables Dirichlet partitioning
* `--alpha` controls heterogeneity (smaller = more skewed)
* `--partition_seed` ensures reproducible client datasets

---

## Running Experiments

All experiment configuration is handled via command-line arguments (see `options.py`).

### Baseline (Centralized Training)

The baseline trains a single model conventionally on the full dataset.

Example (MNIST, MLP, CPU):

```
python src/baseline_main.py \
  --dataset mnist \
  --model mlp \
  --epochs 10 \
  --seed 42
```

Example (CIFAR-10, CNN, GPU):

```
python src/baseline_main.py \
  --dataset cifar \
  --model cnn \
  --epochs 20 \
  --gpu \
  --gpu_id 0 \
  --seed 42
```

---

### Federated Baseline (FedAvg)

Runs standard federated averaging with Dirichlet non-IID splits.

```
python src/baseline_main.py \
  --dataset mnist \
  --model cnn \
  --epochs 30 \
  --num_users 50 \
  --frac 0.2 \
  --local_ep 5 \
  --dirichlet 1 \
  --alpha 0.5 \
  --partition_seed 123 \
  --sample_seed 456 \
  --seed 999 \
  --gpu \
  --gpu_id 0
```

---

### FedBaC

FedBaC extends FedAvg with consensus- and reliability-aware aggregation. Additional hyperparameters control alignment strength, server momentum, and reliability estimation.

Example (FedBaC with consensus regularization and reliability weighting):

```
python src/fedbac_main.py \
  --dataset cifar \
  --model cnn \
  --epochs 30 \
  --num_users 50 \
  --frac 0.2 \
  --local_ep 5 \
  --dirichlet 1 \
  --alpha 0.5 \
  --partition_seed 123 \
  --sample_seed 456 \
  --seed 999 \
  --lambda_reg 1e-5 \
  --gamma 2.0 \
  --beta 0.9 \
  --reliab_window 5 \
  --reliab_alpha 1.0 \
  --gpu \
  --gpu_id 0
```

You can disable individual components (e.g., consensus regularization or reliability weighting) by setting the corresponding hyperparameters to zero, enabling ablation-style experiments.

---

## Configuration

All available arguments and default values are defined in `options.py`. Important categories include:

* Dataset and model selection
* Federated training parameters
* Dirichlet partition controls
* Random seeds for reproducibility
* FedBaC-specific hyperparameters

For reproducibility, you can set the three seeds:

* `--seed` (model initialization)
* `--partition_seed` (client data splits)
* `--sample_seed` (client sampling per round)

---

## License

MIT License
