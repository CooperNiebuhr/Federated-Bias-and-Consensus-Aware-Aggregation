import os
import json
import torch
import datetime
import csv


def prepare_output_dir(dataset: str, method: str) -> str:
    """
    Create a standardized output directory:
    results/<dataset>/<method>/<timestamp>/
    Returns the absolute path.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = os.path.join("results", dataset, method, timestamp)

    # This auto-creates ALL parent directories
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def save_config(out_dir: str, cfg: dict):
    """Save the run configuration as config.json."""
    config_path = os.path.join(out_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=4)


def save_model(out_dir: str, model):
    """
    Save the trained global model as model.pt.
    (model should be a PyTorch nn.Module)
    """
    path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), path)


def save_metrics(out_dir: str, rounds, train_loss, train_acc):
    """
    Save per-round metrics in a CSV file.

    CSV columns:
        round, train_loss, train_acc
    """
    path = os.path.join(out_dir, "metrics.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round", "train_loss", "train_acc"])
        for r, loss, acc in zip(rounds, train_loss, train_acc):
            writer.writerow([r, loss, acc])


def save_final_results(out_dir: str,
                       final_train_acc: float,
                       final_test_acc: float,
                       final_test_loss: float):
    """
    Save final aggregate results (good for the paper) as JSON.
    """
    data = {
        "final_train_acc": float(final_train_acc),
        "final_test_acc": float(final_test_acc),
        "final_test_loss": float(final_test_loss)
    }
    path = os.path.join(out_dir, "final_results.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
