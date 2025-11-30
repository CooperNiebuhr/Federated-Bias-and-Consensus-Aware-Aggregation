import os
import json
import torch
import datetime
import csv
import matplotlib.pyplot as plt


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


def save_loss_plot(out_dir, rounds, train_loss, train_accuracy=None):
    """
    Save a PNG plot of training loss (and optionally accuracy) vs. rounds
    into the run's output directory.
    """
    # Loss curve
    plt.figure()
    plt.plot(rounds, train_loss)
    plt.xlabel("Communication Rounds")
    plt.ylabel("Training Loss")
    plt.title("Training Loss vs Communication Rounds")
    plt.tight_layout()
    loss_path = os.path.join(out_dir, "train_loss.png")
    plt.savefig(loss_path)
    plt.close()

    # Optional accuracy curve
    if train_accuracy is not None:
        plt.figure()
        plt.plot(rounds, train_accuracy)
        plt.xlabel("Communication Rounds")
        plt.ylabel("Train Accuracy")
        plt.title("Train Accuracy vs Communication Rounds")
        plt.tight_layout()
        acc_path = os.path.join(out_dir, "train_accuracy.png")
        plt.savefig(acc_path)
        plt.close()
