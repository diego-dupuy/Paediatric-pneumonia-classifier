"""Evaluate trained models on the test set and compare the experiments.

Usage:
    python -m src.evaluate --experiment weighted_aug   # one experiment
    python -m src.evaluate --compare                   # table with all experiments
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")  # draw figures to files (no screen needed)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, roc_auc_score  # noqa: E402

from .config import CLASS_NAMES, EXPERIMENTS, RESULTS_DIR  # noqa: E402
from .dataset import make_loaders  # noqa: E402
from .model import build_model  # noqa: E402


def load_model(name, device):
    """Rebuild the network and load the weights saved by train.py."""
    model = build_model(pretrained=False)  # same architecture, weights come from the file
    model.load_state_dict(torch.load(RESULTS_DIR / name / "model.pth", map_location=device))
    return model.to(device).eval()  # eval mode: no training behaviour


@torch.no_grad()  # nothing is learnt here, so gradients are not computed
def predict(model, loader, device):
    """Return the real labels, the predicted labels and the probability of pneumonia
    of every image in the loader."""
    labels, preds, probs = [], [], []
    for images, y in loader:
        scores = model(images.to(device))
        labels.extend(y.tolist())
        preds.extend(scores.argmax(1).cpu().tolist())  # highest score wins
        probs.extend(torch.softmax(scores, dim=1)[:, 1].cpu().tolist())  # P(pneumonia)
    return np.array(labels), np.array(preds), np.array(probs)


def clinical_metrics(labels, preds, probs):
    """Metrics from the confusion matrix (positive class = Pneumonia).

    TP: pneumonia detected          FN: pneumonia missed (most dangerous error)
    TN: healthy correctly cleared   FP: healthy flagged as pneumonia (false alarm)
    """
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn)  # share of sick children that are detected
    specificity = tn / (tn + fp)  # share of healthy children correctly cleared
    precision = tp / (tp + fp)  # share of "pneumonia" predictions that are correct
    return {
        "sensitivity": sensitivity, "specificity": specificity, "precision": precision,
        # Accuracy can be misleading with imbalanced classes: always answering
        # "pneumonia" would already give 62 % on this test set.
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
        "f1": 2 * precision * sensitivity / (precision + sensitivity),  # balance of both
        # AUC: probability that a random pneumonia X-ray gets a higher score than a random
        # normal one. Summarises performance over ALL thresholds (0.5 = chance, 1 = perfect).
        "auc": roc_auc_score(labels, probs),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }


def plot_confusion(labels, preds, title, path):
    """Save the 2 x 2 confusion matrix (rows = real class, columns = prediction)."""
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Real label")
    ax.set_title(title)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_history(name, path):
    """Save the training and validation loss per epoch.

    Both curves going down = the model is learning. Training going down while
    validation goes up = overfitting (this is where early stopping acts).
    """
    history = json.loads((RESULTS_DIR / name / "history.json").read_text())
    epochs = [h["epoch"] for h in history]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(epochs, [h["train_loss"] for h in history], label="train")
    ax.plot(epochs, [h["val_loss"] for h in history], label="validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title(f"Training curves ({name})")
    ax.legend()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def evaluate(name, device):
    """Evaluate one experiment and save metrics and figures.

    Metrics are computed on the validation set too: the final model should be CHOSEN
    with validation results, and the test set only REPORTED, so the test numbers are
    not optimistically biased by the choice.
    """
    loaders, _ = make_loaders(EXPERIMENTS[name])
    model = load_model(name, device)
    labels, preds, probs = predict(model, loaders["test"], device)
    metrics = clinical_metrics(labels, preds, probs)
    val = clinical_metrics(*predict(model, loaders["val"], device))
    metrics.update({f"val_{k}": val[k] for k in ("sensitivity", "specificity", "auc")})
    out = RESULTS_DIR / name
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion(labels, preds, f"Test set ({name})", out / "confusion_matrix.png")
    plot_history(name, out / "training_curves.png")
    print(json.dumps(metrics, indent=2))
    return metrics


def compare():
    """Put the test metrics of all evaluated experiments in one table."""
    rows = {}
    for name in EXPERIMENTS:
        path = RESULTS_DIR / name / "metrics.json"
        if path.exists():
            rows[name] = json.loads(path.read_text())
    table = pd.DataFrame(rows).T  # one row per experiment
    table.to_csv(RESULTS_DIR / "comparison.csv")
    print(table.round(3).to_string())
    return table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS)
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()
    if args.compare:
        compare()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluate(args.experiment, device)
