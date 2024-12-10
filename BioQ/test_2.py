import sys
import torch
import scipy

import matplotlib.pyplot as plt

from danq_model import DANQ
from config import device

from sklearn.metrics import auc, f1_score, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score

# Loading Data
print("Loading Data...")
# mat = scipy.io.loadmat('./data/visualization_batch.pt')
# data = mat['testxdata']
# target = mat['testdata']
# data = torch.from_numpy(data).to(dtype=torch.float32)
# target = torch.from_numpy(target).to(dtype=torch.float32)
data, target = torch.load('./data/visualization_batch.pt', weights_only=True)
dataset = torch.utils.data.TensorDataset(data, target)
test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)

batch_precisions = []
batch_recalls = []
batch_f1_scores = []
batch_accuracies = []
all_preds = []
all_labels = []


def iterative_approach(pred, true_labels):
    binary_preds = (pred > 0.5).int()
    binary_labels = true_labels.int()

    # Calculate metrics for this batch
    true_positives = (binary_preds & binary_labels).sum().item()
    false_positives = (binary_preds & (1 - binary_labels)).sum().item()
    false_negatives = ((1 - binary_preds) & binary_labels).sum().item()
    correct_predictions = (binary_preds == binary_labels).sum().item()
    total_predictions = binary_labels.numel()


    precision = true_positives / (true_positives + false_positives + 1e-7)
    recall = true_positives / (true_positives + false_negatives + 1e-7)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
    accuracy = correct_predictions / total_predictions

    # Accumulate batch metrics
    batch_precisions.append(precision)
    batch_recalls.append(recall)
    batch_f1_scores.append(f1_score)
    batch_accuracies.append(accuracy)


def with_auc_scores():
    global all_labels, all_preds
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    plot_roc_curve()
    plot_prc_curve()


def plot_roc_curve():
    roc_auc = roc_auc_score(all_labels.ravel(), all_preds.ravel())
    fpr, tpr, thresholds = roc_curve(all_labels.ravel(), all_preds.ravel())

    print(f"ROC curve Threshold: {thresholds}")
    print(f"ROC AUC Score: {roc_auc:.3f}")

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")  # Diagonal line
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
    plt.savefig("roc_curve.png")


def plot_prc_curve():
    pr_auc = average_precision_score(all_labels.ravel(), all_preds.ravel())
    precision, recall, thresholds = precision_recall_curve(all_labels.ravel(), all_preds.ravel())
    prc_f1_score, prc_auc = f1_score(all_labels.astype(int).ravel(), (all_preds>=0.5).astype(int).ravel()), auc(recall, precision)

    print(f"ROC curve Threshold: {thresholds}")
    print(f"AUC: {prc_auc:.3f}")
    print(f"F1 Score: {prc_f1_score:.3f}")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"PR Curve (AUC = {pr_auc:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
    plt.savefig("prc_curve.png")


def plot_metrics():
    plt.figure(figsize=(12, 6))

    # Precision Plot
    plt.plot(range(1, len(batch_precisions) + 1), batch_precisions, label="Precision", marker="o")
    # Recall Plot
    plt.plot(range(1, len(batch_recalls) + 1), batch_recalls, label="Recall", marker="x")
    # F1-Score Plot
    plt.plot(range(1, len(batch_f1_scores) + 1), batch_f1_scores, label="F1 Score", marker="s")
    plt.plot(range(1, len(batch_accuracies) + 1), batch_accuracies, marker="+", label="Accuracy")

    plt.xlabel("Batch Index")
    plt.ylabel("Metric Value")
    plt.title("Metrics Per Batch")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("batch_metrics.png")


def test(dataloader, model, method="iter"):
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if method == "auc":
                    all_preds.append(pred.cpu())
                    all_labels.append(y.cpu())
            elif method == "iter":
                iterative_approach(pred, y)
            else:
                iterative_approach(pred, y)
    if method == "auc":
        with_auc_scores()
    else:
        plot_metrics()


model = DANQ().to(device)
model.load_state_dict(torch.load(sys.argv[1], weights_only=True, map_location=device))
test(test_dataloader, model, sys.argv[2])

# python test_2.py "./best_model_test_data.pth" "iter"
