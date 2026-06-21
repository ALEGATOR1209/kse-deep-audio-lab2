import torch
import matplotlib.pyplot as plt

import torch.nn as nn
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinarySpecificity
)

def fmt_metrics(metrics, order=("acc", "P", "R", "F1", "spec")):
    return " | ".join(f"{k} {metrics[k].item():.3f}" for k in order)

def evaluate_vad(model, x, loss, metrics, opt=None, i=0, device="cpu"):
  x = tuple(t.to(device) for t in x)

  pred, labels, mask = model(x)
  pred, labels = pred[mask], labels[mask]

  lvalue = loss(pred, labels.float())

  if opt:
    lvalue.backward()
    opt.step()
    opt.zero_grad()

  stats = metrics(torch.sigmoid(pred), labels.int())
  if i % 10 == 0 or i == 1:
    print(f"  batch {i:3d} | loss {lvalue.item():.4f} | " + fmt_metrics(stats))

  return lvalue.item()

def plot_history(history):
  xs = range(1, len(history["train_loss"]) + 1)
  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))

  ax1.plot(xs, history["train_loss"], marker="o", label="train")
  ax1.plot(xs, history["test_loss"],  marker="o", label="test")
  ax1.set(title="Loss", xlabel="epoch", ylabel="loss")
  ax1.legend()

  ax2.plot(xs, history["train_acc"], marker="o", label="train")
  ax2.plot(xs, history["test_acc"],  marker="o", label="test")
  ax2.set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
  ax2.legend()

  ax3.plot(xs, history["train_spec"], marker="o", label="train")
  ax3.plot(xs, history["test_spec"],  marker="o", label="test")
  ax3.set(title="Specificity", xlabel="epoch", ylabel="specificity")
  ax3.legend()

  fig.tight_layout()
  return fig


def train_vad(
  vad,
  device,
  lr,
  epochs,
  dataloader_dev,
  dataloader_test,
):
  vad.to(device)

  frac = 0.936
  pos_weight = torch.tensor([(1 - frac) / frac], device=device)
  loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

  opt = torch.optim.Adam(vad.parameters(), lr=lr)

  train_metrics = MetricCollection({
    "acc": BinaryAccuracy(),
    "P":   BinaryPrecision(),
    "R":   BinaryRecall(),
    "F1":  BinaryF1Score(),
    "spec": BinarySpecificity(),
  }).to(device)

  test_metrics = train_metrics.clone().to(device)

  history = {
    "train_loss": [],
    "test_loss": [],
    "train_acc": [],
    "test_acc": [],
    "train_spec": [],
    "test_spec": [],
  }

  for epoch in range(epochs):
    print(f"===== EPOCH {epoch + 1}/{epochs} - train =====")
    vad.train()
    epoch_loss = 0.0
    for i, x in enumerate(dataloader_dev):
      epoch_loss += evaluate_vad(vad, x, loss, train_metrics, opt, i + 1, device)

    train_stats = train_metrics.compute()
    history["train_loss"].append(epoch_loss / len(dataloader_dev))
    history["train_acc"].append(train_stats["acc"].item())
    history["train_spec"].append(train_stats["spec"].item())
    print(f"train | {fmt_metrics(train_stats)}")
    train_metrics.reset()

    print(f"===== EPOCH {epoch + 1}/{epochs} - test =====")
    vad.eval()
    epoch_loss = 0.0
    with torch.no_grad():
      for i, x in enumerate(dataloader_test):
        epoch_loss += evaluate_vad(vad, x, loss, test_metrics, opt=None, i=i + 1, device=device)

    test_stats = test_metrics.compute()
    history["test_loss"].append(epoch_loss / len(dataloader_test))
    history["test_acc"].append(test_stats["acc"].item())
    history["test_spec"].append(test_stats["spec"].item())
    print(f"test  | {fmt_metrics(test_stats)}")
    test_metrics.reset()

  return history
