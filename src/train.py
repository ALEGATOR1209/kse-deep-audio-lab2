import torch
import matplotlib.pyplot as plt
import itertools
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryF1Score, BinarySpecificity,
    MultilabelAccuracy, MultilabelPrecision, MultilabelRecall, MultilabelF1Score, MultilabelSpecificity,
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
  print(f"  batch {i:3d} | loss {lvalue.item():.4f} | " + fmt_metrics(stats))

  return lvalue.item()

def pit_bce_loss(loss, logits, targets, mask):
    B, W, T, S = logits.shape
    N = B * W

    logits = logits.reshape(N, T, S)
    targets = targets.reshape(N, T, S)
    mask = mask.reshape(N, T)

    targets = targets.float()
    m = mask.unsqueeze(-1).float()
    denom = mask.float().sum(dim=1).clamp(min=1) * S

    losses = []
    for perm in itertools.permutations(range(S)):
        p = logits[:, :, perm]
        bce = loss(p, targets, reduction='none')
        losses.append((bce * m).sum(dim=(1, 2)) / denom)
    return torch.stack(losses, 0).min(dim=0).values.mean()

def pit_align(loss, pred, targets, mask):
  # pred, targets: (N, T, K) — pred raw logits ; mask: (N, T)
  N, T, K = pred.shape
  perms = list(itertools.permutations(range(K)))
  m = mask.unsqueeze(-1).float()
  costs = torch.stack([
      (loss(pred[:, :, p], targets.float(), reduction='none') * m).sum((1, 2))
      for p in perms
  ], dim=0)                                            # (P, N)
  chosen = torch.tensor(perms, device=pred.device)[costs.argmin(0)]   # (N, K)
  idx = chosen[:, None, :].expand(N, T, K)
  return torch.gather(pred, 2, idx)                    # logits, slots aligned to targets


def evaluate_segmentator(model, x, loss, metrics, opt=None, i=0, device="cpu"):
  x = tuple(t.to(device) for t in x)
  pred, labels, mask = model(x)                        # (B,W,T,K), (B,W,T,K), (B,W,T)

  lvalue = pit_bce_loss(loss, pred, labels, mask)
  if opt:
      lvalue.backward(); opt.step(); opt.zero_grad()

  with torch.no_grad():
      B, W, T, K = pred.shape
      p = pred.reshape(B * W, T, K)
      y = labels.reshape(B * W, T, K)
      mk = mask.reshape(B * W, T).bool()

      aligned = pit_align(loss, p, y, mk)              # reorder pred slots to targets
      valid = mk                                       # (N, T) — drop padded frames
      probs = torch.sigmoid(aligned)[valid]            # (n_valid, K)
      tgt   = y.int()[valid]                           # (n_valid, K)
      stats = metrics(probs, tgt)

  print(f"  batch {i:3d} | loss {lvalue.item():.4f} | " + fmt_metrics(stats))
  return lvalue.item()

def plot_history(history):
  xs = range(1, len(history["train_loss"]) + 1)
  fig, ax = plt.subplots(2, 3, figsize=(12, 4))

  ax[0, 0].plot(xs, history["train_loss"], marker="o", label="train")
  ax[0, 0].plot(xs, history["test_loss"],  marker="o", label="test")
  ax[0, 0].set(title="Loss", xlabel="epoch", ylabel="loss")
  ax[0, 0].legend()

  ax[0, 1].plot(xs, history["train_acc"], marker="o", label="train")
  ax[0, 1].plot(xs, history["test_acc"],  marker="o", label="test")
  ax[0, 1].set(title="Accuracy", xlabel="epoch", ylabel="accuracy")
  ax[0, 1].legend()

  ax[0, 2].plot(xs, history["train_spec"], marker="o", label="train")
  ax[0, 2].plot(xs, history["test_spec"],  marker="o", label="test")
  ax[0, 2].set(title="Specificity", xlabel="epoch", ylabel="specificity")
  ax[0, 2].legend()

  ax[1, 0].plot(xs, history["train_rec"], marker="o", label="train")
  ax[1, 0].plot(xs, history["test_rec"],  marker="o", label="test")
  ax[1, 0].set(title="Recall", xlabel="epoch", ylabel="recall")
  ax[1, 0].legend()

  ax[1, 1].plot(xs, history["train_prec"], marker="o", label="train")
  ax[1, 1].plot(xs, history["test_prec"],  marker="o", label="test")
  ax[1, 1].set(title="Precision", xlabel="epoch", ylabel="precision")
  ax[1, 1].legend()

  ax[1, 2].plot(xs, history["train_f1"], marker="o", label="train")
  ax[1, 2].plot(xs, history["test_f1"],  marker="o", label="test")
  ax[1, 2].set(title="F1 Score", xlabel="epoch", ylabel="F1 score")
  ax[1, 2].legend()

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
    "train_rec": [],
    "test_rec": [],
    "train_prec": [],
    "test_prec": [],
    "train_f1": [],
    "test_f1": [],
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
    history["train_rec"].append(train_stats["R"].item())
    history["train_prec"].append(train_stats["P"].item())
    history["train_f1"].append(train_stats["F1"].item())
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
    history["test_rec"].append(test_stats["R"].item())
    history["test_prec"].append(test_stats["P"].item())
    history["test_f1"].append(test_stats["F1"].item())
    print(f"test  | {fmt_metrics(test_stats)}")
    test_metrics.reset()

  return history

def train_segmentator(
  segmentator,
  device,
  lr,
  epochs,
  dataloader_dev,
  dataloader_test,
):
  segmentator.to(device)

  loss = F.binary_cross_entropy_with_logits
  opt = torch.optim.Adam(segmentator.parameters(), lr=lr)

  K = segmentator.kmax

  train_metrics = MetricCollection({
    "acc":  MultilabelAccuracy(num_labels=K),
    "P":    MultilabelPrecision(num_labels=K),
    "R":    MultilabelRecall(num_labels=K),
    "F1":   MultilabelF1Score(num_labels=K),
    "spec": MultilabelSpecificity(num_labels=K),
  }).to(device)


  test_metrics = train_metrics.clone().to(device)

  history = {
    "train_loss": [],
    "test_loss": [],
    "train_acc": [],
    "test_acc": [],
    "train_spec": [],
    "test_spec": [],
    "train_rec": [],
    "test_rec": [],
    "train_prec": [],
    "test_prec": [],
    "train_f1": [],
    "test_f1": [],
  }

  for epoch in range(epochs):
    print(f"===== EPOCH {epoch + 1}/{epochs} - train =====")
    segmentator.train()
    epoch_loss = 0.0
    for i, x in enumerate(dataloader_dev):
      epoch_loss += evaluate_segmentator(segmentator, x, loss, train_metrics, opt, i + 1, device)

    train_stats = train_metrics.compute()
    history["train_loss"].append(epoch_loss / len(dataloader_dev))
    history["train_acc"].append(train_stats["acc"].item())
    history["train_spec"].append(train_stats["spec"].item())
    history["train_rec"].append(train_stats["R"].item())
    history["train_prec"].append(train_stats["P"].item())
    history["train_f1"].append(train_stats["F1"].item())
    print(f"train | {fmt_metrics(train_stats)}")
    train_metrics.reset()

    print(f"===== EPOCH {epoch + 1}/{epochs} - test =====")
    segmentator.eval()
    epoch_loss = 0.0
    with torch.no_grad():
      for i, x in enumerate(dataloader_test):
        epoch_loss += evaluate_segmentator(segmentator, x, loss, test_metrics, opt=None, i=i + 1, device=device)

    test_stats = test_metrics.compute()
    history["test_loss"].append(epoch_loss / len(dataloader_test))
    history["test_acc"].append(test_stats["acc"].item())
    history["test_spec"].append(test_stats["spec"].item())
    history["test_rec"].append(test_stats["R"].item())
    history["test_prec"].append(test_stats["P"].item())
    history["test_f1"].append(test_stats["F1"].item())
    print(f"test  | {fmt_metrics(test_stats)}")
    test_metrics.reset()

  return history
