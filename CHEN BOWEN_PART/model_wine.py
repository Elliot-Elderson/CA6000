#!/usr/bin/env python
# coding: utf-8

# # CA6000 Assignment — Wine Quality Prediction
# ## Part B: Model Training & Evaluation (Neural Network)

# ## 1. Setup: Imports & Random Seed
# 
# - **A fixed seed** (`SEED = 4`) controls the whole experiment — the stratified split, the decision-tree baseline and the MLP initialisation/shuffle — so the notebook reproduces exactly.
# - Training runs on **CPU** (no GPU-specific code — all tensors stay on the default device).

# In[24]:


import os
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, average_precision_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score)

try:
    import tabulate
except ImportError:
    get_ipython().run_line_magic('pip', 'install -q tabulate')

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

os.makedirs("figs", exist_ok=True)


# In[25]:


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

SEED = 4  # fixed seed for the split, the tree baseline and the MLP
set_seed(SEED)


# ## 2. Data Loading & Label Construction
# 
# The cleaned dataset (output of Part A) is loaded directly. We first inspect the distribution of the target `quality`, then construct the **3-class label**:
# 
# | Label | Quality range |
# |---|---|
# | low (0) | 3–4 |
# | medium (1) | 5–6 |
# | high (2) | 7–8 |

# In[26]:


DATA_PATH = "winequality-red.csv"
df = pd.read_csv(DATA_PATH)

print("Shape:", df.shape)
print(df.head().to_markdown(index=False))


# In[27]:


dist = df["quality"].value_counts().sort_index().reset_index()
dist.columns = ["quality", "count"]
dist["percentage"] = (dist["count"] / len(df) * 100).round(2)
print(dist.to_markdown(index=False))


# In[28]:


df["quality_3class"] = pd.cut(df["quality"], bins=[0, 4, 6, 8], labels=[0, 1, 2]).astype(int)

label_summary = pd.DataFrame({
    "3-class label": ["low (0)", "medium (1)", "high (2)"],
    "quality range": ["3-4", "5-6", "7-8"],
    "n_samples": [int((df["quality_3class"] == k).sum()) for k in range(3)],
    "percentage": [round((df["quality_3class"] == k).mean() * 100, 2) for k in range(3)],
})
print(label_summary.to_markdown(index=False))


# ## 3. Train / Validation / Test Split (70 / 15 / 15, stratified)
# 
# - **Why a validation set?** It is used for hyper-parameter selection and early stopping; the **test set is held out until the very end** and evaluated exactly once, so the reported numbers are unbiased.
# - **Why stratified?** The classes are imbalanced; `stratify=` keeps the class proportions identical across all three splits.
# - **Why a fixed seed?** The same split is reproduced on every run.

# In[29]:


FEATURE_COLS = [c for c in df.columns if c not in ("quality", "quality_3class")]
X = df[FEATURE_COLS].values
y = df["quality_3class"].values

X_tr, X_rest, y_tr, y_rest = train_test_split(X, y, test_size=0.30, stratify=y, random_state=SEED)
X_val, X_te, y_val, y_te = train_test_split(X_rest, y_rest, test_size=0.50, stratify=y_rest, random_state=SEED)

def class_breakdown(y_arr):
    return f"{int((y_arr == 0).sum())} / {int((y_arr == 1).sum())} / {int((y_arr == 2).sum())}"

split_df = pd.DataFrame({
    "split": ["train", "validation", "test"],
    "n_samples": [len(X_tr), len(X_val), len(X_te)],
    "fraction": ["70%", "15%", "15%"],
    "low / medium / high": [class_breakdown(y_tr), class_breakdown(y_val), class_breakdown(y_te)],
})
print(split_df.to_markdown(index=False))


# ## 4. Preprocessing: Standardisation & DataLoaders
# 
# Features live on very different scales (e.g. `total sulfur dioxide` ≈ 46 vs `density` ≈ 0.997), which hurts gradient-based training.
# 
# - `StandardScaler` transforms each feature to mean 0 and standard deviation 1. We choose z-score normalisation over min-max scaling because several features are right-skewed and min-max scaling is sensitive to extreme values.
# - **Critical detail:** the scaler is **fit on the training set only** and then applied to validation/test. Fitting it on the full dataset would leak information from the test set into preprocessing.
# - The scaled arrays are wrapped into PyTorch `DataLoader`s, which handle batching and shuffling.

# In[30]:


scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_val_s = scaler.transform(X_val)
X_te_s = scaler.transform(X_te)

raw = pd.DataFrame(X_tr, columns=FEATURE_COLS).agg(["mean", "std"]).T
scaled = pd.DataFrame(X_tr_s, columns=FEATURE_COLS).agg(["mean", "std"]).T
scale_check = pd.DataFrame({
    "mean (raw)": raw["mean"].round(3),
    "std (raw)": raw["std"].round(3),
    "mean (scaled)": scaled["mean"].round(4),
    "std (scaled)": scaled["std"].round(4),
})
print(scale_check.to_markdown())


# In[31]:


BATCH_SIZE = 64

def make_loader(X, y, batch_size, shuffle):
    """Wrap numpy arrays into a PyTorch DataLoader (labels: long, for CrossEntropyLoss)."""
    ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                       torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

train_loader = make_loader(X_tr_s, y_tr, BATCH_SIZE, shuffle=True)
val_loader = make_loader(X_val_s, y_val, BATCH_SIZE, shuffle=False)
test_loader = make_loader(X_te_s, y_te, BATCH_SIZE, shuffle=False)

print("Train batches per epoch:", len(train_loader))


# ## 5. Baseline: Decision Tree (scikit-learn)
# 
# Before training a neural network we need a **reference point**. A decision tree is a strong, interpretable non-linear baseline
# 
# **Small hyper-parameter selection:** we try a few `max_depth` values and pick the best on the **validation set** (the test set stays untouched). The tree is fit on the *scaled* features so it shares the exact same input as the MLP.

# In[ ]:


def fit_dt_baseline(X_tr, y_tr, X_val, y_val, depths=(3, 5, 7, 10, None)):
    """Fit decision trees over a small depth grid, select the best on validation."""
    rows, best = [], None
    for d in depths:
        set_seed(SEED)
        dt = DecisionTreeClassifier(max_depth=d, random_state=SEED)
        dt.fit(X_tr, y_tr)
        val_acc = accuracy_score(y_val, dt.predict(X_val))
        rows.append({"max_depth": str(d), "val_accuracy": round(val_acc, 4)})
        if best is None or val_acc > best[0]:
            best = (val_acc, dt)
    print(pd.DataFrame(rows).to_markdown(index=False))
    print(f"-> best max_depth = {best[1].max_depth}, val_acc = {best[0]:.4f}")
    return best[1]

def plot_confusion_matrix(y_true, y_pred, class_names, title, ax=None):
    cm = confusion_matrix(y_true, y_pred)
    if ax is None:
        ax = plt.gca()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax, cbar=False)
    ax.tick_params(labelsize=7)
    ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("True", fontsize=8)
    ax.set_title(title, fontsize=8)


# In[ ]:


print("== Decision Tree baseline (3-class) ==")
dt = fit_dt_baseline(X_tr_s, y_tr, X_val_s, y_val)

preds_dt = dt.predict(X_te_s)
print("\nTest set — classification report (argmax):")
report_dt = classification_report(y_te, preds_dt,
                                  target_names=["low", "medium", "high"], output_dict=True)
print(pd.DataFrame(report_dt).T.round(4).to_markdown())

fig, ax = plt.subplots(figsize=(2.0, 1.7))
plot_confusion_matrix(y_te, preds_dt, ["low", "medium", "high"],
                      "Decision Tree — Test Confusion Matrix", ax=ax)
plt.tight_layout()
plt.savefig("figs/cm_dt_3class.png", bbox_inches="tight", dpi=150)
plt.show()


# ## 6. Main Model: Multi-Layer Perceptron (PyTorch)
# 
# **Architecture:** `11 → 64 → 32 → 3`, with ReLU activations and Dropout (0.3) after each hidden layer.
# 
# **Design choices:**
# 
# - **Small hidden layers (64, 32):** the dataset only has ~1,120 training samples — a large network would simply memorise it. Starting small and regularising is the safer path.
# - **Dropout (0.3):** randomly disables 30% of neurons during training → a strong regulariser against overfitting.
# - **Output layer without activation:** we output **logits** and let `CrossEntropyLoss` apply the softmax internally — numerically more stable than applying it manually.

# In[34]:


class WineQualityMLP(nn.Module):
    """MLP for wine-quality prediction. Outputs raw logits (no final activation)."""

    def __init__(self, input_dim=11, hidden_dims=(64, 32), output_dim=3, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, x):
        return self.net(x)


INPUT_DIM = X_tr_s.shape[1]
model = WineQualityMLP(INPUT_DIM, hidden_dims=(64, 32), output_dim=3, dropout=0.3)
n_params = sum(p.numel() for p in model.parameters())
print("Architecture:", model)
print("Trainable parameters:", n_params)


# ## 7. Loss Function & Optimizer
# 
# **Loss — `CrossEntropyLoss` with sqrt-inverse-frequency class weights:**
# 
# - The classes are extremely imbalanced (low 3.9% / medium 82.5% / high 13.6%): with a plain loss, the model can minimise it almost entirely by learning "medium".
# - Class weights up-weight the rare classes by the inverse of their training frequency. We use the **sqrt** of the inverse frequency: plain inverse frequency would give `[8.48, 0.40, 2.45]` — the 8.5x weight on "low" (only 44 training samples) over-corrects and makes the model collapse towards "low". The sqrt gives the milder `[2.91, 0.64, 1.57]`.
# - Computed **from the training set only**.
# 
# **Optimiser — Adam with weight decay:** Adam adapts the learning rate per parameter; `weight_decay=1e-4` adds L2 regularisation on top of dropout.
# 
# **Scheduler — `ReduceLROnPlateau`:** halves the learning rate when validation loss stops improving for 10 epochs.

# In[35]:


class_counts = np.bincount(y_tr, minlength=3)
class_weights = torch.tensor(np.sqrt(len(y_tr) / (3 * class_counts)), dtype=torch.float32)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

print("class_weights =", np.round(class_weights.cpu().numpy(), 3))


# ## 8. Training Loop
# 
# Each epoch: **(1)** training pass over mini-batches of 64 (zero gradients → forward → weighted loss → backward → update); **(2)** validation pass with `torch.no_grad()` and `model.eval()` (dropout off); **(3)** `ReduceLROnPlateau` step on the validation loss; **(4)** early-stopping check — the best weights (lowest validation loss) are saved, and training stops after `patience` epochs without improvement, restoring them.

# In[36]:


@torch.no_grad()
def evaluate(model, loader, criterion):
    """Return avg loss, accuracy, predictions, labels and softmax probabilities on a loader."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for xb, yb in loader:
        xb, yb = xb.cpu(), yb.cpu()
        logits = model(xb)
        total_loss += criterion(logits, yb).item() * xb.size(0)
        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)
        all_preds.append(preds.cpu()); all_labels.append(yb.cpu()); all_probs.append(probs.cpu())
    return {
        "loss": total_loss / total,
        "acc": correct / total,
        "preds": torch.cat(all_preds).numpy(),
        "labels": torch.cat(all_labels).numpy(),
        "probs": torch.cat(all_probs).numpy(),
    }


# In[37]:


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs, patience, ckpt_path, print_every=10):
    set_seed(SEED)
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_loss, best_epoch, no_improve = float("inf"), 0, 0

    header = f"{'epoch':>5} | {'train_loss':>10} | {'val_loss':>10} | {'train_acc':>9} | {'val_acc':>9} | {'lr':>9}"
    print(header); print("-" * len(header))

    for epoch in range(1, epochs + 1):
        model.train()
        run_loss, correct, total = 0.0, 0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            run_loss += loss.item() * xb.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == yb).sum().item()
            total += yb.size(0)
        train_loss, train_acc = run_loss / total, correct / total

        val = evaluate(model, val_loader, criterion)
        val_loss, val_acc = val["loss"], val["acc"]

        if scheduler is not None:
            scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss); history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss); history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        if epoch % print_every == 0 or epoch == 1:
            print(f"{epoch:>5} | {train_loss:>10.4f} | {val_loss:>10.4f} | {train_acc:>9.4f} | {val_acc:>9.4f} | {lr:>9.2e}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss, best_epoch, no_improve = val_loss, epoch, 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}: no improvement for {patience} epochs.")
                break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    print(f"Done. Best val_loss = {best_val_loss:.4f} at epoch {best_epoch} (best weights restored).")
    return history


# In[38]:


EPOCHS, PATIENCE = 300, 40
CKPT = "models/wine_mlp_3class.pt"

history = train_model(model, train_loader, val_loader, criterion, optimizer,
                      scheduler, EPOCHS, PATIENCE, CKPT)

val_res = evaluate(model, val_loader, criterion)
print(f"Validation: loss = {val_res['loss']:.4f}, accuracy = {val_res['acc']:.4f}, "
      f"macro F1 = {f1_score(val_res['labels'], val_res['preds'], average='macro'):.4f}")


# In[39]:


def plot_history(history, title, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history["train_loss"], label="train", marker="o", ms=3)
    axes[0].plot(history["val_loss"], label="validation", marker="o", ms=3)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss"); axes[0].set_title("Loss curve")
    axes[0].legend()
    axes[1].plot(history["train_acc"], label="train", marker="o", ms=3)
    axes[1].plot(history["val_acc"], label="validation", marker="o", ms=3)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy"); axes[1].set_title("Accuracy curve")
    axes[1].legend()
    fig.suptitle(title)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

plot_history(history, "3-class MLP (class-weighted)", save_path="figs/training_curves.png")


# ## 9. Test Evaluation (default argmax rule)
# 
# The **test set** is used exactly once here, with the default decision rule `argmax` (the Bayes-optimal rule for accuracy). We report:
# 
# - **Accuracy** — overall correctness (misleading alone: the majority-class rule already scores ~82.5%),
# - **Macro-F1** — the average of the per-class F1 scores, where every class counts equally — the headline metric under imbalance,
# 

# In[40]:


test_res = evaluate(model, test_loader, criterion)
print(f"Test loss = {test_res['loss']:.4f} | Test accuracy = {test_res['acc']:.4f} "
      f"| macro F1 = {f1_score(test_res['labels'], test_res['preds'], average='macro'):.4f}")

report = classification_report(test_res["labels"], test_res["preds"],
                               target_names=["low", "medium", "high"], output_dict=True)
print(pd.DataFrame(report).T.round(4).to_markdown())

fig, ax = plt.subplots(figsize=(2.2, 1.9))
plot_confusion_matrix(test_res["labels"], test_res["preds"], ["low", "medium", "high"],
                      "MLP — Test Confusion Matrix", ax=ax)
plt.tight_layout()
plt.savefig("figs/cm_3class.png", bbox_inches="tight", dpi=150)
plt.show()


# ## 10. Operating-Point Tuning: Probability Multipliers
# 
# `argmax` maximises accuracy, but under class imbalance the metric that matters is **macro-F1** — and the two are not maximised by the same operating point.
# 
# The most intuitive way to adjust the operating point of a 3-class model is to **re-weight the class probabilities before taking the argmax**:
# 
# $$\hat{y} = \arg\max_{c} \left( w_c \cdot P(y = c \mid X) \right)$$
# 
# with $w_{medium} = 1$ and $w_{low}, w_{high} \geq 1$ amplifying the rare classes (equivalently, adding $\log w_c$ to the logits). Samples whose probability is slightly below the winner's can thus still "stand out" if they belong to a rare class.
# 
# We **grid-search** $w_{low}, w_{high} \in \{1.0, 1.5, 2.0, 3.0, 4.0\}$ on the **validation set**, maximising macro-F1, and apply the **same procedure to both models** — a fair comparison.

# In[41]:


WEIGHTS = [1.0, 1.5, 2.0, 3.0, 4.0]

def tune_weights(probs_val, y_val):
    """Grid-search probability multipliers for the rare classes on validation (max macro F1)."""
    best_w, best_f1 = (1.0, 1.0), -1.0
    for w_low in WEIGHTS:
        for w_high in WEIGHTS:
            preds = np.argmax(probs_val * np.array([w_low, 1.0, w_high]), axis=1)
            f1m = f1_score(y_val, preds, average="macro")
            if f1m > best_f1:
                best_w, best_f1 = (w_low, w_high), f1m
    return best_w, best_f1

def weighted_predict(probs, w_low, w_high):
    return np.argmax(probs * np.array([w_low, 1.0, w_high]), axis=1)

w_dt, _ = tune_weights(dt.predict_proba(X_val_s), y_val)
w_mlp, _ = tune_weights(val_res["probs"], y_val)
print(f"tuned multipliers: DT w=[{w_dt[0]}, 1.0, {w_dt[1]}] | MLP w=[{w_mlp[0]}, 1.0, {w_mlp[1]}]")


# In[42]:


probs_dt_te = dt.predict_proba(X_te_s)
p_dt = weighted_predict(probs_dt_te, *w_dt)
p_mlp = weighted_predict(test_res["probs"], *w_mlp)

def per_class_metrics(y_true, p):
    return (f1_score(y_true, p, average=None),
            recall_score(y_true, p, average=None),
            precision_score(y_true, p, average=None))

f1_dt_t, rec_dt_t, prec_dt_t = per_class_metrics(y_te, p_dt)
f1_mlp_t, rec_mlp_t, prec_mlp_t = per_class_metrics(y_te, p_mlp)

summary_tuned = pd.DataFrame({
    "metric": ["operating point (tuned on val)",
               "accuracy",
               "macro F1",
               "F1 (low)",
               "recall (low)",
               "precision (low)",
               "F1 (high)",
               "recall (high)",
               "precision (high)"],
    "Decision Tree": [f"w_low={w_dt[0]:.1f}, w_high={w_dt[1]:.1f}",
                      round(accuracy_score(y_te, p_dt), 4),
                      round(f1_score(y_te, p_dt, average="macro"), 4),
                      round(f1_dt_t[0], 4), round(rec_dt_t[0], 4), round(prec_dt_t[0], 4),
                      round(f1_dt_t[2], 4), round(rec_dt_t[2], 4), round(prec_dt_t[2], 4)],
    "MLP": [f"w_low={w_mlp[0]:.1f}, w_high={w_mlp[1]:.1f}",
            round(accuracy_score(y_te, p_mlp), 4),
            round(f1_score(y_te, p_mlp, average="macro"), 4),
            round(f1_mlp_t[0], 4), round(rec_mlp_t[0], 4), round(prec_mlp_t[0], 4),
            round(f1_mlp_t[2], 4), round(rec_mlp_t[2], 4), round(prec_mlp_t[2], 4)],
})
print(summary_tuned.to_markdown(index=False))

f1_dt_argmax = report_dt["macro avg"]["f1-score"]
f1_mlp_argmax = report["macro avg"]["f1-score"]
print(f"\n- Tuned vs plain argmax (test macro-F1): DT {summary_tuned.loc[2, 'Decision Tree']} vs {round(f1_dt_argmax, 4)}; "
      f"MLP {summary_tuned.loc[2, 'MLP']} vs {round(f1_mlp_argmax, 4)}.")


# ## 11. Precision-Recall Curves (per class, one-vs-rest)
# 
# The comparison above uses hard predictions at a single operating point. The **precision-recall curve** shows the full trade-off for each class (one-vs-rest), and its area — the **average precision (AP)** — is a threshold-free measure of ranking quality.
# 
# The markers on each curve show where the two models' **tuned operating points** (Section 10) land. The dotted line is the no-skill baseline (class prevalence).

# In[43]:


class_names = ["low", "medium", "high"]
ap_rows = []

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for k, (name, ax) in enumerate(zip(class_names, axes)):
    y_k = (y_te == k).astype(int)
    prec_d, rec_d, _ = precision_recall_curve(y_k, probs_dt_te[:, k])
    prec_m, rec_m, _ = precision_recall_curve(y_k, test_res["probs"][:, k])
    ap_d = average_precision_score(y_k, probs_dt_te[:, k])
    ap_m = average_precision_score(y_k, test_res["probs"][:, k])
    ap_rows.append({"class": name,
                    "Decision Tree AP": round(ap_d, 4),
                    "MLP AP": round(ap_m, 4)})

    ax.plot(rec_d, prec_d, label=f"Decision Tree (AP = {ap_d:.4f})", linestyle="--")
    ax.plot(rec_m, prec_m, label=f"MLP (AP = {ap_m:.4f})")
    ax.axhline((y_te == k).mean(), color="k", linestyle=":", alpha=0.6)
    ax.scatter([rec_dt_t[k]], [prec_dt_t[k]], marker="o", s=60, color="tab:blue",
               zorder=5, label="DT tuned point")
    ax.scatter([rec_mlp_t[k]], [prec_mlp_t[k]], marker="s", s=60, color="tab:orange",
               zorder=5, label="MLP tuned point")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(f"class: {name}")
    ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figs/pr_curves_3class.png", bbox_inches="tight", dpi=150)
plt.show()

print(pd.DataFrame(ap_rows).to_markdown(index=False))


# ## 12. Use of AI Tools (required by the assignment)
# 
# **Tool:** Doubao (豆包, ByteDance AI assistant). Usage record: <https://www.doubao.com/thread/xgi7I398K1tqDz2SJ>
# 
# **Knowledge-level assistance:**
# 
# - Learned how `ReduceLROnPlateau` lowers the learning rate automatically so training can converge to a better minimum.
# - Asked whether tuning can be performed at inference time — which methods can optimise the decision rule of a 3-class classifier and how to locate the optimal operating point (leading to Section 10).
# - Asked which metrics best reflect model capability on imbalanced datasets (leading to macro-F1 as the headline metric).
# 
# **Code-level assistance:**
# 
# - Generated the code for all tables and figures in this notebook.
# - Generated the code for the data split, standardisation and `DataLoader` construction; supplemented the decision-tree training code (max-depth selection and its print statements); generated the optimiser, the training/evaluation loop, and the implementation of operating-point tuning with probability multipliers (Section 10).
# 
# **Language / polishing assistance:**
# 
# - Translated all Chinese draft content into English for the accompanying report.
# - Explained the concepts of `argmax` and `CrossEntropyLoss`.
# 
# All AI-generated code was reviewed, run and validated by us; the experiment design and the interpretation of the results are our own.

# ## 13. Conclusion (Model Part)
# 
# - The 3-class MLP outperforms the decision-tree baseline on every headline metric: test macro-F1 **0.6347 vs 0.5285** under the default `argmax` rule (Section 9) at equal accuracy (0.8333 vs 0.8333), and it detects the rare "low" class far better (F1 0.4211 vs 0.1818; recall 0.4000 vs 0.1000).
# - On the "high" class the MLP also wins clearly (F1 0.5833 vs 0.5000; recall 0.6562 vs 0.5000).
# - Operating-point tuning (probability multipliers, grid-searched on validation) moves the MLP only slightly (w = [1, 1, 1.5]; macro-F1 0.6347 → 0.6310) while lifting "high" recall (0.6562 → 0.7500); it still beats the tuned tree (w = [3, 1, 3]) on both rare classes (low 0.4211 vs 0.2857, high 0.5854 vs 0.4571).
# - Per-class PR curves / AP confirm the MLP's ranking advantage on **every** class: low 0.3325 vs 0.3212, medium 0.9246 vs 0.9197, and most decisively high 0.5640 vs 0.4090.
