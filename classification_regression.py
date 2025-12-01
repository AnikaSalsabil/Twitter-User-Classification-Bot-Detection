# ============================================================
# DRY text classification pipeline for Twitter gender dataset
# - Models: MultinomialNB, LogisticRegression, DecisionTree
# - Features: TF-IDF(description + text) + MinMax-scaled numeric
# - Outputs: per-model reports/CM (CSV+PNG), combined metrics CSV,
#            per-class long-form CSV, comparison plots
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.sparse import hstack
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

# ------------------------ Config ----------------------------
DATA_PATH = "./out/df_cleaned.csv"       # cleaned dataset from your previous step
OUT_DIR   = "./out"                      # change to os.path.expanduser("~/out") if you prefer home dir
SEED      = 42
TEST_SIZE = 0.20
TFIDF_MAX_FEATURES = 5000
NUMERIC_COLS = ["gender:confidence"]     # extend here if you add more numeric attrs

# ---------------- Ensure output folder ----------------------
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------- Load data ---------------------------
df = pd.read_csv(DATA_PATH)

# Combine text fields (description + tweet text)
df["combined_text"] = df["description"].astype(str) + " " + df["text"].astype(str)
X_text  = df["combined_text"]
X_extra = df[NUMERIC_COLS].fillna(0)
y       = df["gender"]

# -------------------- Train/Test split ----------------------
X_text_tr, X_text_te, X_extra_tr, X_extra_te, y_tr, y_te = train_test_split(
    X_text, X_extra, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
)

# ----------------- Vectorize + Scale (DRY) ------------------
vectorizer = TfidfVectorizer(stop_words="english", max_features=TFIDF_MAX_FEATURES)
X_tr_tfidf = vectorizer.fit_transform(X_text_tr)
X_te_tfidf = vectorizer.transform(X_text_te)

scaler = MinMaxScaler()
X_tr_extra = scaler.fit_transform(X_extra_tr)
X_te_extra = scaler.transform(X_extra_te)

# Sparse hstack: text + numeric
X_tr_final = hstack([X_tr_tfidf, X_tr_extra])
X_te_final = hstack([X_te_tfidf, X_te_extra])

# Sorted label list for consistent confusion matrices
labels = sorted(y.unique())

# ----------------- Reusable evaluation ----------------------
def train_and_evaluate(model, name, Xtr, ytr, Xte, yte, labels, out_dir=OUT_DIR):
    model.fit(Xtr, ytr)
    y_pred = model.predict(Xte)

    # Core aggregate metrics (macro + weighted + accuracy)
    scores = {
        "model": name,
        "accuracy": accuracy_score(yte, y_pred),
        "precision_macro": precision_score(yte, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(yte, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(yte, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(yte, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(yte, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(yte, y_pred, average="weighted", zero_division=0),
    }

    # Full classification report -> CSV
    report = classification_report(yte, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_csv = os.path.join(out_dir, f"classification_report_{name}.csv")
    report_df.to_csv(report_csv, index=True)

    # Also return per-class metrics in a tidy format for a combined CSV later
    per_class_rows = []
    for cls in labels:
        if cls in report:
            per_class_rows.append({
                "model": name,
                "class": cls,
                "precision": report[cls].get("precision", 0.0),
                "recall": report[cls].get("recall", 0.0),
                "f1": report[cls].get("f1-score", 0.0),
                "support": report[cls].get("support", 0),
            })

    # Confusion matrix: CSV + PNG
    cm = confusion_matrix(yte, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels],
                            columns=[f"pred_{l}" for l in labels])
    cm_csv = os.path.join(out_dir, f"confusion_matrix_{name}.csv")
    cm_df.to_csv(cm_csv, index=True)

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(labels)), labels=labels, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)
    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    cm_png = os.path.join(out_dir, f"confusion_matrix_{name}.png")
    plt.savefig(cm_png, dpi=150)
    plt.close()

    return scores, per_class_rows

# ---------------------- Models (DRY) ------------------------
models = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(
        solver="saga", multi_class="auto", max_iter=2000, n_jobs=-1, random_state=SEED
    ),
    "decision_tree": DecisionTreeClassifier(
        max_depth=None, random_state=SEED
    )
}

# ---------------- Train/Eval all (loop) ---------------------
all_scores = []
all_per_class = []

for name, mdl in models.items():
    scores, per_class_rows = train_and_evaluate(
        mdl, name, X_tr_final, y_tr, X_te_final, y_te, labels, out_dir=OUT_DIR
    )
    all_scores.append(scores)
    all_per_class.extend(per_class_rows)

# ----------------- Save combined metrics --------------------
scores_df = pd.DataFrame(all_scores)
scores_df_path = os.path.join(OUT_DIR, "model_scores_overview.csv")
scores_df.to_csv(scores_df_path, index=False)

per_class_df = pd.DataFrame(all_per_class)
per_class_df_path = os.path.join(OUT_DIR, "per_class_metrics_long.csv")
per_class_df.to_csv(per_class_df_path, index=False)

print("Model scores overview:")
print(scores_df)

# ------------------ Comparison plots ------------------------
def save_bar(metric_key, title, filename):
    plt.figure()
    plt.bar(scores_df["model"], scores_df[metric_key])
    plt.title(title)
    plt.xlabel("Model")
    plt.ylabel(metric_key.replace("_", " ").title())
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=150)
    plt.close()

save_bar("f1_macro",     "Macro F1 by Model",     "macro_f1_by_model.png")
save_bar("accuracy",     "Accuracy by Model",     "accuracy_by_model.png")
save_bar("f1_weighted",  "Weighted F1 by Model",  "weighted_f1_by_model.png")

print(f"All artifacts saved in: {OUT_DIR}")
