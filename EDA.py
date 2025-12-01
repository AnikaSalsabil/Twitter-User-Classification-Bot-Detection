# ===============================================
# EXPLORATORY DATA ANALYSIS (EDA) for Twitter Dataset
# Human vs Non-Human Classification (gender: male/female vs brand)
# ===============================================
# Outputs:
# - Figures saved to ./figures/*.png
# - Text summary saved to ./figures/eda_findings.txt

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# -----------------------------
# Setup
# -----------------------------
sns.set(style="whitegrid")
FIG_DIR = "./out/figures"
os.makedirs(FIG_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")

def clean_text(s):
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", " ", s)          # remove links
    s = re.sub(r"[@#]\w+", " ", s)                   # remove mentions/hashtags
    s = re.sub(r"[^a-z\s]", " ", s)                  # keep letters/spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

# -----------------------------
# 1) Load + Quick Overview  (Aspect 1)
# -----------------------------
# Note: latin1 handles non-UTF8 chars
df = pd.read_csv("twitter_user_data.csv", encoding="latin1", low_memory=False)

print("Dataset Shape:", df.shape)
print("Columns:", df.columns.tolist())

# Persist quick stats
overview_lines = []
overview_lines.append(f"Shape: {df.shape}")
overview_lines.append("Columns: " + ", ".join(df.columns.astype(str).tolist()))
overview_lines.append("\nMissing values per column:\n" + str(df.isnull().sum().sort_values(ascending=False)))
overview_lines.append("\nDtypes:\n" + str(df.dtypes))

with open(os.path.join(FIG_DIR, "eda_findings.txt"), "w", encoding="utf-8") as f:
    f.write("### EDA Findings Summary\n\n")
    f.write("## Overview\n")
    f.write("\n".join(overview_lines) + "\n\n")

# A general numeric snapshot (pairplot) for Aspect 1 (if numeric columns exist)
candidate_num = ["tweet_count", "retweet_count", "fav_number"]
num_cols = [c for c in candidate_num if c in df.columns]

if len(num_cols) >= 2:
    sample_df = df[num_cols].dropna()
    if len(sample_df) > 2000:
        sample_df = sample_df.sample(2000, random_state=42)
    g = sns.pairplot(sample_df, kind="scatter", diag_kind="kde", plot_kws={"alpha":0.6, "s":20, "edgecolor":"k"})
    g.fig.suptitle("Aspect 1: General Pair Plot of Numerical Features", y=1.02, fontsize=12)
    savefig("01_overview_pairplot_numeric.png")

# -----------------------------
# 2) Target Distribution (Aspect 2)
# -----------------------------
if "gender" in df.columns:
    plt.figure(figsize=(6,4))
    ax = sns.countplot(data=df, x="gender", order=df["gender"].value_counts().index, palette="Set2")
    plt.title("Distribution of Gender (Target Variable)")
    plt.xlabel("gender (male/female/brand)")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x()+p.get_width()/2, p.get_height()),
                    ha='center', va='bottom', fontsize=9, rotation=0)
    savefig("02_target_distribution_gender.png")

# -----------------------------
# 3) Numerical Feature Distributions (Aspect 3)
# -----------------------------
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col].dropna(), kde=True, bins=30, color="skyblue")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col); plt.ylabel("Frequency")
    savefig(f"03a_dist_{col}.png")

# Pairplot for Aspect 2 & 3 combined (numeric vs gender)
if "gender" in df.columns and len(num_cols) >= 2:
    sub = df[num_cols + ["gender"]].dropna()
    if len(sub) > 2000:
        sub = sub.sample(2000, random_state=42)
    g = sns.pairplot(sub, hue="gender", diag_kind="kde", palette="Set2",
                     plot_kws={"alpha":0.6, "s":30})
    g.fig.suptitle("Aspect 2&3: Pair Plots of Numerical Features by Gender", y=1.02, fontsize=12)
    savefig("03b_pairplot_numeric_by_gender.png")

# -----------------------------
# 4) Boxplots by Gender (Aspect 4)
# -----------------------------
if "gender" in df.columns:
    for col in num_cols:
        plt.figure(figsize=(6,4))
        sns.boxplot(data=df, x="gender", y=col, palette="Set3")
        plt.title(f"{col} by Gender")
        savefig(f"04a_box_{col}_by_gender.png")

    # Scatter relationships by gender (complements boxplots)
    if len(num_cols) >= 2:
        sub = df[num_cols + ["gender"]].dropna()
        if len(sub) > 2000:
            sub = sub.sample(2000, random_state=42)
        g = sns.pairplot(sub, hue="gender", kind="scatter", palette="Set3",
                         plot_kws={"alpha":0.6, "s":30})
        g.fig.suptitle("Aspect 4: Scatter Relationships of Activity Features by Gender", y=1.02, fontsize=12)
        savefig("04b_pairplot_scatter_by_gender.png")

# -----------------------------
# 5) Correlation Heatmap (Aspect 5)
# -----------------------------
if len(num_cols) >= 2:
    plt.figure(figsize=(7,5))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", vmin=-1, vmax=1)
    plt.title("Correlation Heatmap of Numerical Features")
    savefig("05_corr_heatmap_numeric.png")

    # Pairwise regressions (complements heatmap)
    sub = df[num_cols].dropna()
    if len(sub) > 2000:
        sub = sub.sample(2000, random_state=42)
    g = sns.pairplot(sub, kind="reg", plot_kws={"line_kws": {"color": "red"}, "scatter_kws": {"s":15, "alpha":0.5}})
    g.fig.suptitle("Aspect 5: Pairwise Regressions of Numerical Features", y=1.02, fontsize=12)
    savefig("05b_pairplot_reg_numeric.png")

# -----------------------------
# 6) Missing Data Heatmap (Aspect 6)
# -----------------------------
plt.figure(figsize=(10,5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Missing Data Heatmap")
savefig("06_missing_data_heatmap.png")

# -----------------------------
# 7) Categorical: Top Profile Colors (Aspect 7)
# -----------------------------
for col in ["link_color", "sidebar_color"]:
    if col in df.columns:
        plt.figure(figsize=(9,4))
        df[col].value_counts().head(15).plot(kind="bar", color="coral")
        plt.title(f"Top 15 Most Common {col}")
        plt.xlabel(col); plt.ylabel("Count")
        savefig(f"07a_top15_{col}.png")

# OPTIONAL pairplots: numeric vs color (only if unique colors are low enough to be meaningful)
for col in ["link_color", "sidebar_color"]:
    if col in df.columns and len(num_cols) >= 2:
        # Limit hue categories to top 6 to keep plots readable
        top_vals = df[col].value_counts().head(6).index.tolist()
        sub = df[df[col].isin(top_vals)][num_cols + [col]].dropna()
        if len(sub) > 1500:
            sub = sub.sample(1500, random_state=42)
        if len(sub) > 20 and sub[col].nunique() > 1:
            g = sns.pairplot(sub, hue=col, diag_kind="kde", plot_kws={"alpha":0.6, "s":25})
            g.fig.suptitle(f"Aspect 7: Pair Plot of Activity Features by {col} (Top 6)", y=1.02, fontsize=12)
            savefig(f"07b_pairplot_numeric_by_{col}.png")

# -----------------------------
# 8) Text Exploration (Word Clouds) (Aspect 8)
# -----------------------------
# Use 'description' if present; otherwise fall back to 'text'
text_col = "description" if "description" in df.columns else ("text" if "text" in df.columns else None)
if text_col:
    df["clean_text_field"] = df[text_col].fillna("unknown").apply(clean_text)

    if "gender" in df.columns:
        for label in df["gender"].dropna().unique():
            tdata = " ".join(df.loc[df["gender"] == label, "clean_text_field"])
            if tdata.strip():
                wc = WordCloud(width=1200, height=600, background_color="white").generate(tdata)
                plt.figure(figsize=(10,5))
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
                plt.title(f"Word Cloud of {text_col} by Gender: {label}")
                savefig(f"08_wordcloud_{text_col}_by_{label}.png")
    else:
        # Overall wordcloud
        tdata = " ".join(df["clean_text_field"].astype(str).tolist())
        if tdata.strip():
            wc = WordCloud(width=1200, height=600, background_color="white").generate(tdata)
            plt.figure(figsize=(10,5))
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title(f"Word Cloud of {text_col} (All Records)")
            savefig(f"08_wordcloud_{text_col}_all.png")

# -----------------------------
# 9) Gender Confidence (Aspect 9)
# -----------------------------
conf_col = "gender:confidence" if "gender:confidence" in df.columns else None
if conf_col:
    plt.figure(figsize=(6,4))
    sns.histplot(df[conf_col].dropna(), kde=True, color="purple", bins=30)
    plt.title("Distribution of Gender Confidence")
    savefig("09a_gender_confidence_dist.png")

    # Pairwise plots of confidence vs numerics
    if len(num_cols) >= 1:
        cols = num_cols + [conf_col]
        sub = df[cols].dropna()
        if len(sub) > 2000:
            sub = sub.sample(2000, random_state=42)
        g = sns.pairplot(sub, kind="scatter", diag_kind="kde",
                         plot_kws={"alpha":0.6, "s":25})
        g.fig.suptitle("Aspect 9: Gender Confidence vs Activity Features", y=1.02, fontsize=12)
        savefig("09b_pairplot_confidence_vs_numeric.png")

# -----------------------------
# Findings (auto-written summary)
# -----------------------------
lines = []
lines.append("## Key Findings (Pre-Processing)\n")
# Target balance
if "gender" in df.columns:
    counts = df["gender"].value_counts(dropna=False)
    lines.append("1) Target Balance:")
    lines.append(str(counts))
    lines.append("   - Note: Class imbalance (if present) will affect classification; consider stratified splits and F1/ROC-AUC.\n")

# Numeric distributions
if len(num_cols) > 0:
    lines.append("2) Numeric Distributions:")
    for col in num_cols:
        desc = df[col].describe(percentiles=[0.5,0.9,0.95,0.99]).to_string()
        lines.append(f"- {col} stats:\n{desc}\n"
                     "  * Skew/outliers indicate heavy-tail activity; boxplots show category-level variation.")

# Correlations
if len(num_cols) >= 2:
    corr_vals = df[num_cols].corr().round(2)
    lines.append("3) Correlations (numeric):")
    lines.append(str(corr_vals))
    lines.append("  * Helps avoid multicollinearity; pairs with strongest |corr| influence feature selection.\n")

# Missingness
missing = df.isnull().mean().sort_values(ascending=False)
lines.append("4) Missingness:")
lines.append(str((missing*100).round(1).head(10)) + "  (top-10 shown as %)")
lines.append("  * Columns with very high missingness (e.g., *_gold) should be dropped or handled with care.\n")

# Colors
for col in ["link_color", "sidebar_color"]:
    if col in df.columns:
        top = df[col].value_counts().head(5)
        lines.append(f"5) {col}: top values")
        lines.append(str(top))
        lines.append("  * Aesthetics may be weak predictors alone; useful as auxiliary signals.\n")

# Text (basic)
if text_col:
    avg_len = df[text_col].fillna("").apply(lambda x: len(str(x).split())).mean()
    lines.append(f"6) Text ({text_col}): average token count ~ {avg_len:.1f}")
    lines.append("  * Word clouds show linguistic differences; TF-IDF will be used for modelling.\n")

# Confidence
if conf_col:
    lines.append("7) Gender Confidence:")
    lines.append("  * Distribution suggests potential label noise; low-confidence rows merit caution or relabelling checks.\n")

with open(os.path.join(FIG_DIR, "eda_findings.txt"), "a", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("\n[Done] EDA complete. All figures saved to ./out/figures and findings summarized in ./out/figures/eda_findings.txt")
