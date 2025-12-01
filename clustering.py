# Task 2 - Clustering (KMeans + Hierarchical + SOM)


import os
import pandas as pd
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from minisom import MiniSom
import matplotlib.pyplot as plt
import seaborn as sns

#  folders 
DATA_DIR = "./A2_2025_Released"
OUT_T1 = "./out"
os.makedirs(OUT_T1, exist_ok=True)

# load data 
# cleaned dataset provided from Task 1
df = pd.read_csv(os.path.join(OUT_T1, "df_cleaned.csv"))
true_labels = df["gender"]  # only used for checking patterns

# features
# join description + tweet to get richer text input
df["combined_text"] = df["description"].astype(str) + " " + df["text"].astype(str)

# TF-IDF for text
vectorizer = TfidfVectorizer(stop_words="english", max_features=3000)
X_text = vectorizer.fit_transform(df["combined_text"])

# numeric column (gender confidence is a small signal)
X_num = df[["gender:confidence"]].fillna(0).values

# merge text and numeric features
X = hstack([X_text, X_num])



# 1) KMEANS CLUSTERING
#>>>>>>>>>>>>>>>>>>>>>>>>>
print("\n=== KMeans Clustering ===")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters_kmeans = kmeans.fit_predict(X)
df["cluster_kmeans"] = clusters_kmeans

# reduce features for silhouette score (to avoid memory crash)
svd = TruncatedSVD(n_components=100, random_state=42)
X_reduced = svd.fit_transform(X)
sil = silhouette_score(X_reduced, clusters_kmeans)
print("K-Means Silhouette Score:", sil)

# cross-tab vs gender
ctab_km = pd.crosstab(df["cluster_kmeans"], true_labels)
print("\nKMeans cross-tab:\n", ctab_km)

# save one plot + one results file
plt.figure(figsize=(6,4))
sns.countplot(x="cluster_kmeans", data=df, hue="cluster_kmeans", palette="viridis", legend=False)
plt.title("Cluster distribution (K-Means)")
plt.xlabel("Cluster ID"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(OUT_T1, "kmeans_cluster_distribution.png"))
plt.close()

ctab_km.to_csv(os.path.join(OUT_T1, "kmeans_crosstab.csv"))
df.to_csv(os.path.join(OUT_T1, "df_clustered_kmeans.csv"), index=False)



# 2) HIERARCHICAL CLUSTERING
#>>>>>>>>>>>>>>>>>>>>>>>>>
print("\n=== Hierarchical Clustering ===")
# sample subset (500 rows) so dendrogram is readable
df_sample = df.sample(min(500, len(df)), random_state=42)

# build features for the sample
X_text_s = vectorizer.fit_transform(df_sample["combined_text"])
X_num_s = df_sample[["gender:confidence"]].fillna(0).values
X_sample_dense = hstack([X_text_s, X_num_s]).toarray()

# hierarchical linkage
Z = linkage(X_sample_dense, method="ward")

# dendrogram for visualization
plt.figure(figsize=(10,5))
dendrogram(Z, truncate_mode="level", p=5)
plt.title("Hierarchical Dendrogram (sample)")
plt.xlabel("Samples"); plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(os.path.join(OUT_T1, "hierarchical_dendrogram.png"))
plt.close()

# cut tree into 3 clusters
hc_labels = fcluster(Z, t=3, criterion="maxclust")
df_sample["cluster_hierarchical"] = hc_labels

# cross-tab vs gender
ctab_hc = pd.crosstab(df_sample["cluster_hierarchical"], df_sample["gender"])
print("\nHierarchical cross-tab [sample]:\n", ctab_hc)

# save one results file
ctab_hc.to_csv(os.path.join(OUT_T1, "hierarchical_crosstab_sample.csv"))
df_sample.to_csv(os.path.join(OUT_T1, "df_clustered_hierarchical_sample.csv"), index=False)



# 3) SELF-ORGANIZING MAPS (SOM)
#>>>>>>>>>>>>>>>>>>>>>>>>>
print("\n=== Self-Organizing Maps (SOM) ===")
X_dense = X.toarray()  # SOM needs dense input
som_size = 10
som = MiniSom(x=som_size, y=som_size, input_len=X_dense.shape[1],
              sigma=1.0, learning_rate=0.5, random_seed=42)
som.random_weights_init(X_dense)
som.train_random(X_dense, 1000)

# assign each row to BMU
bmus = np.array([som.winner(x) for x in X_dense])
df["cluster_som"] = [f"{i}-{j}" for i,j in bmus]

# count how many fall into each SOM cell
som_counts = pd.Series(df["cluster_som"]).value_counts()
print("\nTop SOM cells:\n", som_counts.head())
print("Total SOM cells used:", som_counts.shape[0])# also print how many unique SOM cells were used

# SOM visualization
plt.figure(figsize=(6,6))
plt.scatter([i for i,j in bmus], [j for i,j in bmus], c="blue", alpha=0.5, s=10)
plt.title("SOM cluster visualization")
plt.xlabel("SOM X"); plt.ylabel("SOM Y")
plt.tight_layout()
plt.savefig(os.path.join(OUT_T1, "som_visualization.png"))
plt.close()

# save results file
som_counts.to_csv(os.path.join(OUT_T1, "som_cluster_counts.csv"))
df.to_csv(os.path.join(OUT_T1, "df_clustered_som.csv"), index=False)
print("\nDone. Clean results saved in ./out")
