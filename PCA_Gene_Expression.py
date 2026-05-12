import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class_file = "class.tsv"
expr_file = "filtered.tsv.gz"
columns_file = "columns.tsv.gz"

classes = pd.read_csv(
    class_file,
    sep="\t",
    header=None
)

classes.columns = ["class"]

print("\nClass shape:")
print(classes.shape)

expr = pd.read_csv(
    expr_file,
    sep="\t"
)

print("\nExpression matrix shape:")
print(expr.shape)

print("\nFirst few columns:")
print(expr.columns[:10])

print("\nData types:")
print(expr.dtypes.head())

cols = pd.read_csv(
    columns_file,
    sep="\t",
    comment="#"
)

print("\nColumns mapping shape:")
print(cols.shape)


xbp1_rows = cols[
    cols["GeneSymbol"].astype(str).str.upper() == "XBP1"
]

print("\nXBP1 rows:")
print(xbp1_rows)

XBP1_ID = int(xbp1_rows.iloc[0]["ID"])

print("\nXBP1 ID:")
print(XBP1_ID)

gata3_rows = cols[
    cols["GeneSymbol"].astype(str).str.upper() == "GATA3"
]

print("\nGATA3 rows:")
print(gata3_rows)

GATA3_ID = int(gata3_rows.iloc[0]["ID"])

print("\nGATA3 ID:")
print(GATA3_ID)

expr.columns = expr.columns.astype(int)

xbp1 = expr[XBP1_ID]
gata3 = expr[GATA3_ID]

print("\nXBP1 expression:")
print(xbp1.head())

print("\nGATA3 expression:")
print(gata3.head())

colors = classes["class"].map({
    0: "black",
    1: "red"
})

plt.figure(figsize=(7,7))

plt.scatter(
    gata3,
    xbp1,
    c=colors,
    s=35
)

plt.xlabel("GATA3", fontsize=14)
plt.ylabel("XBP1", fontsize=14)

plt.title("Figure 1a Style Plot")

plt.grid(True)

plt.show()

# Keep only numeric columns
expr_numeric = expr.select_dtypes(include=[np.number])

print("\nNumeric matrix shape:")
print(expr_numeric.shape)

X = expr_numeric.values

# IMPORTANT:
# Rows = samples
# Columns = genes

print("\nPCA matrix shape:")
print(X.shape)

# Very common for gene expression PCA

scaler = StandardScaler(with_std=False)

X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)

pcs = pca.fit_transform(X_scaled)

pc1 = pcs[:,0]
pc2 = pcs[:,1]

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(10,4))

# ALL samples
plt.scatter(
    pc1,
    np.ones_like(pc1) * 2,
    c=colors,
    s=25
)

# ER-
er_minus = classes["class"] == 0

plt.scatter(
    pc1[er_minus],
    np.ones(sum(er_minus)) * 1,
    c="black",
    s=25
)

# ER+
er_plus = classes["class"] == 1

plt.scatter(
    pc1[er_plus],
    np.ones(sum(er_plus)) * 0,
    c="red",
    s=25
)

plt.yticks(
    [2,1,0],
    ["All", "ER-", "ER+"]
)

plt.xlabel("Projection onto PC1")

plt.title("Figure 1c Style Plot")

plt.grid(True)

plt.show()

plt.figure(figsize=(7,7))

plt.scatter(
    pc1,
    pc2,
    c=colors,
    s=35
)

plt.xlabel("PC1")
plt.ylabel("PC2")

plt.title("PCA Scatter Plot")

plt.grid(True)

plt.show()

pca_df = pd.DataFrame({
    "PC1": pc1,
    "PC2": pc2,
    "Class": classes["class"]
})

pca_df.to_csv(
    "pca_results.tsv",
    sep="\t",
    index=False
)

print("\nSaved PCA results to pca_results.tsv")

print("\nAssignment completed successfully.")
