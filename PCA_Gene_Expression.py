import numpy as np
import matplotlib.pyplot as plt

# Taking input from user

rows = int(input("Enter number of samples: "))
cols = int(input("Enter number of features: "))

print("\nEnter dataset values row-wise:")

data = []

for i in range(rows):

    row = list(map(float, input().split()))

    data.append(row)

data = np.array(data)

# Standardization
mean = data.mean(axis=0)
std = data.std(axis=0)

standardized_data = (data - mean) / std

# Covariance Matrix
covariance_matrix = np.cov(standardized_data.T)

# Eigenvalues and Eigenvectors

eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

# Sorting Eigenvalues
sorted_index = np.argsort(eigenvalues)[::-1]

sorted_eigenvalues = eigenvalues[sorted_index]
sorted_eigenvectors = eigenvectors[:, sorted_index]

# Select number of components
k = int(input("\nEnter number of principal components: "))

principal_components = sorted_eigenvectors[:, :k]

# Display Eigenvalues

print("\n" + "=" * 40)
print("        EIGENVALUES (sorted)")
print("=" * 40)

for i in range(len(sorted_eigenvalues)):

    value = sorted_eigenvalues[i]

    bar = "█" * int(value * 10)

    print(f"PC{i+1}  |  {value:.4f}  {bar}")

print("=" * 40)
print(f"Total variance: {sorted_eigenvalues.sum():.4f}")

# PCA Projection
reduced_data = standardized_data @ principal_components

# Variance Explained
variance_explained = (
    sorted_eigenvalues / sorted_eigenvalues.sum()
) * 100

print("\nVARIANCE EXPLAINED")
print("-" * 40)

for i in range(len(variance_explained)):

    print(f"PC{i+1} explains: {variance_explained[i]:.1f}%")

print(f"\nTotal variance by top {k} PCs: "
      f"{variance_explained[:k].sum():.1f}%")

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))


# Scree Plot
component_names = [
    f"PC{i+1}" for i in range(len(sorted_eigenvalues))
]

colors = [
    "steelblue" if i < k else "lightgray"
    for i in range(len(sorted_eigenvalues))
]

axes[0].bar(
    component_names,
    sorted_eigenvalues,
    color=colors
)

axes[0].set_title("Scree Plot")
axes[0].set_xlabel("Principal Components")
axes[0].set_ylabel("Eigenvalues")


# Display values on bars
for i in range(len(sorted_eigenvalues)):

    value = sorted_eigenvalues[i]

    axes[0].text(
        i,
        value + 0.01,
        f"{value:.2f}",
        ha="center"
    )


# PCA Scatter Plot
if k >= 2:

    axes[1].scatter(
        reduced_data[:, 0],
        reduced_data[:, 1],
        alpha=0.7
    )

    axes[1].set_xlabel(
        f"PC1 ({variance_explained[0]:.1f}%)"
    )

    axes[1].set_ylabel(
        f"PC2 ({variance_explained[1]:.1f}%)"
    )

    axes[1].set_title("PCA Projection")


plt.tight_layout()
plt.show()