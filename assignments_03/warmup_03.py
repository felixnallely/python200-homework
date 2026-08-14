import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, load_digits
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

#--- Preprocessing Question 1 ---# 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

print("---Preprocessing Q1: ---")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

#--- Preprocessing Question 2 ---# 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("---Preprocessing Q2: ---")
print("Mean of X_trained_scaled:", X_train_scaled.mean(axis=0))

#The fit scaler is used only on X_train to prevent data leakage, this ensures the model evaluates on unseen information from X_test. 

#--- KNN Question 1 ---#
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

preds = knn.predict(X_test)

print("---KNN Q1: ---")
print("Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))

#--- KNN Question 2 ---# 
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)

preds_scaled = knn_scaled.predict(X_test_scaled)

print("---KNN Q2: ---")
print("Accuracy (scaled):", accuracy_score(y_test, preds_scaled))

#Comment: Does scaling improve performance, hurt it, or make no difference? Why might that be for this particular dataset?
# Scaling KNN doesn't make a huge difference, since the Iris features are already vey simliar in scale. So the performance changes very little. 

#--- KNN Question 3 ---#
knn = KNeighborsClassifier(n_neighbors=5)
cv_scores = cross_val_score(knn, X_train, y_train, cv=5)

print("---KNN Q3: ---")
print(cv_scores)           # accuracy on each fold
print(f"Mean: {cv_scores.mean():.3f}")
print(f"Std:  {cv_scores.std():.3f}")

#Comment: Is this result more or less trustworthy than a single train/test split, and why?
# Cross-validation is more trustworthy than a single train/test split because it evaluates on multiple folds instead of one split. 

#--- KNN Question 4 ---# 
k_values = [1, 3, 5, 7, 9, 11, 13, 15]

for k in k_values: 
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)

print("---KNN Q4: ---")
for k in k_values: 
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5)
print(f"k={k}, mean CV accuracy={scores.mean():.4f}")

#Comment: Identifying which k you would choose and why.
# I would choose k=15 because it achieved a more stable cross-validation accuracy, and being a large number it reduces noise
# with class separation. 

#--- Classifier Evaluation Question 1 ---# 
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

plt.figure(figsize=(6, 5))
disp.plot()
plt.title("KNN Confusion Matrix (Iris)-Unscaled")
plt.savefig("assignments_03/outputs/knn_confusion_matrix.png")
plt.close()

#Comment: which pair of species does the model most often confuse (if any)?
# This model did not confuse any species. The model achieved perfect accuracy and contains no misclassifications.

#--- Decision Trees Question 1 ---#
dt = DecisionTreeClassifier(max_depth=3, random_state=42)
dt.fit(X_train, y_train)

y_pred_dt = dt.predict(X_test)

print("--- Decision Tree Q1: ---")
print("Accuracy (Decision Tree):", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))

#Comment 1: comparing the Decision Tree accuracy to KNN.
# The Decision Tree performce almost as well as KNN does, with an accuracy of 0.9667. However KNN is ahead because it captures local 
# neighborhood structure precisely, but the decision tree has a fixed depth of 3 (max_depth=3), which limits how well it seperates overlapping classes. 
#Comment 2:  given that Decision Trees don't rely on distance calculations, would scaled vs. unscaled data affect the result?
# No, unscaled or scaled data would not affect the result because Decision Trees do not rely on distance calculations. Therefore the results for scaled and unscaled 
# data will give identical results. 

#--- Logistic Regression Question 1 ---#
from sklearn.multiclass import OneVsRestClassifier

C_values = [0.01, 1.0, 100]

for C in C_values: 
    log_reg = OneVsRestClassifier(
        LogisticRegression(
            C=C,
            max_iter=1000,
            solver="liblinear"
        )
    )

    log_reg.fit(X_train_scaled, y_train)
    
    coef_sum = sum(np.abs(est.coef_).sum() for est in log_reg.estimators_) 
    #coef_sum = np.abs(log_reg.coef_).sum()
    
    print("---Logistic Regression Q1: ---")
    print(f"C={C}, total |coefficients| sum = {coef_sum:.4f}")

#Comment: what happens to the total coefficient magnitude as C increases? What does this tell you about what regularization is doing?
# As C increases the coefficients get larger. This tells me regularization is penalizing large coefficients by preventing them from blowing up. 

# PCA dataset
from pathlib import Path

digits = load_digits()
X_digits = digits.data    # 1797 images, each flattened to 64 pixel values
y_digits = digits.target  # digit labels 0-9
images   = digits.images  # same data shaped as 8x8 images for plotting

#--- PCA Question 1 ---#
print("---PCA Q1: ----")
print("X_digits shape:", X_digits.shape)
print("images shape:", images.shape)

#Output 
output_dir = Path("assignments_03/outputs")
output_dir.mkdir(parents=True, exist_ok=True)

#1-row subplot
plt.figure(figsize=(12, 3))

for digit in range(10):
    idx = list(y_digits).index(digit)
    img = images[idx]

    plt.subplot(1, 10, digit + 1)
    plt.imshow(img, cmap='gray_r')
    plt.title(str(digit))
    plt.axis('off')
plt.tight_layout()
plt.savefig(output_dir / "sample_digits.png")
plt.close()

print("Saved figure to outputs as sample_digits.png")

#--- PCA Question 2 ---#
pca = PCA()
pca.fit(X_digits)

#PCA scores weightings for each sample
scores = pca.transform(X_digits)

#Scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(scores[:, 0], scores[:, 1], c=y_digits, cmap='tab10', s=10)  # c = color array
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Digits Dataset: PCA 2d Projection")

plt.colorbar(scatter, label='Digit')

plt.tight_layout()
plt.savefig(output_dir / "pca_2d_projection.png")
plt.close()

#Comment: do same-digit images tend to cluster together in this 2D space?
# In terms of this 2D shape, the same-digit images do tend to cluster together. 

#--- PCA Question 3 ---#
cum_var = np.cumsum(pca.explained_variance_ratio_)

plt.figure(figsize=(8, 5))
plt.plot(cum_var, marker='o')
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("Cumulative Explained Variance by PCA Components")
plt.grid(True)

plt.savefig(output_dir / "pca_variance_explained.png", dpi=150, bbox_inches="tight")
plt.close()

#Comment: approximately how many components do you need to explain 80% of the variance?
# Approximately 10 to 20 components are needed to explain the 80% variances since these components usually explain about 80-85% of the variance.

#--- PCA Question 4 ---#
def reconstruct_digit(sample_idx, scores, pca, n_components):
    """Reconstruct one digit using the first n_components principal components."""
    reconstruction = pca.mean_.copy()
    for i in range(n_components):
        reconstruction = reconstruction + scores[sample_idx, i] * pca.components_[i]
    return reconstruction.reshape(8, 8)

#first 5 digits samples 
sample_indices = [0, 1, 2, 3, 4]

#Reconstrution 
n_values = [2, 5, 15, 40]

#Grid of subplots
plt.figure(figsize=(12, 10))
#Original images - Row 0:
for col, idx in enumerate(sample_indices):
    plt.subplot(len(n_values) + 1, len(sample_indices), col + 1)
    plt.imshow(images[idx], cmap='gray_r')
    plt.title(f"Original {y_digits[idx]}")
    plt.axis('off')

# Reconstruction -- Rows 1-4
for row, n in enumerate(n_values):
    for col, idx in enumerate(sample_indices):
        recon = reconstruct_digit(idx, scores, pca, n)
        subplot_index = (row +1) * len(sample_indices) + col + 1 
        plt.subplot(len(n_values) + 1, len(sample_indices), subplot_index)
        plt.imshow(recon, cmap='gray_r')
        plt.title(f"n={n}")
        plt.axis('off')

plt.tight_layout()
plt.savefig(output_dir / "pca_reconstructions.png")

#Comment: at what n do the digits become clearly recognizable, and does that match where the variance curve levels off?
# Around n=15 the digits look more clean and recognizable. Yes, this matches where the variance curve levels off, because 
# the curve flattens around 10-20 components.
