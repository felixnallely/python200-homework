import warnings
import numpy as np
import pandas as pd
import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests
from io import BytesIO
import seaborn as sns
from pathlib import Path 

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline 
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from sklearn.inspection import DecisionBoundaryDisplay

warnings.filterwarnings("ignore", category=RuntimeWarning)

#Directories 
output_dir = "assignments_03/outputs"

#Define column names: 
COLUMN_NAMES = [
    "word_freq_make",        # 0   percent of words that are "make"
    "word_freq_address",     # 1
    "word_freq_all",         # 2
    "word_freq_3d",          # 3   almost never appears
    "word_freq_our",         # 4
    "word_freq_over",        # 5
    "word_freq_remove",      # 6   common in "remove me from this list"
    "word_freq_internet",    # 7
    "word_freq_order",       # 8
    "word_freq_mail",        # 9
    "word_freq_receive",     # 10
    "word_freq_will",        # 11
    "word_freq_people",      # 12
    "word_freq_report",      # 13
    "word_freq_addresses",   # 14
    "word_freq_free",        # 15  classic spam word
    "word_freq_business",    # 16
    "word_freq_email",       # 17
    "word_freq_you",         # 18
    "word_freq_credit",      # 19
    "word_freq_your",        # 20  often high in spam
    "word_freq_font",        # 21  HTML emails
    "word_freq_000",         # 22  "win $ x,000" style offers
    "word_freq_money",       # 23  money related
    "word_freq_hp",          # 24  HP specific
    "word_freq_hpl",         # 25
    "word_freq_george",      # 26  specific HP person
    "word_freq_650",         # 27  area code
    "word_freq_lab",         # 28
    "word_freq_labs",        # 29
    "word_freq_telnet",      # 30
    "word_freq_857",         # 31
    "word_freq_data",        # 32
    "word_freq_415",         # 33
    "word_freq_85",          # 34
    "word_freq_technology",  # 35
    "word_freq_1999",        # 36
    "word_freq_parts",       # 37
    "word_freq_pm",          # 38
    "word_freq_direct",      # 39
    "word_freq_cs",          # 40
    "word_freq_meeting",     # 41
    "word_freq_original",    # 42
    "word_freq_project",     # 43
    "word_freq_re",          # 44  reply threads
    "word_freq_edu",         # 45
    "word_freq_table",       # 46
    "word_freq_conference",  # 47
    "char_freq_;",           # 48  frequency of ';'
    "char_freq_(",           # 49  frequency of '('
    "char_freq_[",           # 50  frequency of '['
    "char_freq_!",           # 51  exclamation marks (often big)
    "char_freq_$",           # 52  dollar sign (money related)
    "char_freq_#",           # 53  hash character
    "capital_run_length_average",  # 54  average length of capital letter runs
    "capital_run_length_longest",  # 55  longest capital run
    "capital_run_length_total",    # 56  total number of capital letters
    "spam_label"                    # 57  1 = spam, 0 = not spam
]

output_dir = Path("assignments_03/outputs")

#load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data"
response = requests.get(url)
response.raise_for_status()

df = pd.read_csv(BytesIO(response.content), header=None)
df.columns = COLUMN_NAMES
df.head()

# --- Task 1: Load and Explore --- #
print("Number of emails:", len(df))
print(df["spam_label"].value_counts())

#Comment: How many emails are in the dataset? How balanced are the two classes? 
#What does that balance (or imbalance) mean for how you should interpret a raw accuracy score?
# There are 4601 emails in this dataset. The two classes are moderately imbalance, not extreme. Raw accuracy 
# is not a reliable metric, other metrics such as precision, recall and F1 score will help determine whether or not 
# the model is actually detecting spam. 

#Boxplot Spam emails vs ham emails
features = ["word_freq_free", "char_freq_!", "capital_run_length_total"]

for feat in features: 
    plt.figure(figsize=(8, 5))
    sns.boxplot(x="spam_label", y=feat, data=df)
    plt.title(f"{feat}: Spam emails Vs Ham emails")
    plt.xlabel("Spam Label (0 = Ham, 1 = Spam)")
    plt.tight_layout()
    plt.savefig(output_dir / f"{feat}_boxplot.png")
    plt.close()

#Comment: What do you notice? Are the differences between classes dramatic or subtle?
# I noticed word_freq_free has noticeably higher spam emails, the difference is clear but not extreme. 
# This feature suggests a strong indicator but not universal.
#
# I noticed for char_freq_! that spam emails tend to use "!" more often and ham emails don't use "!" as often.
# The difference here is moderate but it is also important to know that using "!" signals promotional or an urgent tone. 
#
# For capital_run_length_total I noticed spam emails used larger and longer words. This showed a dramatic difference and 
# suggests a powerful distinguishing feature. 
# 
# Overall many of these features were skewed heavily towards zero, which means most emails didn't conttain these words and characteristics.
# 

# --- Task 2: Prepare Your Data --- #

#Seperate features & target 
X = df.drop(columns=["spam_label"])
y = df["spam_label"]

#Train/test split #using stratify=y test_size=0.3 70% training and 30% testing 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#feature scaling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

#Comment: Preperation 
# - Statified train/test split keeps class balance consistent. 
# - Scaling helps prevent large features from overshadowing smaller features.

# PCA (fit only on training data)
pca = PCA()
pca.fit(X_train_scaled)

#Plot 
plt.figure(figsize=(8, 5))
plt.plot(pca.explained_variance_ratio_.cumsum())
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance")
plt.title("PCA Cumulative Explained Variance")
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "pca_explained_variance.png")
plt.close()

#Find n --> cumulative var 1st 90%
cumulative = pca.explained_variance_ratio_.cumsum()
n = (cumulative >= 0.90).argmax() + 1
print("number of components for 90% variance:", n)

#transform sets using PCA 
X_train_pca = pca.transform(X_train_scaled)[:, :n]
X_test_pca  = pca.transform(X_test_scaled)[:, :n]


# --- Task 3: A Classifier Comparision --- #
results = {}

# -- KNN on unscaled data --
knn_unscaled = KNeighborsClassifier(n_neighbors=5)
knn_unscaled.fit(X_train, y_train)
y_pre_knn_unscaled = knn_unscaled.predict(X_test)

print("\nKNN (unscaled) results:")
print("Accuracy:", accuracy_score(y_test, y_pre_knn_unscaled))
print(classification_report(y_test, y_pre_knn_unscaled))
results["KNN_unscaled"]  = accuracy_score(y_test, y_pre_knn_unscaled)

# -- KNN on scaled data --
knn_scaled = KNeighborsClassifier(n_neighbors=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pre_knn_scaled = knn_scaled.predict(X_test_scaled)

print("\nKNN (scaled) results:")
print("Accuracy:", accuracy_score(y_test, y_pre_knn_scaled))
print(classification_report(y_test, y_pre_knn_scaled))
results["KNN_scaled"]  = accuracy_score(y_test, y_pre_knn_scaled)

# -- KNN on PCA --
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pre_knn_pca = knn_pca.predict(X_test_pca)

print("\nKNN (PCA) results:")
print("Accuracy:", accuracy_score(y_test, y_pre_knn_pca))
print(classification_report(y_test, y_pre_knn_pca))
results["KNN_pca"]  = accuracy_score(y_test, y_pre_knn_pca)

# -- Decision Tree --
depths = [3, 5, 10, None]

print("\nDecision Tree Depth Comparison:")
for d in depths: 
    clf = DecisionTreeClassifier(max_depth=d, random_state=42)
    clf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"max_depth={d}: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

#Comment:  What do you notice as depth increases? What does that tell you about overfitting?
# I notice that as depth increases training accuracy increases but at the same time test accuracy 
# stops improving and even decreases.
# This tells me that higher depths are overfitting meaning its memorizing training data instead of generalizing.

# Pick depth to use in production
#Comment: explaining your reasoning. Why this depth?
# I am picking ten because I think it's training accuracy and testing accuracy are really high. 
#- dt --> decision tree
dt_final = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_final.fit(X_train, y_train)
y_pred_dt = dt_final.predict(X_test)

print("\nDecision Tree (max_depth=10) results:")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))
print(classification_report(y_test, y_pred_dt))
results["Decisiontree_depths10"] = accuracy_score(y_test, y_pred_dt)

# --- Random Forest --
rf = RandomForestClassifier(
    n_estimators=200, max_depth=None, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest results:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))
results["RandomForest"] = accuracy_score(y_test, y_pred_rf)

# -- Logistic Regression (scaled) --
log_scaled = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
log_scaled.fit(X_train_scaled, y_train)
y_pred_log_scaled = log_scaled.predict(X_test_scaled)

print("\nLogistic Regression (scaled) results:")
print("Accuracy:", accuracy_score(y_test, y_pred_log_scaled))
print(classification_report(y_test, y_pred_log_scaled))
results["LogReg_Scaled"] = accuracy_score(y_test, y_pred_log_scaled)

# -- Logistic Regression (PCA) --
log_pca = LogisticRegression(C=1.0, max_iter=1000, solver="liblinear")
log_pca.fit(X_train_pca, y_train)
y_pred_log_pca = log_pca.predict(X_test_pca)

print("\nLogistic Regression (PCA) results:")
print("Accuracy:", accuracy_score(y_test, y_pred_log_pca))
print(classification_report(y_test, y_pred_log_pca))
results["LogReg_PCA"] = accuracy_score(y_test, y_pred_log_pca)

print("\nSummary of test accuracies:")
for name, acc in results.items():
    print(f"{name}: {acc:.4f}")

#Comment: Summary 
# Random forest performs best because its test accuracies are 0.9566 having the highest. 
# KNN scaled worked better than KNN PCA and KNN unscaled, this matches Task 2 because PCA helps
# with features that are high, so raw and scaled features work better. 
# I would prioritize high precision for spam class even if some spam isn't caught because accuracy alone
# isn't ideal and having false positives are very costly. 

# -- Confusion Matrix -- # 
bm_name = max(results, key=results.get)
print("\nBest model by accuracy:", bm_name)

con_mat = confusion_matrix(y_test, y_pred_rf)
disp = ConfusionMatrixDisplay(confusion_matrix=con_mat, display_labels=["ham (0)", "spam (1)"])

plt.figure(figsize=(8, 5))
disp.plot(cmap="Purples")
plt.title("Best Model Confusion Matrix (Random Forest)")
plt.tight_layout()
plt.savefig(output_dir / "best_model_confusion_matrix.png")
plt.close()

#Comment: Given the costs described above, which type of error does your best model make more often?
# The best model as shown by the confusion matrix makes more false negatives errors than anything else. Which 
# is a preferred error since it's better to miss some spam emails over losing real emails. 


# -- Top 10 most important features Bar Chart -- #
dt_importances = pd.Series(dt_final.feature_importances_, index=X.columns)
dt_top10 = dt_importances.sort_values(ascending=False).head(10)

print("\nTop 10 Decision Tree feature importances:")
print(dt_top10)

rf_importances = pd.Series(rf.feature_importances_, index=X.columns)
rf_top10 = rf_importances.sort_values(ascending=False).head(10)

print("\nTop 10 Random Forest feature importances:") 
print(rf_top10)

plt.figure(figsize=(10, 6))
rf_top10.plot(kind="bar", color="purple")
plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig(output_dir / "feature_importances.png", dpi=300)
plt.close()

#Comment: Do the two models agree on which features matter most? 
# Do the results match your intuition about what makes an email spam?
# Yes, both models agree with which features matter more and some of the top 
# features include: char_freq_!, char_freq_$ and capital_run_length_total. Yes, 
# the results match intuition about what makes email spam and its usually words such as "free",
# "money", and "!". 


# --- Task 4: Cross-Validation --- # 
#cv --> cross calidation 
def run_cv(name, model, X_train_data, y_train_data):
    scores = cross_val_score(model, X_train_data, y_train_data, cv=5)
    mean = scores.mean()
    std = scores.std()
    cv_results[name] = (mean, std)
    print(f"\n{name} cross-validation scores: {scores}")
    print(f"{name} mean accuracy: {mean:.4f}")
    print(f"{name} std (stability): {std:.4f}")
cv_results = {}

# -- KNN unscaled --
run_cv("KNN_unscaled", knn_unscaled, X_train, y_train)
# -- KNN Scaled --
run_cv("KNN_scaled", knn_scaled, X_train_scaled, y_train)
# -- KNN PCA --
run_cv("KNN_PCA", knn_pca, X_train_pca, y_train)
# -- Decision Tree (Depth 10) --
run_cv("DecisionTree_depth10", dt_final, X_train, y_train)
# -- Random Forest --
run_cv("RandomForest", rf, X_train, y_train)
# -- Logistic Regression Scaled --
run_cv("LogReg_scaled", log_scaled, X_train_scaled, y_train)
# -- Logistic Regression PCA --
run_cv("LogReg_pca", log_pca, X_train_pca, y_train)

print("\nSummary of cross-validation (mean std):")
for name, (mean, std) in cv_results.items():
    print(f"{name}: mean={mean:.4f}, std={std:.4f}")


#Comment: Which model is the most accurate? Which is the most stable (lowest variance across folds)? 
# Does the ranking match what you saw with the single train/test split?
# The most accurate model is Random Forest because it has the highest mean of 0.9509. The most stable is Random Forest 
# with a std of 0.0053. Logistic Regression is also stable but slightly less accurate. The ranking does match 
# the train/test split which increases the confidence of Random Forest generalizing the best. 

# --- Task 5: Building a Prediction Pipeline --- # 
# -- Pipeline for best tree-based model --
tree_pipeline = Pipeline([
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    ))
])

tree_pipeline.fit(X_train, y_train)
y_pred_tree = tree_pipeline.predict(X_test)

print("Tree Pipeline Results:")
print(classification_report(y_test, y_pred_tree))

# -- Pipeline for best non-tree model --
non_tree_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(C=1.0, max_iter=1000, solver='liblinear'))
])

non_tree_pipeline.fit(X_train, y_train)
y_pred_non_tree = non_tree_pipeline.predict(X_test)

print("Non-Tree Pipeline Results:")
print(classification_report(y_test, y_pred_non_tree))

#Comment on your pipelines: do they have the same structure? Why or why not? What is the practical value of packaging a model this way, 
#especially when handing it off to someone else or deploying it?
# After building both pipelines and running the code, they do have the same structure. The Random Forest pipeline achieved 96% accuracy and the 
# Logistic Regression pipeline achieved 93% accuracy which matches my manual results. Pipelines provide a great practical value because it packages preprocessing and the 
# modeling steps into one single object. This makes the workflow easier and more reliable, it also reduces the chances of mistakes being made. Pipelines also make the model 
# easier to deploy and even hand off to someone else. Since pipelines make sure everything is applied consistently, in the correct order, this allows the improvement of reproducibility 
# and maintainability. 
