#---- scikit-learn API ----
#%%
import numpy as np
from sklearn.linear_model import LinearRegression

#--- Question 1 ---
years  = np.array([1, 2, 3, 5, 7, 10]).reshape(-1, 1)
salary = np.array([45000, 50000, 60000, 75000, 90000, 120000])

model = LinearRegression()
model.fit(years, salary)
test_years = np.array([4, 8]).reshape(-1, 1)
predicted = model.predict(test_years)

print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
print(f"Prediction for 4 years: {predicted[0]}")
print(f"Prediction for 8 years: {predicted[1]}")

#--- Question 2 ---
x = np.array([10, 20, 30, 40, 50])

#print shape
print("Original shape:", x.shape)

#convert to 2D using .reshape()
x_2d = x.reshape(-1, 1)
#print new shape
print("Reshaped shape:", x_2d.shape)

#--- Why scikit-learn needs X to be 2D?
# Because in scikit-kearn the input X must always be a 2D array with shape.
# So even if there is only one feature it's important to convert to a 2D array because a 1D array will cause an error. 


#--- Question 3 ---
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X_clusters, _ = make_blobs(n_samples=120, centers=3, cluster_std=0.8, random_state=7)

#Create, fit, predict model 
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_clusters)

#print cluster center and counts
print("Cluster Centers:\n", kmeans.cluster_centers_)
print("Points per cluster:", np.bincount(labels))

plt.figure(figsize=(8, 6))

#Plot data 
plt.scatter(X_clusters[:, 0], X_clusters[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k')
#plot center cluster
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=200, linewidth=3, label='Centroids')

plt.title("K-means Clusters")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()

#Save figure
plt.savefig('outputs/kmeans_clusters.png')
plt.show()

# %%
#---- Linear Regression ----
#data 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_patients = 100
age    = np.random.randint(20, 65, num_patients).astype(float)
smoker = np.random.randint(0, 2, num_patients).astype(float)
cost   = 200 * age + 15000 * smoker + np.random.normal(0, 3000, num_patients)

#--- Question 1 ---
#Scatter plot 
plt.figure(figsize=(8, 6))
plt.scatter(age, cost, c=smoker, cmap='coolwarm', alpha=0.8, edgecolors='w')
plt.title("Medical Cost vs Age")
plt.xlabel("Age")
plt.ylabel("Medical Cost")

plt.savefig("outputs/cost_vs_age.png", dpi=300)
plt.close()

#--- What you see:
# The scatter plot shows two distinct groups. Each group is sepreated by a huge gap in cost. 
# This suggests the smoker variable is a strong factor that contributes to medical costs, smokers act
# as a major categorical shift indicator. But non-smokers which are shown by the blue color are clustered
# heavily at lower cost baseline.

#--- Question 2 ---
X = age.reshape(-1, 1)
y = cost
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}, y_train: {y_test.shape}")

#--- Question 3 ---
model = LinearRegression()
model.fit(X_train, y_train)

print(f"Slope: {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")
y_predicted = model.predict(X_test)

#calculate rmse and r^2
rmse = np.sqrt(np.mean((y_predicted - y_test) ** 2))
r2 = model.score(X_test, y_test)

print(f"RMSE: {rmse}")
print(f"R2: {r2}")

#--- Question 4 ---
#form question 3 --> Age only 
X_age = age.reshape(-1, 1)
X_train_age, X_test_age, y_train, y_test = train_test_split(X_age, cost, test_size=0.2, random_state=42)
model_age = LinearRegression().fit(X_train_age, y_train)
r2_age = model_age.score(X_test_age, y_test)

#Full model 
X_full = np.column_stack([age, smoker])

#split --> train, test 
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_full, cost, test_size=0.2, random_state=42)
model_full = LinearRegression().fit(X_train_full, y_train_full)
r2_full = model_full.score(X_test_full, y_test_full)

print(f"Question 3: Test R2 (Age): {r2_age:.4f}")
print(f"Question 4: Test R2 (Age & Smoker): {r2_full:.4f}")
print(f"Age Coefficient:", model_full.coef_[0])
print(f"Smoker Coefficient:", model_full.coef_[1])

#-- Smoker Coefficient Interpretation: 
# In practical terms the Smoker coefficient represent the estimated annual medical cost value compared to non-smokers, 
# when age is constant. The smoker flag does help significantly because the test R2 increases from approximately 0.0695 
# to 0.7737, which indicates that smoking status is a strong predictor to medical costs in the dataset and relying on age alone 
# omits the source of variation. 

#--- Question 5 ---
y_predicted = model_full.predict(X_test_full)

#Create plot
plt.figure(figsize=(8, 6))
plt.scatter(y_test_full, y_predicted, color="blue", alpha=0.7, label="Patients")
max_value = max(max(y_test_full), max(y_predicted))
min_value = min(min(y_test_full), min(y_predicted))
plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--", label="Perfect Prediction")

plt.title("Predicted vs Actual")
plt.xlabel("Actual Cost")
plt.ylabel("Predicted Cost")
plt.legend()
plt.grid(True, linestyle=":", alpha=0.6)

#Save 
plt.savefig("outputs/predicted_vs_actual_cost.png", bbox_inches="tight")
plt.close()

#-- Add a comment: what does it mean when a point falls above the diagonal? What about below?
# When a point falls above the diagonal line it means the model overestimated the patient's cost.
# Predicted cost was greater than the actual cost. 
# When a point falls below the line, the model underestimated the patient's cost. The predicted cost was less 
# than the actual cost. 

