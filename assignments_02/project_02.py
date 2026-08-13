#parameters: sep=";", decimal="", encoding="utf-8"
import pandas as pd 
import numpy as np 
from pathlib import Path
from prefect import task, flow, get_run_logger
import matplotlib.pyplot as plt 
import seaborn as sns

#Task 1: Load and Explore 
df = pd.read_csv("assignments_02/student_performance_math.csv", sep=";")

print("Task 1:")
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

#Plot histogram
plt.figure(figsize=(8, 6))
plt.hist(df["G3"], bins=21, edgecolor="black", color="skyblue")
plt.title("Distribution of Final Math Grades")
plt.xlabel("Final Grade")
plt.ylabel("Count")
plt.savefig("assignments_02/outputs/g3_distribution.png")
plt.show()


#Task 2: Preprocess the Data 
df = pd.read_csv("assignments_02/student_performance_math.csv", sep=";")
print("Task 2:")
print("Original Shape:", df.shape)

#Handle G3=0 rows 
df_filtered = df[df["G3"] !=0].copy()
print("Filtered shape:", df_filtered.shape)
# --Why remove G3 rows: 
# The G3=0 means the student was absent during the final exam, so keeping these rows would mix actual grade patterns
# with missing exam cases and cause the model to be distorted. 


#Convert yes/no and sex tp 1/0 in df_filtered (the cleaned dataset)
yes_no_columns = ["schoolsup", "internet", "higher", "activities"]
df[yes_no_columns] = df[yes_no_columns].replace({"yes": 1, "no": 0})
df_filtered[yes_no_columns] = df_filtered[yes_no_columns].replace({"yes": 1, "no": 0})

df["sex"] = df["sex"].replace({"F": 0, "M": 1})
df_filtered["sex"] = df_filtered["sex"].replace({"F": 0, "M": 1})

#Pearson Correlation between absences and G3
corr_original = df["absences"].corr(df["G3"])
corr_filtered = df_filtered["absences"].corr(df_filtered["G3"])

print("Task 2:")
print(f"Correlation (absences vs G3)= orginal data: {corr_original:.4f}")
print(f"Correlation (absences vs G3)= filtered data: {corr_filtered:.4f}")

# --Why filtering changes results?
# In the orginal data it is likely that many students with G3=0 actually had low or relativly low absences
# but still ended up receiving a zero because they skipped the final. Removing the no show students allows this data 
# to reflect a "normal" pattern and helps strenghten correlation. 


#Task 3: Exploratory Data Analysis
numeric_columns = df_filtered.select_dtypes(include=["int64", "float64"]).columns.tolist()
corrs = df_filtered[numeric_columns].corr()["G3"].sort_values()

print("Pearson correlations sorted with G3:")
print(corrs)

#Scatter plot: Absences vs G3 
plt.figure(figsize=(8, 6))
plt.scatter(df_filtered["absences"], df_filtered["G3"], alpha=0.7, color="skyblue")
plt.title("Absences vs final Grade")
plt.xlabel("Absences")
plt.ylabel("Final Grade (G3)")
plt.savefig("assignments_02/outputs/absences_vs_g3.png")
plt.show()

#--- Visulalization Scatter plot 1 comments: 
# After filtering the data the relationship between absences and final grades become clear. 
# It shows that students with more absences tend to have lower G3 scores, althogh the relationship is negative. In the original data
# the correlation was thrown off because being absent for the final, caused
# a misinterpretation when looking at absences and corrleation.  

#Scatter plot: Second period grade vs Final grade (G2 vs G3)
plt.figure(figsize=(8, 6))
plt.scatter(df_filtered["G2"], df_filtered["G3"], alpha=0.7, color="purple")
plt.title("Second period grade (G2) vs Final Grade (G3)")
plt.xlabel("Second period Grade (G2)")
plt.ylabel("Final Grade (G3)")
plt.savefig("assignments_02/outputs/g2_vs_g3.png")
plt.show()

#--- Visualization Scatter plot 2 comments:
# The Second Period Grade is a great predictor of G3. This is because when students perform
# well in the second grading period they will almost always perform well on the final. G2 and G3 have a
# strong positive relationship because good performance early on often leads to great final exam outcomes. 

#Task 4: Baseline Model 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error 

X = df_filtered[["failures"]]
y = df_filtered["G3"]

#train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#fit model 
model = LinearRegression()
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

#metrics 
slope = model.coef_[0]
rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
r2 = model.score(X_test, y_test)

print("Task 4:")
print(f"Slope: {slope:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²: {r2:.4f}")

# Slope: shows how G3 changes with every past faliure.
# RMSE: shows typical error prediction in grade point.  
# R²: shows the change in variation for G3 and how failures account for the variation. 


#Task 5: Build the Full Model 
df_clean = df_filtered.copy()

feature_cols = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures",
                "absences", "freetime", "goout", "Walc", "schoolsup",
                "internet", "higher", "activities", "sex"]
X = df_clean[feature_cols].values
y = df_clean["G3"].values

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Fit model 
model = LinearRegression()
model.fit(X_train, y_train)

y_predicted = model.predict(X_test)

rmse_full = np.sqrt(mean_squared_error(y_test, y_predicted))
r2_train = model.score(X_train, y_train)
r2_test = model.score(X_test, y_test)

print("Task 5:")
print(f"Train R²: {r2_train:.3f}")
print(f"Test R²: {r2_test:.3f}")
print(f"Test RMSE: {rmse_full:.3f}")

print("\nFeature Coefficients:")
for name, coef in zip(feature_cols, model.coef_):
    print(f"{name:12s}: {coef:+.3f}")

# Compare R² baseline:
# - Adding more features helps improve the predictablilty of the model. 
# - The largest positive coefficient is internet (+1.037) the largest negative coefficient is schoolsup (-2.263).
# Suprising signs: 
# - The strong positive coefficients are: studytime, medu, fedu, internet, freetime, sex. And negative coefficients are failures, goout, walc, absences.
#   This was suprising because it shows that freetime and how students spend that free time affects their final grades. 
# Compare train R² and test R²- are they close, is there a gap? What does that tell you about the model?
# - Both train R² and test R² are close with train R² at 0.235 and test R² at 0.263. This tells me that the model is performing well. 
# If you were deploying this model in production, which features would you keep and which would you drop?
# - I would keep the features with larger coefficients and meaningful predictive values, and I would drop weak or unnecessary features.
#   I would keep features such as studytime, failures, goout and absences. I would get rid of weak features that overcomplicates the model such as;
#   activities, and traveltime. 

#Task 6: Evaluate and Summarize 
plt.figure(figsize=(8, 6))
plt.scatter(y_predicted, y_test, alpha=0.7, color="skyblue")
min_val = min(y_test.min(), y_predicted.min())
max_val = max(y_test.max(), y_predicted.max())
plt.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")

plt.title("Predicted vs Actual (Full Model)")
plt.xlabel("Predicted G3")
plt.ylabel("Actual G3")
plt.savefig("assignments_02/outputs/predicted_vs_actual_g3.png")
plt.show()

# Does the model seem to struggle more at the high end, the low end, or is error roughly 
# uniform across grade levels? What does a value above or below the diagonal mean?
# - The error looks to be roughly uniform, there are no major struggles at either end, although there is a slight struggle at the high end.
#   when the models seems to be overestimating the top performing students. Therefore the model performs well.
#   When points are above the diagonal it means the model under predicted (the actual is greater than the predicted). 
#   When points are under/below the diagonal it means the model over predicted (the actual was less than the predicted final grade).

#--- Summary ---
# - The dataset started off as 395 rows but after removing students with G3=0, and using 80/20 split
#   there were 71 rows used for testing. 
# - The RMSE is roughly 2.96 and 2.664, there is a prediction error of 3 points which means that the model could be predicting student scores 3 points higher or lower,
#   it is not a terrible error to have but it is also not precise. 
# - The R² in task 4 was 0.089, meaning failures do correlate with G3 but it is still weak. The R² in task 5 was 0.263, this explains 26% of variation but it is still not strong
#   enough. The R² from task 5 tells us that adding features helped but the model does struggle understanding the complexity of how students perform. 
# - Internet had a positive coefficient of +1.037 and studytime had a +0.311. This means that if a student has internet access they are likely to score a point higher on average, which tells
#   internet access helps with homework and studying. Studytime increases final grades but not as much as we hope to see. The two largest negative coefficients were schoolsup at -2.263 and failures at -0.800.
#   Schoolsup has the largest negative coefficient which means students that get school support score 2.3 points lower and this is because they are already struggling. The failures show how past failures can be a sign of academic
#   struggle. 
# - I was suprised to see that freetime was at +0.014, this was suprising to me because this shows that since freetime was almost at zero. There is no relationship with freetime and final grades. 


#--- Neglected Feature: The Power of G1 ---
feature_columns_g1 = [
    "age", "Medu", "Fedu", "traveltime", "studytime", "failures", "absences", "freetime", "goout", "Walc", "schoolsup",
    "internet", "higher", "activities", "sex", "G1"
]

X = df_clean[feature_columns_g1].values
y = df_clean["G3"].values

#Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#fit model
model_g1 = LinearRegression()
model_g1.fit(X_train, y_train)
#predict
y_predicted = model_g1.predict(X_test)

#G1 R2 
r2_test_g1 = model_g1.score(X_test, y_test)
print("Neglected Feature: The Power of G1")
print(f"Test R² with G1 included: {r2_test_g1:.3f}")

# Does a high R² here mean G1 is causing G3? Is this 
# a useful model for identifying students who might struggle? 
# What might educators need to do if they wanted to intervene early, before G1 is even available?
# - After adding G1 to the model the R² went from 0.263 to 0.765 which is a huge improvement, but this does not mean 
#   G1 is causing G3. This model becomes useful to identify students that might be struggling, but this model can only 
#   be applied once G1 (first period grading) is completed. Because after having G1 teachers are able to see which student
#   needs additional support. If teachers want to intervene before G1 is available teachers would need to use other indicators 
#   such as absences, failures and can even give additional assignments to collect data such as a diagnostic test, and homework. 
#   These are some helpful indicators that can help identify students that are struggling even before any first grade is recorded. 
#   It is very important to note that this model is only useful once G1 exists. 