#%%
#---- Pandas Review ----
#--- Pandas Question 1 ---
import pandas as pd 

data = {
    "name": ["Alice", "Bob", "Carol", "David", "Eve"],
     "grade":  [85, 72, 90, 68, 95],
    "city":   ["Boston", "Austin", "Boston", "Denver", "Austin"],
    "passed": [True, True, True, False, True]
}
df = pd.DataFrame(data)

#--- Pandas Question 1 ---
print("First 3 Rows:")
print(df.head(3))

#--- Pandas Question 2 ---
passed_score = df[(df["passed"] == True) & (df["grade"] > 80)]
print(f"Students Passed: {passed_score}")

#--- Pandas Question 3 ---
df["grade_curved"] = df["grade"] + 5
print(f"New Column: {df[['name', 'grade', 'city', 'passed', 'grade_curved']]}")

#--- Pandas Question 4 ---
df["name_upper"] = df["name"].str.upper()
print(f"Upper Case Name: {df[['name', 'name_upper']]}")

#--- Pandas Question 5 ---
group_by_city = df.groupby("city")["grade"].mean()
print(f"Grade Average by City: {group_by_city}")

#--- Pandas Question 6 ---
df["city"] = df["city"].replace("Austin", "Houston")
print(f"Austin to Houston: {df[['name', 'city']]}")

#--- Pandas Question 7 ---
sort_grade = df.sort_values("grade", ascending=False)
print(f"Top 3 Stuudents by Grade (Descending): {sort_grade.head(3)}")

#%% 
# ---- Numpy Review ----
#--- Numpy Question 1 ---
import numpy as np 
arr_1d = np.array([10, 20, 30, 40, 50])
print("Shape:", arr_1d.shape)
print("Dtype:", arr_1d.dtype) #data type
print("Ndim:", arr_1d.ndim) #number of dimensions

#--- Numpy Question 2 ---
import numpy as np 
arr_2d= np.array([[1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]])
print("Shape:", arr_2d.shape)
print("Size:", arr_2d.size)

#--- Numpy Question 3 ---
print("Top-Left:", arr_2d[0:2, 0:2])

#--- Numpy Question 4 ---
array_zeros = np.zeros((3, 4))
print("3x4 Array of Zeros:", array_zeros)

array_ones = np.ones((2, 5))
print("2x5 Array of Ones:", array_ones)

#--- Numpy Question 5 ---
array_arange = np.arange(0, 50, 5)

print("Array:", array_arange)
print("Shape:", array_arange.shape)
print("Mean:", array_arange.mean())
print("Sum:", array_arange.sum())
print("Standard Deviation:", array_arange.std())

#--- Numpy Question 6 ---
arr_random = np.random.normal(loc=0.0, scale=1.0, size=200)
print("Mean:", arr_random.mean())
print("Standard Deviation:", arr_random.std()) 

# %%
#---- Matplotlib Review ---
#--- Matplotlib Question 1 ---
import matplotlib.pyplot as plt

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.figure(figsize=(6, 4))
plt.plot(x, y, marker='o')

plt.title("Squares")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

#--- Matplotlib Question 2 ---
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

plt.figure(figsize=(6,4))
plt.bar(subjects, scores, color="purple")

plt.title("Subject Scores")
plt.xlabel("Subjects")
plt.ylabel("Scores")
plt.show()

#--- Matplotlib Question 3 ---
x1, y1 = [1, 2, 3, 4, 5], [2, 4, 5, 4, 5]
x2, y2 = [1, 2, 3, 4, 5], [5, 4, 3, 2, 1]

plt.figure(figsize=(6,4))
plt.scatter(x1, y1, color="green", label="Dataset 1")
plt.scatter(x2, y2, color="purple", label="Dataset 2")

plt.title("Two Datasets Scatter Plot")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.legend()
plt.show()

#--- Matplotlib Question 4 ---
#data from question 1
x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

#data from question 2
subjects = ["Math", "Science", "English", "History"]
scores = [88, 92, 75, 83]

fig, (ax1, ax2) = plt.subplots(1, 2)
#ax1 (left --> Line Plot x vs y)
ax1.plot(x, y)
ax1.set_title("Question 1: Line Plot X vs Y")

#ax2 (right --> Bar plot Sugbjects and Scores)
ax2.bar(subjects, scores)
ax2.set_title("Question 2: Subject Scores")
plt.tight_layout()
plt.show()

#%%
#---- Descriptive Statistics Review ----
#--- Descriptive Stats Question 1 ---
import numpy as np
data = [12, 15, 14, 10, 18, 22, 13, 16, 14, 15]

print("Mean:", np.mean(data))
print("Median (even count):", np.median(data))
print("Variance:", np.var(data))
print("Standard:", np.std(data))

#--- Descriptive Stats Question 2 ---
import matplotlib.pyplot as plt

#generate 500 random values 
normal_dist = np.random.normal(loc=65, scale=10, size=500)
plt.figure(figsize=(10,4))

plt.hist(normal_dist, bins=20, color="skyblue", edgecolor="black")
plt.title("Distribution of Scores")
plt.xlabel("Scores")
plt.ylabel("Frequency")
plt.show()

#--- Descriptive Stats Question 3 ---
#Two Box plots
group_a = [55, 60, 63, 70, 68, 62, 58, 65]
group_b = [75, 80, 78, 90, 85, 79, 82, 88]

plt.figure(figsize=(6, 4))
plt.boxplot([group_a, group_b], tick_labels=["Group A", "Group B"])
plt.title("Score Comparison")
plt.ylabel("Value")
plt.show()

#--- Descriptive Stats Question 4 ---
import numpy as np
import matplotlib.pyplot as plt

normal_data = np.random.normal(50, 5, 200)
skewed_data = np.random.exponential(10, 200)

plt.figure(figsize=(6, 4))
plt.boxplot([normal_data, skewed_data], tick_labels=["Normal", "Exponential"])
plt.title("Distribution Comparison")
plt.ylabel("Value")
plt.show()

#--- comment about Distribution ---
#- Which distribution is more skewed?
#  The exponential distribution is more skewed right(longer whisker towards the right).
#  The normal distribution is symmetric.
#
#- Central Tendency?
#  For the normal distribution the Mean is a more appropriate measure of central tendency because the data and box is symmetric and there is no huge outliers.
#  For the exponential distribution the Median is more appropriate because it's strong against the right skew of the tail and outliers. 

#--- Descriptive Stats Question 5 ---
data1 = [10, 12, 12, 16, 18]
data2 = [10, 12, 12, 16, 150]

#Dataset 1: mean, median and mode 
import statistics as stats

print("Data 1- Mean:", np.mean(data1))
print("Data 1- Median:", np.median(data1))
print("Data 1- Mode:", stats.mode(data1))

#Dataset 2: mean, median and mode
print("Data 2- Mean:", np.mean(data2))
print("Data 2- Median:", np.median(data2))
print("Data 2- Mode:", stats.mode(data2))

#--- comment answer ---
#- Why are the median and mean so different for data2?
#  Median is the exact middle number of the dataset when all numbers are placed in order, so the middle number will not be affected by an outlier.
#  While the mean of data2 is large because of the large 150 outlier, since all numbers must be added together and then divided by the total value of numbers. 
#  After finding the average of data2 the mean is affected by the large outlier. 


#%%
#---- Hypothesis Testing Review ----
#--- Hypothesis Question 1 ---
from scipy import stats 

group_a = [72, 68, 75, 70, 69, 73, 71, 74]
group_b = [80, 85, 78, 83, 82, 86, 79, 84]

t_stat, p_val = stats.ttest_ind(group_a, group_b)
print("Hypothesis Q1:")
print("t-statistic:", t_stat)
print("p-value:", p_val)

#--- Hypothesis Question 2 ---
print("Hypothesis Q2:")

if p_val < 0.05: 
    print("The difference is statistically significant.")
else: 
    print("No statistically significant difference detected.")

#--- Hypothesis Question 3 ---
#paired t-test 
before = [60, 65, 70, 58, 62, 67, 63, 66]
after  = [68, 70, 76, 65, 69, 72, 70, 71]

t_stat, p_val = stats.ttest_rel(before, after)
print("Hypothesis Q3:")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value:, {p_val:.6f}")

#--- Hypothesis Question 4 ---
#one-sample t-test 
scores = [72, 68, 75, 70, 69, 74, 71, 73]

t_stat, p_val = stats.ttest_1samp(scores, 70)
print("Hypothesis Q4:")
print(f"t-statistic: {t_stat:.3f}")
print(f"p-value: {p_val:.4f}")

#--- Hypothesis Question 5 ---
print("Hypothesis Q5:")
#one-tail test using Q1 datasets 
t_stat_one_tail, p_val_one_tail = stats.ttest_ind(group_a, group_b, alternative="less")
print(f"t-statistic (one-tailed, group_a < group_b): {t_stat_one_tail:.3f}")
print(f"p-value (one-tailed): {p_val_one_tail:.4f}")


#--- Hypothesis Question 6 ---
print("Hypothesis Q6:")
print("The differenece in average scores is likely due to chance, so we conclude there was an imporvement in study materials.")

#%%
#---- Correlation Review ----
#--- Correlation Question 1 ---
import numpy as np 

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

matrix = np.corrcoef(x, y)
print("Correlation Q1:")
print("Full correlation matrix:")
print(matrix)
print("Correlation coefficient at [0, 1]:")
print(matrix[0, 1])

#--- comment answer ---
#- What do you expect the correlation to be?
#  I expect a positive correlation because the difference between the two datasets is increasing positively. 

#--- Correlation Question 2 ---
from scipy.stats import pearsonr

x = [1, 2, 3, 4, 5, 6, 7, 8, 9,10]
y = [10, 9, 7, 8, 6, 5, 3, 4, 2, 1]
corr, p_value = pearsonr(x, y)

print("Correlation Q2:")
print(f"Correlation Coefficient: {corr}")
print(f"P-value: {p_value}")

#--- Correlation Question 3 ---
import pandas as pd
people = {
    "height": [160, 165, 170, 175, 180],
    "weight": [55,  60,  65,  72,  80],
    "age":    [25,  30,  22,  35,  28]
}
df = pd.DataFrame(people)
corr_matrix = df.corr()
print("Correlation Q3:")
print(corr_matrix)


#--- Correlation Question 4 ---
x = [10, 20, 30, 40, 50]
y = [90, 75, 60, 45, 30]

print("Correlation Q4:")
plt.scatter(x, y, color="blue", marker="o")
plt.title("Negative Correlation")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.show()


#--- Correlation Question 5 ---
#using correlation matrix from Q3
import seaborn as sns

sns.heatmap(corr_matrix, annot=True)
print("Correlation Q5:")
plt.title("Correlation Heatmap")
plt.show()

# %%
#---- Pipelines ----
#--- Pipeline Question 1 ---
import numpy as np
import pandas as pd

arr = np.array([12.0, 15.0, np.nan, 14.0, 10.0, np.nan, 18.0, 14.0, 16.0, 22.0, np.nan, 13.0])

def create_series(arr):
    return pd.Series(arr, name="values")

def clean_data(series):
    return series.dropna()

def summarize_data(series):
    return {
        "mean": float(series.mean()),
        "median": float(series.median()),
        "std": float(series.std()),
        "mode": float(series.mode()[0]),
    }

def data_pipeline(arr):
    s = create_series(arr)
    cleaned = clean_data(s)
    return summarize_data(cleaned)

print("Pipeline Q1:")
result = data_pipeline(arr)

for k, v in result.items():
    print(f"{k}: {v}")
# %%
