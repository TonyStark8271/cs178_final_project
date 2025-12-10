import pandas as pd

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]


df = pd.read_csv(
    "adult.data",
    names=columns,
    sep=",",
    skipinitialspace=True
)


print(df.head())


print("Dataset shape:", df.shape)

# 1. Basic info
print(df.info())
print(df.describe())

# 2. Check missing values
print("Missing values per column:")
print(df.isna().sum())

# 3. Income distribution (label balance)
print("Income value counts:")
print(df["income"].value_counts())
print("Income value counts (normalized):")
print(df["income"].value_counts(normalize=True))


import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Income Distribution Plot
# =========================
plt.figure(figsize=(6, 4))
sns.countplot(x="income", data=df)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

"""
Figure 1 shows the distribution of income labels in the dataset. 
The dataset is clearly imbalanced, with approximately 75.9% of individuals earning ≤50K and 24.1% earning >50K. 
This imbalance should be taken into account during model evaluation.
"""

# =========================
# Age vs Income Plot
# =========================
plt.figure(num=2, figsize=(6, 4))
sns.boxplot(x="income", y="age", data=df)
plt.title("Age vs Income")
plt.xlabel("Income")
plt.ylabel("Age")
plt.tight_layout()
plt.show()

"""
Figure 2: Box plot showing the distribution of age for different income groups. 
Individuals with income >50K generally have a higher median age and a wider upper age range.
"""


# =========================
# Hours per Week vs Income
# =========================
plt.figure(num=3, figsize=(6, 4))
sns.boxplot(x="income", y="hours-per-week", data=df)
plt.title("Hours per Week vs Income")
plt.xlabel("Income")
plt.ylabel("Hours per Week")
plt.tight_layout()
plt.show()

"""
Figure 3: Box plot showing the distribution of weekly working hours for different income groups. 
Higher-income individuals tend to work more hours per week.
"""
