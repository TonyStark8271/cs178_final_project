import pandas as pd
import numpy as np
import seaborn as sns
sns.set_style("whitegrid")
import scipy.stats as ss

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# -------------------------
# Load dataset | dataset shape
# -------------------------

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

df = pd.read_csv(
    "adult.data",
    header=None,
    names=columns,
    na_values="?",
    skipinitialspace=True
)

print("Shape: ", df.shape)
print(df.head())

#---------------------------
#   Income distribution
#---------------------------

plt.figure(figsize=(10, 10))
df["income"].value_counts().plot(kind="bar")
plt.xlabel("Income")
plt.ylabel("Count")
plt.show()


#--------------------------
# Numeric feature distro
#-------------------------
num_cols = df.select_dtypes(include=["int64", "float64"]).columns

for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col].dropna(), bins=30, kde=True)
    plt.title(f"Distribution of {col}")
    plt.tight_layout()
    plt.show()


#-------------------------------
# Categorical feature distro
#-------------------------------
cat_cols = df.select_dtypes(include=["object"]).columns.drop("income")

for col in cat_cols:
    plt.figure(figsize=(8,4))
    df[col].value_counts().head(15).plot(kind="bar")
    plt.title(f"Top categories in {col}")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


#-------------------------------
# Relationship with income
# grouped stats & bar plots
#-------------------------------

# Numeric features vs income (means)
print("\nMean of numeric features by income:")
print(df.groupby("income")[num_cols].mean())

for col in num_cols:
    plt.figure(figsize=(4,8))
    sns.boxplot(x="income", y=col, data=df)
    plt.title(f"{col} by income")
    plt.tight_layout()
    plt.show()

# Categorical features vs income (normalized)
for col in cat_cols:
    ct = pd.crosstab(df[col], df["income"], normalize="index")
    ct.sort_values(">50K" if ">50K" in ct.columns else ct.columns[-1], ascending=False, inplace=True)
    ct.head(10).plot(kind="bar", stacked=True)
    plt.title(f"Income proportion by {col} (top 10)")
    plt.ylabel("Proportion")
    plt.tight_layout()
    plt.show()

#-------------------------------
# Correlation matrix (numeric)
#-------------------------------
# Encode income as 0/1 to include in correlation,
# somehow this looks like a relu activation in nural network,
# we'll see, the data size is comparatively small for neural
# network training, which could be a cause for low precision
# in neural networks in the baseline model plot.
df_corr = df.copy()
df_corr["income_binary"] = (df_corr["income"].str.contains(">50K")).astype(int)

corr = df_corr[num_cols.tolist() + ["income_binary"]].corr()

plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Matrix (numeric features + income)")
plt.tight_layout()
plt.show()


#--------------------------------
# give one-hot encoding a chance
#--------------------------------

df_onehot = pd.get_dummies(df, drop_first=True)

corr = df_onehot.corr()

plt.figure(figsize=(14,12))
sns.heatmap(corr, cmap="coolwarm")
plt.title("Correlation heatmap with one-hot encoded features")
plt.show()


#--------------------------------
# Cramer's V
#--------------------------------

# Function to calculate Cramer's V
# apparently this thing is introduced to us by chatGPT
# we queried:"
# how can I combine the other 7 non-numerical features
# and there correlations also with the income?"
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    r,k = confusion_matrix.shape
    return np.sqrt((chi2/n) / (min(k-1,r-1)))

cat_cols = df.select_dtypes(include='object').columns.drop('income')

for col in cat_cols:
    v = cramers_v(df[col], df['income'])
    print(f"{col}: {v:.3f}")

vals = {col: cramers_v(df[col], df['income']) for col in cat_cols}
pd.Series(vals).sort_values().plot(kind='barh', figsize=(7,4))
plt.title("Categorical feature association with income (Cramer's V)")
plt.show()