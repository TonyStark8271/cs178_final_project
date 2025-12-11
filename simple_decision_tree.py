from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import validation_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# =========================
# Load Adult dataset
# =========================

columns = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
]

train = pd.read_csv(
    "adult.data",
    header=None,
    names=columns,
    na_values="?",
    skipinitialspace=True
)

test = pd.read_csv(
    "adult.test",
    header=None,
    names=columns,
    na_values="?",
    skipinitialspace=True,
    skiprows=1
)

print(train.shape, test.shape)
train.head()
test.head()

test["income"] = test["income"].str.replace(".", "", regex=False).str.strip()

train = train.dropna(subset=["workclass", "occupation", "native-country"])
test = test.dropna(subset=["workclass", "occupation", "native-country"])

X_train = train.drop("income", axis=1)
y_train = train["income"].apply(lambda x: 1 if x == ">50K" else 0)

X_test = test.drop("income", axis=1)
y_test = test["income"].apply(lambda x: 1 if x == ">50K" else 0)

num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns

# One-hot encode categorical features

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# tuning should be within the classifiers
tree = DecisionTreeClassifier(
    max_depth = None,
    min_samples_split = 20,
    min_samples_leaf = 10,
    random_state = 42,
)

clf_tree = Pipeline(
    steps=[
        ("preprocessor", preprocess),
        ("classifier", tree)
    ]
)

clf_tree.fit(X_train, y_train)
y_pred = clf_tree.predict(X_test)

# validation and tuning

depth_range = [2, 3, 4, 5, 6, 8, 10, 12, None]
split_range = [2, 20, 100, 300, 600]
leaf_range = [1, 2, 10, 50, 150, 300]

train_d, val_d = validation_curve(
    clf_tree,
    X_train,
    y_train,
    param_name="classifier__max_depth",
    param_range=depth_range,
    cv=5,
    scoring="average_precision",
    n_jobs=1
)

depth_train_mean = train_d.mean(axis=1)
depth_val_mean = val_d.mean(axis=1)

depth_x = [d if d is not None else 20 for d in depth_range]
depth_labels = [str(d) if d is not None else "None" for d in depth_range]

train_s, val_s = validation_curve(
    clf_tree,
    X_train,
    y_train,
    param_name="classifier__min_samples_split",
    param_range=split_range,
    cv=5,
    scoring="average_precision",
    n_jobs=1
)

split_train_mean = train_s.mean(axis=1)
split_val_mean = val_s.mean(axis=1)

train_l, val_l = validation_curve(
    clf_tree,
    X_train,
    y_train,
    param_name="classifier__min_samples_leaf",
    param_range=leaf_range,
    cv=5,
    scoring="average_precision",
    n_jobs=1
)

leaf_train_mean = train_l.mean(axis=1)
leaf_val_mean = val_l.mean(axis=1)

# plot the tune

fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# depth
axes[0].plot(depth_x, depth_train_mean, marker="o", label="Train AP")
axes[0].plot(depth_x, depth_val_mean, marker="o", label="Val AP")
axes[0].set_xticks(depth_x)
axes[0].set_xticklabels(depth_labels, rotation=45)
axes[0].set_xlabel("max_depth")
axes[0].set_ylabel("AP-score")
axes[0].set_title("Effect of max_depth")
axes[0].grid(alpha=0.3)
axes[0].legend()

# min_samples_split
axes[1].plot(split_range, split_train_mean, marker="o", label="Train AP")
axes[1].plot(split_range, split_val_mean, marker="o", label="Val AP")
axes[1].set_xlabel("min_samples_split")
axes[1].set_title("Effect of min_samples_split")
axes[1].grid(alpha=0.3)
axes[1].legend()

# min_samples_leaf
axes[2].plot(leaf_range, leaf_train_mean, marker="o", label="Train AP")
axes[2].plot(leaf_range, leaf_val_mean, marker="o", label="Val AP")
axes[2].set_xlabel("min_samples_leaf")
axes[2].set_title("Effect of min_samples_leaf")
axes[2].grid(alpha=0.3)
axes[2].legend()

fig.suptitle("Decision Tree Hyperparameters vs AP (Train vs Validation)", fontsize=14)
plt.tight_layout()
plt.show()



# reports
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["<=50K", ">50K"],
            yticklabels=["<=50K", ">50K"])
plt.title("Decision Tree — Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()



# classification results show:
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose()

cls = report_df.loc[["0", "1"], ["precision", "recall", "f1-score"]]

# Plot
cls.plot(kind="bar", figsize=(8,5))
plt.title("Classification Metrics for Classes 0 and 1")
plt.xlabel("Class")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Metric")
plt.tight_layout()
plt.show()