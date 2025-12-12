import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    average_precision_score,
)
from sklearn.model_selection import validation_curve

# =========================================================
# Helper: ensure plot directory exists
# =========================================================
PLOT_DIR = os.path.join("polts", "SVM")
os.makedirs(PLOT_DIR, exist_ok=True)


def save_fig(filename: str):
    """Save current matplotlib figure into polts/SVM."""
    path = os.path.join(PLOT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Saved] {path}")


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
    skipinitialspace=True,
)

test = pd.read_csv(
    "adult.test",
    header=None,
    names=columns,
    na_values="?",
    skipinitialspace=True,
    skiprows=1,
)

print("Train / Test shape:", train.shape, test.shape)

# 修正测试集标签里的 "."
test["income"] = test["income"].str.replace(".", "", regex=False).str.strip()

# 去掉关键类别中含缺失值的行（跟队友保持一致）
train = train.dropna(subset=["workclass", "occupation", "native-country"])
test = test.dropna(subset=["workclass", "occupation", "native-country"])

X_train = train.drop("income", axis=1)
y_train = train["income"].apply(lambda x: 1 if x == ">50K" else 0)

X_test = test.drop("income", axis=1)
y_test = test["income"].apply(lambda x: 1 if x == ">50K" else 0)

# =========================
# Preprocessing
# =========================

num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X_train.select_dtypes(include=["object"]).columns

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ]
)

# ===================================================
# 1. Linear SVM (baseline)
# ===================================================

linear_svm = Pipeline(
    steps=[
        ("preprocessor", preprocess),
        ("classifier", LinearSVC(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
        )),
    ]
)

linear_svm.fit(X_train, y_train)
y_pred_lin = linear_svm.predict(X_test)
y_score_lin = linear_svm.decision_function(X_test)

print("\n===== Linear SVM =====")
print(classification_report(y_test, y_pred_lin))

ap_lin = average_precision_score(y_test, y_score_lin)
print(f"Average Precision (PR-AUC): {ap_lin:.3f}")

# ---- Confusion Matrix: Linear SVM ----
cm_lin = confusion_matrix(y_test, y_pred_lin)
plt.figure(figsize=(4, 4))
sns.heatmap(
    cm_lin,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["<=50K", ">50K"],
    yticklabels=["<=50K", ">50K"],
)
plt.title("Linear SVM — Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
save_fig("linear_confusion_matrix.png")

# ---- PR Curve: Linear SVM ----
prec_lin, rec_lin, _ = precision_recall_curve(y_test, y_score_lin)
plt.figure(figsize=(5, 5))
plt.plot(rec_lin, prec_lin, label=f"Linear SVM (AP = {ap_lin:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve — Linear SVM")
plt.legend(loc="lower left")
save_fig("linear_pr_curve.png")

# ===================================================
# 2. RBF kernel SVM
# ================================================

rbf_svm = Pipeline(
    steps=[
        ("preprocessor", preprocess),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
        )),
    ]
)

rbf_svm.fit(X_train, y_train)
y_pred_rbf = rbf_svm.predict(X_test)
y_score_rbf = rbf_svm.decision_function(X_test)

print("\n===== RBF SVM (baseline) =====")
print(classification_report(y_test, y_pred_rbf))

ap_rbf = average_precision_score(y_test, y_score_rbf)
print(f"Average Precision (PR-AUC): {ap_rbf:.3f}")

# ---- Confusion Matrix: RBF SVM ----
cm_rbf = confusion_matrix(y_test, y_pred_rbf)
plt.figure(figsize=(4, 4))
sns.heatmap(
    cm_rbf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["<=50K", ">50K"],
    yticklabels=["<=50K", ">50K"],
)
plt.title("RBF SVM — Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
save_fig("rbf_confusion_matrix.png")

# ---- PR Curve: RBF SVM ----
prec_rbf, rec_rbf, _ = precision_recall_curve(y_test, y_score_rbf)
plt.figure(figsize=(5, 5))
plt.plot(rec_rbf, prec_rbf, label=f"RBF SVM (AP = {ap_rbf:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve — RBF SVM")
plt.legend(loc="lower left")
save_fig("rbf_pr_curve.png")

# ===================================================
# 3. Validation curve for C (RBF SVM)
# ===================================================

C_range = np.logspace(-2, 2, 5)  # [0.01, 0.1, 1, 10, 100]

model_for_C = Pipeline(
    steps=[
        ("preprocessor", preprocess),
        ("classifier", SVC(
            kernel="rbf",
            gamma=0.01,
            class_weight="balanced",
        )),
    ]
)

train_scores_C, val_scores_C = validation_curve(
    model_for_C,
    X_train,
    y_train,
    param_name="classifier__C",
    param_range=C_range,
    cv=3,
    scoring="average_precision",
    n_jobs=1,
)

train_mean_C = train_scores_C.mean(axis=1)
val_mean_C = val_scores_C.mean(axis=1)

plt.figure(figsize=(6, 4))
plt.semilogx(C_range, train_mean_C, marker="o", label="Train AP")
plt.semilogx(C_range, val_mean_C, marker="o", label="Validation AP")
plt.xlabel("C")
plt.ylabel("Average Precision")
plt.title("Validation Curve for C (RBF SVM, gamma=0.01)")
plt.grid(alpha=0.3)
plt.legend()
save_fig("rbf_validation_curve_C.png")

print("\nAll SVM plots saved under:", PLOT_DIR)
