from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, validation_curve
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from load_data_for_trees import X_train, X_test, y_train, y_test, preprocess
from joblib import parallel_backend
import pandas as pd

"""
#--------------------
#  base for test
#--------------------
rf = RandomForestClassifier(
    n_estimators=500,       # number of trees
    max_depth=16,
    min_samples_split=2,
    min_samples_leaf=5,
    max_features="sqrt",
    bootstrap=True,
    class_weight={0:1, 1:2},      # class balancing
    random_state=42,
)

clf_rf = Pipeline(
    steps=[
        ("preprocessor", preprocess),
        ("classifier", rf)
    ]
)

clf_rf.fit(X_train, y_train)

y_pred = clf_rf.predict(X_test)
y_proba = clf_rf.predict_proba(X_test)[:, 1]

# evaluate
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["<=50K", ">50K"],
            yticklabels=["<=50K", ">50K"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Forest – Confusion Matrix")
plt.tight_layout()
plt.show()

ap = average_precision_score(y_test, y_proba)
print("PR-AUC (Average Precision):", ap)


"""



"""

# This thing is running 5D on my laptop, which takes ages...
# comment this, just go to below method.

rf_base = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

clf_rf_base = Pipeline(
    steps=[
        ("preprocessor", preprocess),
        ("classifier", rf_base),
    ]
)

param_grid = {
    "classifier__n_estimators": [200, 500],
    "classifier__max_depth": [None, 8, 12, 16],
    "classifier__min_samples_leaf": [1, 5, 8],
    "classifier__max_features": ["sqrt", "log2"],
    "classifier__class_weight": [None, "balanced", {0:1, 1:2}],
}

grid_rf = GridSearchCV(
    clf_rf_base,
    param_grid=param_grid,
    cv=2,
    scoring="average_precision",
    n_jobs=-1,
    verbose=2,
)
with parallel_backend("threading", n_jobs=6):
    grid_rf.fit(X_train, y_train)
print("best parameters: ", grid_rf.best_params_)
print("best score: ", grid_rf.best_score_)

results = pd.DataFrame(grid_rf.cv_results_)
results.head()


#------------------------
# n estimator
est_scores = (
    results
    .groupby("param_classifier__n_estimators")["mean_test_score"]
    .mean()
    .reset_index()
    .sort_values("param_classifier__n_estimators")
)

plt.figure(figsize=(5,4))
plt.plot(est_scores["param_classifier__n_estimators"],
         est_scores["mean_test_score"], marker="o")
plt.xlabel("n_estimators")
plt.ylabel("CV score")
plt.title("Effect of n_estimators")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#------------------------
# depth
depth_scores = (
    results
    .groupby("param_classifier__max_depth")["mean_test_score"]
    .mean()
    .reset_index()
    .sort_values("param_classifier__max_depth", key=lambda x: x.fillna(100))
)

x = depth_scores["param_classifier__max_depth"].fillna("None")
y = depth_scores["mean_test_score"]

plt.figure(figsize=(5,4))
plt.plot(x, y, marker="o")
plt.xlabel("max_depth")
plt.ylabel("CV score")
plt.title("Effect of max_depth")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#------------------------
# leaf
leaf_scores = (
    results
    .groupby("param_classifier__min_samples_leaf")["mean_test_score"]
    .mean()
    .reset_index()
    .sort_values("param_classifier__min_samples_leaf")
)

plt.figure(figsize=(5,4))
plt.plot(leaf_scores["param_classifier__min_samples_leaf"],
         leaf_scores["mean_test_score"], marker="o")
plt.xlabel("min_samples_leaf")
plt.ylabel("CV score")
plt.title("Effect of min_samples_leaf")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


#------------------------
# weight
cw_scores = (
    results
    .groupby("param_classifier__class_weight")["mean_test_score"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(5,4))
plt.bar(cw_scores["param_classifier__class_weight"].astype(str),
        cw_scores["mean_test_score"])
plt.xlabel("class_weight")
plt.ylabel("CV score")
plt.title("Effect of class_weight")
plt.tight_layout()
plt.show()

"""

rf_base = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    max_features="sqrt",
    class_weight=None,
    random_state=42,
    n_jobs=1,
)

clf_base = Pipeline(
    steps=[
        ("preprocessor", preprocess),
        ("classifier", rf_base),
    ]
)

param_name = "classifier__n_estimators"
param_range = [50, 100, 200, 300, 500]  # smaller for speed

train_scores, val_scores = validation_curve(
    clf_base,
    X_train, y_train,
    param_name=param_name,
    param_range=param_range,
    cv=5,
    scoring="average_precision",
    n_jobs=1,
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(5,4))
plt.plot(param_range, train_mean, marker="o", label="train")
plt.plot(param_range, val_mean, marker="o", label="cv")
plt.xlabel("n_estimators")
plt.ylabel("average_precision")
plt.title("Validation curve: n_estimators")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

param_name = "classifier__max_depth"
param_range = [None, 6, 8, 10, 12, 16]

train_scores, val_scores = validation_curve(
    clf_base,
    X_train, y_train,
    param_name=param_name,
    param_range=param_range,
    cv=5,
    scoring="average_precision",
    n_jobs=1,
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

x_labels = [str(v) for v in param_range]

plt.figure(figsize=(5,4))
plt.plot(x_labels, train_mean, marker="o", label="train")
plt.plot(x_labels, val_mean, marker="o", label="cv")
plt.xlabel("max_depth")
plt.ylabel("average_precision")
plt.title("Validation curve: max_depth")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

param_name = "classifier__min_samples_leaf"
param_range = [1, 3, 5, 8, 12]

train_scores, val_scores = validation_curve(
    clf_base,
    X_train, y_train,
    param_name=param_name,
    param_range=param_range,
    cv=5,
    scoring="average_precision",
    n_jobs=1,
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(5,4))
plt.plot(param_range, train_mean, marker="o", label="train")
plt.plot(param_range, val_mean, marker="o", label="cv")
plt.xlabel("min_samples_leaf")
plt.ylabel("average_precision")
plt.title("Validation curve: min_samples_leaf")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

param_name = "classifier__class_weight"
param_range = [None, "balanced", {0: 1, 1: 2}]

train_scores, val_scores = validation_curve(
    clf_base,
    X_train, y_train,
    param_name=param_name,
    param_range=param_range,
    cv=5,
    scoring="average_precision",
    n_jobs=1,
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

x_labels = [str(v) for v in param_range]

plt.figure(figsize=(5,4))
plt.bar(x_labels, val_mean)
plt.ylabel("average_precision")
plt.xlabel("class_weight")
plt.title("Validation curve: class_weight")
plt.tight_layout()
plt.show()

param_name = "classifier__max_features"
param_range = ["sqrt", "log2", None]

train_scores, val_scores = validation_curve(
    clf_base,
    X_train, y_train,
    param_name=param_name,
    param_range=param_range,
    cv=5,
    scoring="average_precision",
    n_jobs=1,
)

train_mean = train_scores.mean(axis=1)
val_mean = val_scores.mean(axis=1)

plt.figure(figsize=(5,4))
plt.plot(param_range, train_mean, marker="o", label="train")
plt.plot(param_range, val_mean, marker="o", label="cv")
plt.xlabel("max_features")
plt.ylabel("average_precision")
plt.title("Validation curve: max_features")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()