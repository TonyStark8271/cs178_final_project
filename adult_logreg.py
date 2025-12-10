import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, \
    precision_recall_curve, average_precision_score

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

train = train.dropna(subset=["workclass", "occupation", "native-country"])
test = test.dropna(subset=["workclass", "occupation", "native-country"])
# =========================
# Prepare features and target
# =========================

X_train = train.drop("income", axis=1)
y_train = train["income"].apply(lambda x: 1 if x == ">50K" else 0)

X_test = test.drop("income", axis=1)
y_test = test["income"].apply(lambda x: 1 if x == ">50K." else 0)

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

log_reg = LogisticRegression(
    max_iter=1000,
    solver="lbfgs"
)

clf = Pipeline(
    steps=[
        ("preprocess", preprocess),
        ("model", log_reg)
    ]
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
print("\nLogistic Regression Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

"""
Some plots for better look
"""
# confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = ["<=50K", ">50K"]

plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


# Precision & Recall
prec, recall, thresholds = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)

plt.figure(figsize=(5,5))
plt.plot(recall, prec, label=f"PR curve (AP = {ap:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

# The classifier
preprocess = clf.named_steps["preprocess"]
log_reg = clf.named_steps["model"]

num_features = preprocess.named_transformers_["num"].get_feature_names_out()
cat_features = preprocess.named_transformers_["cat"].get_feature_names_out()

feature_names = list(num_features) + list(cat_features)
coef = log_reg.coef_[0]

classifier_table = pd.DataFrame({
    "feature": feature_names,
    "weight": coef
}).sort_values("weight", ascending=False)
top = classifier_table.reindex(classifier_table.weight.abs().sort_values(ascending=False).index).head(20)

plt.figure(figsize=(8,6))
plt.barh(top["feature"], top["weight"])
plt.axvline(0, color="black")
plt.title("Top Logistic Regression Coefficients (The Classifier)")
plt.xlabel("Weight (Effect on Probability of >50K)")
plt.tight_layout()
plt.show()
