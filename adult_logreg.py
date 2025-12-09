import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# Load Adult dataset
# =========================

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

print("Dataset shape:", df.shape)

# =========================
# Prepare features and target
# =========================

X = df.drop("income", axis=1)
y = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, drop_first=True)
print("Feature matrix shape after encoding:", X_encoded.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =========================
# Train Logistic Regression
# =========================

print("\nTraining Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# =========================
# Evaluation
# =========================

y_pred = log_reg.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print("\nLogistic Regression Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
