import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

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
