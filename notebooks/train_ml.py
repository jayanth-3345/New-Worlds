
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Load dataset
df = pd.read_csv("data/processed/exoplanets_labelled.csv")

# Convert catalog labels into binary classification
df["target"] = (df["habitability_label"] > 0).astype(int)

features = [
    "pl_rade",
    "pl_eqt",
    "pl_insol",
    "st_teff",
    "st_mass",
    "st_rad"
]

X = df[features]
y = df["target"]
groups = df["host_star"]

# Keep the original random split for comparison
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Dummy Baseline": DummyClassifier(strategy="most_frequent"),

    "Logistic Regression": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        ))
    ]),

    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42
        ))
    ])
}

print("\n========== RANDOM SPLIT RESULTS ==========")

for name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(f"\n{name}")
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Balanced Accuracy:", balanced_accuracy_score(y_test, predictions))
    print("Precision:", precision_score(y_test, predictions, zero_division=0))
    print("Recall:", recall_score(y_test, predictions, zero_division=0))
    print("F1 Score:", f1_score(y_test, predictions, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, predictions))

# Group-aware cross-validation
print("\n========== GROUP-AWARE CROSS-VALIDATION ==========")

cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for name, model in models.items():
    scores = []

    for train_idx, test_idx in cv.split(X, y, groups):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]

        model.fit(X_train_fold, y_train_fold)
        predictions = model.predict(X_test_fold)

        scores.append({
            "accuracy": accuracy_score(y_test_fold, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test_fold, predictions),
            "precision": precision_score(y_test_fold, predictions, zero_division=0),
            "recall": recall_score(y_test_fold, predictions, zero_division=0),
            "f1": f1_score(y_test_fold, predictions, zero_division=0)
        })

    results = pd.DataFrame(scores)

    print(f"\n{name}")
    print(results.mean().round(4))
    print("Standard deviation:")
    print(results.std().round(4))

print("\nEvaluation completed.")
# Save group-aware evaluation summary
summary = []

for name, model in models.items():
    scores = []

    for train_idx, test_idx in cv.split(X, y, groups):
        X_train_fold = X.iloc[train_idx]
        X_test_fold = X.iloc[test_idx]
        y_train_fold = y.iloc[train_idx]
        y_test_fold = y.iloc[test_idx]

        model.fit(X_train_fold, y_train_fold)
        predictions = model.predict(X_test_fold)

        scores.append({
            "accuracy": accuracy_score(y_test_fold, predictions),
            "balanced_accuracy": balanced_accuracy_score(y_test_fold, predictions),
            "precision": precision_score(y_test_fold, predictions, zero_division=0),
            "recall": recall_score(y_test_fold, predictions, zero_division=0),
            "f1": f1_score(y_test_fold, predictions, zero_division=0)
        })

    result = pd.DataFrame(scores).mean().to_dict()
    result["model"] = name
    summary.append(result)

pd.DataFrame(summary).to_csv(
    "data/processed/group_aware_ml_results.csv",
    index=False
)

print("Saved: data/processed/group_aware_ml_results.csv")