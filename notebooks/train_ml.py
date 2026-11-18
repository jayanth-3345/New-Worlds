import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("New-Worlds-Exoplanet-Classification")

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
scale_pos_weight = float((y == 0).sum() / max((y == 1).sum(), 1))

print("Class imbalance ratio:", scale_pos_weight)

# Random train-test split
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
    ]),

    "XGBoost": Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=2

    ))
])
}
# Group-aware cross-validation
cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

summary = []

for name, model in models.items():

    print(f"\n========== {name} ==========")

    with mlflow.start_run(run_name=name):

        # Log experiment parameters
        mlflow.log_param("model", name)
        mlflow.log_param("dataset_rows", len(df))
        mlflow.log_param("features", ", ".join(features))
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("split_strategy", "StratifiedGroupKFold")
        mlflow.log_param("random_state", 42)

        # Random split evaluation
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        random_metrics = {
            "random_accuracy": accuracy_score(y_test, predictions),
            "random_balanced_accuracy": balanced_accuracy_score(y_test, predictions),
            "random_precision": precision_score(y_test, predictions, zero_division=0),
            "random_recall": recall_score(y_test, predictions, zero_division=0),
            "random_f1": f1_score(y_test, predictions, zero_division=0)
        }

        print("\nRandom split results:")
        for metric, value in random_metrics.items():
            print(f"{metric}: {value:.4f}")
            mlflow.log_metric(metric, value)

        print("Confusion matrix:\n", confusion_matrix(y_test, predictions))

        # Group-aware cross-validation
        scores = []

        for train_idx, test_idx in cv.split(X, y, groups):
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]

            model.fit(X_train_fold, y_train_fold)
            fold_predictions = model.predict(X_test_fold)

            scores.append({
                "accuracy": accuracy_score(y_test_fold, fold_predictions),
                "balanced_accuracy": balanced_accuracy_score(y_test_fold, fold_predictions),
                "precision": precision_score(y_test_fold, fold_predictions, zero_division=0),
                "recall": recall_score(y_test_fold, fold_predictions, zero_division=0),
                "f1": f1_score(y_test_fold, fold_predictions, zero_division=0)
            })

        results = pd.DataFrame(scores)
        means = results.mean()
        stds = results.std()

        print("\nGroup-aware CV mean:")
        print(means.round(4))
        print("\nGroup-aware CV standard deviation:")
        print(stds.round(4))

        # Log cross-validation metrics
        for metric in means.index:
            mlflow.log_metric(f"cv_{metric}_mean", means[metric])
            mlflow.log_metric(f"cv_{metric}_std", stds[metric])

        # Save results for comparison
        row = means.to_dict()
        row["model"] = name
        summary.append(row)

        # Train final model on all available data and log it
             # Train final model on all available data
        model.fit(X, y)

        # Log model using the appropriate MLflow flavor
        if name == "XGBoost":
            mlflow.xgboost.log_model(
                model.named_steps["model"],
                name="model"
            )
        else:
            mlflow.sklearn.log_model(
                model,
                name="model",
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree"
                ]
            )

        print("\nMLflow run logged successfully.")

# Save group-aware evaluation summary
os.makedirs("data/processed", exist_ok=True)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(
    "data/processed/group_aware_ml_results.csv",
    index=False
)

mlflow.log_artifact("data/processed/group_aware_ml_results.csv")

print("\nSaved: data/processed/group_aware_ml_results.csv")
print("\nAll experiments completed and logged in MLflow.")
