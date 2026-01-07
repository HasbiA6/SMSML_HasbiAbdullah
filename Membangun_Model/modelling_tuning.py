import sys
sys.stdout.reconfigure(encoding="utf-8")

import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===============================
# DAGSHUB + MLFLOW (FINAL)
# ===============================
dagshub.init(
    repo_owner="HasbiA6",
    repo_name="Eksperimen_SML_HasbiAbdullah",
    mlflow=True
)

mlflow.set_tracking_uri(
    "https://dagshub.com/HasbiA6/Eksperimen_SML_HasbiAbdullah.mlflow"
)

# 🔥 LANGSUNG SET EXPERIMENT (TANPA CEK APA PUN)
mlflow.set_experiment("Basic_Insurance_Regression")

print("Tracking URI:", mlflow.get_tracking_uri())

# ===============================
# LOAD DATA
# ===============================
data = pd.read_csv("data_preprocessed.csv")

X = data.drop("charges", axis=1)
y = data["charges"]

num_features = ["age", "bmi", "children"]
cat_features = ["sex", "smoker", "region"]

preprocessor = ColumnTransformer([
    ("num", "passthrough", num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

pipeline = Pipeline([
    ("preprocess", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

param_grid = {
    "model__n_estimators": [100],
    "model__max_depth": [None, 10]
}

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

grid = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="r2",
    n_jobs=-1
)

# ===============================
# TRAINING + LOGGING
# ===============================
with mlflow.start_run(run_name="RandomForest_Tuning"):
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(best_model, "model")

    plt.figure()
    plt.scatter(y_test, preds, alpha=0.5)
    plt.xlabel("Actual Charges")
    plt.ylabel("Predicted Charges")
    plt.title("Actual vs Predicted")
    plt.savefig("prediction_plot.png")

    mlflow.log_artifact("prediction_plot.png")
    mlflow.log_artifact("data_preprocessed.csv")

    print("Run selesai & tersimpan di DagsHub")