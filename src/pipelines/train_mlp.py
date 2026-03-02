import os
import joblib
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split

from src.features.build_feature import build_features
from src.models.custom_mlp import build_custom_mlp
from src.utils.metrics import evaluate_regression
from src.utils.logger import Logger

logger = Logger("train_mlp")

def inverse_price(y, scaler):
    y = scaler.inverse_transform(y.reshape(-1,1))
    return np.expm1(y)
def train_mlp(data_path):
    logger.info("Starting MLP training pipeline")

    # --------------------
    # Load data
    # --------------------
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)

    # --------------------
    # Feature engineering
    # --------------------
    logger.info("Building features")
    features = build_features(data)
    y_scaler = features["scaler_y"]
    X_train, X_test = features["X_train"], features["X_test"]
    y_train, y_test = features["y_train"], features["y_test"]

    # --------------------
    # Split TRAIN → TRAIN + VAL
    # --------------------
    logger.info("Splitting training data for validation")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )

    # --------------------
    # Ensure numpy arrays
    # --------------------
    X_train = np.asarray(X_train)
    X_val   = np.asarray(X_val)
    X_test  = np.asarray(X_test)

    y_train = np.asarray(y_train).reshape(-1, 1)
    y_val   = np.asarray(y_val).reshape(-1, 1)
    y_test  = np.asarray(y_test).reshape(-1, 1)

    # --------------------
    # Training
    # --------------------
    logger.info("Initializing CustomMLP")
    model = build_custom_mlp(X_train.shape[1])

    logger.info("Training CustomMLP model")
    start_time = time.time()
    model.fit(X_train, y_train, X_val, y_val)
    train_time = time.time() - start_time

    # --------------------
    # Evaluation
    # --------------------
    logger.info("Evaluating CustomMLP")

    # Train
    y_train_pred = model.predict(X_train)
    train_metrics = evaluate_regression(
    inverse_price(y_train, y_scaler),
    inverse_price(y_train_pred, y_scaler)
    )


    # Validation
    y_val_pred = model.predict(X_val)

    val_metrics = evaluate_regression(
    inverse_price(y_val, y_scaler),
    inverse_price(y_val_pred, y_scaler)
    )
    # Test
    y_test_pred = model.predict(X_test)
    test_metrics = evaluate_regression(
    inverse_price(y_test, y_scaler),
    inverse_price(y_test_pred, y_scaler)
)

    metrics = {
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
        "train_time": train_time,
        "n_train": len(y_train),
        "n_val": len(y_val),
        "n_test": len(y_test)
    }

    logger.info(f"[TRAIN] {train_metrics}")
    logger.info(f"[VAL]   {val_metrics}")
    logger.info(f"[TEST]  {test_metrics}")
    logger.info(f"Training time: {train_time:.2f}s")

    # --------------------
    # Save model
    # --------------------
    os.makedirs("models", exist_ok=True)
    model_path = "models/mlp_model.joblib"
    joblib.dump(model, model_path)
    logger.success(f"MLP model saved to {model_path}")

    return metrics


if __name__ == "__main__":
    DATA_PATH = "data/staging/data_ban.csv"
    train_mlp(DATA_PATH)
