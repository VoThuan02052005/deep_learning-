import os
import joblib
import pandas as pd
import time
from src.features.build_feature import build_features
from src.models.baseline import MeanBaseline
from src.utils.metrics import evaluate_regression
from src.utils.logger import Logger

logger = Logger("train_baseline")

def train_baseline(data_path):
    logger.info("Starting baseline training pipeline")
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Feature engineering
    logger.info("Building features")
    features = build_features(data)
    X_train, X_test = features["X_train"], features["X_test"]
    y_train, y_test = features["y_train"], features["y_test"]
    
    # Training
    logger.info("Training MeanBaseline model")
    model = MeanBaseline()
    start_time = time.time()
    model.fit(y_train)
    train_time = time.time() - start_time
    
    # Evaluation
    logger.info("Evaluating model")
    y_pred = model.predict(y_test)
    metrics = evaluate_regression(y_test, y_pred)
    metrics["train_time"] = train_time
    
    logger.info(f"Results: {metrics}")
    
    # Save model
    os.makedirs("models", exist_ok=True)
    model_path = "models/baseline_model.joblib"
    joblib.dump(model, model_path)
    logger.success(f"Model saved to {model_path}")
    
    return metrics

if __name__ == "__main__":
    DATA_PATH = "data/staging/data_ban.csv"
    train_baseline(DATA_PATH)
