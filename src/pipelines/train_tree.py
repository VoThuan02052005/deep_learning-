import os
import joblib
import pandas as pd
import time
from src.features.build_feature import build_features
from src.models.tree import build_tree_models
from src.utils.metrics import evaluate_regression
from src.utils.logger import Logger

logger = Logger("train_tree")

def train_tree(data_path):
    logger.info("Starting tree models training pipeline")
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Feature engineering
    logger.info("Building features")
    features = build_features(data)
    X_train, X_test = features["X_train"], features["X_test"]
    y_train, y_test = features["y_train"], features["y_test"]
    
    models = build_tree_models()
    results = {}
    
    # Training and Evaluation
    for name, model in models.items():
        logger.info(f"Training {name} model")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        logger.info(f"Evaluating {name}")
        y_pred = model.predict(X_test)
        metrics = evaluate_regression(y_test, y_pred)
        metrics["train_time"] = train_time
        results[name] = metrics
        
        logger.info(f"{name} Results: {metrics}")
        
        # Save each model
        os.makedirs("models", exist_ok=True)
        model_path = f"models/tree_{name.lower()}_model.joblib"
        joblib.dump(model, model_path)
        logger.success(f"{name} model saved to {model_path}")
    
    return results

if __name__ == "__main__":
    DATA_PATH = "data/staging/data_ban.csv"
    train_tree(DATA_PATH)
