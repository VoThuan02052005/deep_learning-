import pandas as pd
import joblib
import os
from src.features.build_feature import build_features
from src.utils.logger import Logger

logger = Logger("save_preprocessor")

def save_preprocessor(data_path, output_path):
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    logger.info("Building features to extract preprocessor objects")
    features = build_features(data)
    
    preprocessor = {
        "scaler_X": features["scaler_X"],
        "scaler_y": features["scaler_y"],
        "ohe": features["ohe"],
        "num_cols": features["num_cols"],
        "cat_cols": features["cat_cols"],
        "feature_names": features["X_train"].columns.tolist()
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(preprocessor, output_path)
    logger.success(f"Preprocessor objects saved to {output_path}")

if __name__ == "__main__":
    DATA_PATH = "data/staging/data_ban.csv"
    OUTPUT_PATH = "models/preprocessor.pkl"
    save_preprocessor(DATA_PATH, OUTPUT_PATH)
