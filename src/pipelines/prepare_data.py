# import os
# import joblib
# import pandas as pd
# import time
# from src.features.build_feature import build_features
# from src.models.baseline import MeanBaseline
# from src.utils.metrics import evaluate_regression
# from src.utils.logger import Logger

# logger = Logger("train_baseline")

# def train_baseline(data_path):
#     logger.info("Starting baseline training pipeline")
    
#     # Load data
#     logger.info(f"Loading data from {data_path}")
#     data = pd.read_csv(data_path)
    
#     # Feature engineering
#     logger.info("Building features")
#     features = build_features(data)
#     X_train, X_test = features["X_train"], features["X_test"]
#     y_train, y_test = features["y_train"], features["y_test"]
# if __name__ == "__main__":
#     DATA_PATH = "data/staging/data_sau_clean.csv"
#     train_baseline(DATA_PATH)
import os
import joblib
import pandas as pd
import time
from src.features.build_feature import build_features
from src.models.baseline import MeanBaseline
from src.utils.metrics import evaluate_regression
from src.utils.logger import Logger

logger = Logger("train_baseline")

def tranfomer_data(data_path, train_path , test_path):
    logger.info("Starting baseline training pipeline")
    
    # Load data
    logger.info(f"Loading data from {data_path}")
    data = pd.read_csv(data_path)
    
    # Feature engineering
    logger.info("Building features")
    features = build_features(data)

    X_train, X_test = features["X_train"], features["X_test"]
    y_train, y_test = features["y_train"], features["y_test"]

    # Create processed folder
    os.makedirs("data/processed", exist_ok=True)

    # Save processed data
    logger.info("Saving processed data")


    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    y_train = pd.Series(y_train, name="target")
    y_test = pd.Series(y_test, name="target")

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    logger.info("Processed data saved")

if __name__ == "__main__":
    DATA_PATH = "data/staging/data_ban_sau_clean.csv"
    TRAIN_PATH = "data/processed/sell/train_processed.csv"
    TEST_PATH = "data/processed/sell/test_processed.csv"
    tranfomer_data(DATA_PATH, TRAIN_PATH, TEST_PATH)
    DATA_PATH = "data/staging/data_cho_thue.csv"
    TRAIN_PATH = "data/processed/rent/train_processed.csv"
    TEST_PATH = "data/processed/rent/test_processed.csv"
    tranfomer_data(DATA_PATH, TRAIN_PATH, TEST_PATH)