# src/models/baseline.py
import numpy as np

class MeanBaseline:
    def fit(self, y_train):
        self.mean_ = y_train.mean()

    def predict(self, y):
        return np.full_like(y, self.mean_)
