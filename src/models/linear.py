# src/models/linear.py
from sklearn.linear_model import LinearRegression, Ridge, Lasso

def build_linear_models():
    return {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001)
    }
