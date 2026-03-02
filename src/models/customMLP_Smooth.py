
import numpy as np
from src.models.custom_mlp import CustomMLP


class CustomMLPSmooth(CustomMLP):
    """Deep & smooth MLP using Tanh activation"""

    def __init__(self, input_dim):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=[128, 64, 32],
            lr=3e-4,
            epochs=700,
            batch_size=256,
            dropout_rate=0.25,
            l2_lambda=2e-4,
            patience=30,
            lr_patience=10,
            seed=999
        )

    def relu(self, x):
        return np.tanh(x)

    def relu_grad(self, x):
        return 1.0 - np.tanh(x) ** 2


def build_custom_mlp_smooth(input_dim):
    return CustomMLPSmooth(input_dim)
