import numpy as np
from src.models.custom_mlp import CustomMLP


class CustomMLPWide(CustomMLP):
    """Wide MLP with LeakyReLU"""

    def __init__(self, input_dim):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=[256, 128],
            lr=5e-4,
            epochs=600,
            batch_size=256,
            dropout_rate=0.1,
            l2_lambda=5e-5,
            patience=25,
            lr_patience=8,
            seed=123
        )

    # LeakyReLU
    def relu(self, x):
        return np.where(x > 0, x, 0.01 * x)

    def relu_grad(self, x):
        return np.where(x > 0, 1.0, 0.01)


def build_custom_mlp_wide(input_dim):
    return CustomMLPWide(input_dim)
