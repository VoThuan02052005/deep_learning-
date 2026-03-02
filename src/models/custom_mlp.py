import numpy as np
import copy
class CustomMLP:
    def __init__(
        self,
        input_dim,
        hidden_dims=[128, 64],
        output_dim=1,
        lr=1e-3,
        epochs=500,
        batch_size=256,
        l2_lambda=1e-4,
        dropout_rate=0.2,
        patience=20,
        lr_patience=6,
        lr_decay=0.5,
        min_lr=1e-6,
        seed=42
    ):
        np.random.seed(seed)

        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2_lambda = l2_lambda
        self.dropout_rate = dropout_rate
        self.patience = patience
        self.lr_patience = lr_patience
        self.lr_decay = lr_decay
        self.min_lr = min_lr

        dims = [input_dim] + hidden_dims + [output_dim]
        self.weights = [np.random.randn(dims[i], dims[i+1]) * np.sqrt(2./dims[i])
                        for i in range(len(dims)-1)]
        self.biases = [np.zeros((1, dims[i+1])) for i in range(len(dims)-1)]

        self.m_w = [np.zeros_like(w) for w in self.weights]
        self.v_w = [np.zeros_like(w) for w in self.weights]
        self.m_b = [np.zeros_like(b) for b in self.biases]
        self.v_b = [np.zeros_like(b) for b in self.biases]
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.t = 0

        self.train_loss_history = []
        self.val_loss_history = []

    def relu(self, x): return np.maximum(0, x)
    def relu_grad(self, x): return (x > 0).astype(float)

    def forward(self, X, training=True):
        self.zs, self.activations, self.masks = [], [X], []
        for i in range(len(self.weights)-1):
            z = self.activations[-1] @ self.weights[i] + self.biases[i]
            a = self.relu(z)
            if training:
                mask = (np.random.rand(*a.shape) > self.dropout_rate)
                a = a * mask / (1 - self.dropout_rate)
            else:
                mask = None
            self.zs.append(z)
            self.activations.append(a)
            self.masks.append(mask)

        z_out = self.activations[-1] @ self.weights[-1] + self.biases[-1]
        self.zs.append(z_out)
        self.activations.append(z_out)
        return z_out

    def mse_loss(self, y_pred, y_true):
        mse = np.mean((y_pred - y_true)**2)
        l2 = self.l2_lambda * sum(np.sum(w**2) for w in self.weights)
        return mse + l2

    def backward(self, y_true):
        if len(y_true.shape) == 1:
            y_true = np.expand_dims(y_true, axis=0)
        grads_w, grads_b = [], []
        m = y_true.shape[0]
        delta = (self.activations[-1] - y_true) / m

        for i in reversed(range(len(self.weights))):
            dw = self.activations[i].T @ delta + 2*self.l2_lambda*self.weights[i]
            db = np.sum(delta, axis=0, keepdims=True)
            dw = np.clip(dw, -5, 5)

            grads_w.insert(0, dw)
            grads_b.insert(0, db)

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.relu_grad(self.zs[i-1])
                if self.masks[i-1] is not None:
                    delta *= self.masks[i-1] / (1 - self.dropout_rate)
        return grads_w, grads_b

    def adam_update(self, grads_w, grads_b):
        self.t += 1
        for i in range(len(self.weights)):
            self.m_w[i] = self.beta1*self.m_w[i] + (1-self.beta1)*grads_w[i]
            self.v_w[i] = self.beta2*self.v_w[i] + (1-self.beta2)*(grads_w[i]**2)
            m_hat = self.m_w[i] / (1 - self.beta1**self.t)
            v_hat = self.v_w[i] / (1 - self.beta2**self.t)
            self.weights[i] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            self.m_b[i] = self.beta1*self.m_b[i] + (1-self.beta1)*grads_b[i]
            self.v_b[i] = self.beta2*self.v_b[i] + (1-self.beta2)*(grads_b[i]**2)
            mb_hat = self.m_b[i] / (1 - self.beta1**self.t)
            vb_hat = self.v_b[i] / (1 - self.beta2**self.t)
            self.biases[i] -= self.lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

    def fit(self, X_train, y_train, X_val, y_val):
        best_val = np.inf
        wait = lr_wait = 0

        for epoch in range(1, self.epochs+1):
            idx = np.random.permutation(len(X_train))
            X_train, y_train = X_train[idx], y_train[idx]

            for i in range(0, len(X_train), self.batch_size):
                xb = X_train[i:i+self.batch_size]
                yb = y_train[i:i+self.batch_size]
                self.forward(xb, training=True)
                gw, gb = self.backward(yb)
                self.adam_update(gw, gb)

            train_loss = self.mse_loss(self.forward(X_train, False), y_train)
            val_loss   = self.mse_loss(self.forward(X_val, False), y_val)

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            print(f"Epoch {epoch:03d} | Train {train_loss:.4f} | Val {val_loss:.4f}")

            if val_loss < best_val:
                best_val = val_loss
                best_w = copy.deepcopy(self.weights)
                best_b = copy.deepcopy(self.biases)
                wait = lr_wait = 0
            else:
                wait += 1
                lr_wait += 1

            if lr_wait >= self.lr_patience:
                self.lr = max(self.lr*self.lr_decay, self.min_lr)
                lr_wait = 0
                print("Reduce LR")

            if wait >= self.patience:
                print("Early stopping")
                self.weights, self.biases = best_w, best_b
                break

    def predict(self, X):
        return self.forward(X, training=False).ravel()
def build_custom_mlp(input_dim):

    return CustomMLP(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],
        output_dim=1,
        lr=1e-3,
        epochs=2000,
        batch_size=512,
        l2_lambda=1e-4,
        dropout_rate=0.2,
        patience=30,
        lr_patience=8,
        lr_decay=0.5,
        min_lr=1e-6
    )

