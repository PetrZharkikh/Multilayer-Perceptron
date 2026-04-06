import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features) * 0.1
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        self.x = x
        return self.W @ x + self.b
    
    def backward(self, grad_output):
        self.dW = np.outer(grad_output, self.x)
        self.db = grad_output
        return self.W.T @ grad_output
    
    def update(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class ReLU:
    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        return grad_output * self.mask

class MSELoss:
    def forward(self, y_pred, y_true):
        self.diff = y_pred - y_true
        return np.mean(self.diff**2)
    
    def backward(self):
        return 2 * self.diff / self.diff.size
    
class MLP:
    def __init__(self):
        self.layers = [
            Linear(4, 8),
            ReLU(),
            Linear(8, 4)
        ]

        self.loss_fn = MSELoss()
    
    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def loss(self, y_pred, y_true):
        return self.loss_fn.forward(y_pred, y_true)

    def backward(self):
        grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update(self, lr):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lr)

def target_function(x):
    x1, x2, x3, x4 = x
    return np.array([
        x1**2,
        3*x2,
        5*x4 - x3,
        3
    ])

def generate_data(n):
    X = np.random.randn(n, 4)
    Y = np.array([target_function(x) for x in X])
    return X, Y

mlp = MLP()

x = np.array([1.0, 2.0, 3.0, 4.0])
y_true = target_function(x)

y_pred = mlp.forward(x)
loss = mlp.loss(y_pred, y_true)

print("y_pred =", y_pred)
print("y_true =", y_true)
print("loss =", loss)

mlp.backward()
mlp.update(0.01)

print("done")