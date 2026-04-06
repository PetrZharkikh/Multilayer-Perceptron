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
        self.dW += np.outer(grad_output, self.x)
        self.db += grad_output
        return self.W.T @ grad_output
    
    def update(self, lr, batch_size, l2_lambda):
        self.W -= lr * ((self.dW / batch_size) + l2_lambda * self.W)
        self.b -= lr * (self.db / batch_size)

    def zero_grad(self):
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def clip_grad(self, clip_value):
        self.dW = np.clip(self.dW, -clip_value, clip_value)
        self.db = np.clip(self.db, -clip_value, clip_value)

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
            Linear(4, 16),
            ReLU(),
            Linear(16, 16),
            ReLU(),
            Linear(16, 4)
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

    def update(self, lr, batch_size, l2_lambda):
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lr, batch_size, l2_lambda)

    def zero_grad(self):
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()
    
    def clip_grad(self, clip_value):
        for layer in self.layers:
            if hasattr(layer, "clip_grad"):
                layer.clip_grad(clip_value)


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

def create_batches(X, Y, batch_size):
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    X = X[indices]
    Y = Y[indices]

    for start in range(0, len(X), batch_size):
        end = start + batch_size
        yield X[start:end], Y[start:end]

X, Y = generate_data(1000)

mlp = MLP()
lr = 0.01
epochs = 100
batch_size = 32
l2_lambda = 1e-4

for epoch in range(epochs):
    epoch_loss = 0

    for batch_X, batch_Y in create_batches(X, Y, batch_size):
        mlp.zero_grad()
        batch_loss = 0

        for x, y in zip(batch_X, batch_Y):
            y_pred = mlp.forward(x)
            loss = mlp.loss(y_pred, y)
            mlp.backward()
            batch_loss += loss

        #mlp.clip_grad(1.0)
        mlp.update(lr, len(batch_X), l2_lambda)

        epoch_loss += batch_loss

    avg_loss = epoch_loss / len(X)
    print(f"epoch {epoch + 1}: loss = {avg_loss}")

for i in range(5):
    x = X[i]
    y_true = Y[i]
    y_pred = mlp.forward(x)
    print("x =", x)
    print("y_true =", y_true)
    print("y_pred =", y_pred)
    print()