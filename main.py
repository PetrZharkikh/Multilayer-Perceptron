# Реализация многослойного перцептрона (MLP) с нуля на NumPy.
# Включает:
# - Linear слои
# - ReLU активацию
# - MSE Loss
# - Обратное распространение ошибки (backpropagation)
# - Обучение с помощью градиентного спуска
# - Батчи, аккумуляцию градиентов, L2-регуляризацию и gradient clipping

import numpy as np

# Линейный слой: y = W x + b
# Хранит параметры W и b, а также их градиенты

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(out_features, in_features) * 0.1
        self.b = np.zeros(out_features)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        # Сохраняем вход x для использования в backward
        self.x = x
        return self.W @ x + self.b
    
    def backward(self, grad_output):
        # Вычисляем градиенты по параметрам слоя:
        # dW = grad_output ⊗ x (outer product)
        # db = grad_output
        # Возвращаем градиент по входу для предыдущего слоя
        self.dW += np.outer(grad_output, self.x)
        self.db += grad_output
        return self.W.T @ grad_output
    
    def update(self, lr, batch_size, l2_lambda):
        # Обновление параметров по правилу градиентного спуска
        # Добавлена L2-регуляризация для весов
        self.W -= lr * ((self.dW / batch_size) + l2_lambda * self.W)
        self.b -= lr * (self.db / batch_size)

    def zero_grad(self):
        # Обнуляем накопленные градиенты перед новым батчем
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def clip_grad(self, clip_value):
        # Ограничиваем значения градиентов (gradient clipping)
        # Это помогает избежать взрывных градиентов
        self.dW = np.clip(self.dW, -clip_value, clip_value)
        self.db = np.clip(self.db, -clip_value, clip_value)

class ReLU:
    def forward(self, x):
        # Запоминаем маску положительных элементов
        self.mask = x > 0
        return np.maximum(0, x)
    
    def backward(self, grad_output):
        # Градиент проходит только через положительные элементы
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
        # Обнуляем градиенты во всех слоях перед новым батчем
        for layer in self.layers:
            if hasattr(layer, "zero_grad"):
                layer.zero_grad()
    
    def clip_grad(self, clip_value):
        # Применяем clipping ко всем слоям
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

        mlp.clip_grad(1.0)
        # Ограничиваем градиенты
        # Обновляем веса по градиентному спуску
        mlp.update(lr, len(batch_X), l2_lambda)

        epoch_loss += batch_loss

    avg_loss = epoch_loss / len(X)
    print(f"epoch {epoch + 1}: loss = {avg_loss}")

# Проверка качества модели на нескольких примерах
for i in range(5):
    x = X[i]
    y_true = Y[i]
    y_pred = mlp.forward(x)
    print("x =", x)
    print("y_true =", y_true)
    print("y_pred =", y_pred)
    print()