import numpy as np

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        # ==He初始化==
        # 正态分布, ReLu适用
        rng = np.random.RandomState(42)  # 硬编码种子为42
        self.W = rng.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        # self.W = np.random.randn(input_dim, output_dim) * np.sqrt(2. / input_dim)
        self.b = np.zeros(output_dim)
        self.dW = None
        self.db = None
        self.cache = None

        # # ==Xavier初始化==
        # # 均匀分布, s型激活函数适用
        # limit = np.sqrt(6.0 / (input_dim + output_dim))
        # self.W = np.random.uniform(
        #         low=-limit,
        #         high=limit,
        #         size=(input_dim, output_dim)
        #     ).astype(np.float32)
        # self.b = np.zeros(output_dim)
        # self.dW = None
        # self.db = None
        # self.cache = None

    def forward(self, X):
        self.cache = X
        return X @ self.W + self.b

    def backward(self, dout):
        X = self.cache
        self.dW = X.T @ dout / X.shape[0]
        self.db = np.sum(dout, axis=0) / X.shape[0]
        return dout @ self.W.T

class Activation:
    def __init__(self, activation='relu'):
        self.activation = activation
        self.cache = None
        
    def forward(self, X):
        if self.activation == 'relu':
            out = np.maximum(0, X)
        elif self.activation == 'sigmoid':
            out = 1 / (1 + np.exp(-X))
        elif self.activation == 'tanh':
            out = np.tanh(X)
        else:
            raise ValueError("Unsupported activation")
        self.cache = out  # 缓存激活函数的输出值
        return out
    
    def backward(self, dout):
        out = self.cache
        if self.activation == 'relu':
            grad = (out > 0).astype(float)
        elif self.activation == 'sigmoid':
            grad = out * (1 - out)
        elif self.activation == 'tanh':
            grad = 1 - out**2
        return dout * grad

class SoftmaxCrossEntropy:
    def forward(self, X, y):
        exps = np.exp(X - np.max(X, axis=1, keepdims=True))
        probs = exps / np.sum(exps, axis=1, keepdims=True)
        loss = -np.log(probs[np.arange(len(y)), y]).mean()
        self.cache = (probs, y)
        return loss

    def backward(self):
        probs, y = self.cache
        dX = probs.copy()
        dX[np.arange(len(y)), y] -= 1
        return dX / len(y)

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, activation='relu'):
        """
        三层结构：
        input_dim → hidden_dim1 → hidden_dim2 → output_dim
        """
        self.linear1 = LinearLayer(input_dim, hidden_dim1)
        self.act1 = Activation(activation)

        self.linear2 = LinearLayer(hidden_dim1, hidden_dim2)
        self.act2 = Activation(activation)

        self.linear3 = LinearLayer(hidden_dim2, output_dim)
        
        # 所有层的列表（便于批量操作）
        self.layers = [
            self.linear1, self.act1,
            self.linear2, self.act2,
            self.linear3
        ]
        self.loss = SoftmaxCrossEntropy()

    def forward(self, X):
        out = X
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self):
        dout = self.loss.backward()
        for layer in reversed(self.layers):
            dout = layer.backward(dout)

    def get_regularization_loss(self, reg):
        loss = 0
        for layer in [self.linear1, self.linear2, self.linear3]:
            loss += 0.5 * reg * np.sum(layer.W**2)
        return loss

    def save_weights(self, path):
        weights = {
            'W1': self.linear1.W, 'b1': self.linear1.b,
            'W2': self.linear2.W, 'b2': self.linear2.b,
            'W3': self.linear3.W, 'b3': self.linear3.b
        }
        np.savez(path, **weights)

    def load_weights(self, path):
        data = np.load(path)
        self.linear1.W = data['W1']
        self.linear1.b = data['b1']
        self.linear2.W = data['W2']
        self.linear2.b = data['b2']
        self.linear3.W = data['W3']
        self.linear3.b = data['b3']