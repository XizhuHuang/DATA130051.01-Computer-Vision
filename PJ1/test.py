import numpy as np
from model import NeuralNetwork, SoftmaxCrossEntropy, LinearLayer
from utils import load_data



def evaluate(model, loss, X_val, y_val, reg=0.0):
    # 加载最佳模型权重
    model.load_weights("best_model_medium1.npz")
    
    # 前向传播计算测试集的性能
    val_logits = model.forward(X_val)
    val_loss = loss.forward(val_logits, y_val) + model.get_regularization_loss(reg)
    val_acc = compute_accuracy(val_logits, y_val)
    
    # 打印验证集的性能指标
    print("==========test set============")
    print(f"Test Loss: {val_loss:.4f} | Test Accuracy: {val_acc * 100:.2f}%")

def compute_accuracy(logits, y_true):
    y_pred = np.argmax(logits, axis=1)
    return np.mean(y_pred == y_true)


# 加载数据集
(_, _), (_, _), (X_test, y_test) = load_data()

# 初始化模型和损失函数
model = NeuralNetwork(input_dim=3072, hidden_dim1=2048, hidden_dim2=512, output_dim=10, activation='relu')
loss = SoftmaxCrossEntropy()

# 调用 evaluate 函数
evaluate(model, loss, X_test, y_test, reg=0.0001)