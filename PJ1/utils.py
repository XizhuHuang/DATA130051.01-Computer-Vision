import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split


def load_data(data_dir="data\\cifar-10-batches-py"):
    """
    加载本地CIFAR-10数据集
    返回格式与keras.datasets.cifar10.load_data()保持一致
    """
    def unpickle(file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    # 加载训练集
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        batch_dict = unpickle(batch_file)
        train_data.append(batch_dict[b'data'])
        train_labels.append(batch_dict[b'labels'])
    
    X_train = np.concatenate(train_data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_train = np.concatenate(train_labels)
    
    # 加载测试集
    test_file = os.path.join(data_dir, 'test_batch')
    test_dict = unpickle(test_file)
    X_test = test_dict[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    y_test = np.array(test_dict[b'labels'])

    # # 转换为浮点并归一化
    # X_train = X_train.astype(np.float32) / 255.0
    # X_test = X_test.astype(np.float32) / 255.0
    

    # 修改归一化方式（与PyTorch对齐）
    X_train = X_train.astype(np.float32) / 255.0
    X_train = (X_train - 0.5) / 0.5  # 范围变为[-1, 1]
    
    X_test = X_test.astype(np.float32) / 255.0
    X_test = (X_test - 0.5) / 0.5

    # 转换为全连接层需要的展平格式
    X_train = X_train.reshape(X_train.shape[0], -1)  # (50000, 3072)
    X_test = X_test.reshape(X_test.shape[0], -1)     # (10000, 3072)
    
    # 使用 sklearn 的 train_test_split 来分割训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=5000, random_state=42)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# # 测试数据加载
# (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_data()

# print(f"训练集形状: {X_train.shape}")  # 应为 (45000, 3072)
# print(f"验证集形状: {X_val.shape}")    # 应为 (5000, 3072)
# print(f"测试集形状: {X_test.shape}")   # 应为 (10000, 3072)
# print(f"标签范围: {np.unique(y_train)}")  # 应为 [0,1,2,3,4,5,6,7,8,9]