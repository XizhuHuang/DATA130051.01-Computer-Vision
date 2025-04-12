# DATA130051.01-Computer-Vision

本项目为纯Numpy实现的3层全连接神经网络，在CIFAR-10数据集上实现分类。
```
## 环境依赖
- Python 3.8+
- Numpy 1.21+
- Matplotlib 3.5+
```
### 快速开始
#### 数据准备
下载CIFAR-10数据集：https://www.cs.toronto.edu/~kriz/cifar.html 并解压至data/目录。

#### 训练流程
```bash
# 使用默认配置启动训练
python train.py

# 自定义参数训练
python train.py \
    --lr 1.5e-3 \               # 初始学习率 (默认: 0.01)
    --reg 7e-4 \                # L2正则化强度 (默认: 0.0)
    --momentum 0.85 \           # 动量系数 (默认: 0.9)
    --step_size 8 \             # 学习率衰减周期 (默认: 5 epoch)
    --gamma 0.95 \              # 学习率衰减系数 (默认: 0.1)
    --patience 7 \              # 早停检测窗口 (默认: 5 epoch)
    --batch_size 256 \          # 批处理大小 (默认: 64)
    --epochs 200                # 最大训练轮次 (默认: 100)
```

#### 训练过程监控
实时输出训练指标：
```
Epoch 23/150 | Train Loss: 1.4523 | Train Acc: 54.61% | Val Loss: 1.6321 | Val Acc: 48.72% | LR: 0.001128
```

#### 断点续训
当触发早停后，可使用保存的最佳权重继续训练：
```bash
python train.py --resume --model_weights best_model_medium3.npz
```

#### 模型测试

```bash
# 使用最佳权重评估测试集性能
python test.py
```
测试流程说明：
1. 自动加载预训练权重文件best_model_medium1.npz

2. 对测试集执行前向传播计算

3. 输出分类损失与准确率指标

预期输出：
```
==========test set============
Test Loss: 1.6723 | Test Accuracy: 55.92%
```

#### 模型权重下载
- **百度网盘链接**: [点击下载](https://pan.baidu.com/s/1C1mzIldveg3Zv7x_o2bsMw)
- **提取码**: `9n9a`
