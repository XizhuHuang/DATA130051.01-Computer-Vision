<h1 align="center">Midterm Project</h1>
<h3 align="center"> Xizhu Huang </h3>

## Contents
- [Task 1: Caltech-101 分类 - 微调 ImageNet 预训练 CNN](#-task-1-caltech-101-分类---微调-imagenet-预训练-cnn)
  - [项目亮点](#-项目亮点)
  - [项目结构](#-项目结构)
  - [数据准备](#-数据准备)
  - [加载模型参数](#-加载模型参数)
  - [训练和测试](#-训练和测试)
  - [可视化训练过程](#-可视化训练过程)
  - [可调参数](#-可调参数)
  - [安装依赖](#-安装依赖)

- [Task 2: 基于VOC2007数据集的目标检测 —— Mask R-CNN 与 Sparse R-CNN 对比实验](#-task-2-基于voc2007数据集的目标检测--mask-r-cnn-与-sparse-r-cnn-对比实验)
  - [项目目标](#-项目目标)
  - [项目结构](#-项目结构-1)
  - [环境配置](#-环境配置)
  - [数据集准备（VOC2007）](#-数据集准备voc2007)
  - [加载模型参数](#-加载模型参数)
  - [模型训练与测试命令](#-模型训练与测试命令)
  - [训练过程可视化（TensorBoard）](#-训练过程可视化tensorboard)
  - [可视化脚本使用说明](#️-可视化脚本使用说明)

---
## 🧠 Task 1: Caltech-101 分类 - 微调 ImageNet 预训练 CNN

本项目使用 PyTorch 实现对 [Caltech-101](https://data.caltech.edu/records/mzrjq-6wc02) 图像分类数据集的训练与测试，核心是通过微调 ImageNet 上的预训练模型（如 ResNet-18）来提升性能。

---

### ✅ 项目亮点

* 支持微调（fine-tune）与从头训练（scratch）两种方式
* 数据集按标准划分（每类 30 张训练图像）
* 支持 ResNet-18/AlexNet 架构自定义输出层（101 类）
* 加入多种数据增强与标准预处理
* 支持模型断点保存、早停与可视化训练曲线

---

### 📁 项目结构

```
.
├── data/                          # Caltech-101 原始图像数据目录
├── dataloader.py                 # 数据加载与预处理（含30/类训练划分）
├── models.py                     # CNN 模型构建与修改（支持ResNet-18等）
├── train.py                      # 模型训练类（Trainer）
├── test.py                       # 加载模型并评估性能
├── checkpoints/                  # 模型权重保存目录
└── requirements.txt              # 依赖包
```

---

### 📥 数据准备

1. 下载 Caltech-101 数据集：

   * 官方链接：[https://data.caltech.edu/records/mzrjq-6wc02](https://data.caltech.edu/records/mzrjq-6wc02)
2. 解压至本地 `./data/101_ObjectCategories/`
3. 结构如下所示：

```
data/
└── 101_ObjectCategories/
    ├── airplane/
    ├── accordion/
    ├── ...
```

---

### 💾 加载模型参数

项目训练好的模型可通过以下链接获取：

* 🔗 **百度网盘下载链接**：[点击下载]( https://pan.baidu.com/s/1z249iIbLLk7bJ0uFYxWPUQ )
* 🔐 **提取码**：`gjfb`

下载后将 `.pth` 模型文件放置到 `checkpoints/` 目录或自定义路径中。

---


### 🚀 训练和测试

#### 训练模型
##### 微调（Fine-tuning）

```python
from dataloader import get_caltech101_loaders
from models import Caltech101Classifier
from train import Trainer

train_loader, val_loader, _ = get_caltech101_loaders()
model = Caltech101Classifier(arch='resnet18', pretrained=True)

trainer = Trainer(model=model, mode='finetune', lr=0.01)
trainer.train(train_loader, val_loader, epochs=50, save_path='checkpoints/finetune_resnet18.pth')
```

##### 从头训练（Scratch）

```python
model = Caltech101Classifier(arch='resnet18', pretrained=False)

trainer = Trainer(model=model, mode='scratch', lr=0.01)
trainer.train(train_loader, val_loader, epochs=50, save_path='checkpoints/scratch_resnet18.pth')
```

---

#### 🧪 测试模型性能

确保你已下载好训练好的模型，并指定路径：

```bash
python test.py
```

或修改 `test.py`：

```python
test_model(model_path='checkpoints/finetune_resnet18.pth')
```

输出示例：

```
Loaded best model from epoch 38 (val acc: 84.23%)
[TEST RESULT] Loss: 0.5412 | Accuracy: 83.67%
```

---

### 📊 可视化训练过程

```python
trainer.plot_learning_curve()
```

展示 loss / accuracy 曲线对比图：

* 训练 vs 验证损失
* 训练 vs 验证准确率

---

### 🛠 可调参数

| 参数名称          | 说明                           | 默认值        |
| ------------- | ---------------------------- | ---------- |
| `mode`        | 训练模式（`finetune` 或 `scratch`） | `finetune` |
| `lr`          | 主学习率（用于分类头）                  | 0.001      |
| `backbone_lr` | 主干网络学习率（可选）                  | `lr * 0.1` |
| `step_size`   | 学习率衰减步长（epoch）               | 10         |
| `gamma`       | 学习率衰减因子                      | 0.1        |
| `patience`    | 早停轮次数容忍                      | 5          |

---

### 📦 安装依赖

```bash
pip install -r requirements.txt
```

#### 示例内容：

```txt
matplotlib==3.10.3
numpy==2.2.6
scikit_learn==1.6.1
torch==2.6.0
torchvision==0.21.0
tqdm==4.65.2
```
---

## Task 2: 基于VOC2007数据集的目标检测 —— Mask R-CNN 与 Sparse R-CNN 对比实验

### 📌 项目目标

本项目通过 [MMDetection](https://github.com/open-mmlab/mmdetection) 框架，在 **PASCAL VOC2007** 数据集上训练并评估两个代表性的目标检测模型：**Mask R-CNN** 与 **Sparse R-CNN**，并完成以下任务：

1. 在 VOC2007 上训练并测试两个目标检测模型；
2. 可视化 Mask R-CNN 的 Proposal Box 与最终预测结果；
3. 比较两个模型在 VOC 测试集及 3 张非 VOC 外部图片上的目标检测和实例分割效果。

---

### 📁 项目结构
```
TASK2
├── mmdetection/ 
  ├── configs/                        # 配置文件
  │   ├── _base_/                    # 基础配置文件
  │   ├── mask_rcnn/                # Mask R-CNN 配置
  │   └── sparse_rcnn/              # Sparse R-CNN 配置
  ├── tools/                         # 训练与测试脚本
  │   ├── train.py
  │   └── test.py
  ├── demo/
  │   └── external_voc_images/      # 外部图片（非VOC）
  ├── figures/                       # 实验结果可视化图像
  ├── work_dirs/                     # 模型输出目录（日志、权重等）
  │   ├── mask_rcnn_r50_fpn_2x_voc/
  │   └── sparse-rcnn_r50_fpn_ms-480-800-3x_voc/
  ├── voc2coco.py                    # VOC转COCO格式脚本
  ├── vis_dataset.py                # 可视化VOC2007数据集
  ├── vis_external.py               # 可视化外部图像预测结果
  ├── vis_proposal.py               # 可视化Mask R-CNN proposal boxes
  ├── vis_final.py                  # 可视化模型最终预测结果
  ├── vis_test_results.py           # 可视化测试mAP对比图
  └── requirements.txt              # 项目依赖库列表
```

---

### 📦 环境配置

```bash
# 创建 Conda 环境
conda create -n mmdet_voc python=3.8 -y
conda activate mmdet_voc

# 安装 PyTorch 和 CUDA（请根据你的显卡适配版本）
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 安装 MMCV（注意版本匹配）
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

# 安装 MMDetection
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
pip install -e .

# 安装本项目依赖
cd PATH/TO/TASK2  # 请替换为本项目根目录
pip install -r requirements.txt
```

---

### 📥 数据集准备（VOC2007）

1. 下载 VOC2007 数据集：

   官方地址：[http://host.robots.ox.ac.uk/pascal/VOC/voc2007/](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)

   或直接运行以下命令：

   ```bash
   mkdir -p data
   cd data
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   tar -xvf VOCtrainval_06-Nov-2007.tar
   tar -xvf VOCtest_06-Nov-2007.tar
   ```

   解压后数据将保存在 `data/VOCdevkit/VOC2007/` 下。

2. 转换为 COCO 格式：

   ```bash
   python voc2coco.py --voc_path data/VOCdevkit --output_dir data/coco
   ```

---

### 🧠 加载模型参数

本项目训练好的权重文件保存在百度网盘中，可直接下载使用进行测试与可视化：

* 🔗 **百度网盘下载链接**：[点击下载]( https://pan.baidu.com/s/1z249iIbLLk7bJ0uFYxWPUQ )
* 🔐 **提取码**：`gjfb`

将下载后的模型文件放入：

```
work_dirs/
├── mask_rcnn_r50_fpn_2x_voc/epoch_24.pth
├── sparse-rcnn_r50_fpn_ms-480-800-3x_voc/epoch_36.pth
```

---

### 💻 模型训练与测试命令

#### 训练命令

```bash
# Mask R-CNN
python tools/train.py configs/mask_rcnn/mask_rcnn_r50_fpn_2x_voc.py

# Sparse R-CNN
python tools/train.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_ms-480-800-3x_voc.py
```

#### 测试命令（指定模型权重路径）

```bash
# Mask R-CNN
python tools/test.py configs/mask_rcnn/mask_rcnn_r50_fpn_2x_voc.py \
    work_dirs/mask_rcnn_r50_fpn_2x_voc/epoch_24.pth

# Sparse R-CNN
python tools/test.py configs/sparse_rcnn/sparse-rcnn_r50_fpn_ms-480-800-3x_voc.py \
    work_dirs/sparse-rcnn_r50_fpn_ms-480-800-3x_voc/epoch_36.pth
```

---

### 📈 训练过程可视化（TensorBoard）

```bash
tensorboard --logdir=work_dirs/mask_rcnn_r50_fpn_2x_voc/20250525_222927/tensorboard
tensorboard --logdir=work_dirs/sparse-rcnn_r50_fpn_ms-480-800-3x_voc/20250526_222605/tensorboard
```

浏览器中打开：[http://localhost:6006](http://localhost:6006)

---

### 🖼️ 可视化脚本使用说明

* 查看 VOC 数据集样本：`python vis_dataset.py`
* 可视化 Mask R-CNN Proposal Box：`python vis_proposal.py`
* 可视化最终预测结果：`python vis_final.py`
* 对外部图像进行预测：`python vis_external.py`
* 生成检测性能柱状图：`python vis_test_results.py`

---
