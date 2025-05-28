from torchvision import datasets, transforms
from torch.utils.data import Subset, DataLoader
from sklearn.model_selection import train_test_split
from collections import defaultdict
import numpy as np


def get_caltech101_loaders():
    """实现标准数据划分(每类30训练样本)"""
    # 预处理定义
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        # 随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 随机仿射变换
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
        # 颜色抖动（亮度、对比度、饱和度、色调）
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载完整数据集
    full_dataset = datasets.ImageFolder("./data/101_ObjectCategories")

    # 标准划分逻辑（每类选30训练样本）
    class_indices = defaultdict(list)
    for idx, (_, label) in enumerate(full_dataset.samples):
        class_indices[label].append(idx)

    train_indices, test_indices = [], []
    np.random.seed(42)
    for label, indices in class_indices.items():
        shuffled = np.random.permutation(indices)
        train_indices.extend(shuffled[:30])
        test_indices.extend(shuffled[30:])

    # 划分验证集（从训练集取20%）
    train_sub, val_sub = train_test_split(
        train_indices, 
        test_size=0.2,
        stratify=[full_dataset.targets[i] for i in train_indices]
    )

    # 创建子数据集并应用预处理
    def apply_transform(subset, transform):
        subset.dataset.transform = transform
        return subset

    train_dataset = apply_transform(Subset(full_dataset, train_sub), train_transform)
    val_dataset = apply_transform(Subset(full_dataset, val_sub), test_transform)
    test_dataset = apply_transform(Subset(full_dataset, test_indices), test_transform)

    # 创建DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )

    return train_loader, val_loader, test_loader
