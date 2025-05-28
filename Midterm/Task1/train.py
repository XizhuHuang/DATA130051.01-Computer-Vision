import torch
import numpy as np
from tqdm import tqdm
import os
import warnings
from typing import Dict, Tuple, List

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 mode: str = 'finetune',
                 lr: float = 0.001,
                 backbone_lr: float = None,
                 head_lr: float = None,
                 reg: float = 0.0,
                 step_size: int = 10,
                 gamma: float = 0.1,
                 patience: int = 5,
                 momentum: float = 0.9,
                 device: str = None):
        assert mode in ['finetune', 'scratch'], "Invalid mode"
        self.mode = mode
        self.model = model
        self.step_size = step_size
        self.gamma = gamma
        self.patience = patience
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)

        self.backbone_lr = backbone_lr if backbone_lr else lr * 0.1
        self.head_lr = head_lr if head_lr else lr

        self.optimizer = self._configure_optimizer(reg, momentum)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

        self._reset_training_state()
        self.criterion = torch.nn.CrossEntropyLoss()

    def _configure_optimizer(self, reg: float, momentum: float):
        backbone_params = []
        head_params = []

        for name, param in self.model.named_parameters():
            if 'head' in name.lower():  # 假设分类头部的参数名字包含 'head'
                head_params.append(param)
            else:
                backbone_params.append(param)

        # 根据模式配置优化器的学习率
        if self.mode == 'finetune':
            # 这里的学习率分别为主干和头部设置不同的学习率
            return torch.optim.SGD([
                {'params': backbone_params, 'lr': self.backbone_lr},
                {'params': head_params, 'lr': self.head_lr}
            ], momentum=momentum, weight_decay=reg)
        else:
            # scratch 模式下使用相同学习率训练所有参数
            return torch.optim.SGD(self.model.parameters(), lr=self.head_lr,
                                   momentum=momentum, weight_decay=reg)

    def _reset_training_state(self):
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.lr_history = []
        self.best_val_acc = 0.0
        self.counter = 0
        self.early_stop = False

    def compute_accuracy(self, outputs, labels):
        preds = torch.argmax(outputs, dim=1)
        return (preds == labels).sum().item() / len(labels)

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for inputs, labels in tqdm(loader, desc='Training', leave=False):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
            total_samples += labels.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for inputs, labels in tqdm(loader, desc='Validating', leave=False):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                total_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
                total_samples += labels.size(0)

        return total_loss / total_samples, total_correct / total_samples

    def train(self, train_loader, val_loader, epochs=100, save_path='best_model.pth'):
        self._reset_training_state()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        try:
            for epoch in range(epochs):
                if self.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

                train_loss, train_acc = self.train_epoch(train_loader)
                val_loss, val_acc = self.validate(val_loader)

                self.train_losses.append(train_loss)
                self.train_accs.append(train_acc)
                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)
                self.lr_history.append(self.optimizer.param_groups[0]['lr'])

                self.scheduler.step()

                # Save best model
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.counter = 0
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_acc,
                        'mode': self.mode
                    }, save_path)
                else:
                    self.counter += 1
                    if self.counter >= self.patience:
                        self.early_stop = True

                print(f"\nEpoch {epoch+1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
                print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2%}")
                print(f"LR: {self.lr_history[-1]:.2e}")
                print('-' * 50)

        except KeyboardInterrupt:
            warnings.warn("Training interrupted by user")

        if not self.early_stop:
            print("Training completed.")

        return {
            'train_loss': self.train_losses,
            'train_acc': self.train_accs,
            'val_loss': self.val_losses,
            'val_acc': self.val_accs,
            'lr_history': self.lr_history
        }

    def load_best_model(self, save_path):
        checkpoint = torch.load(save_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']} (val acc: {checkpoint['val_acc']:.2%})")

    def plot_learning_curve(self):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            warnings.warn("Matplotlib not installed.")
            return

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc')
        plt.plot(self.val_accs, label='Val Acc')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.tight_layout()
        plt.show()

def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
