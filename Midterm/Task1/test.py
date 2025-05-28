import torch
from data_loader import get_caltech101_loaders
from models import Caltech101Classifier
from train import Trainer

def test_model(model_path='checkpoints/finetune_resnet18_4.pth'):
    # 1. 加载数据
    _, _, test_loader = get_caltech101_loaders()

    # 2. 初始化模型
    model = Caltech101Classifier(arch='resnet18', pretrained=False)  # 加载结构，不加载预训练
    trainer = Trainer(model=model, mode='finetune')  # mode无所谓，这里只用于加载权重和测试

    # 3. 加载训练好的权重
    trainer.load_best_model(model_path)

    # 4. 在测试集上评估
    test_loss, test_acc = trainer.validate(test_loader)
    print(f"\n[TEST RESULT] Loss: {test_loss:.4f} | Accuracy: {test_acc:.2%}")

if __name__ == '__main__':
    test_model(model_path='checkpoints/finetune_resnet18_4.pth')
