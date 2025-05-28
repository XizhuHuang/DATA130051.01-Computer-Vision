import torchvision.models as models
import torch.nn as nn

class Caltech101Classifier(nn.Module):
    def __init__(self, arch='resnet18', pretrained=True):
        """
        :param arch: 模型结构名称 ('resnet18', 'alexnet')
        :param pretrained: 是否加载ImageNet预训练参数
        """
        super().__init__()
        self.arch = arch
        backbone = getattr(models, arch)(pretrained=pretrained)

        if 'resnet' in arch:
            # ResNet 去掉最后的fc
            self.backbone = nn.Sequential(*list(backbone.children())[:-1], nn.Flatten())
            in_features = backbone.fc.in_features
        elif 'alexnet' in arch:
            # AlexNet 拆除最后一层 classifier 并保留 flatten 和 avgpool
            self.backbone = nn.Sequential(
                backbone.features,
                backbone.avgpool,
                nn.Flatten(),
                *backbone.classifier[:-1]
            )
            in_features = backbone.classifier[-1].in_features
        else:
            raise ValueError(f"Unsupported architecture: {arch}")

        self.head = nn.Linear(in_features, 101)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)
    
