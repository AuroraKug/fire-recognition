import torch
from torch import nn


class ResidualBlock(nn.Module):
    '''
    类描述
        带残差的卷积模块 使用stride卷积下采样
    Args
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        stride (int): 步幅
    '''
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out = out + identity
        out = self.act(out)
        return out


class FireCNN(nn.Module):
    '''
    类描述
        残差卷积网络用于火灾图像分类 输出维度等于num_classes
    Args
        num_classes (int): 分类类别数
    '''
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
        )
        self.layer1 = self._make_layer(64, 128, num_blocks=3, stride=2)
        self.layer2 = self._make_layer(128, 256, num_blocks=3, stride=2)
        self.layer3 = self._make_layer(256, 512, num_blocks=3, stride=2)
        self.layer4 = self._make_layer(512, 512, num_blocks=3, stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        blocks = [ResidualBlock(in_channels, out_channels, stride=stride)]
        for _ in range(1, num_blocks):
            blocks.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        # 使用Xavier初始化提升早期训练稳定性
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    '''
    Description
        构建轻量级卷积分类模型 忽略pretrained参数始终随机初始化
    Args
        num_classes (int): 输出类别数
        pretrained (bool): 预训练标志被忽略
    Returns
        model (nn.Module): 可直接用于训练的模型
    '''
    _ = pretrained  # 参数被忽略 确保不使用预训练权重
    model = FireCNN(num_classes=num_classes)
    return model
