import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)


    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)

        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class ResUNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ResUNet, self).__init__()
        self.resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT)

        self.in_conv = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu)
        self.maxpool = self.resnet.maxpool
        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4

        self.bridge = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.decoder4 = Up(512 + 256, 256)
        self.decoder3 = Up(256 + 128, 128)
        self.decoder2 = Up(128 + 64, 64)
        self.decoder1 = Up(64 + 64, 64)

        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=2, stride=2)

    def forward(self, x):
        x1 = self.in_conv(x)
        x1 = self.maxpool(x1)

        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        x5 = self.bridge(x5)

        x = self.decoder4(x5, x4)
        x = self.decoder3(x, x3)
        x = self.decoder2(x, x2)
        x = self.decoder1(x, x1)

        out = self.out_conv(x)
        return out


# 测试模型
if __name__ == "__main__":
    model = ResUNet(num_classes=16)
    x = torch.randn(16, 3, 256, 256)
    y = model(x)
    print(y.shape)
