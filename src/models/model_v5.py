import torch
from torch import nn
import torch.nn.functional as F
from typing import Sequence


class DepthwiseSeparableConv(nn.Module):
    '''
    Description
        深度可分离卷积块 先逐通道卷积再用1x1卷积整合特征
    Args
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        kernel_size (int): 卷积核尺寸 默认3
        stride (int): 步幅 支持1或2
        dilation (int): 空洞率 默认1
    Shape
        Input: (B, C_in, H, W)
        Output: (B, C_out, H/stride, W/stride)
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        dilation: int = 1,
    ):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=False,
        )
        self.dw_bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.pw_bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.dw_bn(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.pw_bn(x)
        x = self.act(x)
        return x


class AttentionBlock(nn.Module):
    '''
    Description
        通道与空间注意力模块 先执行通道注意力再执行空间注意力
    Args
        channels (int): 输入与输出通道数
        reduction (int): 通道注意力的降维比例
    Shape
        Input: (B, C, H, W)
        Output: (B, C, H, W)
    '''
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg = self.avg_pool(x).view(b, c)
        maxp = self.max_pool(x).view(b, c)
        channel_att = self.mlp(avg) + self.mlp(maxp)
        channel_att = self.sigmoid(channel_att).view(b, c, 1, 1)
        x = x * channel_att

        avg_spatial = torch.mean(x, dim=1, keepdim=True)
        max_spatial, _ = torch.max(x, dim=1, keepdim=True)
        spatial = torch.cat([avg_spatial, max_spatial], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial))
        x = x * spatial_att
        return x


class MultiScaleResidualBlock(nn.Module):
    '''
    Description
        多尺度残差块 两条深度可分离卷积路径融合后叠加通道+空间注意力
    Args
        in_channels (int): 输入通道数
        out_channels (int): 输出通道数
        stride (int): 步幅 支持1或2
        reduction (int): 注意力模块的降维比例
        use_dilation (bool): 是否在大核路径使用dilation=2扩大感受野
    Shape
        Input: (B, C_in, H, W)
        Output: (B, C_out, H/stride, W/stride)
    '''
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        reduction: int = 16,
        use_dilation: bool = False,
    ):
        super().__init__()
        mid_channels = max(out_channels // 2, 1)
        dilation = 2 if use_dilation else 1

        self.path1 = DepthwiseSeparableConv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            dilation=1,
        )
        self.path2 = DepthwiseSeparableConv(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=5,
            stride=stride,
            dilation=dilation,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid_channels * 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )
        self.attention = AttentionBlock(out_channels, reduction=reduction)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.out_act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out1 = self.path1(x)
        out2 = self.path2(x)
        out = torch.cat([out1, out2], dim=1)
        out = self.fuse(out)
        out = self.attention(out)
        out = out + self.shortcut(identity)
        out = self.out_act(out)
        return out


class MultiScalePooling(nn.Module):
    '''
    Description
        轻量级多尺度池化模块 通过多个自适应最大池化聚合不同尺度信息后压缩维度
    Args
        in_channels (int): 输入通道数
        pool_sizes (Sequence[int]): 池化输出尺寸列表 如(1, 2, 3)
        out_dim (int): 压缩后的输出维度
        dropout (float): 压缩层后的dropout概率
    Shape
        Input: (B, C, H, W)
        Output: (B, out_dim)
    '''
    def __init__(
        self,
        in_channels: int,
        pool_sizes: Sequence[int] = (1, 2, 3),
        out_dim: int = 512,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.pools = nn.ModuleList([nn.AdaptiveMaxPool2d(output_size=(s, s)) for s in pool_sizes])
        total_dim = sum([s * s for s in pool_sizes]) * in_channels
        self.fc = nn.Sequential(
            nn.Linear(total_dim, out_dim),
            nn.SiLU(),
            nn.Dropout(p=dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        pooled = [pool(x).reshape(b, -1) for pool in self.pools]
        feats = torch.cat(pooled, dim=1)
        feats = self.fc(feats)
        return feats


class LiteGlobalAttention(nn.Module):
    '''
    Description
        轻量全局注意力模块 基于多头空间注意力建模全局依赖
    Args
        channels (int): 输入通道数
        heads (int): 多头数量 默认为4
    Shape
        Input: (B, C, H, W)
        Output: (B, C, H, W)
    '''
    def __init__(self, channels: int, heads: int = 4):
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.out_bn = nn.BatchNorm2d(channels)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = F.normalize(q, p=2, dim=1, eps=1e-6)
        k = F.normalize(k, p=2, dim=1, eps=1e-6)

        q = q.view(b, self.heads, self.head_dim, h * w).permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)
        k = k.view(b, self.heads, self.head_dim, h * w)  # (B, heads, head_dim, HW)
        v = v.view(b, self.heads, self.head_dim, h * w).permute(0, 1, 3, 2)  # (B, heads, HW, head_dim)

        attn = torch.matmul(q, k) * self.scale  # (B, heads, HW, HW)
        attn = attn.clamp(min=-10.0, max=10.0)
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)  # (B, heads, HW, head_dim)
        out = out.permute(0, 1, 3, 2).contiguous().view(b, c, h, w)

        out = self.out_proj(out)
        out = self.out_bn(out)
        out = out + x
        out = self.act(out)
        return out


class FireCNN_v5(nn.Module):
    '''
    Description
        FireCNN_v5 在 v4 基础上于 stage4 输出后加入轻量全局注意力模块
    Args
        num_classes (int): 输出类别数
    Shape
        Input: (B, 3, 320, 320)
        Output: (B, num_classes)
    '''
    def __init__(self, num_classes: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, dilation=1),
        )

        self.stage1 = self._make_stage(in_channels=64, out_channels=128, num_blocks=3, stride=2, use_dilation=False)
        self.stage2 = self._make_stage(in_channels=128, out_channels=256, num_blocks=3, stride=2, use_dilation=False)
        self.stage3 = self._make_stage(in_channels=256, out_channels=384, num_blocks=3, stride=2, use_dilation=True)
        self.stage4 = self._make_stage(in_channels=384, out_channels=512, num_blocks=2, stride=1, use_dilation=True)

        self.lga = LiteGlobalAttention(channels=512, heads=4)
        self.spp = MultiScalePooling(in_channels=512, pool_sizes=(1, 2, 3), out_dim=512, dropout=0.4)
        self.mlp_stage2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.mlp_stage3 = nn.Sequential(
            nn.Linear(384, 128),
            nn.SiLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 + 128 + 128, 384),
            nn.SiLU(),
            nn.Dropout(p=0.5),
            nn.Linear(384, num_classes),
        )
        self._init_weights()

    def _make_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int,
        use_dilation: bool,
    ) -> nn.Sequential:
        blocks = [MultiScaleResidualBlock(in_channels, out_channels, stride=stride, reduction=16, use_dilation=use_dilation)]
        for _ in range(1, num_blocks):
            blocks.append(MultiScaleResidualBlock(out_channels, out_channels, stride=1, reduction=16, use_dilation=use_dilation))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x4 = self.lga(x4)

        f4 = self.spp(x4)

        g2 = torch.flatten(F.adaptive_avg_pool2d(x2, output_size=1), 1)
        f2 = self.mlp_stage2(g2)

        g3 = torch.flatten(F.adaptive_avg_pool2d(x3, output_size=1), 1)
        f3 = self.mlp_stage3(g3)

        fusion = torch.cat([f4, f3, f2], dim=1)
        out = self.classifier(fusion)
        return out


def build_model(num_classes: int, pretrained: bool = False) -> nn.Module:
    '''
    Description
        构建 FireCNN_v5 模型 忽略 pretrained 参数确保随机初始化
    Args
        num_classes (int): 分类类别数
        pretrained (bool): 预训练标志 将被忽略
    Returns
        model (nn.Module): FireCNN_v5 实例
    '''
    if pretrained:
        print("Warning: FireCNN_v5 does not load pretrained weights; using random initialization.")
    model = FireCNN_v5(num_classes=num_classes)
    return model
