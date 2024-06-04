import torch.nn as nn
import torch

from models.Swin_blocks import Scale, ConvFFNModule


class GatingNetworkModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim == 64:
            self.H = 56
            self.W = 29
            self.complex_weight = nn.Parameter(torch.rand(self.H, self.W, dim, 2, dtype=torch.float32) * 0.02)
        elif dim == 128:
            self.H = 28
            self.W = 15
            self.complex_weight = nn.Parameter(torch.rand(self.H, self.W, dim, 2, dtype=torch.float32) * 0.02)


    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm="ortho")
        x = x.permute(0, 3, 1, 2)
        return x


class GatingNetworkLayerModule(nn.Module):
    def __init__(self, dim: int, init_value: float, expansion_ratio: int):
        super().__init__()
        self.conv_ffn = ConvFFNModule(dim, expansion_ratio=expansion_ratio)
        self.gating_network = GatingNetworkModule(dim)
        self.norm1 = nn.GroupNorm(32, dim)
        self.norm2 = nn.GroupNorm(32, dim)
        self.res_scale_1 = Scale(dim=dim, init_value=init_value)
        self.res_scale_2 = Scale(dim=dim, init_value=init_value)
        self.layer_scale_1 = Scale(dim=dim, init_value=init_value)
        self.layer_scale_2 = Scale(dim=dim, init_value=init_value)

    def forward(self, x):
        x = self.res_scale_1(x) + self.norm1(self.layer_scale_1(self.gating_network(x)))
        x = self.res_scale_2(x) + self.norm2(self.layer_scale_1(self.conv_ffn(x)))

        return x


class SpatialPriorModule(nn.Module):
    def __init__(self, dim: int, emb_dim: int):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Hardswish(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Hardswish(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.Hardswish(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim * 2),
            nn.Hardswish()
        )

        self.conv_block_3 = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.Hardswish()
        )

        self.conv_block_4 = nn.Sequential(
            nn.Conv2d(in_channels=dim * 4, out_channels=dim * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim * 4),
            nn.Hardswish()
        )

        self.fc1 = nn.Conv2d(in_channels=dim, out_channels=emb_dim, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(in_channels=dim * 2, out_channels=emb_dim, kernel_size=1, bias=False)
        self.fc3 = nn.Conv2d(in_channels=dim * 4, out_channels=emb_dim, kernel_size=1, bias=False)
        self.fc4 = nn.Conv2d(in_channels=dim * 4, out_channels=emb_dim, kernel_size=1, bias=False)

        self.last = nn.Conv2d(in_channels=emb_dim * 4, out_channels=emb_dim, kernel_size=1, bias=False)

    def forward(self, x):
        x1 = self.stem(x)
        x2 = self.conv_block_2(x1)
        x3 = self.conv_block_3(x2)
        x4 = self.conv_block_4(x3)

        x1 = self.fc1(x1)
        x2 = self.fc2(x2)
        x3 = self.fc3(x3)
        x4 = self.fc4(x4)

        x2 = nn.functional.interpolate(x2, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x3 = nn.functional.interpolate(x3, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x4 = nn.functional.interpolate(x4, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = self.last(torch.cat([x1, x2, x3, x4], dim=1))

        return x