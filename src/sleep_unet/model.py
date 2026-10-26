"""1D ResNet U-Net from ``unet/unet.ipynb``.

The encoder's first stride is 12 so each 5-second sample collapses to one minute.
The head crops the 180-minute context on each side and returns that minute vector.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Full day of 5-second steps, plus 180 minutes of context on each side.
DAY_STEPS = 12 * 60 * 24
CONTEXT_STEPS = 180 * 12
WINDOW_STEPS = DAY_STEPS + 2 * CONTEXT_STEPS
MINUTE_STEPS = 24 * 60


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super().__init__()

        if stride == 1:
            padding = "same"
        else:
            padding = (kernel_size - stride) // 2
        self.layers = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class SEBlock(nn.Module):
    def __init__(self, n_channels, se_ratio):
        super().__init__()

        self.layers = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Conv1d(n_channels, n_channels // se_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(n_channels // se_ratio, n_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, se_ratio):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBNReLU(n_channels, n_channels, kernel_size, stride=1),
            ConvBNReLU(n_channels, n_channels, kernel_size, stride=1),
            SEBlock(n_channels, se_ratio),
        )

    def forward(self, x):
        return x + self.layers(x)


class UNet1d(nn.Module):
    def __init__(
        self,
        input_channels,
        initial_channels,
        initial_kernel_size,
        down_channels,
        down_kernel_size,
        down_stride,
        res_depth,
        res_kernel_size,
        se_ratio,
        out_kernel_size,
    ):
        super().__init__()
        self.down_kernel_size = down_kernel_size
        self.down_stride = down_stride

        self.initial_layers = ConvBNReLU(
            input_channels,
            initial_channels,
            initial_kernel_size,
            stride=1,
            groups=input_channels,
        )

        self.down_layers = nn.ModuleList()
        for i in range(len(down_channels)):
            if i == 0:
                in_channels = initial_channels
            else:
                in_channels = down_channels[i - 1] + input_channels
            out_channels = down_channels[i]
            kernel_size = down_kernel_size[i]
            stride = down_stride[i]

            block = [ConvBNReLU(in_channels, out_channels, kernel_size, stride)]
            for _ in range(res_depth):
                block.append(ResBlock(out_channels, res_kernel_size, se_ratio))
            self.down_layers.append(nn.Sequential(*block))

        self.up_layers = nn.ModuleList()
        for i in range(len(down_channels) - 1, 0, -1):
            in_channels = out_channels + down_channels[i - 1]
            out_channels = down_channels[i]
            kernel_size = down_kernel_size[i]
            self.up_layers.append(ConvBNReLU(in_channels, out_channels, kernel_size, stride=1))

        self.out_layers = nn.Conv1d(down_channels[1], 1, out_kernel_size, padding="same")

    def forward(self, x):
        outs = []
        x_avg = x
        x = self.initial_layers(x)

        for i, layer in enumerate(self.down_layers):
            x_out = layer(x)
            if i == len(self.down_layers) - 1:
                x = x_out
            else:
                outs.append(x_out)
                kernel_size = self.down_kernel_size[i]
                stride = self.down_stride[i]
                padding = (kernel_size - stride) // 2
                x_avg = F.avg_pool1d(x_avg, kernel_size, stride, padding)
                x = torch.cat([x_out, x_avg], dim=1)

        for i, layer in enumerate(self.up_layers):
            scale_factor = self.down_stride[-i - 1]
            x = F.interpolate(x, scale_factor=scale_factor, mode="linear")
            x = torch.cat([x, outs[-i - 1]], dim=1)
            x = layer(x)

        return self.out_layers(x)[:, 0, 180:-180]


def build_unet(
    input_channels: int,
    initial_channels: int,
    *,
    res_depth: int = 3,
    se_ratio: int = 4,
) -> UNet1d:
    """Notebook training settings. ``down_stride[0]`` stays 12."""
    if initial_channels % input_channels != 0:
        raise ValueError(
            "initial_channels must be a multiple of the feature count "
            f"({initial_channels} vs {input_channels})"
        )
    if initial_channels % se_ratio != 0:
        raise ValueError(
            f"initial_channels ({initial_channels}) must be divisible by se_ratio ({se_ratio})"
        )
    return UNet1d(
        input_channels=input_channels,
        initial_channels=initial_channels,
        initial_kernel_size=15,
        down_channels=(initial_channels, initial_channels, initial_channels),
        down_kernel_size=(12, 15, 15),
        down_stride=(12, 9, 5),
        res_depth=res_depth,
        res_kernel_size=15,
        se_ratio=se_ratio,
        out_kernel_size=21,
    )
