"""Tiny 2D U-Net with 3 input channels: CT, coarse tumor probability, Bernoulli entropy."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class BoundaryAwareTinyUNet2d(nn.Module):
    """
    Compact 2D U-Net.
    in_channels: 3 — CT slice + coarse tumor probability + uncertainty (entropy of probability).
    """

    def __init__(self, in_channels: int = 3, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base * 8, base * 16)

        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.dec4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out_conv = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out_conv(self.forward_features(x))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.enc1(x)
        p1 = self.pool(c1)
        c2 = self.enc2(p1)
        p2 = self.pool(c2)
        c3 = self.enc3(p2)
        p3 = self.pool(c3)
        c4 = self.enc4(p3)
        p4 = self.pool(c4)

        b = self.bottleneck(p4)

        u4 = self.up4(b)
        u4 = self._match_pad(u4, c4)
        d4 = self.dec4(torch.cat([u4, c4], dim=1))

        u3 = self.up3(d4)
        u3 = self._match_pad(u3, c3)
        d3 = self.dec3(torch.cat([u3, c3], dim=1))

        u2 = self.up2(d3)
        u2 = self._match_pad(u2, c2)
        d2 = self.dec2(torch.cat([u2, c2], dim=1))

        u1 = self.up1(d2)
        u1 = self._match_pad(u1, c1)
        return self.dec1(torch.cat([u1, c1], dim=1))

    @staticmethod
    def _match_pad(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        dh = up.size(2) - skip.size(2)
        dw = up.size(3) - skip.size(3)
        if dh != 0 or dw != 0:
            up = up[:, :, : skip.size(2), : skip.size(3)]
        return up
