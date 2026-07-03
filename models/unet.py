"""
=========================================================
U-Net Architecture
Project : Object Measurement System
Author  : Jibran Mujtaba
=========================================================
"""

import torch
import torch.nn as nn


# -------------------------------------------------------
# Double Convolution Block
# -------------------------------------------------------
class DoubleConv(nn.Module):
    """
    Conv -> BatchNorm -> ReLU
    Conv -> BatchNorm -> ReLU
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        self.double_conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)

        )

    def forward(self, x):
        return self.double_conv(x)


# -------------------------------------------------------
# Down Sampling Block
# -------------------------------------------------------
class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.down = nn.Sequential(

            nn.MaxPool2d(kernel_size=2),

            DoubleConv(in_channels, out_channels)

        )

    def forward(self, x):
        return self.down(x)


# -------------------------------------------------------
# Up Sampling Block
# -------------------------------------------------------
class Up(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(

            in_channels,

            in_channels // 2,

            kernel_size=2,

            stride=2

        )

        self.conv = DoubleConv(

            in_channels,

            out_channels

        )

    def forward(self, x1, x2):

        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(

            x1,

            [

                diffX // 2,

                diffX - diffX // 2,

                diffY // 2,

                diffY - diffY // 2

            ]

        )

        x = torch.cat([x2, x1], dim=1)

        return self.conv(x)


# -------------------------------------------------------
# Output Layer
# -------------------------------------------------------
class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()

        self.conv = nn.Conv2d(

            in_channels,

            out_channels,

            kernel_size=1

        )

    def forward(self, x):
        return self.conv(x)


# -------------------------------------------------------
# Complete U-Net
# -------------------------------------------------------
class UNet(nn.Module):

    def __init__(self, n_channels=3, n_classes=1):

        super(UNet, self).__init__()

        # Encoder
        self.inc = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)

        self.down2 = Down(128, 256)

        self.down3 = Down(256, 512)

        self.down4 = Down(512, 1024)

        # Decoder
        self.up1 = Up(1024, 512)

        self.up2 = Up(512, 256)

        self.up3 = Up(256, 128)

        self.up4 = Up(128, 64)

        # Output
        self.outc = OutConv(64, n_classes)

    def forward(self, x):

        # Encoder
        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        # Decoder
        x = self.up1(x5, x4)

        x = self.up2(x, x3)

        x = self.up3(x, x2)

        x = self.up4(x, x1)

        logits = self.outc(x)

        return logits


# -------------------------------------------------------
# Test the Network
# -------------------------------------------------------
if __name__ == "__main__":

    print("=" * 60)
    print("Testing U-Net Architecture")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet().to(device)

    print(model)

    print("\nCreating Dummy Input...")

    x = torch.randn((1, 3, 640, 640)).to(device)

    print("Input Shape :", x.shape)

    y = model(x)

    print("Output Shape:", y.shape)

    total_params = sum(p.numel() for p in model.parameters())

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTotal Parameters     :", f"{total_params:,}")

    print("Trainable Parameters:", f"{trainable_params:,}")

    print("\nModel Successfully Built!")

    print("=" * 60)