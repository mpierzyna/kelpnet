from torch import nn
import torch


class ConvBlock(nn.Module):
    """Two convolution layers with batch normalization and ReLU activation.

    - Increase the number of channels from `in_c` to `out_c`.
    - Decrease resolution by a factor of 4.
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    """Apply convolution to input for next layer and return maxpol of it for skip connection"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_ch):
        super().__init__()

        # Encoder
        self.e1 = EncoderBlock(n_ch, 32)

        # Bottleneck
        self.b = ConvBlock(32, 64)

        # Decoder
        self.d1 = DecoderBlock(64, 32)

        # Classifier
        self.outputs = nn.Conv2d(32, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()  # kelp cover is a binary mask

    def forward(self, inputs):
        # Encoder
        s1, p1 = self.e1(inputs)

        # Bottleneck
        b = self.b(p1)

        # Decoder
        d1 = self.d1(b, s1)

        # Classifier
        # No sigmoid here, because we use BCEWithLogitsLoss
        outputs = self.outputs(d1)
        outputs = outputs.squeeze(1)  # remove channel dimension since we only have one channel
        return outputs

 