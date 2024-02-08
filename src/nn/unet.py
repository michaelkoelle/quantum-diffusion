import torch
import warnings
import einops
import math
from .utils import autocrop, autopad, get_label_embedding
from .qconv import QConv2d


def Conv2d(**kwargs):
    """
    Wrapper for QConv2d and torch.nn.Conv2d.
    If qdepth is 0, then torch.nn.Conv2d is used, otherwise QConv2d is used.
    Important kwargs:
        qdepth: int, number of qubits to use for the quantum convolution. If 0, then torch.nn.Conv2d is used.
        in_channels: int, number of input channels
        out_channels: int, number of output channels
        kernel_size: int or tuple, size of the convolution kernel
        padding: int or tuple, padding added to both sides of the input
    """
    qdepth = kwargs.pop("qdepth", 3)
    if qdepth > 0:
        return QConv2d(qdepth=qdepth, **kwargs)  # type: ignore
    else:
        return torch.nn.Conv2d(**kwargs)  # type: ignore


class UpBlock(torch.nn.Module):
    """UpBlock for UNet."""

    def __init__(self, in_channels, out_channels, kernel_size=3, qdepth=3):
        super(UpBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.up_conv = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear"),
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                qdepth=qdepth,
            ),
        )

        self.net = torch.nn.Sequential(
            Conv2d(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=1,
                qdepth=qdepth,
            ),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(out_channels),
            Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=1,
                qdepth=qdepth,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, from_down, from_up):
        from_up = self.up_conv(from_up)
        from_down, from_up = autopad(
            from_down, from_up
        )  # Pad from_up to match from_down
        x = torch.cat([from_up, from_down], dim=1)
        x = self.net(x)
        return x


class DownBlock(torch.nn.Module):
    """DownBlock for UNet"""

    def __init__(self, in_channels, out_channels, pooling, kernel_size=3, qdepth=3):
        super(DownBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.net = torch.nn.Sequential(
            Conv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                qdepth=qdepth,
                padding=1,
            ),
            torch.nn.BatchNorm2d(self.out_channels),
            torch.nn.ReLU(),
            Conv2d(
                in_channels=self.out_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                qdepth=qdepth,
                padding=1,
            ),
            torch.nn.BatchNorm2d(self.out_channels),
            torch.nn.ReLU(),
        )
        if self.pooling:
            self.pooling_layer = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.net(x)
        before_pool = x
        if self.pooling:
            x = self.pooling_layer(x)
        return x, before_pool


class UNetUndirected(torch.nn.Module):
    """
    U-shaped Network for image segmentation.
    Undirected (no labels).
    """

    def __init__(
        self,
        depth=3,
        start_channels=8,
        qdepth=3,
    ):
        super().__init__()
        self.depth = depth
        self.start_channels = start_channels
        self.qdepth = qdepth
        assert self.depth > 0, "Depth must be greater than 0"
        out_channel = -1  # to suppress warnings about uninitialized variables
        down_blocks = []
        for i in range(self.depth):
            in_channel = 1 if i == 0 else out_channel  # 1 for the first layer
            out_channel = self.start_channels * 2**i
            pooling = i < depth - 1  # no pooling in the last layer
            down_blocks.append(
                DownBlock(in_channel, out_channel, pooling=pooling, qdepth=qdepth)
            )

        up_blocks = []
        for i in range(self.depth - 1):
            in_channel = (
                out_channel
            )  # set the input channel to the output channel of the previous layer
            out_channel = out_channel // 2
            up_blocks.append(UpBlock(in_channel, out_channel, qdepth=qdepth))

        self.down_blocks = torch.nn.ModuleList(down_blocks)
        self.up_blocks = torch.nn.ModuleList(up_blocks)
        self.final_conv = Conv2d(
            in_channels=out_channel,
            out_channels=1,
            kernel_size=1,
            padding=0,
            qdepth=qdepth,
        )

    def forward(self, x):
        encoder_outputs = []  # list of skip connections
        for i, block in enumerate(self.down_blocks):
            x, before_pool = block(x)
            encoder_outputs.append(before_pool)

        for i, block in enumerate(self.up_blocks):
            skip = encoder_outputs[-(i + 2)]
            x = block(skip, x)

        x = self.final_conv(x)

        return x

    def extra_repr(self) -> str:
        return f"depth={self.depth}"

    def save_name(self) -> str:
        return f"unet_undirected_d{self.depth}_s{self.start_channels}_d{self.qdepth}"


class UnetDirected(UNetUndirected):
    def forward(self, x, y):
        mask = get_label_embedding(y, x.shape[2], x.shape[3])
        masked_x = x + mask
        return super().forward(masked_x)

    def save_name(self) -> str:
        return f"unet_directed_d{self.depth}_s{self.start_channels}_d{self.qdepth}"
