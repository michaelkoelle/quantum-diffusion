from .unet import DownBlock, UpBlock, UNetUndirected, get_label_embedding
import torch
from .qconv import QConv2d


class DownBlockS(DownBlock):
    def __init__(self, in_channels, out_channels, pooling, kernel_size=3, qdepth=3):
        super().__init__(in_channels, out_channels, pooling, kernel_size, qdepth)
        self.net = torch.nn.Sequential(
            QConv2d(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                kernel_size=self.kernel_size,
                qdepth=qdepth,
                padding=1,
            ),
            torch.nn.BatchNorm2d(self.out_channels),
        )


class UpBlockS(UpBlock):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        qdepth=3,
    ):
        super().__init__(in_channels, out_channels, kernel_size, qdepth=0)
        self.net = torch.nn.Sequential(
            QConv2d(
                in_channels=2 * out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=1,
                qdepth=qdepth,
            ),
            torch.nn.BatchNorm2d(out_channels),
        )
        self.up_conv = torch.nn.Sequential(
            torch.nn.Upsample(scale_factor=2, mode="bilinear"),
            QConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                qdepth=qdepth,
            ),
        )


class UNetUndirectedS(UNetUndirected):
    def __init__(
        self,
        depth=3,
        start_channels=8,
        qdepth=3,
    ):
        super().__init__(depth, start_channels, qdepth=0)
        self.qdepth = qdepth
        down_blocks = [
            DownBlockS(
                in_channels=db.in_channels,
                out_channels=db.out_channels,
                pooling=db.pooling,
                kernel_size=db.kernel_size,
                qdepth=self.qdepth,
            )
            for db in self.down_blocks
        ]
        self.down_blocks = torch.nn.ModuleList(down_blocks)
        up_blocks = [
            UpBlockS(
                in_channels=ub.in_channels,
                out_channels=ub.out_channels,
                kernel_size=ub.kernel_size,
                qdepth=self.qdepth,
            )
            for ub in self.up_blocks
        ]
        self.up_blocks = torch.nn.ModuleList(up_blocks)

    def save_name(self) -> str:
        return f"unet_s_undirected_d{self.depth}_s{self.start_channels}_d{self.qdepth}"


class UnetDirectedS(UNetUndirectedS):
    def forward(self, x, y):
        mask = get_label_embedding(y, x.shape[2], x.shape[3])
        masked_x = x + mask
        return super().forward(masked_x)

    def save_name(self) -> str:
        return f"unet_s_directed_d{self.depth}_s{self.start_channels}_d{self.qdepth}"
