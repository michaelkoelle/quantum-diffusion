from __future__ import annotations
import torch
import einops
from .utils import get_label_embedding


class DeepConvUndirected(torch.nn.Module):
    """Deep Convolutional Neural Network. Undirected"""

    def __init__(self, channels: list[int], shape: tuple[int, int]):
        super().__init__()
        assert channels[0] == channels[-1], "Input and output channels must be equal"
        self.channels = channels
        layers = []
        for i in range(len(channels) - 1):
            layers.append(
                torch.nn.Conv2d(
                    in_channels=channels[i],
                    out_channels=channels[i + 1],
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Sigmoid())
        self.net = torch.nn.Sequential(*layers)
        self.shape = shape

    def forward(self, x):
        assert len(x.shape) == 4, "Input must be 4D tensor"
        return self.net(x)

    def __repr__(self):
        return f"DeepConvUndirected({self.net})"

    def save_name(self) -> str:
        return f"deep_conv_undirected_{'_'.join(map(str, self.channels))}"


class DeepConvDirectedMulti(torch.nn.Module):
    """Deep Convolutional Neural Network. Directed"""

    def __init__(self, channels: list[int]):
        super().__init__()
        assert channels[0] == channels[-1], "Input and output channels must be equal"
        self.channels = channels
        layers = []
        for i in range(len(channels) - 1):
            layers.append(
                torch.nn.Conv2d(
                    in_channels=channels[i] + 1,
                    out_channels=channels[i + 1],
                    kernel_size=3,
                    padding=1,
                )
            )
            layers.append(torch.nn.ReLU())
        layers[-1] = torch.nn.Sigmoid()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, y):
        assert len(x.shape) == 4, "Input must be 4D tensor"
        y = einops.repeat(y, "b -> b 1 h w", h=x.shape[2], w=x.shape[3])
        for l in self.layers:
            if isinstance(l, torch.nn.Conv2d):
                x = torch.cat((x, y), dim=1)  # Concatenate label channel
            x = l(x)
        return x

    def __repr__(self):
        return f"DeepConvDirectedMulti({self.layers})"

    def save_name(self) -> str:
        return f"deep_conv_directed_multi_{'_'.join(map(str, self.channels))}"


class DeepConvDirectedSingle(DeepConvUndirected):
    def forward(self, x, y):
        assert len(x.shape) == 4, "Input must be 4D tensor"
        y = y.unsqueeze(-1)
        mask = get_label_embedding(y, self.shape[0], self.shape[1])
        masked_x = x + mask
        return self.net(masked_x)

    def __repr__(self):
        return f"DeepConvDirectedSingle({self.net})"

    def save_name(self) -> str:
        return f"deep_conv_directed_single_{'_'.join(map(str, self.channels))}"
