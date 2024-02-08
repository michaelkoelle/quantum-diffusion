"""
Module contains all neural networks used in the project.
Some are not used anymore, but are kept for reference and marked as deprecated.
"""

from .qconv import QConv2d
from .qdense import *
from .unet import UnetDirected, UNetUndirected
from .dense import DenseDirected, DenseUndirected
from .utils import *
from .unet_simple import UNetUndirectedS, UnetDirectedS
from .conv import *
