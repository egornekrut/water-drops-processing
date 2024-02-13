from functools import partial
from typing import Callable

import torch
from torch import Tensor
from torch.nn import (AdaptiveAvgPool3d, BatchNorm3d, Conv3d, Dropout, Linear,
                      Module, Sequential, SiLU, Sigmoid)
from torchvision.ops import StochasticDepth


class FrameClassModel(Module):
    def __init__(self, batch_size: int) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.model = Sequential(
            Conv3d(
                in_channels=1,
                out_channels=16,
                kernel_size=(1, 5, 5),
                stride=(1, 3, 3),
                padding=0,
                bias=True,
            ),
            BatchNorm3d(16),
            SiLU(True),
            MBConvPlusBlock(16, 16, time_dim_ch=5),
            MBConvPlusBlock(16, 32, stride=2, kernel_time_size=3, time_dim_ch=5),
            MBConvPlusBlock(32, 32),
            MBConvPlusBlock(32, 64, stride=2, kernel_time_size=3, time_dim_ch=3),
            MBConvPlusBlock(64, 64),
            AdaptiveAvgPool3d((1, 1, 1)),
        )
        self.classifier = Sequential(
            Dropout(p=0.2, inplace=True),
            Linear(64, 1),
        )

    def forward(self, x: Tensor):
        feats = self.model(x).view(self.batch_size, -1)

        return self.classifier(feats)

class MBConvPlusBlock(Module):
    def __init__(
            self,
            in_ch: int,
            out_ch: int,
            exp_ratio: float = 2,
            kernel_size: int = 3,
            kernel_time_size: int = 1,
            stride: int = 1,
            st_depth_prob: float = 0.2,
            time_dim_ch=1,
        ) -> None:
        super().__init__()
        self.middle_channels = int(in_ch * exp_ratio)
        self.use_res_conn = stride == 1 and in_ch == out_ch
        self.layers = Sequential(
            Conv3d(
                in_channels=in_ch,
                out_channels=self.middle_channels,
                kernel_size=(1, 1, 1),
                stride=1,
                padding=0,
            ),
            BatchNorm3d(self.middle_channels),
            SiLU(True),
            Conv3d(
                in_channels=self.middle_channels,
                out_channels=self.middle_channels,
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, stride, stride),
                groups=self.middle_channels,
                padding=(0, kernel_size // 2, kernel_size // 2),
            ),
            BatchNorm3d(self.middle_channels),
            SiLU(True),
            SqueezeExcitation(
                self.middle_channels,
                max(1, self.middle_channels // 4),
                activation=partial(SiLU, True),
                time_dim_ch=time_dim_ch,
            ),
            Conv3d(
                in_channels= self.middle_channels,
                out_channels=out_ch,
                kernel_size=(kernel_time_size, 1, 1),
                stride=1,
                padding=0,
            ),
            BatchNorm3d(out_ch),
            StochasticDepth(p=st_depth_prob, mode='row'),
        )

    def forward(self, in_tensor: Tensor):
        output = self.layers(in_tensor)
        if self.use_res_conn:
            output += in_tensor

        return output


class SqueezeExcitation(Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., torch.nn.Module] = torch.nn.ReLU,
        scale_activation: Callable[..., torch.nn.Module] = torch.nn.Sigmoid,
        time_dim_ch: int = 3,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool3d((time_dim_ch, 1, 1))
        self.fc1 = torch.nn.Conv3d(input_channels, squeeze_channels, (1, 1, 1))
        self.fc2 = torch.nn.Conv3d(squeeze_channels, input_channels, (1, 1, 1))
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input: Tensor) -> Tensor:
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input: Tensor) -> Tensor:
        scale = self._scale(input)
        return scale * input
