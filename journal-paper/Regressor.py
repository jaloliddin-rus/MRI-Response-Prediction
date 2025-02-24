from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import torch
import torch.nn as nn
from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.convutils import calculate_out_shape, same_padding
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import Reshape
from monai.utils import ensure_tuple, ensure_tuple_rep
from torch.nn import TransformerEncoder, TransformerEncoderLayer

__all__ = ["Regressor"]


class Regressor(nn.Module):
    def __init__(
        self,
        in_shape: Sequence[int],
        out_shape: Sequence[int],
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 2,
        act=Act.PRELU,
        norm=Norm.INSTANCE,
        dropout: float | None = None,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels, *self.in_shape = ensure_tuple(in_shape)
        self.dimensions = len(self.in_shape)
        self.channels = ensure_tuple(channels)
        self.strides = ensure_tuple(strides)
        self.out_shape = ensure_tuple(out_shape)
        self.kernel_size = ensure_tuple_rep(kernel_size, self.dimensions)
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.net = nn.Sequential()

        echannel = self.in_channels

        padding = same_padding(kernel_size)

        self.final_size = np.asarray(self.in_shape, dtype=int)
        self.reshape = Reshape(*self.out_shape)

        # encode stage
        for i, (c, s) in enumerate(zip(self.channels, self.strides)):
            layer = self._get_layer(echannel, c, s, i == len(channels) - 1)
            echannel = c  # use the output channel number as the input for the next loop
            self.net.add_module("layer_%i" % i, layer)
            self.final_size = calculate_out_shape(self.final_size, kernel_size, s, padding)  # type: ignore

        self.final = self._get_final_layer((echannel,) + self.final_size)

    def _get_layer(
        self, in_channels: int, out_channels: int, strides: int, is_last: bool
    ) -> ResidualUnit | Convolution:
        """
        Returns a layer accepting inputs with `in_channels` number of channels and producing outputs of `out_channels`
        number of channels. The `strides` indicates downsampling factor, ie. convolutional stride. If `is_last`
        is True this is the final layer and is not expected to include activation and normalization layers.
        """

        layer: ResidualUnit | Convolution

        if self.num_res_units > 0:
            layer = ResidualUnit(
                subunits=self.num_res_units,
                last_conv_only=is_last,
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
            )
        else:
            layer = Convolution(
                conv_only=is_last,
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
            )

        return layer

    def _get_final_layer(self, in_shape: Sequence[int]):
        linear = nn.Linear(int(np.prod(in_shape)), int(np.prod(self.out_shape)))
        return nn.Sequential(nn.Flatten(), linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = self.final(x)
        x = self.reshape(x)
        
        # Flatten or reduce the 11-dimension to match (16, 50)
        #x = x.mean(dim=1)  # For example, average over the 11-dimension
        return x

class CustomRegressor(Regressor):
    def __init__(self, mri_param_size=3):
        super(CustomRegressor, self).__init__(in_shape=(9, 64, 64, 64), out_shape=(11, 50),
                                              channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2, 1))

        self.mri_param_size = mri_param_size
        
        # Adjust the final layer to incorporate MRI parameters
        self.final = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256 * 4 * 4 * 4 + mri_param_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 11 * 50),
            torch.nn.Sigmoid()
        )
        self.reshape = Reshape(11, 50)

    def forward(self, x: torch.Tensor, mri_params: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        x = x.flatten(1)
        # Concatenate flattened image features with MRI parameters
        x = torch.cat([x, mri_params], dim=1)
        x = self.final(x)
        x = self.reshape(x)
        return x