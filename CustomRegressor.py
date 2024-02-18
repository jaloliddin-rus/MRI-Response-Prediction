# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 15:35:02 2023

@author: Zahir
"""
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
        return x

class CustomRegressor(Regressor):
    def __init__(self):
        # Adjusting strides to ensure spatial dimensions don't get reduced to 1x1x1
        super(CustomRegressor, self).__init__(in_shape=(9, 64, 64, 64), out_shape=(50,),
                                              channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2, 1))
        
        # Adjusting the final layer to match the output from the convolutional layers
        # Assuming the spatial dimensions are reduced to 4x4x4 after all convolutions
        self.final = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.2),  # Add dropout with 50% probability
            torch.nn.Linear(256 * 4 * 4 * 4, 50),  # Adjusted based on the new channels and strides
            torch.nn.Sigmoid()
        )

class CustomRegressorCNN(nn.Module):
    def __init__(self):
        super(CustomRegressorCNN, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            self._conv_block(9, 16, stride=2),
            self._conv_block(16, 32, stride=2),
            self._conv_block(32, 64, stride=2),
            self._conv_block(64, 128, stride=2),
            self._conv_block(128, 256, stride=1)  # Keeping spatial dimensions at 4x4x4
        )
        
        # Decoder (you can expand this if needed)
        self.decoder = nn.Sequential(
            self._conv_block(256, 128),
            self._conv_block(128, 64)
        )
        
        # Final layers
        self.final = nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(64 * 4 * 4 * 4, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 50),
            torch.nn.Sigmoid()
        )
        
    def _conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final(x)
        return x
    
class CustomRegressorTransformer(nn.Module):
    def __init__(self):
        super(CustomRegressorTransformer, self).__init__()
        
        # Feature extraction using convolutional layers
        self.feature_extractor = nn.Sequential(
            self._conv_block(9, 16, stride=2),
            self._conv_block(16, 32, stride=2),
            self._conv_block(32, 64, stride=2),
            self._conv_block(64, 128, stride=2),
            self._conv_block(128, 256, stride=1)  # Keeping spatial dimensions at 4x4x4
        )
        
        # Transformer encoder
        encoder_layers = TransformerEncoderLayer(d_model=256, nhead=8)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers=6)
        
        # Final layers for regression
        self.final = nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256 * 4 * 4 * 4, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(256, 50),
            torch.nn.Sigmoid()
        )
        
    def _conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Extract features
        x = self.feature_extractor(x)
        
        # Prepare for transformer: flatten spatial dimensions and permute dimensions
        x = x.view(x.size(0), 256, -1).permute(2, 0, 1)  # Shape: [S, B, C]
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Prepare for final layers: permute back and reshape
        x = x.permute(1, 2, 0).contiguous().view(x.size(1), -1)
        
        # Final regression layers
        x = self.final(x)
        
        return x
