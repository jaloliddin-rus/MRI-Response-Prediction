from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from monai.networks.blocks import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm

__all__ = ["AutoEncoder"]

class AutoEncoder(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        channels: Sequence[int],
        strides: Sequence[int],
        kernel_size: Sequence[int] | int = 3,
        up_kernel_size: Sequence[int] | int = 3,
        num_res_units: int = 0,
        inter_channels: list | None = None,
        inter_dilations: list | None = None,
        num_inter_units: int = 2,
        act: tuple | str | None = Act.PRELU,
        norm: tuple | str = Norm.INSTANCE,
        dropout: tuple | str | float | None = None,
        bias: bool = True,
        padding: Sequence[int] | int | None = None,
    ) -> None:
        super().__init__()
        self.dimensions = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = list(channels)
        self.strides = list(strides)
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.padding = padding
        self.num_inter_units = num_inter_units
        self.inter_channels = inter_channels if inter_channels is not None else []
        self.inter_dilations = list(inter_dilations or [1] * len(self.inter_channels))

        if len(channels) != len(strides):
            raise ValueError("Autoencoder expects matching number of channels and strides")

        self.encoded_channels = in_channels
        decode_channel_list = list(channels[-2::-1]) + [out_channels]

        self.encode, self.encoded_channels = self._get_encode_module(self.encoded_channels, channels, strides)
        self.intermediate, self.encoded_channels = self._get_intermediate_module(self.encoded_channels, num_inter_units)
        self.decode, _ = self._get_decode_module(self.encoded_channels, decode_channel_list, strides[::-1] or [1])

        # MRI parameter processing
        self.mri_param_fc = nn.Linear(3, 64)  # Adjust the output size as needed
        
        # Combine encoded features and MRI parameters
        self.combine_features = nn.Conv3d(self.encoded_channels + 64, self.encoded_channels, kernel_size=1)
        
        # Final layers for signal prediction
        self.final_conv = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        self.final_fc = nn.Linear(out_channels, 11 * 50)

    def _get_encode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> tuple[nn.Sequential, int]:
        """
        Returns the encode part of the network by building up a sequence of layers returned by `_get_encode_layer`.
        """
        encode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_encode_layer(layer_channels, c, s, False)
            encode.add_module("encode_%i" % i, layer)
            layer_channels = c

        return encode, layer_channels

    def _get_intermediate_module(self, in_channels: int, num_inter_units: int) -> tuple[nn.Module, int]:
        """
        Returns the intermediate block of the network which accepts input from the encoder and whose output goes
        to the decoder.
        """
        # Define some types
        intermediate: nn.Module
        unit: nn.Module

        intermediate = nn.Identity()
        layer_channels = in_channels

        if self.inter_channels:
            intermediate = nn.Sequential()

            for i, (dc, di) in enumerate(zip(self.inter_channels, self.inter_dilations)):
                if self.num_inter_units > 0:
                    unit = ResidualUnit(
                        spatial_dims=self.dimensions,
                        in_channels=layer_channels,
                        out_channels=dc,
                        strides=1,
                        kernel_size=self.kernel_size,
                        subunits=self.num_inter_units,
                        act=self.act,
                        norm=self.norm,
                        dropout=self.dropout,
                        dilation=di,
                        bias=self.bias,
                        padding=self.padding,
                    )
                else:
                    unit = Convolution(
                        spatial_dims=self.dimensions,
                        in_channels=layer_channels,
                        out_channels=dc,
                        strides=1,
                        kernel_size=self.kernel_size,
                        act=self.act,
                        norm=self.norm,
                        dropout=self.dropout,
                        dilation=di,
                        bias=self.bias,
                        padding=self.padding,
                    )

                intermediate.add_module("inter_%i" % i, unit)
                layer_channels = dc

        return intermediate, layer_channels

    def _get_decode_module(
        self, in_channels: int, channels: Sequence[int], strides: Sequence[int]
    ) -> tuple[nn.Sequential, int]:
        """
        Returns the decode part of the network by building up a sequence of layers returned by `_get_decode_layer`.
        """
        decode = nn.Sequential()
        layer_channels = in_channels

        for i, (c, s) in enumerate(zip(channels, strides)):
            layer = self._get_decode_layer(layer_channels, c, s, i == (len(strides) - 1))
            decode.add_module("decode_%i" % i, layer)
            layer_channels = c

        return decode, layer_channels

    def _get_encode_layer(self, in_channels: int, out_channels: int, strides: int, is_last: bool) -> nn.Module:
        """
        Returns a single layer of the encoder part of the network.
        """
        mod: nn.Module
        if self.num_res_units > 0:
            mod = ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=in_channels,
                out_channels=out_channels,
                strides=strides,
                kernel_size=self.kernel_size,
                subunits=self.num_res_units,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                padding=self.padding,
                last_conv_only=is_last,
            )
            return mod
        mod = Convolution(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            padding=self.padding,
            conv_only=is_last,
        )
        return mod

    def _get_decode_layer(self, in_channels: int, out_channels: int, strides: int, is_last: bool) -> nn.Sequential:
        """
        Returns a single layer of the decoder part of the network.
        """
        decode = nn.Sequential()

        conv = Convolution(
            spatial_dims=self.dimensions,
            in_channels=in_channels,
            out_channels=out_channels,
            strides=strides,
            kernel_size=self.up_kernel_size,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            padding=self.padding,
            conv_only=is_last and self.num_res_units == 0,
            is_transposed=True,
        )

        decode.add_module("conv", conv)

        if self.num_res_units > 0:
            ru = ResidualUnit(
                spatial_dims=self.dimensions,
                in_channels=out_channels,
                out_channels=out_channels,
                strides=1,
                kernel_size=self.kernel_size,
                subunits=1,
                act=self.act,
                norm=self.norm,
                dropout=self.dropout,
                bias=self.bias,
                padding=self.padding,
                last_conv_only=is_last,
            )

            decode.add_module("resunit", ru)

        return decode

    
    def forward(self, x: torch.Tensor, mri_params: torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.intermediate(x)
        
        # Process MRI parameters
        mri_features = self.mri_param_fc(mri_params)
        mri_features = mri_features.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mri_features = mri_features.expand(-1, -1, x.shape[2], x.shape[3], x.shape[4])
        
        # Combine encoded features and MRI parameters
        combined = torch.cat([x, mri_features], dim=1)
        combined = self.combine_features(combined)
        
        x = self.decode(combined)
        x = self.final_conv(x)
        
        # Global average pooling
        x = x.mean(dim=[2, 3, 4])
        
        # Final fully connected layer to predict signals
        x = self.final_fc(x)
        
        # Reshape to (batch_size, 11, 50)
        x = x.view(-1, 11, 50)
        
        return x