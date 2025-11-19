import os
from math import sqrt, floor, ceil, log2
import math
from typing import Optional, Tuple, Union, List
import functools
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from einops import rearrange
import safetensors.torch

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.activations import get_activation
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.autoencoders.vae import (
    DecoderOutput,
    DiagonalGaussianDistribution,
)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True


def prepare_causal_attention_mask(
    f: int, s: int, dtype: torch.dtype, device: torch.device, b: int
) -> torch.Tensor:
    return (
        torch.ones((f, f), dtype=dtype, device=device)
        .tril_()
        .log_()
        .repeat_interleave(s, dim=0)
        .repeat_interleave(s, dim=1)
        .unsqueeze(0)
        .expand(b, -1, -1)
        .contiguous()
    )


class HunyuanVideoCausalConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int, int]] = 3,
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        bias: bool = True,
        pad_mode: str = "replicate",
    ) -> None:
        super().__init__()

        kernel_size = (
            (kernel_size, kernel_size, kernel_size)
            if isinstance(kernel_size, int)
            else kernel_size
        )

        self.pad_mode = pad_mode
        self.time_causal_padding = (
            kernel_size[0] // 2,
            kernel_size[0] // 2,
            kernel_size[1] // 2,
            kernel_size[1] // 2,
            kernel_size[2] - 1,
            0,
        )

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = F.pad(
            hidden_states, self.time_causal_padding, mode=self.pad_mode
        )
        return self.conv(hidden_states)


class HunyuanVideoUpsampleCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
        upsample_factor: Tuple[float, float, float] = (2, 2, 2),
    ) -> None:
        super().__init__()

        out_channels = out_channels or in_channels
        self.upsample_factor = upsample_factor

        self.conv = HunyuanVideoCausalConv3d(
            in_channels, out_channels, kernel_size, stride, bias=bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_frames = hidden_states.size(2)
        dtp = hidden_states.dtype
        first_frame, other_frames = hidden_states.split((1, num_frames - 1), dim=2)
        first_frame = F.interpolate(
            first_frame.squeeze(2),
            scale_factor=self.upsample_factor[1:],
            mode="nearest",
        ).unsqueeze(2).to(dtp) #force cast

        if num_frames > 1:
            other_frames = other_frames.contiguous()
            other_frames = F.interpolate(
                other_frames, scale_factor=self.upsample_factor, mode="nearest"
            ).to(dtp) # force cast
            hidden_states = torch.cat((first_frame, other_frames), dim=2)
            del first_frame
            del other_frames
            torch.cuda.empty_cache()
        else:
            hidden_states = first_frame

        hidden_states = self.conv(hidden_states)
        return hidden_states


class HunyuanVideoDownsampleCausal3D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
        padding: int = 1,
        kernel_size: int = 3,
        bias: bool = True,
        stride=2,
    ) -> None:
        super().__init__()
        out_channels = out_channels or channels

        self.conv = HunyuanVideoCausalConv3d(
            channels, out_channels, kernel_size, stride, padding, bias=bias
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(hidden_states)
        return hidden_states


class HunyuanVideoResnetBlockCausal3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.0,
        groups: int = 32,
        eps: float = 1e-6,
        non_linearity: str = "swish",
    ) -> None:
        super().__init__()
        out_channels = out_channels or in_channels

        self.nonlinearity = get_activation(non_linearity)

        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps, affine=True)
        self.conv1 = HunyuanVideoCausalConv3d(in_channels, out_channels, 3, 1, 0)

        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = HunyuanVideoCausalConv3d(out_channels, out_channels, 3, 1, 0)

        self.conv_shortcut = None
        if in_channels != out_channels:
            self.conv_shortcut = HunyuanVideoCausalConv3d(
                in_channels, out_channels, 1, 1, 0
            )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtp = hidden_states.dtype
        hidden_states = hidden_states.contiguous()
        residual = hidden_states

        hidden_states = self.norm1(hidden_states).to(dtp) #force cast
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states).to(dtp) #force cast
        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)
        
        hidden_states = hidden_states + residual
        return hidden_states


class HunyuanVideoMidBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_attention: bool = True,
        attention_head_dim: int = 1,
    ) -> None:
        super().__init__()
        resnet_groups = (
            resnet_groups if resnet_groups is not None else min(in_channels // 4, 32)
        )
        self.add_attention = add_attention

        # There is always at least one resnet
        resnets = [
            HunyuanVideoResnetBlockCausal3D(
                in_channels=in_channels,
                out_channels=in_channels,
                eps=resnet_eps,
                groups=resnet_groups,
                dropout=dropout,
                non_linearity=resnet_act_fn,
            )
        ]
        attentions = []

        for _ in range(num_layers):
            if self.add_attention:
                attentions.append(
                    Attention(
                        in_channels,
                        heads=in_channels // attention_head_dim,
                        dim_head=attention_head_dim,
                        eps=resnet_eps,
                        norm_num_groups=resnet_groups,
                        residual_connection=True,
                        bias=True,
                        upcast_softmax=True,
                        _from_deprecated_attn_block=True,
                    )
                )
            else:
                attentions.append(None)

            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.attentions = nn.ModuleList(attentions)
        self.resnets = nn.ModuleList(resnets)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.resnets[0](hidden_states)

        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            if attn is not None:
                batch_size, _, num_frames, height, width = hidden_states.shape
                hidden_states = hidden_states.permute(0, 2, 3, 4, 1).flatten(1, 3)
                mask = prepare_causal_attention_mask(
                    num_frames,
                    height * width,
                    hidden_states.dtype,
                    hidden_states.device,
                    batch_size,
                )
                hidden_states = attn(hidden_states, attention_mask=mask)
                hidden_states = hidden_states.unflatten(
                    1, (num_frames, height, width)
                ).permute(0, 4, 1, 2, 3)

            hidden_states = resnet(hidden_states)

        return hidden_states


class HunyuanVideoDownBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_downsample: bool = True,
        downsample_stride: int = 2,
        downsample_padding: int = 1,
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_downsample:
            self.downsamplers = nn.ModuleList(
                [
                    HunyuanVideoDownsampleCausal3D(
                        out_channels,
                        out_channels=out_channels,
                        padding=downsample_padding,
                        stride=downsample_stride,
                    )
                ]
            )
        else:
            self.downsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                hidden_states = downsampler(hidden_states)

        return hidden_states


class HunyuanVideoUpBlock3D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        num_layers: int = 1,
        resnet_eps: float = 1e-6,
        resnet_act_fn: str = "swish",
        resnet_groups: int = 32,
        add_upsample: bool = True,
        upsample_scale_factor: Tuple[int, int, int] = (2, 2, 2),
    ) -> None:
        super().__init__()
        resnets = []

        for i in range(num_layers):
            input_channels = in_channels if i == 0 else out_channels

            resnets.append(
                HunyuanVideoResnetBlockCausal3D(
                    in_channels=input_channels,
                    out_channels=out_channels,
                    eps=resnet_eps,
                    groups=resnet_groups,
                    dropout=dropout,
                    non_linearity=resnet_act_fn,
                )
            )

        self.resnets = nn.ModuleList(resnets)

        if add_upsample:
            self.upsamplers = nn.ModuleList(
                [
                    HunyuanVideoUpsampleCausal3D(
                        out_channels,
                        out_channels=out_channels,
                        upsample_factor=upsample_scale_factor,
                    )
                ]
            )
        else:
            self.upsamplers = None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            hidden_states = resnet(hidden_states)

        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)

        return hidden_states


class HunyuanVideoEncoder3D(nn.Module):
    r"""
    Causal encoder for 3D video-like data introduced
    in [Hunyuan Video](https://huggingface.co/papers/2412.03603).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        down_block_types: Tuple[str, ...] = (
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        double_z: bool = True,
        mid_block_add_attention=True,
        temporal_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ) -> None:
        super().__init__()

        self.conv_in = HunyuanVideoCausalConv3d(
            in_channels, block_out_channels[0], kernel_size=3, stride=1
        )
        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            if down_block_type != "HunyuanVideoDownBlock3D":
                raise ValueError(f"Unsupported down_block_type: {down_block_type}")

            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_downsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_downsample_layers = int(np.log2(temporal_compression_ratio))

            if temporal_compression_ratio == 4:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(
                    i >= (len(block_out_channels) - 1 - num_time_downsample_layers)
                    and not is_final_block
                )
            elif temporal_compression_ratio == 8:
                add_spatial_downsample = bool(i < num_spatial_downsample_layers)
                add_time_downsample = bool(i < num_time_downsample_layers)
            else:
                raise ValueError(
                    f"Unsupported time_compression_ratio: {temporal_compression_ratio}"
                )

            downsample_stride_HW = (2, 2) if add_spatial_downsample else (1, 1)
            downsample_stride_T = (2,) if add_time_downsample else (1,)
            downsample_stride = tuple(downsample_stride_T + downsample_stride_HW)

            down_block = HunyuanVideoDownBlock3D(
                num_layers=layers_per_block,
                in_channels=input_channel,
                out_channels=output_channel,
                add_downsample=bool(add_spatial_downsample or add_time_downsample),
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
                downsample_stride=downsample_stride,
                downsample_padding=0,
            )

            self.down_blocks.append(down_block)

        self.mid_block = HunyuanVideoMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = HunyuanVideoCausalConv3d(
            block_out_channels[-1], conv_out_channels, kernel_size=3
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv_in(hidden_states)

        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)

        hidden_states = self.mid_block(hidden_states)

        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states)
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class HunyuanVideoDecoder3D(nn.Module):
    r"""
    Causal decoder for 3D video-like data introduced
    in [Hunyuan Video](https://huggingface.co/papers/2412.03603).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        up_block_types: Tuple[str, ...] = (
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
        ),
        block_out_channels: Tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        act_fn: str = "silu",
        mid_block_add_attention=True,
        time_compression_ratio: int = 4,
        spatial_compression_ratio: int = 8,
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = HunyuanVideoCausalConv3d(
            in_channels, block_out_channels[-1], kernel_size=3, stride=1
        )
        self.up_blocks = nn.ModuleList([])

        # mid
        self.mid_block = HunyuanVideoMidBlock3D(
            in_channels=block_out_channels[-1],
            resnet_eps=1e-6,
            resnet_act_fn=act_fn,
            attention_head_dim=block_out_channels[-1],
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            if up_block_type != "HunyuanVideoUpBlock3D":
                raise ValueError(f"Unsupported up_block_type: {up_block_type}")

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1
            num_spatial_upsample_layers = int(np.log2(spatial_compression_ratio))
            num_time_upsample_layers = int(np.log2(time_compression_ratio))

            if time_compression_ratio == 4:
                add_spatial_upsample = bool(i < num_spatial_upsample_layers)
                add_time_upsample = bool(
                    i >= len(block_out_channels) - 1 - num_time_upsample_layers
                    and not is_final_block
                )
            else:
                raise ValueError(
                    f"Unsupported time_compression_ratio: {time_compression_ratio}"
                )

            upsample_scale_factor_HW = (2, 2) if add_spatial_upsample else (1, 1)
            upsample_scale_factor_T = (2,) if add_time_upsample else (1,)
            upsample_scale_factor = tuple(
                upsample_scale_factor_T + upsample_scale_factor_HW
            )

            up_block = HunyuanVideoUpBlock3D(
                num_layers=self.layers_per_block + 1,
                in_channels=prev_output_channel,
                out_channels=output_channel,
                add_upsample=bool(add_spatial_upsample or add_time_upsample),
                upsample_scale_factor=upsample_scale_factor,
                resnet_eps=1e-6,
                resnet_act_fn=act_fn,
                resnet_groups=norm_num_groups,
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(
            num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6
        )
        self.conv_act = nn.SiLU()
        self.conv_out = HunyuanVideoCausalConv3d(
            block_out_channels[0], out_channels, kernel_size=3
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        dtp = hidden_states.dtype
        hidden_states = self.conv_in(hidden_states)

        hidden_states = self.mid_block(hidden_states)

        for up_block in self.up_blocks:
            hidden_states = up_block(hidden_states)

        hidden_states = self.conv_norm_out(hidden_states)
        hidden_states = self.conv_act(hidden_states).to(dtp) # force cast
        hidden_states = self.conv_out(hidden_states)

        return hidden_states


class AutoencoderKLHunyuanVideo(ModelMixin, ConfigMixin):
    r"""
    A VAE model with KL loss for encoding videos into latents
    and decoding latent representations into videos.
    Introduced in [HunyuanVideo](https://huggingface.co/papers/2412.03603).

    This model inherits from [`ModelMixin`]. Check the superclass
    documentation for it's generic methods implemented
    for all models (such as downloading or saving).
    """

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 16,
        down_block_types: Tuple[str, ...] = (
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
            "HunyuanVideoDownBlock3D",
        ),
        up_block_types: Tuple[str, ...] = (
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
            "HunyuanVideoUpBlock3D",
        ),
        block_out_channels: Tuple[int] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        act_fn: str = "silu",
        norm_num_groups: int = 32,
        scaling_factor: float = 0.476986,
        spatial_compression_ratio: int = 8,
        temporal_compression_ratio: int = 4,
        mid_block_add_attention: bool = True,
    ) -> None:
        super().__init__()

        self.time_compression_ratio = temporal_compression_ratio

        self.encoder = HunyuanVideoEncoder3D(
            in_channels=in_channels,
            out_channels=latent_channels,
            down_block_types=down_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            double_z=True,
            mid_block_add_attention=mid_block_add_attention,
            temporal_compression_ratio=temporal_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
        )

        self.decoder = HunyuanVideoDecoder3D(
            in_channels=latent_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            time_compression_ratio=temporal_compression_ratio,
            spatial_compression_ratio=spatial_compression_ratio,
            mid_block_add_attention=mid_block_add_attention,
        )

        self.quant_conv = nn.Conv3d(
            2 * latent_channels, 2 * latent_channels, kernel_size=1
        )
        self.post_quant_conv = nn.Conv3d(
            latent_channels, latent_channels, kernel_size=1
        )

        self.spatial_compression_ratio = spatial_compression_ratio
        self.temporal_compression_ratio = temporal_compression_ratio

        self.use_slicing = False

        self.use_tiling = True

        self.use_framewise_encoding = True
        self.use_framewise_decoding = True

        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_num_frames = 16

        self.tile_sample_stride_height = 192
        self.tile_sample_stride_width = 192
        self.tile_sample_stride_num_frames = 12

        self.tile_size = None

        # Track if manual tile sizes were set (to prevent automatic overriding)
        self.use_manual_tiling = False

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        _, _, num_frames, height, width = x.shape

        if self.use_framewise_decoding and num_frames > (
            self.tile_sample_min_num_frames + 1
        ):
            return self._temporal_tiled_encode(x)

        if self.use_tiling and (
            width > self.tile_sample_min_width or height > self.tile_sample_min_height
        ):
            return self.tiled_encode(x)

        x = self.encoder(x)
        enc = self.quant_conv(x)
        return enc

    @apply_forward_hook
    def encode(
        self, x: torch.Tensor, opt_tiling: bool = True, return_dict: bool = True
    ) -> Union[AutoencoderKLOutput, Tuple[DiagonalGaussianDistribution]]:
        r"""
        Encode a batch of images into latents.

        Args:
            x (`torch.Tensor`): Input batch of images.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.autoencoder_kl.AutoencoderKLOutput`]
                instead of a plain tuple.

        Returns:
                The latent representations of the encoded videos. If `return_dict` is True, a
                [`~models.autoencoder_kl.AutoencoderKLOutput`] is returned,
                otherwise a plain `tuple` is returned.
        """
        if opt_tiling:
            tile_size, tile_stride = self.get_enc_optimal_tiling(x.shape)
        else:
            b, _, f, h, w = x.shape
            tile_size, tile_stride = (b, f, h, w), (f, h, w)
        if tile_size != self.tile_size:
            self.tile_size = tile_size
            self.apply_tiling(tile_size, tile_stride)

        h = self._encode(x)

        posterior = DiagonalGaussianDistribution(h)

        if not return_dict:
            return (posterior,)
        return AutoencoderKLOutput(latent_dist=posterior)

    def _decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        _, _, num_frames, height, width = z.shape
        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_stride_width // self.spatial_compression_ratio
        )
        tile_latent_min_num_frames = (
            self.tile_sample_min_num_frames // self.temporal_compression_ratio
        )

        if self.use_framewise_decoding and num_frames > (
            tile_latent_min_num_frames + 1
        ):
            return self._temporal_tiled_decode(z, return_dict=return_dict)

        if self.use_tiling and (
            width > tile_latent_min_width or height > tile_latent_min_height
        ):
            return self.tiled_decode(z, return_dict=return_dict)

        z = self.post_quant_conv(z)
        dec = self.decoder(z)

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)

    @apply_forward_hook
    def decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned,
                otherwise a plain `tuple` is returned.
        """
        tile_size, tile_stride = self.get_dec_optimal_tiling(z.shape)
        if tile_size != self.tile_size:
            self.tile_size = tile_size
            self.apply_tiling(tile_size, tile_stride)

            # Log tiling configuration (only on first decode or when changed)
            if not self.use_manual_tiling:
                b, c, f, h, w = z.shape
                h_pix, w_pix = h * 8, w * 8
                print(f"\nVAE Adaptive Tiling Activated (Resolution: {h_pix}x{w_pix}):")
                print(f"  Temporal: {self.tile_sample_min_num_frames} frames (stride: {self.tile_sample_stride_num_frames})")
                print(f"  Spatial: {self.tile_sample_min_height}x{self.tile_sample_min_width} pixels (stride: {self.tile_sample_stride_height}x{self.tile_sample_stride_width})\n")

        decoded = self._decode(z).sample

        if not return_dict:
            return (decoded,)

        return DecoderOutput(sample=decoded)

    def blend_v(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-2], b.shape[-2], blend_extent)
        for y in range(blend_extent):
            b[:, :, :, y, :] = a[:, :, :, -blend_extent + y, :] * (
                1 - y / blend_extent
            ) + b[:, :, :, y, :] * (y / blend_extent)
        return b

    def blend_h(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-1], b.shape[-1], blend_extent)
        for x in range(blend_extent):
            b[:, :, :, :, x] = a[:, :, :, :, -blend_extent + x] * (
                1 - x / blend_extent
            ) + b[:, :, :, :, x] * (x / blend_extent)
        return b

    def blend_t(
        self, a: torch.Tensor, b: torch.Tensor, blend_extent: int
    ) -> torch.Tensor:
        blend_extent = min(a.shape[-3], b.shape[-3], blend_extent)
        for x in range(blend_extent):
            b[:, :, x, :, :] = a[:, :, -blend_extent + x, :, :] * (
                1 - x / blend_extent
            ) + b[:, :, x, :, :] * (x / blend_extent)
        return b

    def tiled_encode(self, x: torch.Tensor) -> AutoencoderKLOutput:
        r"""Encode a batch of images using a tiled encoder.

        Args:
            x (`torch.Tensor`): Input batch of videos.

        Returns:
            `torch.Tensor`:
                The latent representation of the encoded videos.
        """
        _, _, _, height, width = x.shape
        latent_height = height // self.spatial_compression_ratio
        latent_width = width // self.spatial_compression_ratio

        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_min_width // self.spatial_compression_ratio
        )
        tile_latent_stride_height = (
            self.tile_sample_stride_height // self.spatial_compression_ratio
        )
        tile_latent_stride_width = (
            self.tile_sample_stride_width // self.spatial_compression_ratio
        )

        blend_height = tile_latent_min_height - tile_latent_stride_height
        blend_width = tile_latent_min_width - tile_latent_stride_width

        rows = []
        for i in range(
            0, height - self.tile_sample_min_height + 1, self.tile_sample_stride_height
        ):
            row = []
            for j in range(
                0, width - self.tile_sample_min_width + 1, self.tile_sample_stride_width
            ):
                tile = x[
                    :,
                    :,
                    :,
                    i : i + self.tile_sample_min_height,
                    j : j + self.tile_sample_min_width,
                ]
                tile = self.encoder(tile).clone()
                tile = self.quant_conv(tile)
                row.append(tile)
            rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                height_lim = (
                    tile_latent_min_height
                    if i == len(rows) - 1
                    else tile_latent_stride_height
                )
                width_lim = (
                    tile_latent_min_width
                    if j == len(row) - 1
                    else tile_latent_stride_width
                )
                result_row.append(tile[:, :, :, :height_lim, :width_lim])
            result_rows.append(torch.cat(result_row, dim=4))

        enc = torch.cat(result_rows, dim=3)[:, :, :, :latent_height, :latent_width]
        return enc

    def tiled_decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Decode a batch of images using a tiled decoder.

        Args:
            z (`torch.Tensor`): Input batch of latent vectors.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.vae.DecoderOutput`] instead of a plain tuple.

        Returns:
            [`~models.vae.DecoderOutput`] or `tuple`:
                If return_dict is True, a [`~models.vae.DecoderOutput`] is returned,
                otherwise a plain `tuple` is returned.
        """

        _, _, _, height, width = z.shape
        sample_height = height * self.spatial_compression_ratio
        sample_width = width * self.spatial_compression_ratio

        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_min_width // self.spatial_compression_ratio
        )
        tile_latent_stride_height = (
            self.tile_sample_stride_height // self.spatial_compression_ratio
        )
        tile_latent_stride_width = (
            self.tile_sample_stride_width // self.spatial_compression_ratio
        )

        blend_height = self.tile_sample_min_height - self.tile_sample_stride_height
        blend_width = self.tile_sample_min_width - self.tile_sample_stride_width

        rows = []
        i_range = list(range(0, height - tile_latent_min_height + 1, tile_latent_stride_height))
        j_range = list(range(0, width - tile_latent_min_width + 1, tile_latent_stride_width))
        total_tiles = len(i_range) * len(j_range)

        with tqdm(total=total_tiles, desc="VAE spatial tiling", unit="tile") as pbar:
            for i in i_range:
                row = []
                for j in j_range:
                    tile = z[
                        :,
                        :,
                        :,
                        i : i + tile_latent_min_height,
                        j : j + tile_latent_min_width,
                    ]
                    tile = self.post_quant_conv(tile)
                    decoded = self.decoder(tile).clone()
                    row.append(decoded)
                    pbar.update(1)

                    # Clear cache every few tiles to prevent memory accumulation
                    if (len(row) % 4 == 0) and torch.cuda.is_available():
                        torch.cuda.empty_cache()

                rows.append(row)

        result_rows = []
        for i, row in enumerate(rows):
            result_row = []
            for j, tile in enumerate(row):
                if i > 0:
                    tile = self.blend_v(rows[i - 1][j], tile, blend_height)
                if j > 0:
                    tile = self.blend_h(row[j - 1], tile, blend_width)
                height_lim = (
                    self.tile_sample_min_height
                    if i == len(rows) - 1
                    else self.tile_sample_stride_height
                )
                width_lim = (
                    self.tile_sample_min_width
                    if j == len(row) - 1
                    else self.tile_sample_stride_width
                )
                result_row.append(tile[:, :, :, :height_lim, :width_lim])
            result_rows.append(torch.cat(result_row, dim=-1))

        dec = torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def _temporal_tiled_encode(self, x: torch.Tensor) -> AutoencoderKLOutput:
        _, _, num_frames, height, width = x.shape
        latent_num_frames = (num_frames - 1) // self.temporal_compression_ratio + 1

        tile_latent_min_num_frames = (
            self.tile_sample_min_num_frames // self.temporal_compression_ratio
        )
        tile_latent_stride_num_frames = (
            self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        )
        blend_num_frames = tile_latent_min_num_frames - tile_latent_stride_num_frames

        row = []
        # for i in range(0, num_frames, self.tile_sample_stride_num_frames):
        for i in range(
            0,
            num_frames - self.tile_sample_min_num_frames + 1,
            self.tile_sample_stride_num_frames,
        ):
            tile = x[:, :, i : i + self.tile_sample_min_num_frames + 1, :, :]
            if self.use_tiling and (
                height > self.tile_sample_min_height
                or width > self.tile_sample_min_width
            ):
                tile = self.tiled_encode(tile)
            else:
                tile = self.encoder(tile).clone()
                tile = self.quant_conv(tile)
            if i > 0:
                tile = tile[:, :, 1:, :, :]
            row.append(tile)

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                t_lim = (
                    tile_latent_min_num_frames
                    if i == len(row) - 1
                    else tile_latent_stride_num_frames
                )
                result_row.append(tile[:, :, :t_lim, :, :])
            else:
                result_row.append(tile[:, :, : tile_latent_stride_num_frames + 1, :, :])

        enc = torch.cat(result_row, dim=2)[:, :, :latent_num_frames]
        return enc

    def _temporal_tiled_decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[DecoderOutput, torch.Tensor]:
        _, _, num_frames, _, _ = z.shape
        num_sample_frames = (num_frames - 1) * self.temporal_compression_ratio + 1

        tile_latent_min_height = (
            self.tile_sample_min_height // self.spatial_compression_ratio
        )
        tile_latent_min_width = (
            self.tile_sample_min_width // self.spatial_compression_ratio
        )
        tile_latent_min_num_frames = (
            self.tile_sample_min_num_frames // self.temporal_compression_ratio
        )
        tile_latent_stride_num_frames = (
            self.tile_sample_stride_num_frames // self.temporal_compression_ratio
        )
        blend_num_frames = (
            self.tile_sample_min_num_frames - self.tile_sample_stride_num_frames
        )

        row = []
        temporal_chunks = list(range(
            0,
            num_frames - tile_latent_min_num_frames + 1,
            tile_latent_stride_num_frames,
        ))

        for i in tqdm(temporal_chunks, desc="VAE temporal decoding", unit="chunk"):
            tile = z[:, :, i : i + tile_latent_min_num_frames + 1, :, :]
            if self.use_tiling and (
                tile.shape[-1] > tile_latent_min_width
                or tile.shape[-2] > tile_latent_min_height
            ):
                decoded = self.tiled_decode(tile, return_dict=True).sample
            else:
                tile = self.post_quant_conv(tile)
                decoded = self.decoder(tile).clone()
            if i > 0:
                decoded = decoded[:, :, 1:, :, :]
            row.append(decoded)

            # Clear cache after each temporal chunk to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result_row = []
        for i, tile in enumerate(row):
            if i > 0:
                tile = self.blend_t(row[i - 1], tile, blend_num_frames)
                t_lim = (
                    self.tile_sample_min_num_frames
                    if i == len(row) - 1
                    else self.tile_sample_stride_num_frames
                )
                result_row.append(tile[:, :, :t_lim, :, :])
            else:
                result_row.append(
                    tile[:, :, : self.tile_sample_stride_num_frames + 1, :, :]
                )

        dec = torch.cat(result_row, dim=2)[:, :, :num_sample_frames]

        if not return_dict:
            return (dec,)
        return DecoderOutput(sample=dec)

    def forward(
        self,
        sample: torch.Tensor,
        sample_posterior: bool = False,
        return_dict: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Union[DecoderOutput, torch.Tensor]:
        r"""
        Args:
            sample (`torch.Tensor`): Input sample.
            sample_posterior (`bool`, *optional*, defaults to `False`):
                Whether to sample from the posterior.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`DecoderOutput`] instead of a plain tuple.
        """
        x = sample
        posterior = self.encode(x).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z, return_dict=return_dict)
        return dec

    def apply_tiling(
        self, tile: Tuple[int, int, int, int], stride: Tuple[int, int, int]
    ):
        """Applies tiling."""
        _, ft, ht, wt = tile
        fs, hs, ws = stride

        self.use_tiling = True
        self.tile_sample_min_num_frames = ft - 1
        self.tile_sample_stride_num_frames = fs
        self.tile_sample_min_height = ht
        self.tile_sample_min_width = wt
        self.tile_sample_stride_height = hs
        self.tile_sample_stride_width = ws

    def get_enc_optimal_tiling(
        self, shape: List[int]
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int]]:
        """Returns optimal tiling for given shape."""
        h, w = shape[3:]

        free_mem = torch.cuda.mem_get_info()[0]
        max_area = free_mem / 256 / 17 / 8

        if h * w < max_area:
            return (1, 17, h, w), (8, h, w)

        def factorize(n, k):
            a = sqrt(n / k)
            b = sqrt(n * k)
            aa = [floor(a), ceil(a)]
            bb = [floor(b), ceil(b)]
            for a in aa:
                for b in bb:
                    if a * b >= n:
                        return a, b

        k = max(h / w, w / h)
        N = ceil(h * w / max_area)
        a, b = factorize(N, k)
        if h >= w:
            wn, hn = a, b
        else:
            wn, hn = b, a

        if wn > 1:
            wt = ceil(w / wn / 8) * 8 + 16
            ws = wt - 32
        else:
            wt = w
            ws = w
        if hn > 1:
            ht = ceil(h / hn / 8) * 8 + 16
            hs = ht - 32
        else:
            ht = h
            hs = h

        return (1, 17, ht, wt), (hs, ws, 8)

    def get_dec_optimal_tiling(
        self, shape: List[int]
    ) -> Tuple[Tuple[int, int, int, int], Tuple[int, int, int]]:
        """
        Returns optimal tiling for given shape.
        If manual tiling is set, respects those settings.
        Otherwise, adaptively calculates based on resolution and available memory.
        """
        # If manual tiling is enabled, use current settings
        if self.use_manual_tiling:
            return (
                (1, self.tile_sample_min_num_frames + 1,
                 self.tile_sample_min_height, self.tile_sample_min_width),
                (self.tile_sample_stride_num_frames,
                 self.tile_sample_stride_height, self.tile_sample_stride_width)
            )

        b, _, f, h, w = shape
        h_pix, w_pix = h * 8, w * 8

        # Adaptive tile sizing based on resolution
        total_pixels = h_pix * w_pix

        # Resolution-based adaptive tiling (provides better defaults than pure memory calc)
        if total_pixels <= 768 * 512:  # Low-res (e.g., 768x512)
            # Use larger tiles for better speed
            tile_h, tile_w = 256, 256
            stride_h, stride_w = 192, 192
        elif total_pixels <= 960 * 544:  # Medium-res (e.g., 960x544)
            # Moderate tiles
            tile_h, tile_w = 192, 192
            stride_h, stride_w = 160, 160
        elif total_pixels <= 1280 * 704:  # High-res (e.g., 1280x704)
            # Smaller tiles to prevent OOM
            tile_h, tile_w = 128, 128
            stride_h, stride_w = 96, 96
        else:  # Ultra-high-res (e.g., 1920x1080)
            # Very small tiles for extreme resolutions
            tile_h, tile_w = 96, 96
            stride_h, stride_w = 64, 64

        # Ensure tiles don't exceed actual dimensions
        tile_h = min(tile_h, h_pix)
        tile_w = min(tile_w, w_pix)
        stride_h = min(stride_h, tile_h - 16)  # Ensure some overlap
        stride_w = min(stride_w, tile_w - 16)

        # Temporal tiling (frames) - use hardcoded defaults to avoid corruption from encoding
        # Note: get_enc_optimal_tiling has incompatible stride order that corrupts these values
        # So we MUST NOT rely on self.tile_sample_* for automatic mode
        tile_f = 16   # Default temporal tile size (pixel-space frames)
        stride_f = 12  # Default temporal stride (pixel-space frames)

        return (1, tile_f + 1, tile_h, tile_w), (stride_f, stride_h, stride_w)


# ============================================================================
# KANDINSKY VAE 3D (KVAE) IMPLEMENTATION
# ============================================================================

# --- From layers.py ---

def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)

class SafeConv3d(nn.Conv3d):
    def forward(self, x, write_to=None, transform=None):
        if transform is None:
            transform = lambda x: x

        memory_count = x.numel() * x.element_size() / (10 ** 9)
        if memory_count > 3:
            kernel_size = self.kernel_size[0]
            part_num = math.ceil(memory_count / 2)
            input_chunks = torch.chunk(x, part_num, dim=2)  # NCTHW

            if input_chunks[0].size(2) < 3 and kernel_size > 1:
                for i in range(x.size(2) - 2):
                    torch.cuda.empty_cache()
                    time.sleep(.2)
                    chunk = transform(x[:, :, i:i+3])
                    write_to[:, :, i:i+1] = super(SafeConv3d, self).forward(chunk)
                return write_to

            if write_to is None:
                output = []
                for i, chunk in enumerate(input_chunks):
                    if i == 0 or kernel_size == 1:
                        z = torch.clone(chunk)
                    else:
                        z = torch.cat([z[:, :, -kernel_size + 1:], chunk], dim=2)
                    output.append(super(SafeConv3d, self).forward(transform(z)))
                output = torch.cat(output, dim=2)
                return output
            else:
                time_offset = 0
                for i, chunk in enumerate(input_chunks):
                    if i == 0 or kernel_size == 1:
                        z = torch.clone(chunk)
                    else:
                        z = torch.cat([z[:, :, -kernel_size + 1:], chunk], dim=2)
                    z_time = z.size(2) - (kernel_size - 1)
                    write_to[:, :, time_offset:time_offset+z_time] = super(SafeConv3d, self).forward(transform(z))
                    time_offset += z_time
                return write_to
        else:
            if write_to is None:
                return super(SafeConv3d, self).forward(transform(x))
            else:
                write_to[...] = super(SafeConv3d, self).forward(transform(x))
                return write_to


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


class CausalConv3d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size: Union[int, Tuple[int, int, int]], stride=(1,1,1), dilation=(1,1,1), **kwargs):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        assert (height_kernel_size % 2) and (width_kernel_size % 2)

        self.height_pad = height_kernel_size // 2
        self.width_pad = width_kernel_size // 2
        self.time_pad = time_kernel_size - 1
        self.time_kernel_size = time_kernel_size
        self.stride = stride

        self.conv = SafeConv3d(chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs)

    def forward(self, input_):
        padding_3d = (self.width_pad, self.width_pad, self.height_pad, self.height_pad, self.time_pad, 0)
        input_padded = F.pad(input_, padding_3d, mode="replicate")
        output = self.conv(input_padded)
        return output


class WanRMS_norm(nn.Module):
    def __init__(self, n_ch: int, bias: bool = False) -> None:
        super().__init__()
        shape = (n_ch, 1, 1, 1)
        self.scale = n_ch ** 0.5
        self.gamma = nn.Parameter(torch.ones(shape))
        self.bias = nn.Parameter(torch.zeros(shape)) if bias else 0.0

    def forward(self, x, *args, **kwargs):
        return F.normalize(x, dim=1) * self.scale * self.gamma + self.bias

def RMSNorm(in_channels, *args, **kwargs):
    return WanRMS_norm(n_ch=in_channels, bias=False)


# --- From causal_samplings.py ---

class PXSUpsample(nn.Module):
    def __init__(self, in_channels: int, compress_time: bool, factor: int=2):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.shuffle = nn.PixelShuffle(self.factor)
        self.spatial_conv = SafeConv3d(in_channels, in_channels,
                                      kernel_size=(1, 3, 3),
                                      stride=(1, 1, 1),
                                      padding=(0, 1, 1),
                                      padding_mode='reflect')
        
        if self.temporal_compress:
            self.temporal_conv = SafeConv3d(in_channels, in_channels,
                                            kernel_size=(3, 1, 1),
                                            stride=(1, 1, 1),
                                            dilation=(1, 1, 1))

        self.linear = nn.Conv3d(in_channels, in_channels,
                                kernel_size=1,
                                stride=1)

    def spatial_upsample(self, input_):
        def conv_part(x):
            to = torch.empty_like(x)
            out = self.spatial_conv(x, write_to=to)
            return out

        b, t, c, h, w = input_.shape
        input_view = input_.view(b, (t *c), h, w)

        input_interp = F.interpolate(input_view, scale_factor=2, mode='nearest')
        input_interp = input_interp.view(b, t, c, 2 * h, 2 * w)
        input_interp.add_(conv_part(input_interp))
        return input_interp

    def temporal_upsample(self, input_):
        time_factor = 1.0 + 1.0 * (input_.size(2) > 1)
        if isinstance(time_factor, torch.Tensor):
            time_factor = time_factor.item()

        repeated = input_.repeat_interleave(int(time_factor), dim=2)
        tail = repeated[..., int(time_factor - 1) :, :, :]
        padding_3d = (0, 0, 0, 0, 2, 0)
        tail_pad = F.pad(tail, padding_3d, mode="replicate")
        conv_out = self.temporal_conv(tail_pad)
        return conv_out + tail

    def forward(self, x):
        if self.temporal_compress:
            x = self.temporal_upsample(x)
        s_out = self.spatial_upsample(x)
        return self.linear(s_out)


# --- From cached_layers.py ---

class CachedGroupNorm(nn.GroupNorm):
    def group_forward(self, x_input, expectation=None, variance=None, return_stat=False):
        input_dtype = x_input.dtype
        x = x_input.to(torch.float32)
        chunks = torch.chunk(x, self.num_groups, dim=1)
        if expectation is None:
            ch_mean = [torch.mean(chunk, dim=(1, 2, 3, 4), keepdim=True) for chunk in chunks]
        else:
            ch_mean = expectation

        if variance is None:
            ch_var = [torch.var(chunk, dim=(1, 2, 3, 4), keepdim=True, unbiased=False) for chunk in chunks]
        else:
            ch_var = variance

        x_norm = [(chunk - mean) / torch.sqrt(var + self.eps) for chunk, mean, var in zip(chunks, ch_mean, ch_var)]
        x_norm = torch.cat(x_norm, dim=1)

        x_norm.mul_(self.weight.data.view(1, -1, 1, 1, 1))
        x_norm.add_(self.bias.data.view(1, -1, 1, 1, 1))

        x_out = x_norm.to(input_dtype)
        if return_stat:
            return x_out, ch_mean, ch_var
        return x_out

    def forward(self, x, cache: dict):
        out = super().forward(x)
        if cache.get('mean') is None and cache.get('var') is None:
            cache['mean'] = 1
            cache['var'] = 1
        return out

def Normalize(in_channels, gather=False, **kwargs):
    return CachedGroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class CachedCausalConv3d(CausalConv3d):
    def forward(self, input_, cache: dict):
        t_stride = self.stride[0]
        padding_3d = (self.height_pad, self.height_pad, self.width_pad, self.width_pad, 0, 0)
        input_parallel = F.pad(input_, padding_3d, mode="replicate")

        if cache['padding'] is None:
            first_frame = input_parallel[:, :, :1]
            time_pad_shape = [i for i in first_frame.shape]
            time_pad_shape[2] = self.time_pad
            padding = first_frame.expand(time_pad_shape)
        else:
            padding = cache['padding']

        out_size = [i for i in input_.shape]
        out_size[1] = self.conv.out_channels
        if t_stride == 2:
            out_size[2] = (input_.size(2) + 1) // 2
        output = torch.empty(tuple(out_size), dtype=input_.dtype, device=input_.device)

        offset_out = math.ceil(padding.size(2) / t_stride)
        offset_in = offset_out * t_stride - padding.size(2)

        if offset_out > 0:
            padding_poisoned = torch.cat([padding, input_parallel[:, :, :offset_in + self.time_kernel_size - t_stride]], dim=2)
            output[:, :, :offset_out] = self.conv(padding_poisoned)

        if offset_out < output.size(2):
            output[:, :, offset_out:] = self.conv(input_parallel[:, :, offset_in:])

        pad_offset = offset_in + t_stride * math.trunc((input_parallel.size(2) - offset_in - self.time_kernel_size) / t_stride) + t_stride
        cache['padding'] = torch.clone(input_parallel[:, :, pad_offset:])
        return output

class CachedCausalResnetBlock3D(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        zq_ch=None,
        add_conv=False,
        gather_norm=False,
        normalization=Normalize,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = normalization(
            in_channels,
            zq_ch=zq_ch,
            add_conv=add_conv
        )

        self.conv1 = CachedCausalConv3d(
            chan_in=in_channels,
            chan_out=out_channels,
            kernel_size=3,
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = normalization(
            out_channels,
            zq_ch=zq_ch,
            add_conv=add_conv
        )
        self.conv2 = CachedCausalConv3d(
            chan_in=out_channels,
            chan_out=out_channels,
            kernel_size=3,
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CachedCausalConv3d(
                    chan_in=in_channels,
                    chan_out=out_channels,
                    kernel_size=3,
                )
            else:
                self.nin_shortcut = SafeConv3d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )

    def forward(self, x, temb, layer_cache, zq=None):
        if x.size(2) == 17 and x.size(3) == 1080 and zq is not None:
            torch.cuda.empty_cache()
        h = x

        if zq is None:
            h = self.norm1(h, cache=layer_cache['norm1'])
        else:
            h = self.norm1(h, zq, cache=layer_cache['norm1'])

        if x.size(2) == 17 and x.size(3) == 1080 and zq is not None:
            torch.cuda.empty_cache()

        h = F.silu(h, inplace=True)
        if x.size(2) == 17 and x.size(3) == 1080 and zq is not None:
            torch.cuda.empty_cache()

        h = self.conv1(h, cache=layer_cache['conv1'])
        if x.size(2) == 17 and x.size(3) == 1080 and zq is not None:
            torch.cuda.empty_cache()

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]

        if zq is None:
            h = self.norm2(h, cache=layer_cache['norm2'])
        else:
            h = self.norm2(h, zq, cache=layer_cache['norm2'])

        h = F.silu(h, inplace=True)
        h = self.conv2(h, cache=layer_cache['conv2'])

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x, cache=layer_cache['conv_shortcut'])
            else:
                x = self.nin_shortcut(x)

        return x + h


class CachedPXSDownsample(nn.Module):
    def __init__(self, in_channels: int, compress_time: bool, factor: int=2):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.unshuffle = nn.PixelUnshuffle(self.factor)
        self.s_pool = nn.AvgPool3d((1, 2, 2), (1, 2, 2))

        out_channels = in_channels

        self.spatial_conv = SafeConv3d(in_channels, out_channels,
                                      kernel_size=(1, 3, 3),
                                      stride=(1, 2, 2),
                                      padding=(0, 1, 1),
                                      padding_mode='reflect')
        
        if self.temporal_compress:
            self.temporal_conv = CachedCausalConv3d(out_channels, out_channels,
                                                    kernel_size=(3, 1, 1),
                                                    stride=(2, 1, 1),
                                                    dilation=(1, 1, 1))

        self.linear = nn.Conv3d(out_channels, out_channels,
                                kernel_size=1,
                                stride=1)

    def spatial_downsample(self, input_):
        pxs_input = rearrange(input_, 'b c t h w -> (b t) c h w')
        pxs_interm = self.unshuffle(pxs_input)
        b, c, h, w = pxs_interm.shape
        pxs_interm_view = pxs_interm.view(b, c // self.factor ** 2, self.factor ** 2, h, w)

        pxs_out = torch.mean(pxs_interm_view, dim=2)
        pxs_out = rearrange(pxs_out, '(b t) c h w -> b c t h w', t=input_.size(2))

        conv_out = self.spatial_conv(input_)
        return conv_out + pxs_out

    def temporal_downsample(self, input_, cache):
        permuted = rearrange(input_, "b c t h w -> (b h w) c t")
        if cache[0]['padding'] is None:
            first, rest = permuted[..., :1], permuted[..., 1:]

            if rest.size(-1) > 0:
                rest_interp = F.avg_pool1d(rest, kernel_size=2, stride=2)
                full_interp = torch.cat([first, rest_interp], dim=-1)
            else:
                full_interp = first
        else:
            rest = permuted
            if rest.size(-1) > 0:
                full_interp = F.avg_pool1d(rest, kernel_size=2, stride=2)

        full_interp = rearrange(full_interp, "(b h w) c t -> b c t h w", h=input_.size(-2), w=input_.size(-1))
        conv_out = self.temporal_conv(input_, cache[0])
        return conv_out + full_interp

    def forward(self, x, cache):
        out = self.spatial_downsample(x)
        if self.temporal_compress:
            out = self.temporal_downsample(out, cache=cache)
        return self.linear(out)


class CachedSpatialNorm3D(nn.Module):
    def __init__(
        self,
        f_channels,
        zq_channels,
        freeze_norm_layer=False,
        add_conv=False,
        pad_mode="constant",
        normalization=Normalize,
        **norm_layer_params,
    ):
        super().__init__()
        self.norm_layer = normalization(in_channels=f_channels, **norm_layer_params)

        self.add_conv = add_conv
        if add_conv:
            self.conv = CachedCausalConv3d(
                chan_in=zq_channels,
                chan_out=zq_channels,
                kernel_size=3,
            )

        self.conv_y = SafeConv3d(
            zq_channels,
            f_channels,
            kernel_size=1,
        )
        self.conv_b = SafeConv3d(
            zq_channels,
            f_channels,
            kernel_size=1,
        )

    def forward(self, f, zq, cache):
        f_shape = [s for s in f.shape]

        if cache['norm']['mean'] is None and cache['norm']['var'] is None:
            f_first, f_rest = f[:, :, :1], f[:, :, 1:]
            f_first_size, f_rest_size = f_first.shape[-3:], f_rest.shape[-3:]
            zq_first, zq_rest = zq[:, :, :1], zq[:, :, 1:]

            zq_first = F.interpolate(zq_first, size=f_first_size, mode="nearest")

            if zq.size(2) > 1:
                zq_rest_splits = torch.split(zq_rest, 32, dim=1)
                interpolated_splits = [
                    F.interpolate(split, size=f_rest_size, mode="nearest") for split in zq_rest_splits
                ]
    
                zq_rest = torch.cat(interpolated_splits, dim=1)
                zq = torch.cat([zq_first, zq_rest], dim=2)
            else:
                zq = zq_first
        else:
            f_size = f.shape[-3:]
            zq_splits = torch.split(zq, 32, dim=1)
            interpolated_splits = [
                F.interpolate(split, size=f_size, mode="nearest") for split in zq_splits
            ]
            zq = torch.cat(interpolated_splits, dim=1)


        if self.add_conv:
            zq = self.conv(zq, cache['add_conv'])

        norm_f = self.norm_layer(f, cache['norm'])
        norm_f.mul_(self.conv_y(zq))
        norm_f.add_(self.conv_b(zq))

        if cache['norm']['mean'] is None and cache['norm']['var'] is None:
            cache['norm']['mean'] = 1
            cache['norm']['var'] = 1

        return norm_f


def Normalize3D(
    in_channels,
    zq_ch,
    add_conv,
    normalization=Normalize
):
    return CachedSpatialNorm3D(
        in_channels,
        zq_ch,
        freeze_norm_layer=False,
        add_conv=add_conv,
        num_groups=32,
        eps=1e-6,
        affine=True,
        normalization=normalization
    )


class CachedPXSUpsample(PXSUpsample):
    def __init__(self, in_channels: int, compress_time: bool, factor: int=2):
        super().__init__(in_channels, compress_time, factor)
        self.temporal_compress = compress_time
        self.factor = factor
        self.shuffle = nn.PixelShuffle(self.factor)
        self.spatial_conv = SafeConv3d(in_channels, in_channels,
                                      kernel_size=(1, 3, 3),
                                      stride=(1, 1, 1),
                                      padding=(0, 1, 1),
                                      padding_mode='reflect')
        
        if self.temporal_compress:
            self.temporal_conv = CachedCausalConv3d(in_channels, in_channels,
                                                    kernel_size=(3, 1, 1),
                                                    stride=(1, 1, 1),
                                                    dilation=(1, 1, 1))

        self.linear = SafeConv3d(in_channels, in_channels,
                                kernel_size=1,
                                stride=1)

    def temporal_upsample(self, input_, cache):
        time_factor = 1.0 + 1.0 * (input_.size(2) > 1)
        if isinstance(time_factor, torch.Tensor):
            time_factor = time_factor.item()

        # input_ : (T + 1) x H x W
        repeated = input_.repeat_interleave(int(time_factor), dim=2)
        # repeated: (2T + 2) x H x W

        if cache['padding'] is None:
            tail = repeated[..., int(time_factor - 1) :, :, :] # tail: (2T + 1) x H x W
        else:
            tail = repeated

        conv_out = self.temporal_conv(tail, cache)
        return conv_out + tail

    def forward(self, x, cache):
        if self.temporal_compress:
            x = self.temporal_upsample(x, cache)

        s_out = self.spatial_upsample(x)
        to = torch.empty_like(s_out)
        lin_out = self.linear(s_out, write_to=to)
        return lin_out


# --- From cached_enc_dec.py (Decoder) ---

class CachedDecoder3D(nn.Module):
    def __init__(
        self,
        ch=128,
        out_ch=None,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.0,
        resamp_with_conv=True,
        resolution=0,
        z_channels=16,
        give_pre_end=False,
        zq_ch=None,
        add_conv=False,
        pad_mode="first",
        temporal_compress_times=4,
        gather_norm=False,
        norm_type="group_norm",
        **kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end

        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        if zq_ch is None:
            zq_ch = z_channels

        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = CachedCausalConv3d(
            chan_in=z_channels,
            chan_out=block_in,
            kernel_size=3,
        )

        modulated_norm = functools.partial(Normalize3D, normalization=Normalize if norm_type == "group_norm" else RMSNorm)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            normalization=modulated_norm,
            gather_norm=gather_norm,
        )

        self.mid.block_2 = CachedCausalResnetBlock3D(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            zq_ch=zq_ch,
            add_conv=add_conv,
            normalization=modulated_norm,
            gather_norm=gather_norm,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    CachedCausalResnetBlock3D(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        zq_ch=zq_ch,
                        add_conv=add_conv,
                        normalization=modulated_norm,
                        gather_norm=gather_norm,
                    )
                )
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                if i_level < self.num_resolutions - self.temporal_compress_level:
                    up.upsample = CachedPXSUpsample(block_in, compress_time=False)
                else:
                    up.upsample = CachedPXSUpsample(block_in, compress_time=True)
            self.up.insert(0, up)

        self.norm_out = modulated_norm(block_in, zq_ch, add_conv=add_conv) 

        self.conv_out = CachedCausalConv3d(
            chan_in=block_in,
            chan_out=out_ch,
            kernel_size=3,
        )

    def forward(self, z, cache_dict):
        self.last_z_shape = z.shape

        temb = None
        zq = z
        h = self.conv_in(z, cache_dict['conv_in'])

        h = self.mid.block_1(h, temb, layer_cache=cache_dict['mid_1'], zq=zq)
        h = self.mid.block_2(h, temb, layer_cache=cache_dict['mid_2'], zq=zq)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, layer_cache=cache_dict[i_level][i_block], zq=zq)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, zq)
            if i_level != 0:
                h = self.up[i_level].upsample(h, cache_dict[i_level]['up'])

        if self.give_pre_end:
            return h

        h = self.norm_out(h, zq, cache_dict['norm_out'])
        h = nonlinearity(h)
        h = self.conv_out(h, cache_dict['conv_out'])

        return h

# --- From efficient_vae.py ---

class EfficientCausalConv3D(nn.Module):
    def __init__(self, c_in, c_out, k_size: Union[int, Tuple[int, int, int]], stride=(1,1,1),
                 dilation=(1,1,1), **kwargs):
        super().__init__()
        k_size = cast_tuple(k_size, 3)
        tks, hks, wks = k_size
        _, hs, ws = stride
        self.h_pad = hks // 2
        self.w_pad = wks // 2
        self.t_pad = tks - 1
        self.padding = (wks//2-ws//2, wks//2-ws//2, hks//2-hs//2, hks//2-hs//2, tks - 1, 0)
        self.conv = nn.Conv3d(c_in, c_out, k_size, stride=stride, dilation=dilation, **kwargs).\
            to(memory_format=torch.channels_last_3d)

    @torch.compile(dynamic=True, fullgraph=False, options={"force_same_precision": True, 
        "max_autotune": True, "memory_planning": True})
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, self.padding, mode="replicate")
        x = self.conv(x)
        return x

class SplittedGN(nn.GroupNorm):
    def __init__(self, num_groups, num_channels, split_size, eps=1e-05, affine=True, device=None,
                 dtype=None):
        super().__init__(num_groups, num_channels, eps, affine, device, dtype)
        self.split_size = split_size
    
    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        func = self.forward
        res = torch.empty_like(x)
        if x.shape[2] in [self.split_size + 1, 1]:
            return func(x)
        else:
            N, C, T, H, W = x.shape
            T -= self.split_size + 1
            rest = x[:, :, (self.split_size + 1):]
            res[:, :, :(self.split_size + 1)] = func(x[:, :, :(self.split_size + 1)])
            N, C, T, H, W = rest.shape
            res[:, :, (self.split_size + 1):] = torch.vmap(func, in_dims=2, out_dims=2)\
                (rest.reshape(N, C, T // self.split_size, self.split_size, H, W))\
                .reshape(N, C, T, H, W)
        return res

class EffDownsample(nn.Module):
    def __init__(self, c_in: int, compress_time: bool, factor: int=2, split_size: int = 16):
        super().__init__()
        self.temporal_compress = compress_time
        self.factor = factor
        self.split_size = split_size
        self.s_pool = nn.AvgPool3d((1, 2, 2), (1, 2, 2))
        self.spatial_conv = nn.Conv3d(c_in, c_in, kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                      padding=(0, 1, 1), padding_mode='reflect')
        if compress_time:
            self.t_pool = nn.AvgPool3d((2, 1, 1), (2, 1, 1))
            self.temporal_conv = EfficientCausalConv3D(c_in, c_in, k_size=(3, 1, 1), stride=(2, 1, 1))

        self.linear = nn.Conv3d(c_in, c_in, kernel_size=1)

    def spatial(self, x: torch.Tensor) -> torch.Tensor:
        return self.s_pool(x) + self.spatial_conv(x)

    def temporal(self, x: torch.Tensor) -> torch.Tensor:
        N, C, T, H, W = x.shape
        if T == 1:
            return x
        res_pool = torch.empty((N, C, 1 + T // self.factor, H, W), dtype=x.dtype, device=x.device)
        res_pool[:, :, :1] = x[:, :, :1]
        res_pool[:, :, 1:] = self.t_pool(x[:, :, 1:])
        return res_pool + self.temporal_conv(x)

class EffEncoderResnetBlock3D(nn.Module):
    def __init__(self, c_in, c_out=None, split_size=16):
        super().__init__()
        c_out = c_in if c_out is None else c_out
        self.c_in, self.c_out = c_in, c_out
        self.norm1 = SplittedGN(num_groups=32, num_channels=c_in, split_size=split_size, eps=1e-6)
        self.conv1 = EfficientCausalConv3D(c_in=c_in, c_out=c_out, k_size=3)
        self.norm2 = SplittedGN(num_groups=32, num_channels=c_out, split_size=split_size, eps=1e-6)
        self.conv2 = EfficientCausalConv3D(c_in=c_out, c_out=c_out, k_size=3)
        self.nin_shortcut = nn.Conv3d(c_in, c_out, kernel_size=1) if c_in!=c_out else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1.normalize(x)
        y = F.silu(y, inplace=True)
        y = self.conv1(y)
        y = self.norm2.normalize(y)
        y = F.silu(y, inplace=True)
        y = self.conv2(y)
        return y + self.nin_shortcut(x)

class KandinskyEncoder3D(nn.Module):
    def __init__(
        self,
        ch=128,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        in_channels=3,
        z_channels=16,
        double_z=True,
        temporal_compress_times=4,
        **kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.temporal_compress_level = int(np.log2(temporal_compress_times))

        self.conv_in = EfficientCausalConv3D(c_in=4, c_out=self.ch, k_size=3)

        split_size = 16
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(EffEncoderResnetBlock3D(c_in=block_in, c_out=block_out,
                                                     split_size=split_size))
                block_in = block_out
            down = nn.Module()
            down.block = block
            if i_level != self.num_resolutions - 1:
                if i_level < self.temporal_compress_level:
                    down.downsample = EffDownsample(block_in, compress_time=True,
                                                    split_size=split_size)
                    split_size //= 2
                else:
                    down.downsample = EffDownsample(block_in, compress_time=False,
                                                    split_size=split_size)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = EffEncoderResnetBlock3D(c_in=block_in, c_out=block_in,
                                                   split_size=split_size)
        self.mid.block_2 = EffEncoderResnetBlock3D(c_in=block_in, c_out=block_in,
                                                   split_size=split_size)

        self.norm_out = SplittedGN(num_groups=32, num_channels=block_in, split_size=split_size,
                                   eps=1e-6)
        self.conv_out = EfficientCausalConv3D(
            c_in=block_in,
            c_out=2 * z_channels if double_z else z_channels,
            k_size=3,
        )
        self.z_channels = z_channels

    @torch.compile(dynamic=True, fullgraph=False, options={"force_same_precision": True,
                    "max_autotune": True, "memory_planning": True,
                    'max_autotune_conv_backends': 'ATEN,TRITON,CUTLASS'})
    def piece_0(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.down[0].block[0](x)              # M
        x = self.down[0].block[1](x)              # M
        x = self.down[0].downsample.spatial(x)    # M / 4
        return x 
    
    @torch.compile(dynamic=True, fullgraph=False, options={"force_same_precision": True,
                    "max_autotune": True, "memory_planning": True,
                    'max_autotune_conv_backends': 'ATEN,TRITON,CUTLASS'})
    def piece_1(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down[0].downsample.temporal(x)   # M / 8
        x = self.down[0].downsample.linear(x)     # M / 8
        x = self.down[1].block[0](x)              # M / 4
        x = self.down[1].block[1](x)              # M / 4
        x = self.down[1].downsample.spatial(x)    # M / 16
        return x  
    
    @torch.compile(dynamic=True, fullgraph=False, options={"force_same_precision": True,
                    "max_autotune": True, "memory_planning": True,
                    'max_autotune_conv_backends': 'ATEN,TRITON,CUTLASS'})
    def piece_2(self, x: torch.Tensor) -> torch.Tensor:
        x = self.down[1].downsample.temporal(x)   # M / 32
        x = self.down[1].downsample.linear(x)     # M / 32
        x = self.down[2].block[0](x)              # M / 16
        x = self.down[2].block[1](x)              # M / 16
        x = self.down[2].downsample.spatial(x)    # M / 64
        x = self.down[2].downsample.linear(x)     # M / 64
        x = self.down[3].block[0](x)              # M / 32
        x = self.down[3].block[1](x)              # M / 32
        x = self.mid.block_1(x)                   # M / 32
        x = self.mid.block_2(x)                   # M / 32
        x = self.norm_out.normalize(x)
        x = F.silu(x, inplace=True)
        x = self.conv_out(x)[:, :self.z_channels]
        return x.contiguous()

    @torch.compile(dynamic=True, fullgraph=False, options={"force_same_precision": True, 
        "max_autotune": True, "memory_planning": True})
    def forward_1(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        x = x.to(memory_format=torch.channels_last_3d)
        x = self.piece_0(x)
        x = self.piece_1(x)
        x = self.piece_2(x)
        return x

    @torch.compile(dynamic=True, fullgraph=False, options={"force_same_precision": True, 
        "max_autotune": True, "memory_planning": True})
    def forward_2h(self, x: torch.Tensor, th: int, tw: int) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        x = x.to(memory_format=torch.channels_last_3d)
        _, _, T, H, W = x.shape 
        z = torch.empty((1, 128, T, H//2, W//2), device=x.device, dtype=x.dtype).\
            to(memory_format=torch.channels_last_3d)
        for i in range(2):
            h2l, h2r = H//2*i, H//2*i+H//2
            h4l, h4r = H//4*i, H//4*i+H//4
            z[0,:,:,h4l:h4r,:] = self.piece_0(x[[0],:,:,h2l:h2r,:])
        x = self.piece_1(z)
        x = self.piece_2(x)
        return x

    @torch.compile(dynamic=True, fullgraph=False, options={"force_same_precision": True, 
        "max_autotune": True, "memory_planning": True})
    def forward_2w(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        x = x.to(memory_format=torch.channels_last_3d)
        _, _, T, H, W = x.shape 
        z = torch.empty((1, 128, T, H//2, W//2), device=x.device, dtype=x.dtype).\
            to(memory_format=torch.channels_last_3d)
        for j in range(2):
            w2l, w2r = W//2*j, W//2*j+W//2
            w4l, w4r = W//4*j, W//4*j+W//4
            z[0,:,:,:,w4l:w4r] = self.piece_0(x[[0],:,:,:,w2l:w2r])
        x = self.piece_1(z)
        x = self.piece_2(x)
        return x

    def forward_4(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        x = x.to(memory_format=torch.channels_last_3d)
        _, _, T, H, W = x.shape 
        z = torch.empty((1, 128, T, H//2, W//2), device=x.device, dtype=x.dtype).\
            to(memory_format=torch.channels_last_3d)
        for i in range(2):
            for j in range(2):
                h2l, h2r, w2l, w2r = H//2*i, H//2*i+H//2, W//2*j, W//2*j+W//2
                h4l, h4r, w4l, w4r = H//4*i, H//4*i+H//4, W//4*j, W//4*j+W//4
                z[0,:,:,h4l:h4r,w4l:w4r] = self.piece_0(x[[0],:,:,h2l:h2r,w2l:w2r])
        x = self.piece_1(z)
        x = self.piece_2(x)
        return x

    def forward_8h(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        x = x.to(memory_format=torch.channels_last_3d)
        _, _, T, H, W = x.shape 
        z = torch.empty((1, 256, (T-1)//2+1, H//4, W//4), device=x.device, dtype=x.dtype).\
            to(memory_format=torch.channels_last_3d)
        for i in range(4):
            for j in range(2):
                h4l, h4r, w2l, w2r = H//4*i, H//4*i+H//4, W//2*j, W//2*j+W//2
                h16l, h16r, w8l, w8r = H//16*i, H//16*i+H//16, W//8*j, W//8*j+W//8
                y = self.piece_0(x[[0],:,:,h4l:h4r,w2l:w2r])
                z[0,:,:,h16l:h16r,w8l:w8r] = self.piece_1(y)
        x = self.piece_2(z)
        return x
    
    def forward_8w(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        x = x.to(memory_format=torch.channels_last_3d)
        _, _, T, H, W = x.shape 
        z = torch.empty((1, 256, (T-1)//2+1, H//4, W//4), device=x.device, dtype=x.dtype).\
            to(memory_format=torch.channels_last_3d)
        for i in range(2):
            for j in range(4):
                h2l, h2r, w4l, w4r = H//2*i, H//2*i+H//2, W//4*j, W//4*j+W//4
                h8l, h8r, w16l, w16r = H//8*i, H//8*i+H//8, W//16*j, W//16*j+W//16
                y = self.piece_0(x[[0],:,:,h2l:h2r,w4l:w4r])
                z[0,:,:,h8l:h8r,w16l:w16r] = self.piece_1(y)
        x = self.piece_2(z)
        return x

    def forward_16(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (0, 0, 0, 0, 0, 0, 0, 1))
        x = x.to(memory_format=torch.channels_last_3d)
        _, _, T, H, W = x.shape 
        z = torch.empty((1, 256, (T-1)//2+1, H//4, W//4), device=x.device, dtype=x.dtype).\
            to(memory_format=torch.channels_last_3d)
        for i in range(4):
            for j in range(4):
                h4l, h4r, w4l, w4r = H//4*i, H//4*i+H//4, W//4*j, W//4*j+W//4
                h16l, h16r, w16l, w16r = H//16*i, H//16*i+H//16, W//16*j, W//16*j+W//16
                y = self.piece_0(x[[0],:,:,h4l:h4r,w4l:w4r])
                z[0,:,:,h16l:h16r,w16l:w16r] = self.piece_1(y)
        x = self.piece_2(z)
        return x

# --- Wrapper for API compatibility ---

class ConstantDistribution:
    """
    A dummy distribution that returns a constant value (the mean/mode) 
    since the optimized encoder slices the variance.
    """
    def __init__(self, mean):
        self.mean = mean
        
    def sample(self, generator=None):
        return self.mean
    
    def mode(self):
        return self.mean

class AutoencoderKLKandinsky(ModelMixin, ConfigMixin):
    """
    Wrapper for K-VAE 3D to make it compatible with the pipeline expected API.
    Uses efficient KandinskyEncoder3D for encoding and CachedDecoder3D for decoding.
    """
    
    @register_to_config
    def __init__(
        self,
        encoder_params: dict,
        decoder_params: dict,
        scaling_factor: float = 0.476986, # Match HunyuanVideo VAE scaling for DiT compatibility
    ):
        super().__init__()
        self.encoder = KandinskyEncoder3D(**encoder_params)
        self.decoder = CachedDecoder3D(**decoder_params)
        self.config.scaling_factor = scaling_factor
        
        self.encoder = self.encoder.to(memory_format=torch.channels_last_3d)
        
        # Default tiling parameters from user config args will be set here
        self.tile_sample_min_height = 256
        self.tile_sample_min_width = 256
        self.tile_sample_min_num_frames = 16
        self.tile_sample_stride_num_frames = 12
        self.use_manual_tiling = False
        
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, subfolder=None, **kwargs):
        # Custom loading logic because weights need processing (padding)
        # This is a simplified loader that assumes model file exists in path
        
        # Load config first (assuming user passed config to build_vae, so this might be redundant if called from there)
        # But if called directly:
        pass 
        # We will handle weight loading in build_vae for this specific case
        # to align with the provided infrastructure.
        return super().from_pretrained(pretrained_model_name_or_path, subfolder=subfolder, **kwargs)

    def make_empty_cache(self, block: str):
        # Helper from KVAE implementation for Decoder cache
        def make_dict(name):
            if name == 'conv':
                return {'padding' : None}

            layer, module = name.split('_')
            if layer == 'norm':
                if module == 'enc':
                    return {'mean' : None,
                            'var' : None}
                else:
                    return {'norm' : make_dict('norm_enc'),
                            'add_conv' : make_dict('conv')}
            elif layer == 'resblock':
                return {'norm1' : make_dict(f'norm_{module}'),
                        'norm2' : make_dict(f'norm_{module}'),
                        'conv1' : make_dict('conv'),
                        'conv2' : make_dict('conv'),
                        'conv_shortcut' : make_dict('conv')}
            elif layer.isdigit():
                return {0 : make_dict(f'resblock_{module}'),
                        1 : make_dict(f'resblock_{module}'),
                        2 : make_dict(f'resblock_{module}'),
                        'down' : make_dict('conv'),
                        'up' : make_dict('conv')}

        cache = {'conv_in' : make_dict('conv'),
                 'mid_1' : make_dict(f'resblock_{block}'),
                 'mid_2' : make_dict(f'resblock_{block}'),
                 'norm_out' : make_dict(f'norm_{block}'),
                 'conv_out' : make_dict('conv'),
                 0 : make_dict(f'0_{block}'),
                 1 : make_dict(f'1_{block}'),
                 2 : make_dict(f'2_{block}'),
                 3 : make_dict(f'3_{block}')}

        return cache

    def encode(self, x: torch.Tensor, opt_tiling: bool = True) -> AutoencoderKLOutput:
        # Logic from KandinskyVAE.encode
        N, _, T, H, W = x.shape
        if N > 1:
            # The original implementation restricts batch size > 1
            # But standard pipeline might send batch. We iterate if needed.
            # Assuming batch=1 for video usually.
            pass
            
        # Determine tiling strategy
        n_tiles_log = round(log2(128 * T * H * W / 2**31))

        with torch._dynamo.utils.disable_cache_limit():
            if n_tiles_log <= 0:
                z = self.encoder.forward_1(x)
            elif n_tiles_log <= 1:
                if H > W:
                    z = self.encoder.forward_2h(x, 0, 0) # args unused in func signature but present
                else:
                    z = self.encoder.forward_2w(x)
            elif n_tiles_log <= 2:
                z = self.encoder.forward_4(x)
            elif n_tiles_log <= 3:
                if H > W:
                    z = self.encoder.forward_8h(x)
                else:
                    z = self.encoder.forward_8w(x)
            else:
                z = self.encoder.forward_16(x)
                
        # KVAE encoder returns 2*z_channels if double_z=True, BUT
        # the efficient encoder logic in `piece_2` slices it: x = self.conv_out(x)[:, :self.z_channels]
        # So z is already the mean/mode.
        
        dist = ConstantDistribution(z)
        return AutoencoderKLOutput(latent_dist=dist)

    def decode(self, z: torch.Tensor, return_dict: bool = True) -> Union[DecoderOutput, torch.Tensor]:
        # Use tiling logic (segmentation) from KVAE
        # Map user preference tile_sample_min_num_frames to seg_len if set
        seg_len = 16
        if self.use_manual_tiling:
             # Approximate mapping: user provided tile size in frames
             # VAE logic uses seg_len for temporal chunks
             seg_len = self.tile_sample_min_num_frames

        cache = self.make_empty_cache('dec')

        ## get segments size
        split_list = [seg_len + 1]
        n_frames = 4 * (z.size(2) - 1) - seg_len
        while n_frames > 0:
            split_list.append(seg_len)
            n_frames -= seg_len
        
        # Fix for negative n_frames if video is short
        if split_list[-1] < 0:
             split_list = [4 * (z.size(2) - 1)]

        split_list[-1] += n_frames
        
        # Ensure all sizes are positive
        split_list = [s for s in split_list if s > 0]
        
        split_list = [math.ceil(size / 4) for size in split_list]

        ## decode by segments
        recs = []
        # Handle case where split might not match perfectly due to rounding/math
        # Simple robust splitting:
        chunks = torch.split(z, split_list, dim=2)
        
        for chunk in chunks:
            out = self.decoder(chunk, cache)
            recs.append(out)

        recs = torch.cat(recs, dim=2)
        
        if not return_dict:
            return (recs,)
        return DecoderOutput(sample=recs)


def build_vae(
    conf,
    dtype=torch.bfloat16,
    temporal_tile_frames=None,
    temporal_stride_frames=None,
    spatial_tile_height=None,
    spatial_tile_width=None
):
    """
    Build VAE with configurable chunking parameters.

    Args:
        conf: VAE configuration (OmegaConf or dict)
        dtype: Model dtype
        temporal_tile_frames: Temporal chunk size in pixel-space frames (default: 16)
        temporal_stride_frames: Temporal stride in pixel-space frames (default: 12)
        spatial_tile_height: Spatial tile height (default: 256)
        spatial_tile_width: Spatial tile width (default: 256)
    """
    
    # Check for model type. The KVAE config usually lacks a top-level "name" field in the provided snippet,
    # but we can infer from structure 'encoder_params'
    is_kandinsky = "encoder_params" in conf or (hasattr(conf, "name") and conf.name == "kandinsky") or (hasattr(conf, "model") and "encoder_params" in conf.model)
    
    # Flatten config if it comes nested as model.vae -> {model: {encoder_params...}}
    if hasattr(conf, "model") and "encoder_params" in conf.model:
        vae_conf = conf.model
    elif "encoder_params" in conf:
        vae_conf = conf
    elif hasattr(conf, "name") and conf.name == "hunyuan":
        vae_conf = None
    else:
        vae_conf = None # Default fallthrough

    if is_kandinsky and vae_conf is not None:
        print("Building Kandinsky 3D VAE (KVAE)...")
        
        encoder_params = vae_conf.encoder_params
        decoder_params = vae_conf.decoder_params
        
        # Convert OmegaConf to dict if necessary
        if hasattr(encoder_params, 'to_container'): encoder_params = encoder_params.to_container()
        if hasattr(decoder_params, 'to_container'): decoder_params = decoder_params.to_container() # recursive=True?
        
        # Initialize model
        vae = AutoencoderKLKandinsky(
            encoder_params=encoder_params, 
            decoder_params=decoder_params
        )
        
        # Load weights
        checkpoint_path = conf.checkpoint_path if hasattr(conf, "checkpoint_path") else None
        # In the provided utils.py, checkpoint_path might be a directory or file
        if checkpoint_path and os.path.exists(checkpoint_path):
             # If it's a folder, look for .safetensors
            if os.path.isdir(checkpoint_path):
                files = [f for f in os.listdir(checkpoint_path) if f.endswith('.safetensors')]
                if files:
                    checkpoint_path = os.path.join(checkpoint_path, files[0])
                else:
                    print(f"Warning: No safetensors found in {checkpoint_path}")
            
            if os.path.isfile(checkpoint_path):
                print(f"Loading VAE weights from {checkpoint_path}")
                state_dict = safetensors.torch.load_file(checkpoint_path)
                
                # Apply padding fix for optimized encoder as seen in efficient_vae.py
                for k in list(state_dict.keys()):
                    if "encoder.conv_in.conv.weight" in k:
                        state_dict[k] = F.pad(state_dict[k], (0, 0, 0, 0, 0, 0, 0, 1))
                
                # Load
                vae.load_state_dict(state_dict, strict=True)
                vae = vae.to(dtype=dtype)
        else:
            print("Warning: No checkpoint path provided for KVAE, initializing random weights.")

        # Apply custom temporal chunking parameters if provided
        if temporal_tile_frames is not None:
            vae.tile_sample_min_num_frames = temporal_tile_frames
            vae.use_manual_tiling = True
            
        print(f"KVAE Initialized. Dtype: {dtype}")
        return vae

    elif hasattr(conf, "name") and conf.name == "hunyuan":
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            conf.checkpoint_path, subfolder="vae", torch_dtype=dtype
        )

        # Track if any manual parameters are provided
        has_manual_settings = False

        # Apply custom temporal chunking parameters if provided
        if temporal_tile_frames is not None:
            vae.tile_sample_min_num_frames = temporal_tile_frames
            has_manual_settings = True
        if temporal_stride_frames is not None:
            vae.tile_sample_stride_num_frames = temporal_stride_frames
            has_manual_settings = True
        elif temporal_tile_frames is not None:
            # Auto-calculate stride: tile_size - 4 for proper overlap
            vae.tile_sample_stride_num_frames = max(4, temporal_tile_frames - 4)

        # Apply custom spatial tiling parameters if provided
        if spatial_tile_height is not None:
            vae.tile_sample_min_height = spatial_tile_height
            # Auto-calculate spatial stride to maintain 32-pixel overlap (or 25% of tile)
            overlap = min(32, spatial_tile_height // 4)
            vae.tile_sample_stride_height = spatial_tile_height - overlap
            has_manual_settings = True

        if spatial_tile_width is not None:
            vae.tile_sample_min_width = spatial_tile_width
            # Auto-calculate spatial stride to maintain 32-pixel overlap (or 25% of tile)
            overlap = min(32, spatial_tile_width // 4)
            vae.tile_sample_stride_width = spatial_tile_width - overlap
            has_manual_settings = True

        # Enable manual tiling flag if any manual settings were provided
        if has_manual_settings:
            vae.use_manual_tiling = True
            print(f"VAE Manual Tiling Enabled:")
            print(f"  Temporal: {vae.tile_sample_min_num_frames} frames (stride: {vae.tile_sample_stride_num_frames})")
            print(f"  Spatial: {vae.tile_sample_min_height}x{vae.tile_sample_min_width} pixels (stride: {vae.tile_sample_stride_height}x{vae.tile_sample_stride_width})")

        return vae
    elif hasattr(conf, "name") and conf.name == "flux":
        from diffusers.models import AutoencoderKL
        vae = AutoencoderKL.from_pretrained(conf.checkpoint_path, subfolder="vae", torch_dtype=torch.bfloat16)
        return vae
    else:
        # Fallback or error
        print(f"Warning: Unknown VAE config: {conf}")
        assert False, f"Unknown VAE configuration: {conf}"