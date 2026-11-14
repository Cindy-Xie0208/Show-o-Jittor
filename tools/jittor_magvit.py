"""Native Jittor MagViT v2 encoder/decoder used by Show-o 1.

This is a direct inference-only port of the pinned official VQGAN.  The
module hierarchy intentionally follows ``modeling_magvitv2.py`` so that the
official safetensors keys can be assigned one-for-one.
"""

import math

import jittor as jt
from jittor import nn


def nonlinearity(x):
    return x * jt.sigmoid(x)


def normalize(channels):
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True)


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, dropout=0.0, temb_channels=0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv(in_channels, out_channels, 3, padding=1)
        self.temb_proj = nn.Linear(temb_channels, out_channels) if temb_channels > 0 else None
        self.norm2 = normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv(out_channels, out_channels, 3, padding=1)
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv(in_channels, out_channels, 1)

    def execute(self, x, temb=None):
        h = self.conv1(nonlinearity(self.norm1(x)))
        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb)).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(self.dropout(nonlinearity(self.norm2(h))))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.in_channels = channels
        self.norm = normalize(channels)
        self.q = nn.Conv(channels, channels, 1)
        self.k = nn.Conv(channels, channels, 1)
        self.v = nn.Conv(channels, channels, 1)
        self.proj_out = nn.Conv(channels, channels, 1)

    def execute(self, x):
        h = self.norm(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        b, c, height, width = q.shape
        q = q.reshape((b, c, height * width)).transpose((0, 2, 1))
        k = k.reshape((b, c, height * width))
        weights = jt.nn.softmax(jt.matmul(q, k) * (float(c) ** -0.5), dim=2)
        v = v.reshape((b, c, height * width))
        attended = jt.matmul(v, weights.transpose((0, 2, 1))).reshape((b, c, height, width))
        return x + self.proj_out(attended)


class Downsample(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv(channels, channels, 3, stride=2, padding=0)

    def execute(self, x):
        if self.with_conv:
            x = jt.nn.pad(x, [0, 1, 0, 1], mode="constant", value=0)
            return self.conv(x)
        return jt.nn.pool(x, 2, 2, "mean")


class Upsample(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv(channels, channels, 3, padding=1)

    def execute(self, x):
        x = jt.nn.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x) if self.with_conv else x


class EncoderStage(nn.Module):
    def __init__(self, block, attn, downsample=None):
        super().__init__()
        self.block = nn.ModuleList(block)
        self.attn = nn.ModuleList(attn)
        if downsample is not None:
            self.downsample = downsample


class DecoderStage(nn.Module):
    def __init__(self, block, attn, upsample=None):
        super().__init__()
        self.block = nn.ModuleList(block)
        self.attn = nn.ModuleList(attn)
        if upsample is not None:
            self.upsample = upsample


class VQGANEncoder(nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 2, 2, 4, 4), num_res_blocks=(4, 3, 4, 3, 4),
                 attn_resolutions=(5,), dropout=0.0, in_ch=3, out_ch=3, resolution=256,
                 z_channels=13, double_z=False):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_ch = in_ch
        self.conv_in = nn.Conv(in_ch, ch, 3, padding=1)
        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        stages = []
        for level in range(self.num_resolutions):
            block = []
            attn = []
            block_in = ch * in_ch_mult[level]
            block_out = ch * ch_mult[level]
            for block_index in range(num_res_blocks[level]):
                block.append(ResnetBlock(block_in, block_out, dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = Downsample(block_in, True) if level != self.num_resolutions - 1 else None
            stages.append(EncoderStage(block, attn, down))
            if down is not None:
                curr_res //= 2
        self.down = nn.ModuleList(stages)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv(block_in, 2 * z_channels if double_z else z_channels, 3, padding=1)
        self.quant_conv = nn.Conv(z_channels, z_channels, 1)

    def execute(self, x):
        hidden = [self.conv_in(x)]
        for level, stage in enumerate(self.down):
            for block_index, block in enumerate(stage.block):
                h = block(hidden[-1], None)
                if len(stage.attn) > 0:
                    h = stage.attn[block_index](h)
                hidden.append(h)
            if hasattr(stage, "downsample"):
                hidden.append(stage.downsample(hidden[-1]))
        h = self.mid.block_1(hidden[-1], None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        return self.quant_conv(self.conv_out(nonlinearity(self.norm_out(h))))


class LFQuantizer(nn.Module):
    def __init__(self, codebook_dim=13):
        super().__init__()
        self.codebook_size = 2 ** codebook_dim
        self.e_dim = codebook_dim
        indices = jt.arange(self.codebook_size).int()
        bits = jt.arange(codebook_dim - 1, -1, -1).int()
        embedding = ((indices.unsqueeze(1) >> bits) & 1).float32() * 2 - 1
        self.embedding = embedding.stop_grad()
        self.power_vals = (2 ** jt.arange(codebook_dim - 1, -1, -1)).int().stop_grad()

    def get_indices(self, z_q):
        return (self.power_vals.reshape((1, -1, 1, 1)) * (z_q > 0).int()).sum(1, keepdims=True).int()

    def get_codebook_entry(self, indices, shape=None):
        if shape is None:
            side = int(math.sqrt(indices.shape[-1]))
            shape = (side, side)
        h, w = shape
        b = indices.shape[0]
        z = self.embedding[indices.reshape((-1,))].reshape((b, h, w, -1))
        return z.transpose((0, 3, 1, 2))

    def execute(self, z):
        z = z.transpose((0, 2, 3, 1))
        z_flat = z.reshape((-1, self.e_dim))
        quantized = jt.where(z_flat > 0, jt.ones_like(z_flat), -jt.ones_like(z_flat))
        quantized = quantized.reshape(z.shape).transpose((0, 3, 1, 2))
        return {"z": quantized, "indices": self.get_indices(quantized)}


class VQGANDecoder(nn.Module):
    def __init__(self, ch=128, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=(4, 4, 3, 4, 3),
                 attn_resolutions=(5,), dropout=0.0, in_ch=3, out_ch=3, resolution=256, z_channels=13):
        super().__init__()
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.z_channels = z_channels
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[-1]
        curr_res = resolution // (2 ** (self.num_resolutions - 1))
        self.conv_in = nn.Conv(z_channels, block_in, 3, padding=1)
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, dropout)
        stages = [None] * self.num_resolutions
        for level in reversed(range(self.num_resolutions)):
            block = []
            attn = []
            block_out = ch * ch_mult[level]
            for _ in range(num_res_blocks[level]):
                block.append(ResnetBlock(block_in, block_out, dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = Upsample(block_in, True) if level != 0 else None
            stages[level] = DecoderStage(block, attn, up)
            if up is not None:
                curr_res *= 2
        self.up = nn.ModuleList(stages)
        self.norm_out = normalize(block_in)
        self.conv_out = nn.Conv(block_in, out_ch, 3, padding=1)
        self.post_quant_conv = nn.Conv(z_channels, z_channels, 1)

    def execute(self, z):
        h = self.mid.block_1(self.conv_in(self.post_quant_conv(z)), None)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, None)
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_index, block in enumerate(stage.block):
                h = block(h, None)
                if len(stage.attn) > 0:
                    h = stage.attn[block_index](h)
            if hasattr(stage, "upsample"):
                h = stage.upsample(h)
        return self.conv_out(nonlinearity(self.norm_out(h)))


class MAGVITv2(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = VQGANEncoder()
        self.decoder = VQGANDecoder()
        self.quantize = LFQuantizer()

    def get_code(self, pixel_values):
        hidden = self.encoder(pixel_values)
        quantized = self.quantize(hidden)["z"]
        return self.quantize.get_indices(quantized).reshape((pixel_values.shape[0], -1))

    def decode_code(self, codebook_indices, shape=None):
        return self.decoder(self.quantize.get_codebook_entry(codebook_indices, shape))
