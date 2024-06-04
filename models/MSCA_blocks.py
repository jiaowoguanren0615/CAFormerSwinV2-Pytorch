import torch
from torch import nn
from torchvision import ops
import itertools
import math

from timm.layers import trunc_normal_
from torch.nn import functional

import torch.nn.functional as F

from pytorch_wavelets import DTCWTForward, DTCWTInverse


class ConvFFNModule(nn.Module):
    def __init__(self, in_channels: int, expansion_ratio: int = 4, out_channels: int = None, eps: float = 1e-5):
        """
        in_channels: input dimension
        out_channels: output dimension
        """
        super().__init__()
        out_channels = in_channels if not out_channels else out_channels
        hid_channels = in_channels * expansion_ratio
        self.pwconv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=hid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, hid_channels, eps=eps)
        )

        self.pwconv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hid_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels, eps=eps)
        )

        self.dwconv_block = nn.Sequential(
            nn.Conv2d(in_channels=hid_channels, out_channels=hid_channels, kernel_size=3, padding=1,
                      groups=hid_channels, bias=False),
            nn.GroupNorm(32, hid_channels, eps=eps)
        )

        self.activation = nn.Hardswish()

    def forward(self, x):
        x = self.pwconv_block_1(x)
        x = self.activation(x)
        x = x + self.dwconv_block(x)
        x = self.activation(x)
        x = self.pwconv_block_2(x)
        return x



class LayerNormProxy(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.layer_norm_proxy = nn.Sequential(
            ops.Permute([0, 2, 3, 1]),
            nn.LayerNorm(dim, eps=eps),
            ops.Permute([0, 3, 1, 2])
        )

    def forward(self, x):
        return self.layer_norm_proxy(x)


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class StemModule(nn.Module):
    def __init__(self, dim: int = 96, kernel_size: int = 3, stride: int = 2, padding: int = 1, eps: float = 1e-5):
        """
        dim: return dimension of the stem module
        """
        super().__init__()
        self.stem_block = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=dim // 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(16, dim // 2, eps=eps),
            nn.Hardswish(),
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(16, dim // 2, eps=eps),
            nn.Hardswish(),
            nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(32, dim, eps=eps),
            nn.Hardswish(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(32, dim, eps=eps),
        )

    def forward(self, x):
        return self.stem_block(x)


class RevStemModule(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.rev_stem_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(32, dim),
            nn.Hardswish(),
            nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=2, stride=2, bias=False),
            nn.GroupNorm(32, dim),
            nn.Hardswish(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.Hardswish(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, dim),
            nn.Hardswish(),
            nn.Conv2d(in_channels=dim, out_channels=2, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.rev_stem_block(x)


class LPUModule(nn.Module):
    def __init__(self, dim: int):
        """
        dim: return dimension of the lpu module
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)

    def forward(self, x):
        x = x.clone() + self.conv(x)

        return x


class Scale(nn.Module):
    """
    Scale vector by element multiplications.
    """
    def __init__(self, dim, init_value=1.0, trainable=True):
        super().__init__()
        self.scale = nn.Parameter(init_value * torch.ones(dim), requires_grad=trainable)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2) * self.scale
        return x.transpose(1, 2).reshape(B, C, H, W)


class StrideMHAModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, resolution: int = 56, eps: int = 1e-5,
                 num_heads: int = 8):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.num_heads = num_heads
        self.head_dims = in_channels // num_heads
        self.scale = self.head_dims ** -0.5
        self.resolution = resolution
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True).to("cuda")
        self.out_channels = out_channels
        stride = resolution // 7
        self.stride_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )

        self.upsample = nn.Upsample(scale_factor=stride, mode="bilinear")

        self.to_q_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels, eps=eps)
        )

        self.to_k_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels, eps=eps)
        )

        self.to_v_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels, eps=eps)
        )

        self.to_out_block = nn.Sequential(
            nn.Hardswish(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=out_channels, eps=eps)
        )

        self.locality = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, groups=num_heads),
            nn.BatchNorm2d(num_features=out_channels, eps=eps)
        )

        self.talking_head_1 = nn.Conv2d(in_channels=num_heads, out_channels=num_heads, kernel_size=1, bias=False)
        self.talking_head_2 = nn.Conv2d(in_channels=num_heads, out_channels=num_heads, kernel_size=1, bias=False)

        points = list(itertools.product(range(self.resolution // stride), range(self.resolution // stride)))
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets))).to("cuda")
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(49, 49).to("cuda"))

    def forward(self, x):
        x = self.stride_conv(x)
        B, C, H, W = x.shape
        q = self.to_q_block(x).flatten(2).reshape(B, self.num_heads, -1, W * H).permute(0, 1, 3, 2)
        k = self.to_k_block(x).flatten(2).reshape(B, self.num_heads, -1, W * H).permute(0, 1, 2, 3)
        v = self.to_v_block(x)
        v_local = self.locality(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, W * H).permute(0, 1, 3, 2)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to("cuda"))).exp()
        attn = ((functional.normalize(q, dim=-1) @ functional.normalize(k, dim=-1)) * logit_scale + (
        self.attention_biases[:, self.attention_bias_idxs]))
        attn = self.talking_head_1(attn)
        attn = attn.softmax(-1)
        attn = self.talking_head_2(attn)
        attn = attn @ v
        attn = attn.transpose(2, 3).reshape(B, self.out_channels, H, W) + v_local
        attn = self.upsample(attn)
        attn = self.to_out_block(attn)

        return attn


class MHAModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, resolution: int = 56, eps: int = 1e-5,
                 num_heads: int = 8):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.num_heads = num_heads
        self.head_dims = in_channels // num_heads
        self.scale = self.head_dims ** -0.5
        self.resolution = resolution
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True).to("cuda")
        self.out_channels = out_channels
        stride = resolution // 7

        self.query_stride_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )
        self.key_stride_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )
        self.value_stride_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels),
            nn.BatchNorm2d(in_channels)
        )

        self.upsample = nn.Upsample(scale_factor=stride, mode="bilinear")

        self.to_q_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels, eps=eps)
        )

        self.to_k_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels, eps=eps)
        )

        self.to_v_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels, eps=eps)
        )

        self.to_out_block = nn.Sequential(
            nn.Hardswish(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels, eps=eps)
        )

        self.locality = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, groups=num_heads,
                      bias=False),
            nn.GroupNorm(32, out_channels, eps=eps)
        )

        self.talking_head_1 = nn.Conv2d(in_channels=num_heads, out_channels=num_heads, kernel_size=1, bias=False)
        self.talking_head_2 = nn.Conv2d(in_channels=num_heads, out_channels=num_heads, kernel_size=1, bias=False)

        points = list(itertools.product(range(self.resolution // stride), range(self.resolution // stride)))
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets))).to("cuda")
        self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(49, 49).to("cuda"))

    def forward(self, query, key, value):
        query = self.query_stride_conv(query)
        key = self.key_stride_conv(key)
        value = self.value_stride_conv(value)
        B, C, H, W = query.shape
        q = self.to_q_block(query).flatten(2).reshape(B, self.num_heads, -1, W * H).permute(0, 1, 3, 2)
        k = self.to_k_block(key).flatten(2).reshape(B, self.num_heads, -1, W * H).permute(0, 1, 2, 3)
        v = self.to_v_block(value)
        v_local = self.locality(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, W * H).permute(0, 1, 3, 2)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to("cuda"))).exp()
        attn = ((functional.normalize(q, dim=-1) @ functional.normalize(k, dim=-1)) * logit_scale + (
        self.attention_biases[:, self.attention_bias_idxs]))
        attn = self.talking_head_1(attn)
        attn = attn.softmax(-1)
        attn = self.talking_head_2(attn)
        attn = attn @ v
        attn = attn.transpose(2, 3).reshape(B, self.out_channels, H, W) + v_local
        attn = self.upsample(attn)
        attn = self.to_out_block(attn)

        return attn


class MSCAModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        eps = 1e-5
        out_channels = out_channels if out_channels is not None else in_channels
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              bias=False) if out_channels is not None else nn.Identity()
        self.pwconv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels)
        )

        self.pwconv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels)
        )

        self.conv0 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2,
                      groups=out_channels, bias=False),
            nn.GroupNorm(32, out_channels)
        )

        self.conv1_block = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 7), padding=(0, 3),
                      groups=out_channels, bias=False),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(7, 1), padding=(3, 0),
                      groups=out_channels, bias=False),
            nn.GroupNorm(32, out_channels)
        )

        self.conv2_block = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 11), padding=(0, 5),
                      groups=out_channels, bias=False),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(11, 1), padding=(5, 0),
                      groups=out_channels, bias=False),
            nn.GroupNorm(32, out_channels)
        )

        self.conv3_block = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 21), padding=(0, 10),
                      groups=out_channels, bias=False),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(21, 1), padding=(10, 0),
                      groups=out_channels, bias=False),
            nn.GroupNorm(32, out_channels)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels)
        )

        self.activation = nn.Hardswish()

    def forward(self, x):
        x = self.proj(x)
        temp_1 = x.clone()
        x = self.pwconv_block_1(x)
        x = self.activation(x)

        temp = x.clone()
        attn = self.conv0(x)

        attn1 = self.conv1_block(attn)
        attn2 = self.conv2_block(attn)
        attn3 = self.conv3_block(attn)

        attn = attn + attn1 + attn2 + attn3

        attn = self.conv4(attn)

        attn = attn * temp

        x = self.pwconv_block_2(attn)
        x = x + temp_1
        return x



class DAModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_groups: int, num_heads, kernel_size: int, stride: int,
                 resolution: int):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels

        self.resolution = resolution
        self.num_groups = num_groups
        self.group_dims = in_channels // num_groups
        self.num_heads = num_heads
        self.head_dims = out_channels // num_heads
        self.scale = self.head_dims ** -0.5
        self.num_group_heads = self.num_heads // self.num_groups
        padding = kernel_size // 2 if kernel_size != stride else 0

        self.conv_offset = nn.Sequential(
            nn.Conv2d(self.group_dims, self.group_dims, kernel_size=kernel_size, stride=stride, padding=padding,
                      groups=self.group_dims, bias=False),
            nn.GroupNorm(2, self.group_dims),
            nn.Hardswish(),
            nn.Conv2d(in_channels=self.group_dims, out_channels=2, kernel_size=1, bias=False)
        )

        self.to_q = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels)
        )
        self.to_k = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels)
        )
        self.to_v = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels)
        )

        self.to_out = nn.Sequential(
            nn.Hardswish(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(32, out_channels)
        )
        self.rpe_table = nn.Parameter(torch.zeros(self.num_heads, self.resolution * 2 - 1, self.resolution * 2 - 1))
        trunc_normal_(self.rpe_table, std=0.01)

    @torch.no_grad()
    def _get_ref_points(self, H_key, W_key, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_key - 0.5, H_key, dtype=dtype, device=device),
            torch.linspace(0.5, W_key - 0.5, W_key, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W_key - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H_key - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.num_groups, -1, -1, -1)  # B * g H W 2

        return ref

    @torch.no_grad()
    def _get_q_grid(self, H, W, B, dtype, device):
        ref_y, ref_x = torch.meshgrid(
            torch.arange(0, H, dtype=dtype, device=device),
            torch.arange(0, W, dtype=dtype, device=device),
            indexing='ij'
        )
        ref = torch.stack((ref_y, ref_x), -1)
        ref[..., 1].div_(W - 1.0).mul_(2.0).sub_(1.0)
        ref[..., 0].div_(H - 1.0).mul_(2.0).sub_(1.0)
        ref = ref[None, ...].expand(B * self.num_groups, -1, -1, -1)  # B * g H W 2

        return ref

    def forward(self, query, key, value):
        B, C, H, W = query.shape
        dtype, device = query.dtype, query.device

        q = self.to_q(query)
        q_offset = q.reshape([B * self.num_groups, self.group_dims, H, W])
        offset = self.conv_offset(q_offset).contiguous()
        Hk, Wk = offset.shape[2:]
        num_sample = Hk * Wk

        offset = offset.permute([0, 2, 3, 1])
        reference = self._get_ref_points(Hk, Wk, B, dtype, device)

        pos = (offset + reference).clamp(-1., +1)

        k_sampled = functional.grid_sample(
            input=key.reshape(B * self.num_groups, self.group_dims, H, W),
            grid=pos[..., (1, 0)],
            mode='bilinear', align_corners=True
        ).reshape(B, C, 1, num_sample)

        v_sampled = functional.grid_sample(
            input=value.reshape(B * self.num_groups, self.group_dims, H, W),
            grid=pos[..., (1, 0)],
            mode='bilinear', align_corners=True
        ).reshape(B, C, 1, num_sample)

        q = q.reshape(B * self.num_heads, self.head_dims, H * W)
        k = self.to_k(k_sampled).reshape(B * self.num_heads, self.head_dims, num_sample)
        v = self.to_v(v_sampled).reshape(B * self.num_heads, self.head_dims, num_sample)

        attn = attn = torch.einsum('b c m, b c n -> b m n', q, k)
        attn = attn.mul(self.scale)

        rpe_table = self.rpe_table
        rpe_bias = rpe_table[None, ...].expand(B, -1, -1, -1)
        q_grid = self._get_q_grid(H, W, B, dtype, device)
        displacement = (q_grid.reshape(B * self.num_groups, H * W, 2).unsqueeze(2) - pos.reshape(B * self.num_groups,
                                                                                                 num_sample,
                                                                                                 2).unsqueeze(1)).mul(
            0.5)
        Ha, Wa = rpe_bias.shape[2:]
        attn_bias = functional.grid_sample(
            input=rpe_bias.reshape(B * self.num_groups, self.num_group_heads, Ha, Wa),
            grid=displacement[..., (1, 0)],
            mode="bilinear",
            align_corners=True
        )
        attn_bias = attn_bias.reshape(B * self.num_heads, H * W, num_sample)
        attn = attn + attn_bias

        attn = functional.softmax(attn, dim=2)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, H, W)
        x = self.to_out(out)
        return x


class AgentAttnModule(nn.Module):
    def __init__(self, dim: int, num_heads: int, agent_num: int, num_patches: int = 16,
                 res_scale_init_value: float = 0.0, sr_ratio: int = 1):
        super().__init__()
        self.dim = dim
        self.sr_ratio = sr_ratio
        window_size = int(num_patches ** 0.5)
        self.window_size = window_size
        self.num_heads = num_heads
        self.agent_num = agent_num
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim)
        )
        self.k = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim)
        )
        self.v = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim)
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim)
        )
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, 7, 7))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size, 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size, agent_num))
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.GroupNorm(num_groups=2, num_channels=dim)

    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        query = self.q(x).flatten(2).transpose(1, 2)
        if self.sr_ratio > 1:
            x_ = self.sr(x)
            x_ = self.norm(x_)
            key = self.k(x_).flatten(2).transpose(1, 2)
            value = self.v(x_).flatten(2).transpose(1, 2)
        else:
            key = self.k(x).flatten(2).transpose(1, 2)
            value = self.v(x).flatten(2).transpose(1, 2)

        agent_tokens = self.pool(query.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).transpose(1, 2)
        query = query.reshape(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.reshape(B, N // self.sr_ratio ** 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.reshape(B, N // self.sr_ratio ** 2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(B, self.agent_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        kv_size = self.window_size // self.sr_ratio
        position_bias1 = F.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, self.num_heads, self.agent_num, -1).repeat(B, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ key.transpose(-2, -1) + position_bias)
        agent_v = agent_attn @ value

        agent_bias1 = F.interpolate(self.na_bias, size=self.window_size, mode="bilinear")
        agent_bias1 = agent_bias1.reshape(1, self.num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(B, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, self.num_heads, -1, self.agent_num).repeat(B, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((query * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        x = q_attn @ agent_v
        x = x.transpose(1, 2).reshape(B, N, C)
        value = value.transpose(1, 2).reshape(B, H // self.sr_ratio, W // self.sr_ratio, C).permute(0, 3, 1, 2)
        if self.sr_ratio > 1:
            value = F.interpolate(value, size=(H, W), mode='bilinear')
        x = x + self.dwc(value).permute(0, 2, 3, 1).reshape(B, N, C)
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return self.proj(x)


class AgentAttnLayerModule(nn.Module):
    def __init__(self, in_channels: int, agent_num: int, num_heads: int, res_scale_init_value: float, num_patches: int,
                 expansion_ratio: int = 4, sr_ratio: int = 1):
        super().__init__()
        self.lpu = LPUModule(dim=in_channels)
        self.attn_block = nn.Sequential(
            AgentAttnModule(dim=in_channels, num_heads=num_heads, agent_num=agent_num,
                            res_scale_init_value=res_scale_init_value, num_patches=num_patches, sr_ratio=sr_ratio)
        )
        self.conv_ffn_block = ConvFFNModule(in_channels=in_channels, expansion_ratio=expansion_ratio)
        self.res_scale_1 = Scale(dim=in_channels,
                                 init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.res_scale_2 = Scale(dim=in_channels,
                                 init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.layer_scale_1 = Scale(dim=in_channels,
                                   init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.layer_scale_2 = Scale(dim=in_channels,
                                   init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, in_channels)

    #         self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        x = self.lpu(x)
        x = self.res_scale_1(x) + self.norm1(self.layer_scale_1(self.attn_block(x)))
        x = self.res_scale_2(x) + self.norm2(self.layer_scale_2(self.conv_ffn_block(x)))
        #         x = self.proj(x)

        return x


class ScatteringModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim == 64:
            self.hidden_size = dim
            self.num_blocks = 4
            self.block_size = self.hidden_size // self.num_blocks
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 56, 56, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)

        if dim == 128:  # [b, 128,28,28]
            self.hidden_size = dim
            self.num_blocks = 4
            self.block_size = self.hidden_size // self.num_blocks
            self.complex_weight_ll = nn.Parameter(torch.randn(dim, 28, 28, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_1 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_2 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b1 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)
            self.complex_weight_lh_b2 = nn.Parameter(
                torch.randn(2, self.num_blocks, self.block_size, dtype=torch.float32) * 0.02)

        self.xfm = DTCWTForward(J=1, biort='near_sym_b', qshift='qshift_b')
        self.ifm = DTCWTInverse(biort='near_sym_b', qshift='qshift_b')
        self.softshrink = 0.0

    def multiply(self, input, weights):
        return torch.einsum('...bd,bdk->...bk', input, weights)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.to(torch.float32)

        xl, xh = self.xfm(x)
        xl = xl * self.complex_weight_ll

        xh[0] = torch.permute(xh[0], (5, 0, 2, 3, 4, 1))
        xh[0] = xh[0].reshape(xh[0].shape[0], xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], xh[0].shape[4],
                              self.num_blocks, self.block_size)

        x_real = xh[0][0]
        x_imag = xh[0][1]

        x_real_1 = F.relu(
            self.multiply(x_real, self.complex_weight_lh_1[0]) - self.multiply(x_imag, self.complex_weight_lh_1[1]) +
            self.complex_weight_lh_b1[0])
        x_imag_1 = F.relu(
            self.multiply(x_real, self.complex_weight_lh_1[1]) + self.multiply(x_imag, self.complex_weight_lh_1[0]) +
            self.complex_weight_lh_b1[1])

        x_real_2 = self.multiply(x_real_1, self.complex_weight_lh_2[0]) - self.multiply(x_imag_1,
                                                                                        self.complex_weight_lh_2[1]) + \
                   self.complex_weight_lh_b2[0]
        x_imag_2 = self.multiply(x_real_1, self.complex_weight_lh_2[1]) + self.multiply(x_imag_1,
                                                                                        self.complex_weight_lh_2[0]) + \
                   self.complex_weight_lh_b2[1]

        xh[0] = torch.stack([x_real_2, x_imag_2], dim=-1).float()
        xh[0] = F.softshrink(xh[0], lambd=self.softshrink) if self.softshrink else xh[0]
        xh[0] = xh[0].reshape(B, xh[0].shape[1], xh[0].shape[2], xh[0].shape[3], self.hidden_size, xh[0].shape[6])
        xh[0] = torch.permute(xh[0], (0, 4, 1, 2, 3, 5))

        x = self.ifm((xl, xh))
        return x


class QueryDownSamplingModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=1, stride=2, padding=0)
        self.local = nn.Conv2d(in_channels=dim // 2, out_channels=dim // 2, kernel_size=3, stride=2, padding=1,
                               groups=dim // 2)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=1),
            nn.GroupNorm(32, dim, eps=1e-5)
        )

    def forward(self, x):
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)
        return q


class QueryUpSamplingModule(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.pool = nn.Upsample(scale_factor=2.0)
        self.local = nn.ConvTranspose2d(in_channels=dim * 2, out_channels=dim * 2, kernel_size=2, stride=2,
                                        groups=dim * 2)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1),
            nn.GroupNorm(32, dim, eps=1e-5)
        )

    def forward(self, x):
        local_q = self.local(x)
        pool_q = self.pool(x)
        q = local_q + pool_q
        q = self.proj(q)
        return q


class DualPathAttentionDownSamplingModule(nn.Module):
    def __init__(self, dim: int, resolution: int, eps: int = 1e-5, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dims = dim // num_heads
        self.scale = self.head_dims ** -0.5
        self.resolution = resolution * 2
        self.resolution2 = resolution
        self.N2 = self.resolution2 ** 2
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True).to("cuda")

        self.to_q_block = QueryDownSamplingModule(dim)

        self.to_k_block = nn.Sequential(
            nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim, eps=eps)
        )

        self.to_v_block = nn.Sequential(
            nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim, eps=eps)
        )

        self.to_out_block = nn.Sequential(
            nn.Hardswish(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim, eps=eps)
        )

        self.locality = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=2, padding=1, groups=num_heads),
            nn.GroupNorm(32, dim, eps=eps)
        )

        self.talking_head_1 = nn.Conv2d(in_channels=num_heads, out_channels=num_heads, kernel_size=1, bias=False)
        self.talking_head_2 = nn.Conv2d(in_channels=num_heads, out_channels=num_heads, kernel_size=1, bias=False)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=dim // 2, out_channels=dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, dim)
        )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        points_ = list(itertools.product(
            range(self.resolution2), range(self.resolution2)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * math.ceil(self.resolution / self.resolution2) - p2[0] + (size - 1) / 2),
                    abs(p1[1] * math.ceil(self.resolution / self.resolution2) - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))).to("cuda")
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N).to("cuda"))

    def forward(self, x):
        B, C, H, W = x.shape
        x_proj = self.proj(x)
        q = self.to_q_block(x).flatten(2).reshape(B, self.num_heads, -1, self.N2).permute(0, 1, 3, 2)
        k = self.to_k_block(x).flatten(2).reshape(B, self.num_heads, -1, W * H).permute(0, 1, 2, 3)
        v = self.to_v_block(x)
        v_local = self.locality(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, W * H).permute(0, 1, 3, 2)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to("cuda"))).exp()
        attn = ((functional.normalize(q) @ functional.normalize(k)) * self.scale + (
        self.attention_biases[:, self.attention_bias_idxs]))
        attn = self.talking_head_1(attn)
        attn = attn.softmax(-1)
        attn = self.talking_head_2(attn)
        attn = attn @ v
        attn = attn.transpose(2, 3).reshape(B, C * 2, self.resolution2, self.resolution2) + v_local
        attn = self.to_out_block(attn)
        attn = attn + x_proj
        return attn


class DualPathAttentionUpSamplingModule(nn.Module):
    def __init__(self, dim: int, resolution: int, eps: int = 1e-5, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dims = dim // num_heads
        self.scale = self.head_dims ** -0.5
        self.resolution = resolution // 2
        self.resolution2 = resolution
        self.N2 = self.resolution2 ** 2
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True).to("cuda")

        self.to_q_block = QueryUpSamplingModule(dim)

        self.to_k_block = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim, eps=eps)
        )

        self.to_v_block = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim, eps=eps)
        )

        self.to_out_block = nn.Sequential(
            nn.Hardswish(),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=False),
            nn.GroupNorm(32, dim, eps=eps)
        )

        self.locality = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim, out_channels=dim, kernel_size=2, stride=2, groups=num_heads),
            nn.GroupNorm(32, dim, eps=eps)
        )

        self.talking_head_1 = nn.Conv2d(in_channels=num_heads, out_channels=num_heads, kernel_size=1, bias=False)
        self.talking_head_2 = nn.Conv2d(in_channels=num_heads, out_channels=num_heads, kernel_size=1, bias=False)
        self.proj = nn.Sequential(
            nn.ConvTranspose2d(in_channels=dim * 2, out_channels=dim, kernel_size=2, stride=2),
            nn.GroupNorm(32, dim)
        )

        points = list(itertools.product(range(self.resolution), range(self.resolution)))
        points_ = list(itertools.product(
            range(self.resolution2), range(self.resolution2)))
        N = len(points)
        N_ = len(points_)
        attention_offsets = {}
        idxs = []
        for p1 in points_:
            for p2 in points:
                size = 1
                offset = (
                    abs(p1[0] * math.ceil(self.resolution / self.resolution2) - p2[0] + (size - 1) / 2),
                    abs(p1[1] * math.ceil(self.resolution / self.resolution2) - p2[1] + (size - 1) / 2))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets))).to("cuda")
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N_, N).to("cuda"))

    def forward(self, x):
        B, C, H, W = x.shape
        x_proj = self.proj(x)
        q = self.to_q_block(x).flatten(2).reshape(B, self.num_heads, -1, self.N2).permute(0, 1, 3, 2)
        k = self.to_k_block(x).flatten(2).reshape(B, self.num_heads, -1, W * H).permute(0, 1, 2, 3)
        v = self.to_v_block(x)
        v_local = self.locality(v)
        v = v.flatten(2).reshape(B, self.num_heads, -1, W * H).permute(0, 1, 3, 2)
        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01).to("cuda"))).exp()
        attn = ((functional.normalize(q) @ functional.normalize(k)) * self.scale + (
        self.attention_biases[:, self.attention_bias_idxs]))
        attn = self.talking_head_1(attn)
        attn = attn.softmax(-1)
        attn = self.talking_head_2(attn)
        attn = attn @ v
        attn = attn.transpose(2, 3).reshape(B, C // 2, self.resolution2, self.resolution2) + v_local
        attn = self.to_out_block(attn)
        attn = attn + x_proj
        return attn


class MSCALayerModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, expansion_ratio: int = 4,
                 res_scale_init_value: float = None):
        super().__init__()
        out_channels = out_channels if out_channels is not None else in_channels
        self.lpu = LPUModule(dim=in_channels)
        self.attn_block = nn.Sequential(
            MSCAModule(in_channels=in_channels)
        )
        self.conv_ffn_block = ConvFFNModule(in_channels=in_channels, expansion_ratio=expansion_ratio)
        self.res_scale_1 = Scale(dim=in_channels,
                                 init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.res_scale_2 = Scale(dim=in_channels,
                                 init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.layer_scale_1 = Scale(dim=in_channels,
                                   init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.layer_scale_2 = Scale(dim=in_channels,
                                   init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.norm2 = nn.GroupNorm(32, in_channels)
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                              bias=False) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        x = self.lpu(x)
        x = self.res_scale_1(x) + self.norm1(self.layer_scale_1(self.attn_block(x)))
        x = self.res_scale_2(x) + self.norm2(self.layer_scale_2(self.conv_ffn_block(x)))
        x = self.proj(x)

        return x


class DALayerModule(nn.Module):
    def __init__(self, dim: int, res_scale_init_value: float, num_groups: int, num_heads: int, kernel_size: int,
                 stride: int, resolution: int):
        super().__init__()
        self.lpu = LPUModule(dim=dim)
        self.attn_block = DAModule(in_channels=dim, out_channels=None, num_groups=num_groups, num_heads=num_heads,
                                   kernel_size=kernel_size, stride=stride, resolution=resolution)

        self.conv_ffn_block = ConvFFNModule(in_channels=dim)
        self.res_scale_1 = Scale(dim=dim,
                                 init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.res_scale_2 = Scale(dim=dim,
                                 init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.norm1 = nn.GroupNorm(32, dim)
        self.norm2 = nn.GroupNorm(32, dim)

    def forward(self, x):
        x = self.lpu(x)
        x = self.res_scale_1(x) + self.norm1(self.attn_block(x, x, x))
        x = self.res_scale_2(x) + self.norm2(self.conv_ffn_block(x))

        return x


class ScatteringLayerModule(nn.Module):
    def __init__(self, dim: int, res_scale_init_value: int):
        super().__init__()
        self.lpu = LPUModule(dim=dim)
        self.scat = ScatteringModule(dim=dim)
        self.conv_ffn_block = ConvFFNModule(in_channels=dim)
        self.res_scale_1 = Scale(dim=dim,
                                 init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.res_scale_2 = Scale(dim=dim,
                                 init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.norm1 = nn.GroupNorm(32, dim)
        self.norm2 = nn.GroupNorm(32, dim)

    def forward(self, x):
        x = self.lpu(x)
        x = self.res_scale_1(x) + self.norm1(self.scat(x))
        x = self.res_scale_2(x) + self.norm2(self.conv_ffn_block(x))

        return x


class MHALayerModule(nn.Module):
    def __init__(self, dim: int, res_scale_init_value: float, resolution: int, num_heads: int):
        super().__init__()
        self.lpu = LPUModule(dim=dim)
        self.attn_block = nn.Sequential(
            StrideMHAModule(in_channels=dim, resolution=resolution, num_heads=num_heads),
        )
        self.conv_ffn_block = ConvFFNModule(in_channels=dim)
        self.res_scale_1 = Scale(dim=dim,
                                 init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.res_scale_2 = Scale(dim=dim,
                                 init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.layer_scale_1 = Scale(dim=dim,
                                   init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.layer_scale_2 = Scale(dim=dim,
                                   init_value=res_scale_init_value) if res_scale_init_value is not None else nn.Identity()
        self.norm1 = nn.GroupNorm(32, dim)
        self.norm2 = nn.GroupNorm(32, dim)

    def forward(self, x):
        x = self.lpu(x)
        x = self.res_scale_1(x) + self.norm1(self.attn_block(self.layer_scale_1(x)))
        x = self.res_scale_2(x) + self.norm2(self.conv_ffn_block(self.layer_scale_2(x)))

        return x