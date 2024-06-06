import torch
from torch import nn
from .MSCA_blocks import LPUModule, LayerNormProxy, StarReLU, Scale


class MLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None):
        """
        in_channels: input dimension
        out_channels: output dimension
        """
        super().__init__()
        out_channels = in_channels if not out_channels else out_channels
        hid_channels = in_channels * 4
        self.pw1 = nn.Conv2d(in_channels=in_channels, out_channels=hid_channels, kernel_size=1, bias=False)
        self.acti = StarReLU()
        self.pw2 = nn.Conv2d(in_channels=hid_channels, out_channels=out_channels, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(in_channels=hid_channels, out_channels=hid_channels, kernel_size=3, padding=1,
                              groups=hid_channels, bias=False)

    def forward(self, x):
        x = self.pw1(x)
        x = x + self.conv(x)
        x = self.acti(x)
        x = self.pw2(x)

        return x


class WindowSelfAttentionModule(nn.Module):
    def __init__(self, dim: int, num_heads: int, window_size: int):
        """
        dim: output dimension
        num_heads: number of attention head
        window_size: the window size for wsa module
        """
        super().__init__()

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_heads = num_heads
        self.head_dims = dim // num_heads
        self.window_size = window_size
        self.to_qkv = nn.Linear(in_features=dim, out_features=dim * 3, bias=False)
        self.to_out = nn.Linear(in_features=dim, out_features=dim, bias=False)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=False),
            nn.Hardswish(),
            nn.Linear(512, num_heads, bias=False)
        )
        self.softmax = nn.Softmax(1)

        # relative coords table
        relative_coords_h = torch.arange(-(window_size - 1), window_size, dtype=torch.float32).to(self.device)
        relative_coords_w = torch.arange(-(window_size - 1), window_size, dtype=torch.float32).to(self.device)
        relative_coords_grid = torch.meshgrid([relative_coords_h, relative_coords_w])
        relative_coords_table = torch.stack(relative_coords_grid).permute(1, 2, 0).contiguous().unsqueeze(0)
        relative_coords_table[:, :, :, 0] /= window_size - 1
        relative_coords_table[:, :, :, 1] /= window_size - 1
        relative_coords_table *= 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / torch.log2(torch.tensor(8)).to(self.device)
        self.register_buffer("relative_coords_table", relative_coords_table)

        # relative_position_index
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords_grid = torch.meshgrid([coords_h, coords_w])
        coords = torch.stack(coords_grid)
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_coords = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_coords)

        self.logits_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        q, k, v = self.to_qkv(x).reshape(B, N, 3, self.num_heads, -1).permute([2, 0, 3, 1, 4])
        attn = (torch.nn.functional.normalize(q, dim=-1) @ torch.nn.functional.normalize(k, dim=-1).transpose(-2, -1))
        logits_scale = torch.clamp(self.logits_scale, max=torch.log(torch.tensor(1. / 0.01).to(self.device))).exp()
        attn = attn * logits_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(self.window_size * self.window_size,
                                                             self.window_size * self.window_size, - 1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = (attn @ v).transpose(1, 2).reshape(B, N, C)
        attn = self.to_out(attn)

        return attn



class SwinTransformerBlock(nn.Module):
    def __init__(self, is_shifted: bool, window_size: int, dim: int, num_heads: int, res_scale_init_value: float,
                 input_resolution: list[int]):
        """
        is_shifted: indicate if this block is shifted or not
        window_size: window size of (s)wsa block
        dim: output dimension
        num_heads: number of attention head
        res_scale_init_value: residual scale init value
        input_resolution: the resolution of the input image
        """
        super().__init__()
        self.is_shifted = is_shifted
        self.window_size = window_size
        self.shift_size = window_size // 2
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.lpu = LPUModule(dim=dim)
        self.window_attn = WindowSelfAttentionModule(dim=dim, num_heads=num_heads, window_size=window_size)
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -window_size), slice(-window_size, self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size, self.shift_size), slice(-self.shift_size, None))

            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mB, mH, mW, mC = img_mask.shape
            mask_windows = img_mask.view(mB, mH // window_size, window_size, mW // window_size, window_size, mC)
            mask_windows = mask_windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, mC)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)
        self.norm1 = LayerNormProxy(dim, eps=8e-8)
        self.norm2 = LayerNormProxy(dim, eps=8e-8)
        self.mlp = MLP(in_channels=dim)
        self.res_scale1 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()
        self.res_scale2 = Scale(dim=dim, init_value=res_scale_init_value) if res_scale_init_value else nn.Identity()

    def forward(self, x):
        H, W = self.input_resolution
        B, C, H, W = x.shape
        x = self.lpu(x)
        x_temp = x
        x = x.flatten(2).transpose(1, 2)

        if self.is_shifted:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size, self.window_size, C)
        x = x.view(-1, self.window_size * self.window_size, C)
        x = self.window_attn(x, mask=self.attn_mask)
        x = x.view(-1, self.window_size, self.window_size, C)
        B = int(x.shape[0] / (H * W / self.window_size / self.window_size))
        x = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)

        if self.is_shifted:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x = x.transpose(1, 2).reshape(B, C, H, W)

        x = self.res_scale1(x_temp) + self.norm1(x)
        x = self.res_scale2(x) + self.norm2(self.mlp(x))

        return x


class RSIRWinAttnModule(nn.Module):
    def __init__(self, dim: int, window_size: int, mode: str, num_heads: int):
        super().__init__()
        self.window_size = window_size
        self.attn = WindowSelfAttentionModule(dim=dim, num_heads=num_heads, window_size=window_size)
        self.mode = mode

    def window_partition(self, x, window_size):
        B, C, H, W = x.shape
        x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
        window = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)
        return window

    def window_reverse(self, window, window_size, H, W):
        B = int(window.shape[0] / (H * W / window_size / window_size))
        x = window.view(B, H // window_size, W // window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1).flatten(1, 2)
        return x

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W
        device, dtype = x.device, x.dtype
        x = x.flatten(2).transpose(1, 2)
        if self.mode == "rs":
            sample_map = torch.rand(B, L, device=device)
        else:
            with torch.no_grad():
                sample_map = torch.mean(x, dim=2)
        ids_shuffle = torch.argsort(sample_map, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        x_shuffle = torch.gather(x, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, C))
        x_windows = self.window_partition(x_shuffle.view(B, C, H, W), self.window_size)
        attn_windows = self.attn(x_windows)
        x = self.window_reverse(attn_windows, self.window_size, H, W)
        x_restore = torch.gather(x, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, C))
        x = x.transpose(1, 2).reshape(B, C, H, W)
        return x


class RSIRWinModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_heads: int, window_size: int):
        super().__init__()
        #         self.rs_attn = RSIRWinAttnModule(dim=in_channels // 2, mode="ir", window_size=window_size, num_heads=num_heads//2)
        self.ir_attn = RSIRWinAttnModule(dim=in_channels // 1, mode="ir", window_size=window_size,
                                         num_heads=num_heads // 1)

    def forward(self, x):
        #         x1, x2 = torch.chunk(x, 2, 1)
        #         x = torch.cat([self.rs_attn(x1), self.ir_attn(x2)], 1)
        x = self.ir_attn(x)
        return x


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


class RSIRWinLayerModule(nn.Module):
    def __init__(self, dim: int, num_heads: int, scale_init_value: float):
        super().__init__()
        self.attn = RSIRWinModule(in_channels=dim, out_channels=None, num_heads=num_heads, window_size=7)
        self.conv_ffn = ConvFFNModule(dim)
        self.norm1 = nn.GroupNorm(32, dim)
        self.norm2 = nn.GroupNorm(32, dim)
        self.layer_scale_1 = Scale(dim, scale_init_value) if scale_init_value is not None else nn.Identity()
        self.layer_scale_2 = Scale(dim, scale_init_value) if scale_init_value is not None else nn.Identity()
        self.res_scale_1 = Scale(dim, scale_init_value) if scale_init_value is not None else nn.Identity()
        self.res_scale_2 = Scale(dim, scale_init_value) if scale_init_value is not None else nn.Identity()

    def forward(self, x):
        x = self.res_scale_1(x) + self.norm1(self.layer_scale_1(self.attn(x)))
        x = self.res_scale_2(x) + self.norm2(self.layer_scale_2(self.conv_ffn(x)))
        return x
