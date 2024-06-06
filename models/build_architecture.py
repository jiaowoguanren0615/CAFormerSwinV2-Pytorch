import torch.nn as nn
import torch
from .Swin_blocks import RSIRWinLayerModule, RSIRWinModule, RSIRWinAttnModule, ConvFFNModule
from .GetingNet_blocks import GatingNetworkModule, GatingNetworkLayerModule, SpatialPriorModule
from .MSCA_blocks import DualPathAttentionDownSamplingModule, StemModule, MSCALayerModule, AgentAttnLayerModule, \
    DALayerModule, RevStemModule, DAModule



class ExtractorModule(nn.Module):
    def __init__(self, dim, resolution, num_heads, num_groups, kernel_size, stride):
        super().__init__()
        self.conv_ffn = ConvFFNModule(dim)
        self.c_attn = DAModule(in_channels=dim, out_channels=dim, resolution=resolution, num_heads=num_heads, num_groups=num_groups,
                               kernel_size=kernel_size, stride=stride)
        self.norm = nn.GroupNorm(32, dim)

    def forward(self, query, key, value):
        x = query + self.c_attn(self.norm(query), self.norm(key), self.norm(value))
        #         x = x + self.s_attn(self.norm(x), self.norm(x), self.norm(x))

        x = x + self.conv_ffn(self.norm(x))

        return x

class StageModule(nn.Module):
    def __init__(self, stage_index: int, dim: int, num_layers: int, res_scale_init_value: float, resolution: int,
                 num_groups: int, num_heads: int, kernel_size: int, stride: int, expansion_ratio: int, sr_ratio: int):
        super().__init__()
        #         self.extractor = ExtractorModule(dim=dim, resolution=resolution, num_heads=num_heads, num_groups=num_groups, kernel_size=kernel_size, stride=stride)
        if stage_index == 0:
            self.down = StemModule(dim)
        else:
            self.down = DualPathAttentionDownSamplingModule(dim=dim, resolution=resolution)
        self.pos_emb = nn.Parameter(torch.zeros(1, dim, resolution, resolution))
        if stage_index != 3:
            self.norm = nn.GroupNorm(32, dim)
        else:
            self.norm = nn.Identity()

        #         if stage_index == 0:
        #             self.c_down = SpatialPriorModule(3, dim)
        #         else:
        #             self.c_down = DualPathAttentionDownSamplingModule(dim=dim, resolution=resolution)

        self.layers = nn.ModuleList([])
        self.stage_index = stage_index
        if stage_index < 2:
            for i in range(num_layers):
                #                 self.layers.append(RSIRWinLayerModule(dim=dim, scale_init_value=res_scale_init_value, num_heads=num_heads))
                #                 if i < 2:
                #                     self.layers.append(GatingNetworkLayerModule(dim=dim, init_value=res_scale_init_value, expansion_ratio=expansion_ratio))
                #                 else:
                self.layers.append(MSCALayerModule(in_channels=dim, res_scale_init_value=res_scale_init_value,
                                                   expansion_ratio=expansion_ratio))
        #                 self.layers.append(ScatteringLayerModule(dim=dim, res_scale_init_value=res_scale_init_value))

        else:
            for _ in range(num_layers):
                #                 self.layers.append(DALayerModule(dim=dim, res_scale_init_value=res_scale_init_value, resolution=resolution, kernel_size=kernel_size, stride=stride, num_heads=num_heads, num_groups=num_groups))
                #                 self.layers.append(RSIRWinLayerModule(dim=dim, scale_init_value=res_scale_init_value, num_heads=num_heads))
                #                 self.layers.append(MSCALayerModule(in_channels=dim, res_scale_init_value=res_scale_init_value, expansion_ratio=expansion_ratio))
                #                 self.layers.append(MHALayerModule(dim=dim, res_scale_init_value=res_scale_init_value, resolution=resolution, num_heads=num_heads))
                self.layers.append(AgentAttnLayerModule(in_channels=dim, res_scale_init_value=res_scale_init_value,
                                                        num_heads=num_heads, agent_num=resolution ** 2,
                                                        sr_ratio=sr_ratio, num_patches=resolution ** 2))

    def forward(self, x):
        #         if c == None:
        #             c = self.c_down(x)
        #         else:
        #             c = self.c_down(c)
        x = self.down(x)
        x = x + self.pos_emb

        for layer in self.layers:
            if self.stage_index <= 2:
                x = layer(x)
            else:
                x = layer(x)
        #         c = self.extractor(c, x, x)
        x = self.norm(x)

        return x


class PPM(nn.Module):
    def __init__(self, dim: int, hid_dim: int, bins: list[int]):
        super().__init__()
        self.features = nn.ModuleList([])
        for bin in bins:
            self.features.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(bin),
                    nn.Conv2d(dim, hid_dim, kernel_size=1, bias=False),
                    nn.GroupNorm(32, hid_dim, eps=1e-5),
                    nn.Hardswish()
                )
            )
        self.last = nn.Conv2d(hid_dim * 8, hid_dim, kernel_size=1)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for feature in self.features:
            out.append(nn.functional.interpolate(feature(x), size=x_size[2:], mode="bilinear", align_corners=True))
        return self.last(torch.cat(out, 1))



class ConvModule(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, reduction_ratio: int = 1, kernel_size: int = 1,
                 stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=False)
        self.norm = nn.GroupNorm(32, out_channels, eps=1e-5)
        self.acti = nn.Hardswish()
        self.conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2,
                               bias=False, groups=out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,
                               bias=False, groups=out_channels)
        self.conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, bias=False,
                               groups=out_channels)

    def forward(self, x):
        x = self.acti(self.norm(self.conv(x)))
        x = self.conv1(x) + self.conv2(x) + self.conv3(x)
        return x


class PixelDecoderAttnModule(nn.Module):
    def __init__(self, num_layers: int, num_heads: int, num_groups: int, dim: int, kernel_size: int, stride: int):
        super().__init__()
        self.layers = nn.ModuleList([])
        for i in range(num_layers):
            self.layers.append(DALayerModule(dim=dim, res_scale_init_value=1.0, resolution=56, kernel_size=kernel_size, stride=stride, num_heads=num_heads, num_groups=num_groups))


class DecoderStageModule(nn.Module):
    def __init__(self, dim: int, resolution: int):
        super().__init__()
        temp = 224 // resolution
        self.conv1 = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, bias=False)
        #         self.conv3 = nn.Conv2d(in_channels=dim*2, out_channels=dim*2, kernel_size=5, padding=2, bias=False, groups=dim*2)
        #         self.up = DualPathAttentionUpSamplingModule(dim=dim, resolution=resolution, eps=1e-5, num_heads=8)
        self.up = nn.Upsample(scale_factor=2.0)
        self.proj = MSCALayerModule(in_channels=dim, out_channels=dim, expansion_ratio=4, res_scale_init_value=1.0)
        #         self.proj = RSIRWinLayerModule(dim=dim, scale_init_value=1.0, num_heads=8)
        self.conv4 = MSCALayerModule(in_channels=dim, out_channels=dim, expansion_ratio=4, res_scale_init_value=1.0)

    #         self.conv5 = ConvModule(in_channels=dim, out_channels=dim, reduction_ratio=temp)

    def forward(self, x_proj, x):
        x = self.conv1(x) + self.conv2(x)
        x_proj = self.proj(x_proj)
        x = self.up(x)
        x = self.conv4(x + x_proj)
        #         x = self.conv5(x)
        return x


class DownGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(DownGroup, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, inputs):
        x = self.pool(inputs)
        x = self.conv(x)

        return x


class UpGroup(nn.Module):
    def __init__(self, scale_factor, in_channels=320, out_channels=64):
        super(UpGroup, self).__init__()
        self.up = nn.Upsample(scale_factor=(scale_factor, scale_factor), mode='bilinear')
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1)

    def forward(self, inputs):
        x = self.up(inputs)
        x = self.conv(x)

        return x


class UnetDecoder(nn.Module):
    def __init__(self, dim_list: list[int] = [256, 128, 64], resolution_list=[14, 28, 56], num_classes=2, num_stages=4):
        super().__init__()
        self.num_stages = num_stages
        self.proj = MSCALayerModule(in_channels=512, out_channels=512, expansion_ratio=4, res_scale_init_value=1.0)
        #         self.proj = RSIRWinLayerModule(dim=512, scale_init_value=1.0, num_heads=8)
        self.stages = nn.ModuleList([])
        for i, (dim, resolution) in enumerate(zip(dim_list, resolution_list)):
            self.stages.append(DecoderStageModule(dim=dim, resolution=resolution))
        self.rev_stem = RevStemModule()

    def forward(self, x_encoder):
        x_encoder.reverse()
        features = [self.proj(x_encoder[0])]
        x = features[0]

        for i, x_encode in enumerate(x_encoder[1:]):
            x = self.stages[i](x_encode, x)

        return self.rev_stem(x)


class UperNetHead(nn.Module):
    def __init__(self, in_channels_list: list[int] = [64, 128, 256, 512], out_channels: int = 128,
                 bins: list[int] = [1, 2, 3, 6], num_classes=2, img_size=224):
        super().__init__()

        self.img_size = img_size
        self.ppm = PPM(dim=in_channels_list[-1], hid_dim=out_channels, bins=bins)
        self.fpn_in = nn.ModuleList([])
        self.fpn_out = nn.ModuleList([])


        temp = 2
        for in_channels in in_channels_list[:-1]:
            temp *= 2
            self.fpn_in.append(
                nn.Sequential(
                    ConvModule(in_channels=in_channels, out_channels=out_channels, reduction_ratio=temp),
                    #                     MSCALayerModule(in_channels=in_channels, out_channels=out_channels, expansion_ratio=6, res_scale_init_value=1.0)
                    #                     RSIRWinLayerModule(dim=out_channels, scale_init_value=1.0, num_heads=8),
                    #                     RSIRWinLayerModule(dim=out_channels, scale_init_value=1.0, num_heads=8)
                    #                     DALayerModule(dim=out_channels, res_scale_init_value=1.0, resolution=224//temp, kernel_size=7, stride=5, num_heads=8, num_groups=4)
                )
            )
            self.fpn_out.append(
                nn.Sequential(
                    ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,
                               reduction_ratio=temp),
                    #                 DALayerModule(dim=out_channels, res_scale_init_value=1.0, resolution=224//temp, kernel_size=7, stride=5, num_heads=8, num_groups=4)
                    #                 MSCALayerModule(in_channels=out_channels, out_channels=out_channels, expansion_ratio=6, res_scale_init_value=1.0)
                    #                 RSIRWinLayerModule(dim=out_channels, scale_init_value=1.0, num_heads=8)
                )
            )

        self.bottle_neck = nn.Sequential(
            ConvModule(in_channels=len(in_channels_list) * out_channels, out_channels=out_channels, kernel_size=3,
                       padding=1, reduction_ratio=4),
            MSCALayerModule(in_channels=out_channels, out_channels=out_channels, expansion_ratio=4,
                            res_scale_init_value=1.0)
        )

        self.last1 = nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)

    #         self.last2 = nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)
    #         self.last3 = nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)
    #         self.last4 = nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)
    #         self.last5 = nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)
    #         self.last_layer_list = nn.ModuleList([self.last2, self.last3, self.last4, self.last5])

    def forward(self, x):
        f = self.ppm(x[-1])
        fpn_features = [f]

        for i in reversed(range(len(x) - 1)):
            feature = self.fpn_in[i](x[i])
            f = feature + nn.functional.interpolate(f, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features[-1] = nn.functional.interpolate(fpn_features[i], size=(self.img_size, self.img_size), mode="bilinear",
                                                     align_corners=False)
        fpn_features.reverse()
        for i in range(1, len(x)):
            fpn_features[i] = nn.functional.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:],
                                                        mode="bilinear", align_corners=False)

        x = self.bottle_neck(torch.cat(fpn_features, dim=1))
        x = self.last1(x)
        #         for i in range(len(fpn_features)):
        #             fpn_features[i] = self.last_layer_list[i](fpn_features[i])

        #         fpn_features.append(x)
        return x


class PixelDecoderModule(nn.Module):
    def __init__(self, in_channels_list: list[int] = [64, 128, 256, 512], out_channels: int = 128,
                 bins: list[int] = [1, 2, 3, 6], num_classes=2):
        super().__init__()
        self.ppm = PPM(dim=in_channels_list[-1], hid_dim=out_channels, bins=bins)
        self.fpn_in = nn.ModuleList([])
        self.fpn_out = nn.ModuleList([])
        self.input_projs = nn.ModuleList([])

        temp = 2
        for in_channels in in_channels_list[:-1]:
            temp *= 2
            self.input_projs.append(
                ConvModule(in_channels=in_channels, out_channels=out_channels, reduction_ratio=temp))
            self.fpn_in.append(ConvModule(in_channels=in_channels, out_channels=out_channels, reduction_ratio=temp))
            self.fpn_out.append(
                ConvModule(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,
                           reduction_ratio=temp))

        self.bottle_neck = ConvModule(in_channels=len(in_channels_list) * out_channels, out_channels=out_channels,
                                      kernel_size=3, padding=1, reduction_ratio=4)

        self.last1 = nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)

    #         self.last2 = nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)
    #         self.last3 = nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)
    #         self.last4 = nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)
    #         self.last5 = nn.Conv2d(in_channels=out_channels, out_channels=num_classes, kernel_size=1)
    #         self.last_layer_list = nn.ModuleList([self.last2, self.last3, self.last4, self.last5])

    def forward(self, x):
        attn_src = []
        for i in reversed(range(len(x) - 1)):
            attn_src.append(input)
        f = self.ppm(x[-1])
        fpn_features = [f]

        for i in reversed(range(len(x) - 1)):
            feature = self.fpn_in[i](x[i])
            f = feature + nn.functional.interpolate(f, size=feature.shape[-2:], mode="bilinear", align_corners=False)
            fpn_features.append(self.fpn_out[i](f))

        fpn_features.reverse()
        for i in range(1, len(x)):
            fpn_features[i] = nn.functional.interpolate(fpn_features[i], size=fpn_features[0].shape[-2:],
                                                        mode="bilinear", align_corners=False)

        x = self.bottle_neck(torch.cat(fpn_features, dim=1))
        x = self.last1(x)
        #         for i in range(len(fpn_features)):
        #             fpn_features[i] = self.last_layer_list[i](fpn_features[i])

        #         fpn_features.append(x)
        return x