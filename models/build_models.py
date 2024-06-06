import torch
import torch.nn as nn
from .build_architecture import StageModule, UperNetHead



class MSCAEfficientFormerV2(nn.Module):
    def __init__(self, img_size=224, num_classes=2, dim_list=[64, 128, 256, 512], num_layer_list=[2, 2, 6, 2],
                 res_scale_init_value_list=[1.0, 1.0, 1.0, 1.0], resolution_list=[56, 28, 14, 7],
                 kernel_size_list=[9, 7, 5, 3], stride_list=[7, 5, 3, 1], num_head_list=[2, 4, 8, 8],
                 num_group_list=[1, 2, 4, 8], expansion_ratio_list=[4, 4, 4, 4], sr_ratio_list=[8, 4, 1 ,1], **kwargs):
        """
        img_size: resolution of image;
        num_classes: number classes of datasets;
        dim_list: [dim, 2 * dim, 4 * dim, 8 * dim];
        resolution_list: [img_size // 4, img_size // 8, img_size // 16, img_size // 32];
        """
        super().__init__()
        self.stages = nn.ModuleList([])

        for i, \
        (dim, num_layers, res_scale_init_value, resolution, kernel_size, stride, num_heads, num_groups, expansion_ratio,
        sr_ratio) in enumerate(
                zip(dim_list, num_layer_list, res_scale_init_value_list, resolution_list, kernel_size_list, stride_list,
                    num_head_list, num_group_list, expansion_ratio_list, sr_ratio_list)):
            self.stages.append(
                StageModule(stage_index=i, dim=dim, num_layers=num_layers, res_scale_init_value=res_scale_init_value,
                            resolution=resolution, kernel_size=kernel_size, stride=stride, num_groups=num_groups,
                            num_heads=num_heads, expansion_ratio=expansion_ratio, sr_ratio=sr_ratio))

        self.decoder = UperNetHead(in_channels_list=dim_list, out_channels=dim_list[1],
                                   num_classes=num_classes, img_size=img_size)

    def forward(self, x):
        outputs = []
        #         c = None
        for stage in self.stages:
            x = stage(x)
            outputs.append(x)
        x = self.decoder(outputs)
        return x


# if __name__ == '__main__':
#     net = MSCAEfficientFormerV2()
#     # summary(net, input_size=(1, 3, 224, 224))
#     x = torch.randn((1, 3, 224, 224))
#     output = net(x)
#     print(output.size())
#     print(output.argmax(1).size())