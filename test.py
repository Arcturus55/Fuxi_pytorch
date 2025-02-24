import torch
from src.fuxi_net import FuXiNet

depths = 2
in_channels = 96
out_channels = 192
h_size = 720
w_size = 1440
level_feature_size = 5
pressure_level_num = 13
surface_feature_size = 4
batch_size = 1
feature_num = 69
kernel_size = (2, 4, 4)

x = torch.rand(batch_size, h_size * w_size, feature_num, dtype=torch.float32).to('cuda')
fuxi_model = FuXiNet(depths=depths,
                     in_channels=in_channels,
                     out_channels=out_channels,
                     h_size=h_size,
                     w_size=w_size,
                     level_feature_size=level_feature_size,
                     pressure_level_num=pressure_level_num,
                     surface_feature_size=surface_feature_size,
                     kernel_size=kernel_size).to('cuda')

output, output_surface = fuxi_model(x)
print(output.shape, output_surface.shape)
