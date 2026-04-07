import torch
import torch.nn as nn
import torch.nn.functional as F


class CEPool(nn.Module):
    def __init__(self, c_in, c_out=128, stride=2):
        super().__init__()
        
        # 1. Ramo Max-Pool 
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=stride, padding=1)
        
        # 2. Ramo Avg-Pool 
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)
        
        # 3. Ramo Spatial Depthwise 
        self.dw_conv_spatial = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride, padding=1, groups=c_in, bias=False),
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True)
        )

        # 4. Ramo Channel Pointwise 
        # CENet usa questo per pesare le feature prima di ridurle
        self.dw_conv_channel = nn.Sequential(
            nn.Conv2d(c_in, c_in, kernel_size=1, bias=False), # Mescola canali
            nn.BatchNorm2d(c_in),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) # Poi riduce
        )

        # Fusione dei 4 rami (4 * c_in -> c_out)
        self.fuse = nn.Sequential(
            nn.Conv2d(4 * c_in, c_out, kernel_size=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [N, c_in, He, We]
        
        x_max = self.max_pool(x)
        x_avg = self.avg_pool(x)
        x_spat = self.dw_conv_spatial(x)    
        x_chan = self.dw_conv_channel(x)    

        # Concatenazione
        x_cat = torch.cat([x_max, x_avg, x_spat, x_chan], dim=1)  # [N, 4*c_in, Hd, Wd]
        
        out = self.fuse(x_cat)
        return out