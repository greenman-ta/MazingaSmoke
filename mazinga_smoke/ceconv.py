import torch
import torch.nn as nn

class ConvBNReLU(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None, g=1):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=k, stride=s, padding=p, groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(c_out)
        self.act  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConvBNReLU(nn.Module):
    def __init__(self, c, k=3):
        super().__init__()
        p = k // 2
        self.dw = ConvBNReLU(c, c, k=k, p=p, g=c)  # depthwise
        self.pw = ConvBNReLU(c, c, k=1, p=0, g=1)  # pointwise

    def forward(self, x):
        return self.pw(self.dw(x))


class ChannelAxisConv(nn.Module):
    """
    Channel-axis conv: Conv3d su [B,1,C,H,W] con kernel (k,1,1) lungo C
    Output: [B,C,H,W], poi BN2d(C) + ReLU
    """
    def __init__(self, C, k=3):
        super().__init__()
        assert k % 2 == 1, 
        self.conv3d = nn.Conv3d(
            1, 1,
            kernel_size=(k, 1, 1),
            padding=(k // 2, 0, 0),
            bias=False
        )
        self.bn  = nn.BatchNorm2d(C)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # x: [B,C,H,W] -> [B,1,C,H,W] -> conv3d -> [B,C,H,W]
        y = self.conv3d(x.unsqueeze(1)).squeeze(1)
        return self.act(self.bn(y))


class CEConv(nn.Module):
    """
    CEConv block:
      branches: DSConv k=5, DSConv k=3, DSConv k=1, ChAxis k=3, ChAxis k=5
      concat -> 1x1 fuse
      skip-connection + ReLU
    """
    def __init__(self, C, drop_p=0.0):
        super().__init__()
        self.b_ds5 = DSConvBNReLU(C, k=5)
        self.b_ds3 = DSConvBNReLU(C, k=3)
        self.b_ds1 = ConvBNReLU(C, C, k=1, p=0)

        self.b_ca3 = ChannelAxisConv(C, k=3)
        self.b_ca5 = ChannelAxisConv(C, k=5)

        # fuse: conv 1x1 per tornare a C canali
        self.fuse = nn.Sequential(
            nn.Conv2d(5 * C, C, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C)
        )

        self.drop = nn.Dropout2d(drop_p) if drop_p > 0 else nn.Identity()
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x):
        y = torch.cat([
            self.b_ds5(x),
            self.b_ds3(x),
            self.b_ds1(x),
            self.b_ca3(x),
            self.b_ca5(x),
        ], dim=1)                 # [B,5C,H,W]

        y = self.fuse(y)          # [B,C,H,W]
        y = self.drop(y)

        return self.out_act(x + y)  