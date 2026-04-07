# backbone_ghostnet_features.py
import torch
import torch.nn as nn
import timm

class GhostV2Backbone(nn.Module):
    """
    GhostNetV2 backbone.
    Input:  x [B*T, 3, H, W]
    Output:
        f_int: [B, T, C_int, H_int, W_int]  )
        f_deep : [B, T, D,   H_deep, W_deep]  
    """
    def __init__(self,
                 pretrained: bool = True,
                 T: int = 36,
                 d_model: int = 128):
        super().__init__()
        self.T = T
        self.stage_int_idx = 2
        self.stage_deep_idx  = 4
        self.d_model = d_model

        # modello timm in modalità features_only
        self.backbone = timm.create_model(
            'ghostnetv2_100',
            pretrained=pretrained,
            features_only=True
        )

        #attributi da esporre
        info = self.backbone.feature_info
        self.C_int = info.channels()[self.stage_int_idx]   
        self.C_deep  = info.channels()[self.stage_deep_idx]    

        # proiezione del DEEP a D=128 
        self.proj_deep = nn.Conv2d(self.C_deep, d_model, kernel_size=1, bias=True)

        # FREEZE
        FREEZE_PREFIXES = ["conv_stem", "bn1", "blocks_0", "blocks_1"]

        frozen = 0
        for name, p in self.backbone.named_parameters():
            #print(name)
            if any(name.startswith(pref) for pref in FREEZE_PREFIXES):
                #print(f"Freezing {name}")
                p.requires_grad = False
                frozen += 1

        print(f"[-GhostV2Backbone] frozen n: {frozen}")
    
    #Da [B*T, C, H, W] a [B, T, C, H, W]
    def _reshape_bt(self, ft, B):
        BT, C, H, W = ft.shape
        return ft.view(B, self.T, C, H, W)
    
    def forward(self, x: torch.Tensor):

        #ottengo B
        B_T = x.shape[0]
        B = B_T // self.T
        
        #estraggo features
        feats = self.backbone(x)   

        # estraggo feature map intermedia e deep
        f_int_bt = feats[self.stage_int_idx]    # [B*T, C_int, H_int, W_int]
        f_deep_bt  = feats[self.stage_deep_idx]     # [B*T, C_deep, H_deep, W_deep]

        # proiezione deep a D
        f_deep_bt = self.proj_deep(f_deep_bt)       # [B*T, D, H_deep, W_deep]

        # reshaping in [B, T, C, H, W]
        f_int  = self._reshape_bt(f_int_bt,  B)
        f_deep = self._reshape_bt(f_deep_bt, B)

        return f_deep, f_int