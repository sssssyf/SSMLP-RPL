import torch
import torch.nn as nn

from timm.models.layers import DropPath, trunc_normal_


t_stride = 1

model_path = {
    's_sip_nods_s4': '/mnt/WXRC0020/users/junhao.zhang/tmp/slowfast/tools/s_sip_nods.pth',
}
def conv_3xnxn(inp, oup, kernel_size=3, stride=3,padding=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, padding, padding))


def conv_1xnxn(inp, oup, kernel_size=3, stride=3,padding=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, padding, padding))


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Spa_FC(nn.Module):
    def __init__(self, dim, segment_dim=8,tmp=7, C=3,qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim

        self.tmp=tmp
        dim2=C*tmp
        self.mlp_h = nn.Linear(dim2, dim2, bias=qkv_bias)
        self.mlp_w = nn.Linear(dim2, dim2, bias=qkv_bias)
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)

        # init weight problem
        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, H, W, C = x.shape

        S = C // self.segment_dim
        tmp=self.tmp
        # H
        h = x.transpose(3,2).reshape(B, T,H*W//tmp,tmp, self.segment_dim, S).permute(0, 1, 2, 4, 3, 5).reshape(B, T,  H*W//tmp,self.segment_dim,tmp* S)
        h = self.mlp_h(h).reshape(B, T,  H*W//tmp,self.segment_dim, tmp,S).permute(0, 1, 2, 4, 3, 5).reshape(B, T, W, H, C).transpose(3,2)
        # W
        w = x.reshape(B, T, H* W//tmp,tmp, self.segment_dim, S).permute(0, 1, 2, 4, 3, 5).reshape(B, T,  H*W//tmp,self.segment_dim, tmp* S)
        w = self.mlp_w(w).reshape(B, T, H*W//tmp,self.segment_dim, tmp,S).permute(0, 1, 2, 4, 3, 5).reshape(B, T, H, W, C)
        # C
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 4, 1, 2, 3).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class Spe_FC(nn.Module):
    def __init__(self, dim,segment_dim,band,C ,qkv_bias=False, proj_drop=0.):
        super().__init__()

        self.segment_dim =segment_dim
        dim2 = band*C

        self.mlp_t = nn.Linear(dim2, dim2, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, T, H, W, C = x.shape

        S = C // self.segment_dim


        # T
        t = x.reshape(B, T, H, W, self.segment_dim, S).permute(0, 4, 2, 3, 1, 5).reshape(B, self.segment_dim, H, W, T * S)
        t = self.mlp_t(t).reshape(B, self.segment_dim, H, W, T, S).permute(0, 4, 2, 3, 1, 5).reshape(B, T, H, W, C)

        x = t

        x = self.proj(x)
        x = self.proj_drop(x)

        return x
class PermutatorBlock(nn.Module):
    def __init__(self, dim, segment_dim,tmp, band,C,mlp_ratio=4., qkv_bias=False,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.s_norm1 = norm_layer(dim)
        self.s_fc = Spe_FC(dim, segment_dim,band,C,qkv_bias=qkv_bias)
        self.fc = Spa_FC(dim, segment_dim=segment_dim, tmp=tmp, C=C,qkv_bias=qkv_bias)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        xs = x + self.s_fc(self.s_norm1(x))
        x = x + self.drop_path(self.fc(self.norm1(xs))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj1 = conv_3xnxn(in_chans, embed_dim//2, kernel_size=1, stride=1,padding=0)
        self.norm1= nn.BatchNorm3d(embed_dim//2)
        self.act=nn.GELU()
        self.proj2 = conv_1xnxn(embed_dim//2, embed_dim, kernel_size=3, stride=1,padding=1)
        self.norm2 = nn.BatchNorm3d(embed_dim)

    def forward(self, x):
        x = self.proj1(x)
        x= self.norm1(x)
        x=self.act(x)
        x = self.proj2(x)
        x = self.norm2(x)
        return x

class Downsample(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = conv_1xnxn(in_embed_dim, out_embed_dim, kernel_size=3, stride=2,padding=1)
        self.norm=nn.LayerNorm(out_embed_dim)

    def forward(self, x):
        x = x.permute(0, 4, 1, 2, 3)
        x = self.proj(x)  # B, C, T, H, W
        x = x.permute(0, 2, 3, 4, 1)
        x=self.norm(x)
        return x

class SSMLP(nn.Module):
    """ MorphMLP
    """

    def __init__(self,Patch, BAND, CLASSES_NUM,layers,embed_dims,segment_dim):
        super().__init__()
        global t_stride


        num_classes = CLASSES_NUM

        in_chans = 1
        layers = layers
        segment_dim = segment_dim
        mlp_ratios =3
        embed_dims =embed_dims

        tmp = Patch
        qkv_bias = True
        C=int(embed_dims/segment_dim)

        drop_path_rate =0.1
        norm_layer = nn.LayerNorm

        skip_lam = 1.0

        self.num_classes = num_classes

        self.patch_embed1 = PatchEmbed( in_chans=in_chans, embed_dim=embed_dims)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, layers)]  # stochastic depth decay rule
        # for item in dpr:
        #     print(item)

        # stage1
        self.blocks1 = nn.ModuleList([])
        for i in range(layers):
            self.blocks1.append(
                PermutatorBlock(embed_dims, segment_dim,tmp=tmp,band=BAND,C=C, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias, drop_path=dpr[i], skip_lam=skip_lam)
            )

        self.norm = norm_layer(embed_dims)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        # Classifier head
        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            if 't_fc.mlp_t.weight' in name:
                nn.init.constant_(p, 0)
            if 't_fc.mlp_t.bias' in name:
                nn.init.constant_(p, 0)
            if 't_fc.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_fc.proj.bias' in name:
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_pretrained_model(self, cfg):
        if cfg.MORPH.PRETRAIN_PATH:
            checkpoint = torch.load(cfg.MORPH.PRETRAIN_PATH, map_location='cpu')
            if self.num_classes != 1000:
                del checkpoint['head.weight']
                del checkpoint['head.bias']
            return checkpoint
        else:
            return None

    def forward_features(self, x):
        x=x.view(x.shape[0],1,x.shape[1],x.shape[2],x.shape[3])
        x = self.patch_embed1(x)
        # B,C,T,H,W -> B,T,H,W,C
        x = x.permute(0, 2, 3,4, 1)

        for blk in self.blocks1:
            x = blk(x)

        #x = x.permute(0, 4, 2, 3, 1)
        B, T,H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x,bool):

        x = self.forward_features(x)
        #x = self.avgpool(x)
        #x = self.norm(x.squeeze())
        x = self.norm(x)
        #x = torch.flatten(x, 1)

        return x.mean(1),self.head(x.mean(1))
        #return x, self.head(x)