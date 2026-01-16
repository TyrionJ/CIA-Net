import torch
from sympy.polys import domains
from torch import nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


def min_max_norm(x):
    for i in range(x.shape[0]):
        x[i] = (x[i] - x[i].min()) / (x[i].max() - x[i].min())
    return x


class ConvDropoutNormReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), groups=1):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, groups=groups)
        self.dropout = nn.Dropout(p=0.1)
        self.norm = nn.InstanceNorm3d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.nonlin = nn.LeakyReLU(negative_slope=0.01, inplace=True)

    def forward(self, x):
        return self.nonlin(self.norm(self.dropout(self.conv(x))))


class MLPwithConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLPwithConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class SAMgr(nn.Module):
    def __init__(self, dim_size, num_heads, qkv_bias: bool = False, only_sa = False):
        super().__init__()

        assert dim_size % num_heads == 0, 'hidden_size must be divisible by num_heads'

        self.num_heads = num_heads
        self.out_proj = nn.Linear(dim_size, dim_size)
        self.qkv = nn.Linear(dim_size, dim_size * 3, bias=qkv_bias)
        self.only_sa = only_sa
        self.input_rearrange = Rearrange("b h (qkv l d) -> qkv b l h d", qkv=3, l=num_heads)
        self.out_rearrange = Rearrange("b h l d -> b l (h d)")
        self.head_dim = dim_size // num_heads
        self.scale = torch.tensor(self.head_dim ** -0.5)
        self.sx, self.sy, self.sz = None, None, None

    def get_QKV(self, x):
        self.sx, self.sy, self.sz = x.shape[2:]
        x = Rearrange("b c x y z -> b (x y z) c")(x)
        output = self.input_rearrange(self.qkv(x))
        Q, K, V = output[0], output[1], output[2]

        return Q, K, V

    @staticmethod
    def cal_attn(Q, K, V, shape, att_weight=None):
        assert V is not None, 'V must be provided'
        if att_weight is None:
            assert Q is not None and K is not None, 'Q and K must be provided'
            scale = torch.tensor(Q.shape[-1] ** -0.5)
            att_weight = (torch.einsum("blxd,blyd->blxy", Q, K) * scale).softmax(dim=-1)

        SA = torch.einsum("bhxy,bhyd->bhxd", att_weight, V)
        SA = Rearrange("b h l d -> b l (h d)")(SA)
        SA = Rearrange("b (x y z) c -> b c x y z", x=shape[0], y=shape[1])(SA)
        return SA

    def forward(self, x):
        if isinstance(x, tuple):
            Q, K, V = x
        else:
            Q, K, V = self.get_QKV(x)
        att_mat = (torch.einsum("blxd,blyd->blxy", Q, K) * self.scale).softmax(dim=-1)
        SA = torch.einsum("bhxy,bhyd->bhxd", att_mat, V)
        SA = self.out_rearrange(SA)
        SA = self.out_proj(SA)
        SA = Rearrange("b (x y z) c -> b c x y z", x=self.sx, y=self.sy)(SA)

        att_weight = Rearrange("b h (x y z) -> b h x y z", x=self.sx, y=self.sy)(att_mat.detach().mean(dim=2))
        att_weight = min_max_norm(att_weight)

        return SA if self.only_sa else (SA, att_mat, att_weight)


class SaF(nn.Module):
    def __init__(self, in_chns, modal_num, head_num):
        super().__init__()

        self.encoders = nn.ModuleList(
            [
                nn.Sequential(
                    ConvDropoutNormReLU(in_chns, 64, (3, 3, 3), (1, 1, 1)),
                    ConvDropoutNormReLU(64, 64, (3, 3, 3), (1, 1, 1)),
                ),
                nn.Sequential(
                    ConvDropoutNormReLU(64, 128, (3, 3, 3), (2, 2, 2)),
                    ConvDropoutNormReLU(128, 128, (3, 3, 3), (1, 1, 1))
                ),
                nn.Sequential(
                    ConvDropoutNormReLU(128, 256, (3, 3, 3), (2, 2, 2)),
                    ConvDropoutNormReLU(256, 256, (3, 3, 3), (1, 1, 1))
                )
            ]
        )
        self.sa_mgr = SAMgr(256, head_num)
        self.skip_convs = nn.ModuleList([
            ConvDropoutNormReLU(modal_num, 128, (3, 3, 3), (1, 1, 1)),
            ConvDropoutNormReLU(modal_num, 64, (3, 3, 3), (1, 1, 1))
        ])
        self.main_ca_mgr = SAMgr(128, 1)
        self.sub_ca_mgr = SAMgr(128, 1)

        self.trans_convs = nn.ModuleList([
            nn.ConvTranspose3d(256, 128, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        ])

        self.stages = nn.ModuleList([
            nn.Sequential(ConvDropoutNormReLU(256, 128, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(128, 128, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(128, 64, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(64, 64, (3, 3, 3), (1, 1, 1)))
        ])

        self.seg_layers = nn.ModuleList([
            nn.Conv3d(128, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(64, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        ])

    def forward(self, x, net_in):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
        x, _, attn = self.sa_mgr(x)

        s_in, x_a = net_in.shape[2:], attn.shape[2:]
        zipped = zip(self.trans_convs, self.stages, self.seg_layers, self.skip_convs)
        out_skips, deep_segs = [], []
        for n, (trans_conv, stage, seg_layer, skip_conv) in enumerate(zipped):
            x = trans_conv(x)
            s_sk = x.shape[2:]

            sf1 = [k / j for j, k in zip(s_in, s_sk)]
            sf2 = [k / j for j, k in zip(x_a, s_sk)]
            _x = F.interpolate(net_in, scale_factor=sf1, mode='trilinear')
            _a = F.interpolate(attn, scale_factor=sf2, mode='trilinear')

            fused = skip_conv(_x * _a)
            x = stage(torch.concatenate([x, skips[-(n + 2)] + fused], dim=1))

            out_skips.append(x)
            deep_segs.append(seg_layer(x))

        # q1, k1, v1 = self.main_ca_mgr.get_QKV(out_skips[0])
        # q2, k2, v2 = self.sub_ca_mgr.get_QKV(deep_segs[0].repeat(1, 128, 1, 1, 1))
        # out_skips[0] = self.main_ca_mgr.pure_cal_attn(q1, k2, v2)
        out_skips[0] = out_skips[0] + out_skips[0] * torch.sigmoid(deep_segs[0])
        out_skips[1] = out_skips[1] + out_skips[1] * torch.sigmoid(deep_segs[1])

        return out_skips[::-1], deep_segs[::-1]


class CIA(nn.Module):
    def __init__(self, in_chns, domains, num_heads):
        super().__init__()

        self.sa_mgrs = nn.ModuleList([SAMgr(in_chns, num_heads) for _ in range(domains)])

    def forward(self, in_domains: list[torch.tensor]):
        IAs, CA= [], None
        prodA, sumV = 1, 0
        for sa_mgr, x_domain in zip(self.sa_mgrs, in_domains):
            Q, K, V = sa_mgr.get_QKV(x_domain)
            sa, attn_mat, _ = sa_mgr((Q, K, V))
            prodA *= attn_mat
            IAs.append(sa)
            sumV += V
        prodA = torch.softmax(prodA, dim=1)
        sumV /= len(in_domains)
        CA = SAMgr.cal_attn(Q=None, K=None, V=sumV, shape=in_domains[0].shape[2:], att_weight=prodA)
        IAs = [IA + domain for IA, domain in zip(IAs, in_domains)]

        return IAs, CA


class PositionConv(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super().__init__()
        self.pos_to_conv = nn.Linear(in_dim, out_dim * kernel_size ** 2)
        self.conv = nn.Conv3d(in_dim, out_dim, kernel_size, padding=1)

    def forward(self, x, pos_emb):
        dynamic_weight = self.pos_to_conv(pos_emb).view(
            x.size(0), self.conv.out_channels,
            self.conv.in_channels, *self.conv.kernel_size
        )
        return F.conv3d(x, dynamic_weight, padding=1)
