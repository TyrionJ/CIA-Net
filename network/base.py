from torch import nn, cat

from network.nn import ConvDropoutNormReLU, SaF, SAMgr


class BaseNet(nn.Module):
    def __init__(self, domain_chns: list, out_chns: int, **kwargs):
        super().__init__()

        cia_heads = kwargs.get('cia_heads', 1)
        baf_heads = kwargs.get('baf_heads', 1)

        self.domains = len(domain_chns)
        in_chns = [0] + domain_chns
        self.in_group = [i if i == 0 else in_chns[i-1] + in_chns[i] for i in range(len(in_chns))]

        self.in_layers = nn.ModuleList([])
        self.domain_encoders = nn.ModuleList([])
        # channel per domain
        cpd = 16

        for _ in domain_chns:
            self.in_layers.append(ConvDropoutNormReLU(1, cpd, (3, 3, 3), (1, 1, 1)))

            self.domain_encoders.append(nn.ModuleList([
                nn.Sequential(
                    ConvDropoutNormReLU(cpd, cpd * 2, (3, 3, 3), (1, 1, 1)),
                    ConvDropoutNormReLU(cpd * 2, cpd * 2, (3, 3, 3), (1, 1, 1)),
                ),
                nn.Sequential(
                    ConvDropoutNormReLU(cpd * 2, cpd * 4, (3, 3, 3), (2, 2, 2)),
                    ConvDropoutNormReLU(cpd * 4, cpd * 4, (3, 3, 3), (1, 1, 1))
                ),
                nn.Sequential(
                    ConvDropoutNormReLU(cpd * 4, cpd * 8, (3, 3, 3), (2, 2, 2)),
                    ConvDropoutNormReLU(cpd * 8, cpd * 8, (3, 3, 3), (1, 1, 1))
                )
            ]))

        self.saf_module = SaF(cpd * 2 * self.domains, self.domains, baf_heads)
        self.sa_modules = nn.ModuleList([
            nn.Sequential(
                SAMgr(cpd * 16, cia_heads, only_sa=True),
                ConvDropoutNormReLU(cpd * 16, cpd * 16),
            ),
            nn.Sequential(
                SAMgr(cpd * 32, cia_heads, only_sa=True),
                ConvDropoutNormReLU(cpd * 32, cpd * 32, (3, 3, 3), (2, 2, 2)),
            )
        ])

        self.trans_convs = nn.ModuleList([
            nn.ConvTranspose3d(cpd * 32, cpd * 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(cpd * 16, cpd * 8, kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            nn.ConvTranspose3d(cpd * 8, cpd * 4, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        ])

        self.stages = nn.ModuleList([
            nn.Sequential(ConvDropoutNormReLU(cpd * 32, cpd * 16, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(cpd * 16, cpd * 16, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(cpd * 16, cpd * 8, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(cpd * 8, cpd * 8, (3, 3, 3), (1, 1, 1))),
            nn.Sequential(ConvDropoutNormReLU(cpd * 8, cpd * 4, (3, 3, 3), (1, 1, 1)),
                          ConvDropoutNormReLU(cpd * 4, cpd * 4, (3, 3, 3), (1, 1, 1)))
        ])

        self.seg_layers = nn.ModuleList([
            nn.Conv3d(cpd * 16, out_chns, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(cpd * 8, out_chns, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(cpd * 4, out_chns, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        ])
        self.sup_depth = 3
        self.deep_supervision = False

    def forward_frag(self, domain_encoders_out, baf_out):
        SA_in1 = cat([domain_encoders_out[i][-1] for i in range(self.domains)], dim=1)
        SA_out1 = self.sa_modules[0](SA_in1)
        SA_out2 = self.sa_modules[1](cat([SA_out1, SA_in1], dim=1))

        skips = [SA_out2, SA_out1] + baf_out[::-1]
        x = skips.pop(0)

        net_deep_segs = []
        zipped = zip(self.trans_convs, self.stages, self.seg_layers)
        for n, (trans_conv, stage, seg_layer) in enumerate(zipped):
            x = trans_conv(x)
            x = stage(cat([x, skips[n]], dim=1))
            net_deep_segs.append(seg_layer(x))

        return net_deep_segs

    def forward(self, x, **_):
        x = cat([x[:, n:n + 1] for n in self.in_group[:-1]], dim=1)
        domain_encoders_out = [[] for _ in range(self.domains)]

        for n, in_layer in enumerate(self.in_layers):
            group_in = in_layer(x[:, n:n+1])
            for domain_encoder in self.domain_encoders[n]:
                group_in = domain_encoder(group_in)
                domain_encoders_out[n].append(group_in)

        en_x = cat([domain_encoders_out[i][0] for i in range(self.domains)], dim=1)
        baf_out = []
        for encoder in self.saf_module.encoders[:2]:
            en_x = encoder(en_x)
            baf_out.append(en_x)
        baf_deep_segs = []

        net_deep_segs = self.forward_frag(domain_encoders_out, baf_out)
        return (net_deep_segs[::-1], baf_deep_segs) if self.deep_supervision else net_deep_segs[-1]


if __name__ == '__main__':
    import torch
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    data = torch.randn(1, 13, 96, 96, 64).to(device)

    net = BaseNet([5, 8], 116).to(device)
    net.deep_supervision = True
    net = net.to(device)
    net_segs, baf_segs = net(data)
    _ = 1
