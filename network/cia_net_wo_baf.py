from torch import nn, cat

from network.nn import ConvDropoutNormReLU, SaF, CIA, MLPwithConv


class CIANetWoBAF(nn.Module):
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

        for in_chns in domain_chns:
            self.in_layers.append(ConvDropoutNormReLU(in_chns, cpd, (3, 3, 3), (1, 1, 1)))

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
        self.cia_modules = nn.ModuleList([
            CIA(cpd * 8, self.domains, cia_heads),
            CIA(cpd * 16, self.domains, cia_heads)
        ])
        self.cia_mlps = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    MLPwithConv(cpd * 8 * self.domains, cpd * 8),
                    ConvDropoutNormReLU(cpd * 8, cpd * 8),
                ),
                nn.Sequential(
                    MLPwithConv(cpd * 16, cpd * 16),
                    ConvDropoutNormReLU(cpd * 16, cpd * 16),
                )
            ]),
            nn.ModuleList([
                nn.Sequential(
                    MLPwithConv(cpd * 16 * self.domains, cpd * 16),
                    ConvDropoutNormReLU(cpd * 16, cpd * 16),
                ),
                nn.Sequential(
                    MLPwithConv(cpd * 32, cpd * 32),
                    ConvDropoutNormReLU(cpd * 32, cpd * 32, stride=(2, 2, 2)),
                )
            ])
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
        cia_in1 = [domain_encoders_out[i][-1] for i in range(self.domains)]
        IAs1, CA1 = self.cia_modules[0](cia_in1)
        CIA_out1 = self.cia_mlps[0][0](cat(IAs1, dim=1))
        CIA_out1 = self.cia_mlps[0][1](cat([CIA_out1, CA1], dim=1))

        cia_in2 = [cat([IA, CA1], dim=1) for IA in IAs1]
        IAs2, CA2 = self.cia_modules[1](cia_in2)
        CIA_out2 = self.cia_mlps[1][0](cat(IAs2, dim=1))
        CIA_out2 = self.cia_mlps[1][1](cat([CIA_out2, CA2], dim=1))

        skips = [CIA_out2, CIA_out1] + baf_out[::-1]
        x = skips.pop(0)

        net_deep_segs = []
        zipped = zip(self.trans_convs, self.stages, self.seg_layers)
        for n, (trans_conv, stage, seg_layer) in enumerate(zipped):
            x = trans_conv(x)
            x = stage(cat([x, skips[n]], dim=1))
            net_deep_segs.append(seg_layer(x))

        return net_deep_segs

    def forward(self, x, **_):
        domain_encoders_out = [[] for _ in range(self.domains)]

        for n, in_layer in enumerate(self.in_layers):
            group_in = in_layer(x[:, self.in_group[n]:self.in_group[n+1]])
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = torch.randn(1, 13, 144, 144, 80).to(device)

    net = CIANetWoBAF([5, 8], 116).to(device)
    net.deep_supervision = True
    net = net.to(device)
    net_segs, baf_segs = net(data, epoch=30)
    _ = 1
