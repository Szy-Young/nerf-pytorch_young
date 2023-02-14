import torch
import torch.nn as nn
import torch.nn.functional as F


class FourierEmbedding:
    def __init__(self,
                 n_freq=10,
                 log_sample=True,
                 input_dim=3,
                 include_input=True):
        self.embed_fns = []
        self.output_dim = 0

        # Identity mapping
        if include_input:
            self.embed_fns.append(lambda x: x)
            self.output_dim += input_dim

        # Fourier embedding
        if log_sample:
            freq_bands = 2.**torch.linspace(0., n_freq-1, steps=n_freq)
        else:
            freq_bands = torch.linspace(2.**0., 2.**(n_freq-1), steps=n_freq)
        for freq in freq_bands:
            self.embed_fns.append(lambda x, freq=freq : torch.sin(freq * x))
            self.embed_fns.append(lambda x, freq=freq : torch.cos(freq * x))
            self.output_dim += 2 * input_dim

    def __call__(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class NeRF(nn.Module):
    def __init__(self,
                 n_layer=8,
                 n_dim=256,
                 input_dim=3,
                 input_view_dim=3,
                 skips=[4],
                 use_viewdir=False,
                 rgb_act='sigmoid',
                 density_act='relu'):
        super().__init__()
        self.skips = skips
        self.use_viewdir = use_viewdir

        # Point -> feature
        self.point_fc = nn.ModuleList()
        self.point_fc.append(nn.Linear(input_dim, n_dim))
        for l in range(n_layer - 1):
            if l in skips:
                self.point_fc.append(nn.Linear(n_dim + input_dim, n_dim))
            else:
                self.point_fc.append(nn.Linear(n_dim, n_dim))

        if use_viewdir:
            self.view_fc = nn.ModuleList([nn.Linear(n_dim + input_view_dim, n_dim // 2)])
            self.feat_fc = nn.Linear(n_dim, n_dim)
            self.density_fc = nn.Linear(n_dim, 1)
            self.rgb_fc = nn.Linear(n_dim // 2, 3)
        else:
            self.density_fc = nn.Linear(n_dim, 1)
            self.rgb_fc = nn.Linear(n_dim, 3)

        # # Output branch for density
        # self.density_fc = nn.Linear(n_dim, 1)
        #
        # # Output branch for RGB color
        # if use_viewdir:
        #     self.view_fc = nn.ModuleList([nn.Linear(n_dim + input_view_dim, n_dim // 2)])
        #     self.feat_fc = nn.Linear(n_dim, n_dim)
        #     self.rgb_fc = nn.Linear(n_dim // 2, 3)
        # else:
        #     self.rgb_fc = nn.Linear(n_dim, 3)

        # Output activations
        act_fns = {'sigmoid': lambda x: torch.sigmoid(x),
                   'relu': lambda x: F.relu(x),
                   'softplus': lambda x: F.softplus(x),
                   'shifted_softplus': lambda x: F.softplus(x - 1)}
        self.rgb_act = act_fns[rgb_act]
        self.density_act = act_fns[density_act]


    def forward(self, point, view=None, density_noise_std=0.):
        """
        :param point: (N, C) torch.Tensor.
        :param view: (N, C) torch.Tensor, if provided.
        :param density_noise_std: Noise added to raw density output
        :return:
            rgb: (N, 3) torch.Tensor.
            density: (N,) torch.Tensor.
        """
        h = point
        # Point -> feature
        for l in range(len(self.point_fc)):
            h = self.point_fc[l](h)
            h = F.relu(h)
            if l in self.skips:
                h = torch.cat([point, h], 1)

        # Output branch for density
        density = self.density_fc(h)

        # Output branch for RGB color
        if self.use_viewdir:
            feat = self.feat_fc(h)
            h = torch.cat([feat, view], 1)
            for l in range(len(self.view_fc)):
                h = self.view_fc[l](h)
                h = F.relu(h)
            rgb = self.rgb_fc(h)
        else:
            rgb = self.rgb_fc(h)

        # Add noise to raw density output
        if density_noise_std > 0.:
            noise = density_noise_std * torch.randn(density.shape)
            density += noise

        # Output activations
        rgb = self.rgb_act(rgb)
        density = self.density_act(density)
        density = density.squeeze(1)

        return rgb, density


    # def forward(self, point, view=None, density_noise_std=0.):
    #     """
    #     :param point: (Nr, Np, C) torch.Tensor.
    #     :param view: (Nr, Np, C) torch.Tensor, if provided.
    #     :param density_noise_std: Noise added to raw density output
    #     :return:
    #         rgb: (Nr, Np, 3) torch.Tensor.
    #         density: (Nr, Np) torch.Tensor.
    #     """
    #     h = point
    #     # Point -> feature
    #     for l in range(len(self.point_fc)):
    #         h = self.point_fc[l](h)
    #         h = F.relu(h)
    #         if l in self.skips:
    #             h = torch.cat([point, h], -1)
    #
    #     # Output branch for density
    #     density = self.density_fc(h)
    #
    #     # Output branch for RGB color
    #     if self.use_viewdir:
    #         feat = self.feat_fc(h)
    #         h = torch.cat([feat, view], -1)
    #         for l in range(len(self.view_fc)):
    #             h = self.view_fc[l](h)
    #             h = F.relu(h)
    #         rgb = self.rgb_fc(h)
    #     else:
    #         rgb = self.rgb_fc(h)
    #
    #     # Add noise to raw density output
    #     if density_noise_std > 0.:
    #         noise = density_noise_std * torch.randn(density.shape)
    #         density += noise
    #
    #     # Output activations
    #     rgb = self.rgb_act(rgb)
    #     density = self.density_act(density)
    #     density = density.squeeze(2)
    #
    #     return rgb, density


if __name__ == '__main__':
    torch.manual_seed(0)
    point = torch.randn(8, 1024, 3)
    view = torch.randn(8, 1024, 3)

    point, view = point.reshape(-1, 3), view.reshape(-1, 3)
    point_embedding = FourierEmbedding(n_freq=10)
    point = point_embedding(point)
    view_embedding = FourierEmbedding(n_freq=4)
    view = view_embedding(view)

    nerf = NeRF(use_viewdir=True,
                input_dim=point_embedding.output_dim,
                input_view_dim=view_embedding.output_dim)
    rgb, density = nerf(point, view)
    print(rgb)
    print(rgb.shape)
    print('Number of parameters:', sum(p.numel() for p in nerf.parameters() if p.requires_grad))