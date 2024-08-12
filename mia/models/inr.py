import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from einops import rearrange, einsum
from math import sqrt

from mia.models.layers import (
    Sine,
    FixedFourierFeatures,
    RandomFourierFeatures,
)

from mia.ops.grid_sample_gradfix import grid_sample


class SirenLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=1.0,
        w0_learnable=False,
        use_bias=True,
        c=6.0,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out

        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)

        # initialize layers following SIREN paper
        w_std = sqrt(c / dim_in) / w0
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)

        self.activation = Sine(w0, w0_learnable)

    def forward(self, x, scale=1.0, shift=0.0):
        return self.activation(scale * self.linear(x) + shift)

    def forward_lowrank_gate(self, x, u_mat, v_mat):
        """lowrank gate modulation

        Args:
            x (torch.Tensor): input [batch_size, ..., dim_in]
            u_mat (torch.Tensor): U matrix [batch_size, rank, dim_out]
            v_mat (torch.Tensor): V matrix [batch_size, rank, dim_in]

        Returns:
            torch.Tensor: activation((sigmoid(UV) * W)x + b) [batch_size, ..., dim_out]
        """
        rank = u_mat.shape[1]
        G = einsum(u_mat, v_mat, "b r do, b r di -> b do di") / sqrt(rank)
        G = torch.sigmoid(G)

        modulated_weight = G * self.linear.weight.unsqueeze(0)
        bias = 0.0 if self.linear.bias is None else self.linear.bias
        out = einsum(modulated_weight, x, "b o i, b ... i -> b ... o")
        out = out + bias
        return self.activation(out)


class SirenLayerFirst(SirenLayer):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        w0_learnable=False,
        use_bias=True,
    ):
        super().__init__(
            dim_in,
            dim_out,
            w0,
            w0_learnable,
            use_bias,
        )

        # initialize layers following SIREN paper
        w_std = 1 / dim_in
        nn.init.uniform_(self.linear.weight, -w_std, w_std)
        if use_bias:
            nn.init.uniform_(self.linear.bias, -w_std, w_std)


class INRLayerLast(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        use_bias=True,
        out_bias=0.0,
    ):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        self.out_bias = out_bias

    def forward(self, x):
        return self.linear(x) + self.out_bias


class INRLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        use_bias=True,
    ):
        super().__init__()
        self.linear = nn.Linear(dim_in, dim_out, bias=use_bias)
        self.activation = nn.ReLU()

    def forward(self, x, scale=1.0, shift=0.0):
        return self.activation(scale * self.linear(x) + shift)

    def forward_lowrank_gate(self, x, u_mat, v_mat):
        """lowrank gate modulation

        Args:
            x (torch.Tensor): input [batch_size, ..., dim_in]
            u_mat (torch.Tensor): U matrix [batch_size, rank, dim_out]
            v_mat (torch.Tensor): V matrix [batch_size, rank, dim_in]

        Returns:
            torch.Tensor: activation((sigmoid(UV) * W)x + b) [batch_size, ..., dim_out]
        """
        rank = u_mat.shape[1]
        G = einsum(u_mat, v_mat, "b r do, b r di -> b do di") / sqrt(rank)
        G = torch.sigmoid(G)

        modulated_weight = G * self.linear.weight.unsqueeze(0)
        bias = 0.0 if self.linear.bias is None else self.linear.bias
        out = einsum(modulated_weight, x, "b o i, b ... i -> b ... o")
        out = out + bias
        return self.activation(out)


class LowRankINRLayer(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        rank=1,
        use_bias=True,
    ):
        super().__init__()
        self.dim_in = dim_in
        self.linear = nn.Linear(rank, dim_out, bias=use_bias)
        self.activation = nn.ReLU()

    def forward(self, x, v_mat):
        weight = einsum(self.linear.weight, v_mat, "do r, b r di -> b do di")
        x = einsum(weight, x, "b do di, b n di -> b n do")
        return self.activation(x)


class FourierFeatureINRLayerFirst(INRLayer):
    def __init__(
        self,
        dim_in,
        dim_out,
        use_bias=True,
        ff_dim=128,
    ):
        super().__init__(
            dim_in,
            dim_out,
            use_bias,
        )

        assert ff_dim % 2 == 0 and ff_dim > 0

        self.linear = nn.Linear(dim_in * ff_dim // 2, dim_out, bias=use_bias)
        self.convert_pos_emb = FixedFourierFeatures(ff_dim // 2)

    def forward(self, x, scale=1.0, shift=0.0):
        x = self.convert_pos_emb(x)
        return self.activation(scale * self.linear(x) + shift)


class RandomFourierFeatureINRLayerFirst(INRLayer):
    def __init__(
        self,
        dim_in,
        dim_out,
        use_bias=True,
        ff_dim=128,
        sigma=10,
    ):
        super().__init__(
            dim_in,
            dim_out,
            use_bias,
        )

        assert ff_dim % 2 == 0 and ff_dim > 0

        self.linear = nn.Linear(ff_dim, dim_out, bias=use_bias)
        self.convert_pos_emb = RandomFourierFeatures(ff_dim, dim_in, sigma)

    def forward(self, x, scale=1.0, shift=0.0):
        x = self.convert_pos_emb(x)
        return self.activation(scale * self.linear(x) + shift)


class INRModule(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden,
        num_layers,
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        self.dim_in = dim_in
        self.dim_out = dim_out

        self.encoder = None
        self.body = nn.ModuleList()

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.body:
            x = layer(x)

        x = self.decoder(x)
        return x

    def shift_modulated_forward(self, x0, modulations, modulate_first=False):
        if modulations is None:
            return self(x0)

        """
        Shape of x0
        1D - (bsz, num_points, 1)
        2D - (bsz, num_points, 2)
        3D - (bsz, num_points, 3)

        Shpae of modulations
        1D - (bsz, nl * dim_hidden, mss)
        2D - (bsz, nl * dim_hidden, mss, mss)
        3D - (bsz, nl * dim_hidden, mss, mss, mss)
        """

        Dx = x0.shape[-1]
        bsz, D, mss = modulations.shape[:3]
        nl = D // self.dim_hidden

        l = 0
        if mss == 1:
            # no spatial functa
            shifts = rearrange(modulations, "b (nl dim) ... -> nl b 1 (dim ...)", nl=nl)  # (nl, bsz, 1, dim_hidden)

        else:
            # spatial functa
            shifts = grid_sample(modulations, x0)  # (bsz, num_points, nl * dim_hidden)
            shifts = rearrange(shifts, "b n (nl dim) -> nl b n dim", nl=nl)  # (nl, bsz, num_points, dim_hidden)

        if modulate_first:
            x = self.encoder(x0, shift=shifts[l])
            l += 1
        else:
            x = self.encoder(x0)

        for layer in self.body:
            if l < nl:
                x = layer(x, shift=shifts[l])
                l += 1
            else:
                x = layer(x)

        x = self.decoder(x)
        return x

    def scale_modulated_forward(self, x0, modulations, modulate_first=False):
        if modulations is None:
            return self(x0)

        """
        Shape of x0
        1D - (bsz, num_points, 1)
        2D - (bsz, num_points, 2)
        3D - (bsz, num_points, 3)

        Shpae of modulations
        1D - (bsz, nl * dim_hidden, mss)
        2D - (bsz, nl * dim_hidden, mss, mss)
        3D - (bsz, nl * dim_hidden, mss, mss, mss)
        """

        Dx = x0.shape[-1]
        bsz, D, mss = modulations.shape[:3]
        nl = D // self.dim_hidden

        l = 0
        if mss == 1:
            # no spatial functa
            scales = rearrange(modulations, "b (nl dim) ... -> nl b 1 (dim ...)", nl=nl)  # (nl, bsz, 1, dim_hidden)

        else:
            # spatial functa
            scales = grid_sample(modulations, x0)  # (bsz, num_points, nl * dim_hidden)
            scales = rearrange(scales, "b n (nl dim) -> nl b n dim", nl=nl)  # (nl, bsz, num_points, dim_hidden)

        if modulate_first:
            x = self.encoder(x0, scale=scales[l])
            l += 1
        else:
            x = self.encoder(x0)

        for layer in self.body:
            if l < nl:
                x = layer(x, scale=scales[l])
                l += 1
            else:
                x = layer(x)

        x = self.decoder(x)
        return x

    def scaleshift_modulated_forward(self, x0, modulations, modulate_first=False):
        if modulations is None:
            return self(x0)

        """
        Shape of x0
        1D - (bsz, num_points, 1)
        2D - (bsz, num_points, 2)
        3D - (bsz, num_points, 3)

        Shpae of modulations
        1D - (bsz, nl * dim_hidden, mss)
        2D - (bsz, nl * dim_hidden, mss, mss)
        3D - (bsz, nl * dim_hidden, mss, mss, mss)
        """

        Dx = x0.shape[-1]
        bsz, D, mss = modulations.shape[:3]
        nl = D // (self.dim_hidden * 2)

        l = 0
        if mss == 1:
            # no spatial functa
            scaleshifts = rearrange(
                modulations, "b (nl p dim) ... -> nl p b 1 (dim ...)", nl=nl, p=2
            )  # (nl, 2, bsz, 1, dim_hidden)

        else:
            # spatial functa
            scaleshifts = grid_sample(modulations, x0)  # (bsz, num_points, nl * dim_hidden)
            scaleshifts = rearrange(
                scaleshifts, "b n (nl p dim) -> nl p b n dim", nl=nl, p=2
            )  # (nl, p, bsz, num_points, dim_hidden)

        if modulate_first:
            x = self.encoder(x0, scale=scaleshifts[l, 0], shift=scaleshifts[l, 1])
            l += 1
        else:
            x = self.encoder(x0)

        for layer in self.body:
            if l < nl:
                x = layer(x, scale=scaleshifts[l, 0], shift=scaleshifts[l, 1])
                l += 1
            else:
                x = layer(x)

        x = self.decoder(x)
        return x


class BasicINR(INRModule):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden,
        num_layers,
        use_bias=True,
        out_bias=0.0,
    ):
        super().__init__(
            dim_in,
            dim_out,
            dim_hidden,
            num_layers,
        )

        self.encoder = INRLayer(
            dim_in=self.dim_in,
            dim_out=dim_hidden,
            use_bias=use_bias,
        )

        for _ in range(num_layers - 2):
            self.body.append(
                INRLayer(
                    dim_in=dim_hidden,
                    dim_out=dim_hidden,
                    use_bias=use_bias,
                )
            )

        self.decoder = INRLayerLast(
            dim_in=dim_hidden,
            dim_out=self.dim_out,
            use_bias=use_bias,
            out_bias=out_bias,
        )


class Siren(INRModule):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden,
        num_layers,
        w0=30.0,
        w0_initial=30.0,
        use_bias=True,
        out_bias=0.0,
    ):
        super().__init__(dim_in, dim_out, dim_hidden, num_layers)

        self.encoder = SirenLayerFirst(
            dim_in=self.dim_in,
            dim_out=dim_hidden,
            w0=w0_initial,
            w0_learnable=False,
            use_bias=use_bias,
        )

        for _ in range(num_layers - 2):
            self.body.append(
                SirenLayer(
                    dim_in=dim_hidden,
                    dim_out=dim_hidden,
                    w0=w0,
                    w0_learnable=False,
                    use_bias=use_bias,
                )
            )

        self.decoder = INRLayerLast(
            dim_in=dim_hidden,
            dim_out=self.dim_out,
            use_bias=use_bias,
            out_bias=out_bias,
        )


class FourierFeatureINR(INRModule):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden,
        num_layers,
        ff_dim,
        use_bias=True,
        out_bias=0.0,
    ):
        super().__init__(
            dim_in,
            dim_out,
            dim_hidden,
            num_layers,
        )

        self.encoder = FourierFeatureINRLayerFirst(
            dim_in=self.dim_in,
            dim_out=dim_hidden,
            use_bias=use_bias,
            ff_dim=ff_dim,
        )

        for _ in range(num_layers - 2):
            self.body.append(
                INRLayer(
                    dim_in=dim_hidden,
                    dim_out=dim_hidden,
                    use_bias=use_bias,
                )
            )

        self.decoder = INRLayerLast(
            dim_in=dim_hidden,
            dim_out=self.dim_out,
            use_bias=use_bias,
            out_bias=out_bias,
        )


class RandomFourierFeatureINR(INRModule):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden,
        num_layers,
        ff_dim,
        sigma=10.0,
        use_bias=True,
        out_bias=0.0,
    ):
        super().__init__(
            dim_in,
            dim_out,
            dim_hidden,
            num_layers,
        )

        self.encoder = RandomFourierFeatureINRLayerFirst(
            dim_in=self.dim_in,
            dim_out=dim_hidden,
            use_bias=use_bias,
            ff_dim=ff_dim,
            sigma=sigma,
        )

        for _ in range(num_layers - 2):
            self.body.append(
                INRLayer(
                    dim_in=dim_hidden,
                    dim_out=dim_hidden,
                    use_bias=use_bias,
                )
            )

        self.decoder = INRLayerLast(
            dim_in=dim_hidden,
            dim_out=self.dim_out,
            use_bias=use_bias,
            out_bias=out_bias,
        )


class LowRankINR(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        dim_hidden,
        num_layers,
        rank,
        ff_dim,
        sigma=10.0,
        use_bias=True,
        out_bias=0.0,
    ):
        super().__init__()
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers

        self.dim_in = dim_in
        self.dim_out = dim_out

        if sigma == 0.0:
            self.encoder = INRLayer(
                dim_in=self.dim_in,
                dim_out=dim_hidden,
                use_bias=use_bias,
            )
        else:
            self.encoder = RandomFourierFeatureINRLayerFirst(
                dim_in=self.dim_in,
                dim_out=dim_hidden,
                use_bias=use_bias,
                ff_dim=ff_dim,
                sigma=sigma,
            )

        self.body = nn.ModuleList()
        for i in range(num_layers - 2):
            if i == 0:
                self.body.append(
                    LowRankINRLayer(
                        dim_in=dim_hidden,
                        dim_out=dim_hidden,
                        rank=rank,
                        use_bias=use_bias,
                    )
                )
            else:
                self.body.append(
                    INRLayer(
                        dim_in=dim_hidden,
                        dim_out=dim_hidden,
                        use_bias=use_bias,
                    )
                )

        self.decoder = INRLayerLast(
            dim_in=dim_hidden,
            dim_out=self.dim_out,
            use_bias=use_bias,
            out_bias=out_bias,
        )

    def forward(self, x):
        x = self.encoder(x)
        for layer in self.body:
            x = layer(x)

        x = self.decoder(x)
        return x

    def lowrank_modulated_forward(self, x, modulation):
        if modulation is None:
            return self(x)

        """
        Shape of x
        1D - (bsz, num_points, 1)
        2D - (bsz, num_points, 2)
        3D - (bsz, num_points, 3)

        Shape of modulation
        1D - (bsz, n, dim_hidden)
        2D - (bsz, n, dim_hidden)
        3D - (bsz, n, dim_hidden)
        """

        x = self.encoder(x)

        # apply modulation
        for i, layer in enumerate(self.body):
            if i == 0:
                x = layer(x, modulation)
            else:
                x = layer(x)

        x = self.decoder(x)

        return x
