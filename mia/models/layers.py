import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from math import sqrt


# Helpers
# ------------------------------------------------------------------------
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# Transformers, Perceivers
# ------------------------------------------------------------------------
class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim=None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs["context"]
            normed_context = self.norm_context(context)
            kwargs.update(context=normed_context)

        return self.fn(x, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head**-0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context=None, mask=None, topk=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        sim = einsum(q, k, "b h i d, b h j d -> b h i j") * self.scale

        if exists(mask):
            max_neg_value = -torch.finfo(sim.dtype).max
            sim.masked_fill_(mask.bool(), max_neg_value)

        if topk is not None:
            max_neg_value = -torch.finfo(sim.dtype).max
            B, nH, nQ, nK = sim.shape
            topk = min(topk, nK)
            _, topk_idx = torch.topk(sim, topk, sorted=False, dim=-1)
            topk_mask = torch.ones_like(sim, dtype=torch.bool)
            topk_mask[
                torch.arange(B, device=sim.device).view(B, 1, 1, 1),
                torch.arange(nH, device=sim.device).view(1, nH, 1, 1),
                torch.arange(nQ, device=sim.device).view(1, 1, nQ, 1),
                topk_idx,
            ] = False
            sim.masked_fill_(topk_mask, max_neg_value)

        attn = sim.softmax(dim=-1)
        attn = self.dropout(attn)

        out = einsum(attn, v, "b h i j, b h j d -> b h i d")
        out = rearrange(out, "b h n d -> b n (h d)", h=h)

        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x, mask=None, topk=None):
        for attn, ffn in self.layers:
            x = attn(x, mask=mask, topk=topk)
            x = ffn(x)
        return x


class Perceiver(nn.Module):
    def __init__(
        self,
        dim,
        context_dim,
        num_querys,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
    ):
        super().__init__()
        context_dim = default(context_dim, dim)
        self.cross_attn_layer = nn.ModuleList(
            [
                PreNorm(
                    dim,
                    Attention(dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout),
                    context_dim=context_dim,
                ),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]
        )

        self.self_attn_layers = nn.ModuleList()
        for _ in range(depth):
            self.self_attn_layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

        self.num_querys = num_querys
        self.querys = nn.Parameter(torch.randn(1, num_querys, dim) * 0.2)

    def forward(self, context, mask=None, topk=None):
        attn, ffn = self.cross_attn_layer

        x = repeat(self.querys, "1 n d -> b n d", b=context.shape[0])
        x = attn(x, context=context, mask=mask, topk=topk) + x
        x = ffn(x) + x

        for attn, ffn in self.self_attn_layers:
            x = attn(x)
            x = ffn(x)
        return x


# Fourier Features
# ------------------------------------------------------------------------
class LearnableFourierFeatures(nn.Module):
    def __init__(self, xdim, fdim):
        super().__init__()
        self.fdim = fdim
        self.mat = nn.Linear(xdim, fdim // 2, bias=False)

    def forward(self, x):
        x = self.mat(x)
        x = torch.cat([torch.cos(x), torch.sin(x)], -1) * 1 / sqrt(self.fdim)
        return x


class FixedFourierFeatures(nn.Module):
    def __init__(self, fdim):
        super().__init__()
        w = torch.exp(torch.linspace(0, 8, fdim // 2))
        self.register_buffer("w", w)

    def forward(self, x):
        x = einsum(x, self.w, "... d, fdim -> ... d fdim")
        x = torch.pi * rearrange(x, "... d fdim -> ... (d fdim)")
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return x


class RandomFourierFeatures(nn.Module):
    def __init__(self, fdim, in_dim, sigma):
        super().__init__()
        B = torch.randn(fdim // 2, in_dim) * sigma
        self.register_buffer("B", B)

    def forward(self, x):
        """
        x : (bsz, n, in_dim)
        self.B : (fdim // 2, in_dim)
        return : (bsz, n, fdim)
        """
        x = 2 * torch.pi * x @ self.B.T
        x = torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
        return x


# Activations
# ------------------------------------------------------------------------
class Sine(nn.Module):
    def __init__(self, w0=1.0, w0_learnable=False):
        super().__init__()
        self.w0 = nn.Parameter(torch.tensor(w0), requires_grad=w0_learnable)

    def forward(self, x):
        return torch.sin(self.w0 * x)


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class LatentReshape1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        b, mss, d = x.shape
        x = rearrange(x, "b l d -> b d l")
        return x


class ModulateReshape1D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        b, mss, d = x.shape
        x = rearrange(x, "b d l -> b l d")
        return x


class LatentReshape2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        b, mss, d = x.shape
        if sqrt(mss) == int(sqrt(mss)):
            h = w = int(sqrt(mss))
        else:
            # this is the case when H:W = 1:2
            h = int(sqrt(mss / 2))
            w = int(h * 2)
        x = rearrange(x, "b (h w) d -> b d h w", h=h, w=w)
        return x


class ModulateReshape2D(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        x = rearrange(x, "b d h w -> b (h w) d")
        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        depth=2,
        act_layer=nn.ReLU,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        if depth == 1:
            layers = [nn.Linear(in_features, out_features)]
        elif depth > 1:
            layers = [nn.Linear(in_features, hidden_features)]
            for _ in range(depth - 2):
                layers += [act_layer(), nn.Linear(hidden_features, hidden_features)]
            layers += [act_layer(), nn.Linear(hidden_features, out_features)]
        else:
            layers = [nn.Identity()]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
