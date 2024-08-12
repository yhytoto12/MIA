import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, einsum
from einops.layers.torch import Rearrange

from mia.models.layers import (
    Transformer,
    LatentReshape2D,
    LatentReshape1D,
    Mlp,
)

from mia.models.inr import (
    Siren,
    FourierFeatureINR,
    RandomFourierFeatureINR,
    BasicINR,
    LowRankINR,
)

from mia.utils import (
    get_input_dims,
    get_output_dims,
    get_out_bias,
    get_input_range,
)


class MultimodalMetaModel(nn.Module):
    def __init__(
        self,
        args,
        modes,
        latent_spatial_shapes,
        latent_dims,
        inr_dict,
        grad_encoder_dict,
        meta_sgd_dict,
    ) -> None:
        super().__init__()

        self.args = args
        self.modes = modes
        self.num_modes = len(modes)
        self.dim_hidden = inr_dict["dim_hidden"]
        self.num_layers = inr_dict["num_layers"]

        self.inr_type = inr_dict["inr_type"]
        self.inr_dict = inr_dict
        self.modulate_scale = inr_dict["modulate_scale"]
        self.modulate_shift = inr_dict["modulate_shift"]
        self.modulate_first = inr_dict["modulate_first"]
        if self.modulate_first:
            self.num_modulate_layers = inr_dict["num_layers"] - 1
        else:
            self.num_modulate_layers = inr_dict["num_layers"] - 2
        self.latent_spatial_shapes = latent_spatial_shapes

        if "composer" in self.inr_type:
            self.latent_dims = {mode: self.dim_hidden for mode in modes}  # if latent_dims is None else latent_dims
            latent_dims = None
        else:
            self.latent_dims = latent_dims

        self.grad_encoder_dict = grad_encoder_dict
        self.meta_sgd_dict = meta_sgd_dict

        # Init INR layers
        # ---------------------------------------------------------------------------------------

        self.dims_in = get_input_dims(modes)
        self.dims_out = get_output_dims(modes)

        self.inr = nn.ModuleDict()

        if self.inr_type == "siren":
            for mode in self.modes:
                self.inr[mode] = Siren(
                    dim_in=self.dims_in[mode],
                    dim_out=self.dims_out[mode],
                    dim_hidden=self.dim_hidden,
                    num_layers=self.num_layers,
                    w0=inr_dict["w0"],
                    w0_initial=inr_dict["w0_initial"],
                    out_bias=get_out_bias(mode),
                )
        elif self.inr_type == "rffn":
            for mode in self.modes:
                self.inr[mode] = RandomFourierFeatureINR(
                    dim_in=self.dims_in[mode],
                    dim_out=self.dims_out[mode],
                    dim_hidden=self.dim_hidden,
                    num_layers=self.num_layers,
                    ff_dim=inr_dict["ff_dim"],
                    sigma=inr_dict["sigma"],
                    use_bias=True,
                    out_bias=get_out_bias(mode),
                )
        elif self.inr_type == "ffn":
            for mode in self.modes:
                self.inr[mode] = FourierFeatureINR(
                    dim_in=self.dims_in[mode],
                    dim_out=self.dims_out[mode],
                    dim_hidden=self.dim_hidden,
                    num_layers=self.num_layers,
                    ff_dim=inr_dict["ff_dim"],
                    use_bias=True,
                    out_bias=get_out_bias(mode),
                )
        elif self.inr_type == "basic":
            for mode in self.modes:
                self.inr[mode] = BasicINR(
                    dim_in=self.dims_in[mode],
                    dim_out=self.dims_out[mode],
                    dim_hidden=self.dim_hidden,
                    num_layers=self.num_layers,
                    use_bias=True,
                    out_bias=get_out_bias(mode),
                )
        elif self.inr_type == "composer":
            for mode in self.modes:
                self.inr[mode] = LowRankINR(
                    dim_in=self.dims_in[mode],
                    dim_out=self.dims_out[mode],
                    dim_hidden=self.dim_hidden,
                    num_layers=self.num_layers,
                    rank=latent_spatial_shapes[mode],
                    ff_dim=inr_dict["ff_dim"],
                    sigma=inr_dict["sigma"],
                    use_bias=True,
                    out_bias=get_out_bias(mode),
                )

        self.latent_shapes = dict()

        if "composer" in self.inr_type:
            self.modulate_dim = self.dim_hidden
        else:
            if self.modulate_scale and self.modulate_shift:
                self.modulate_dim = self.dim_hidden * 2 * self.num_modulate_layers
            else:
                self.modulate_dim = self.dim_hidden * self.num_modulate_layers

        for mode in self.modes:
            if "composer" in self.inr_type:
                self.latent_shapes[mode] = latent_spatial_shapes[mode]
            else:
                self.latent_shapes[mode] = latent_spatial_shapes[mode] ** self.dims_in[mode]
                if "era5" == self.args.dataset_config["name"]:
                    # ERA5 needs 1:2 grid
                    self.latent_shapes[mode] *= 2

        self.latent_prior_embeds = nn.ParameterDict()

        for mode in self.modes:
            self.latent_prior_embeds[mode] = nn.Parameter(
                torch.randn(1, self.latent_shapes[mode], self.latent_dims[mode]) * 0.2, requires_grad=True
            )

        self.latent_to_modulate = nn.ModuleDict()

        if "composer" in self.inr_type:
            # Composers for INRs
            for mode in self.modes:
                self.latent_to_modulate[mode] = (
                    nn.Sequential(
                        nn.LayerNorm(self.latent_dims[mode]),
                        nn.Linear(self.latent_dims[mode], self.modulate_dim),
                    )
                    if latent_dims is not None
                    else nn.Identity()
                )
        else:
            # Functa for INRs
            for mode in self.modes:
                if self.dims_in[mode] == 1:  # 1D data
                    self.latent_to_modulate[mode] = nn.Sequential(
                        nn.LayerNorm(self.latent_dims[mode]),  # (bsz, mss, D)
                        LatentReshape1D(),  # (bsz, D, mss)
                        nn.Conv1d(self.latent_dims[mode], self.modulate_dim, 3, 1, 1),  # (bsz, D, mss)
                    )
                elif self.dims_in[mode] == 2:  # 2D data
                    self.latent_to_modulate[mode] = nn.Sequential(
                        nn.LayerNorm(self.latent_dims[mode]),  # (bsz, mss * mss, D)
                        LatentReshape2D(),  # (bsz, D, mss, mss)
                        nn.Conv2d(self.latent_dims[mode], self.modulate_dim, 3, 1, 1),  # (bsz, D, mss, mss)
                    )

        # meta sgd
        # ---------------------------------------------------------------------------------------

        lr_init = meta_sgd_dict["inner_lr_init"]
        self.meta_lr = nn.ParameterDict()

        for mode in self.modes:
            if self.args.use_alfa:
                self.meta_lr[mode] = nn.Parameter(
                    torch.zeros(1, self.latent_shapes[mode], 1) + lr_init,
                    requires_grad=meta_sgd_dict["use_meta_sgd"],
                )
            else:
                self.meta_lr[mode] = nn.Parameter(
                    torch.zeros(1, self.latent_shapes[mode], self.latent_dims[mode]) + lr_init,
                    requires_grad=meta_sgd_dict["use_meta_sgd"],
                )
            if lr_init == 0:
                nn.init.uniform_(self.meta_lr[mode], 0.005, 1.0)

        # uncertainty-based loss weighting
        if self.args.loss_weight_mode == "uncertainty":
            self.logvars = nn.ParameterDict()

            for mode in self.modes:
                self.logvars[mode] = nn.Parameter(
                    torch.zeros(
                        1,
                    )
                    + self.args.logvar_init[mode],
                    requires_grad=True,
                )

        self.use_gap = self.args.use_gap
        self.use_alfa = self.args.use_alfa
        self.use_grad_encoder = grad_encoder_dict["um_depth"] + grad_encoder_dict["mm_depth"] > 0

        # Define State Fusion Transformers (SFTs)
        # ---------------------------------------------------------------------------------------
        if self.use_grad_encoder:
            self.grad_encoder_projection_mlp = nn.ModuleDict()
            self.grad_encoder_state_to_gradient = nn.ModuleDict()
            self.grad_encoder_pos_embed_type = grad_encoder_dict["pos_embed_type"]
            self.grad_encoder_fuser = nn.ModuleDict() if grad_encoder_dict["use_fuser"] else False
            self.log_grad_scaler = nn.ParameterDict()
            grad_scaler_init = grad_encoder_dict["grad_scaler_init"] if "grad_scaler_init" in grad_encoder_dict else 1.0

            if grad_encoder_dict["use_grad"]:
                self.grad_encoder_grad_ln = nn.ModuleDict()

            if grad_encoder_dict["use_latent"]:
                self.grad_encoder_latent_ln = nn.ModuleDict()

            for mode in self.modes:
                self.log_grad_scaler[mode] = nn.Parameter(
                    torch.tensor(grad_scaler_init).log(),
                    requires_grad=self.args.grad_encoder_grad_scaler_learnable,
                )

                num_channels = 0
                if grad_encoder_dict["use_grad"]:
                    num_channels += self.latent_dims[mode]
                    self.grad_encoder_grad_ln[mode] = nn.LayerNorm(self.latent_dims[mode], eps=1e-5)

                if grad_encoder_dict["use_latent"]:
                    num_channels += self.latent_dims[mode]
                    self.grad_encoder_latent_ln[mode] = nn.LayerNorm(self.latent_dims[mode], eps=1e-5)

                assert num_channels > 0

                self.grad_encoder_projection_mlp[mode] = nn.Sequential(
                    Mlp(
                        in_features=num_channels,
                        hidden_features=grad_encoder_dict["dim"],
                        out_features=grad_encoder_dict["dim"],
                        depth=grad_encoder_dict["projection_mlp_depth"],
                    ),
                )

                if grad_encoder_dict["use_fuser"]:
                    num_states = 1
                    if grad_encoder_dict["um_depth"] > 0:
                        num_states += 1
                    if grad_encoder_dict["mm_depth"] > 0:
                        num_states += 1
                    self.grad_encoder_fuser[mode] = nn.Sequential(
                        Rearrange("b n d -> b d n"),
                        nn.GroupNorm(
                            num_groups=num_states,
                            num_channels=grad_encoder_dict["dim"] * num_states,
                        ),
                        Rearrange("b d n -> b n d"),
                        Mlp(
                            in_features=grad_encoder_dict["dim"] * num_states,
                            hidden_features=grad_encoder_dict["dim"],
                            out_features=grad_encoder_dict["dim"],
                            depth=grad_encoder_dict["depth_fuser"],
                        ),
                        nn.LayerNorm(grad_encoder_dict["dim"]),
                    )

                self.grad_encoder_state_to_gradient[mode] = nn.Linear(
                    grad_encoder_dict["dim"],
                    self.latent_dims[mode],
                )

                self.grad_encoder_pos_embeds = nn.ParameterDict()
                if self.grad_encoder_pos_embed_type in ["fixed"]:
                    for mode in self.modes:
                        self.grad_encoder_pos_embeds[mode] = self._create_fourier_embeds(grad_encoder_dict["dim"], mode)
                elif self.grad_encoder_pos_embed_type in ["learned"]:
                    for mode in self.modes:
                        self.grad_encoder_pos_embeds[mode] = nn.Parameter(
                            torch.randn(self.latent_shapes[mode], grad_encoder_dict["dim"]) * 0.2,
                            requires_grad=True,
                        )

            self.grad_encoder_um = nn.ModuleDict()
            for mode in self.modes:
                self.grad_encoder_um[mode] = Transformer(
                    dim=grad_encoder_dict["dim"],
                    depth=grad_encoder_dict["um_depth"],
                    heads=grad_encoder_dict["heads"],
                    dim_head=grad_encoder_dict["dim_head"],
                    mlp_dim=int(grad_encoder_dict["dim"] * grad_encoder_dict["mlp_ratio"]),
                    dropout=grad_encoder_dict["dropout"],
                )
            self.grad_encoder_mm = Transformer(
                dim=grad_encoder_dict["dim"],
                depth=grad_encoder_dict["mm_depth"],
                heads=grad_encoder_dict["heads"],
                dim_head=grad_encoder_dict["dim_head"],
                mlp_dim=int(grad_encoder_dict["dim"] * grad_encoder_dict["mlp_ratio"]),
                dropout=grad_encoder_dict["dropout"],
            )

        if self.args.use_alfa:
            self.alfa = nn.ModuleDict()
            self.beta_init_dict = nn.ParameterDict()
            for mode in self.modes:
                input_dim = self.latent_shapes[mode] * 2
                if self.args.dim_alfa > 0:
                    hidden_dim = self.args.dim_alfa
                else:
                    hidden_dim = input_dim

                self.alfa[mode] = Mlp(
                    in_features=input_dim,
                    hidden_features=hidden_dim,
                    out_features=input_dim,
                    depth=self.args.depth_alfa,
                )
                self.beta_init_dict[mode] = nn.Parameter(
                    torch.ones(1, self.latent_shapes[mode], 1),
                    requires_grad=meta_sgd_dict["use_meta_sgd"],
                )

        self.use_gap = self.args.use_gap
        if self.use_gap:
            self.M = nn.ParameterDict()
            for mode in self.modes:
                shape = min(self.latent_shapes[mode], self.latent_dims[mode])
                self.M[mode] = nn.Parameter(0.928 * torch.ones(shape), requires_grad=True)

    def get_inr_params(self):
        non_inr_keywords = ["grad_enc", "logvars"]

        params = {}
        for k, v in dict(self.named_parameters()).items():
            non_inr_keywords_exist = []
            for non_inr_keyword in non_inr_keywords:
                non_inr_keywords_exist += [non_inr_keyword in k]
            if sum(non_inr_keywords_exist) == 0:
                # print('inr_params:', k)
                params[k] = v
        return params

    def get_non_inr_params(self):
        non_inr_keywords = ["grad_enc"]

        params = {}
        for k, v in dict(self.named_parameters()).items():
            non_inr_keywords_exist = []
            for non_inr_keyword in non_inr_keywords:
                non_inr_keywords_exist += [non_inr_keyword in k]
            if sum(non_inr_keywords_exist) > 0:
                # print('non_inr_params:', k)
                params[k] = v
        return params

    def get_logvars(self):
        non_inr_keywords = ["logvars"]

        params = {}
        for k, v in dict(self.named_parameters()).items():
            non_inr_keywords_exist = []
            for non_inr_keyword in non_inr_keywords:
                non_inr_keywords_exist += [non_inr_keyword in k]
            if sum(non_inr_keywords_exist) > 0:
                params[k] = v
        return params

    def get_parameters(self, keys=None):
        if keys is None:
            params = [v for k, v in self.named_parameters()]
        else:
            if isinstance(keys, (list, tuple)):
                params = [v for k, v in self.named_parameters() if len([key for key in keys if key in k]) > 0]
            elif isinstance(keys, str):
                params = [v for k, v in self.named_parameters() if keys in k]

        return params

    def _create_fourier_embeds(self, dim, mode):
        w = torch.exp(torch.linspace(0, 8, dim // 2 // self.dims_in[mode]))
        coords = []
        input_range = get_input_range(mode)
        for dim_idx in range(self.dims_in[mode]):
            if self.latent_spatial_shapes[mode] == 1:
                coords.append(torch.linspace(-0, 0, self.latent_spatial_shapes[mode]))
            else:
                if "era5" == self.args.dataset_config["name"]:
                    # ERA5 needs 1:2 grid
                    coords.append(
                        torch.linspace(-input_range, input_range, self.latent_spatial_shapes[mode] * (dim_idx + 1))
                    )
                else:
                    coords.append(torch.linspace(-input_range, input_range, self.latent_spatial_shapes[mode]))

        coords = torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)
        coords = einsum(coords, w, "... d, fdim -> ... d fdim").view(*coords.shape[:-1], -1)
        coords = torch.cat([torch.cos(torch.pi * coords), torch.sin(torch.pi * coords)], dim=-1)
        coords = coords.reshape(1, -1, dim)
        coords = nn.Parameter(coords, requires_grad=False)
        return coords

    def init_latent(self, batch_size):
        latent_prior_dict = {}
        for mode in self.modes:
            latent_prior_dict[mode] = repeat(self.latent_prior_embeds[mode], "1 ... -> bsz ...", bsz=batch_size)

        return latent_prior_dict

    def fuse_states(self, grad_dict, latent_dict):
        if not self.use_grad_encoder and not self.use_alfa and not self.use_gap:
            return grad_dict, latent_dict

        elif self.use_grad_encoder:
            ordinal_state_dict = {}
            unimodal_state_dict = {}
            multimodal_state_dict = {}
            modified_grad_dict = {}

            n_tokens = [0]
            states = []
            for mode in self.modes:
                B, N, D = grad_dict[mode].shape

                input_features = []
                if self.grad_encoder_dict["use_grad"]:
                    input_features += [self.grad_encoder_grad_ln[mode](grad_dict[mode])]
                if self.grad_encoder_dict["use_latent"]:
                    input_features += [self.grad_encoder_latent_ln[mode](latent_dict[mode])]

                input_features = torch.cat(input_features, dim=-1)
                input_features = self.grad_encoder_projection_mlp[mode](input_features)

                state = input_features
                if self.grad_encoder_fuser:
                    ordinal_state_dict[mode] = state

                if self.use_grad_encoder:
                    state += self.grad_encoder_pos_embeds[mode]

                state = self.grad_encoder_um[mode](state)

                if self.grad_encoder_fuser:
                    unimodal_state_dict[mode] = state

                n_tokens += [n_tokens[-1] + state.shape[1]]

                states += [state]

            states = torch.cat(states, dim=1)
            states = self.grad_encoder_mm(states)

            for i, mode in enumerate(self.modes):
                state = states[:, n_tokens[i] : n_tokens[i + 1], :]
                multimodal_state_dict[mode] = state

            for mode in self.modes:
                if self.grad_encoder_fuser:
                    state_features = [ordinal_state_dict[mode]]
                    if self.grad_encoder_dict["um_depth"] > 0:
                        state_features += [unimodal_state_dict[mode]]
                    if self.grad_encoder_dict["mm_depth"] > 0:
                        state_features += [multimodal_state_dict[mode]]
                    state_features = torch.cat(state_features, dim=-1)
                    state_features = self.grad_encoder_fuser[mode](state_features)
                    modified_grad_dict[mode] = self.grad_encoder_state_to_gradient[mode](state_features)
                else:
                    modified_grad_dict[mode] = self.grad_encoder_state_to_gradient[mode](multimodal_state_dict[mode])

            return modified_grad_dict, latent_dict

        elif self.use_alfa:
            modified_latent_dict = {}
            modified_grad_dict = {}
            for mode in self.modes:
                states = torch.cat([grad_dict[mode].mean(-1), latent_dict[mode].mean(-1)], -1).flatten(1)
                states = self.alfa[mode](states)
                states = states.reshape(states.shape[0], -1, 1, 2)
                beta, alpha = states[..., 0], states[..., 1]
                modified_latent_dict[mode] = self.beta_init_dict[mode] * beta * latent_dict[mode]
                modified_grad_dict[mode] = alpha * grad_dict[mode]

            return modified_grad_dict, modified_latent_dict

        elif self.use_gap:
            modified_grad_dict = {}
            for mode in self.modes:
                M = self.M[mode]
                M = repeat(torch.diag(F.softplus(M, beta=2)), "n m -> b n m", b=grad_dict[mode].shape[0])
                if self.args.use_gap_approx:
                    if grad_dict[mode].shape[1] >= grad_dict[mode].shape[2]:
                        modified_grad_dict[mode] = grad_dict[mode] @ M
                    else:
                        modified_grad_dict[mode] = M @ grad_dict[mode]
                else:
                    u, _, _ = torch.svd(grad_dict[mode].detach())
                    preconditioner = u @ (M @ u.transpose(2, 1))
                    modified_grad_dict[mode] = preconditioner @ grad_dict[mode]

            return modified_grad_dict, latent_dict

    def get_grad_scale(self, mode):
        if self.grad_encoder_dict["use_grad_scaler"]:
            return self.log_grad_scaler[mode].exp()
        else:
            return 1

    def modulated_forward_single(self, x, latent, mode):
        # 1D - (bsz, lss, ld) -> (bsz, D, lss)
        # 2D - (bsz, lss * lss, ld) -> (bsz, D, lss, lss)
        modulations = self.latent_to_modulate[mode](latent)

        if "composer" in self.inr_type:
            x = self.inr[mode].lowrank_modulated_forward(x, modulations)
        elif self.modulate_scale and self.modulate_shift:
            x = self.inr[mode].scaleshift_modulated_forward(x, modulations, self.modulate_first)
        elif self.modulate_scale:
            x = self.inr[mode].scale_modulated_forward(x, modulations, self.modulate_first)
        elif self.modulate_shift:
            x = self.inr[mode].shift_modulated_forward(x, modulations, self.modulate_first)
        else:
            x = self.inr[mode](x)

        return x
