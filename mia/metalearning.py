import mia.losses as losses
from mia.models.models import MultimodalMetaModel
from typing import List, Dict
import torch


def inner_loss_step(
    meta_model: MultimodalMetaModel,
    modes: List[str],
    xs_support_dict: Dict[str, torch.Tensor],
    ys_support_dict: Dict[str, torch.Tensor],
    ms_support_dict: Dict[str, torch.Tensor],
    inner_lr_dict: Dict[str, torch.Tensor],
    latent_dict: Dict[str, torch.Tensor],
    is_train: bool = False,
):
    meta_model.zero_grad()

    loss_dict = {}
    grad_dict = {}

    with torch.enable_grad():
        for mode in modes:
            ys_recon = meta_model.modulated_forward_single(xs_support_dict[mode], latent_dict[mode], mode)
            loss_per_signal = losses.per_element_loss_fn(
                ys_recon, ys_support_dict[mode], element_mask=ms_support_dict[mode].bool()
            )
            loss_per_signal = loss_per_signal.sum(dim=1) / ((1 - ms_support_dict[mode]).sum(dim=1) + 1e-9)
            loss_dict[mode] = loss_per_signal.nansum()

        for mode in modes:
            loss = loss_dict[mode]

            grad = torch.autograd.grad(
                loss,
                latent_dict[mode],
                create_graph=is_train,
                retain_graph=True,
            )[
                0
            ] * meta_model.get_grad_scale(mode)

            if meta_model.args.grad_scale_pos == "pre":
                grad_scale = inner_lr_dict[mode].expand_as(grad)
                grad = grad_scale * grad

            grad_dict[mode] = grad

        # Enhance gradients or latents by SFT or ALFA or GAP
        # If self.use_grad_encoder or self.use_alfa is not True, then just return the grad_dict and latent_dict`
        grad_dict, latent_dict = meta_model.fuse_states(grad_dict, latent_dict)

        for mode in modes:
            if meta_model.args.grad_scale_pos == "pre":
                latent_dict[mode] = latent_dict[mode] - grad_dict[mode]
            elif meta_model.args.grad_scale_pos == "post":
                latent_dict[mode] = latent_dict[mode] - inner_lr_dict[mode] * grad_dict[mode]

    return latent_dict, grad_dict
