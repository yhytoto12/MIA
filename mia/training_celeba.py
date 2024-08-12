from mia.conversion import Converter
from mia.models.models import MultimodalMetaModel
from mia.training import BaseTrainer

import torch
from typing import Dict
from einops import rearrange
import torchvision
from torchvision.utils import make_grid


class CelebATrainer(BaseTrainer):
    def __init__(
        self,
        func_rep: MultimodalMetaModel,
        converter: Dict[str, Converter],
        args,
        train_dataset,
        test_dataset,
        model_path="",
    ):
        super().__init__(
            func_rep,
            converter,
            args,
            train_dataset,
            test_dataset,
            model_path,
        )

    @torch.no_grad()
    def visualize(
        self,
        num_vis,
        xs_support_dict,
        ys_support_dict,
        ms_support_dict,
        ids_restore_dict,
        xs_query_dict,
        ys_query_dict,
        ys_recon_dict,
        ys_recon_init_dict,
        data_dict,
    ):
        vis = []

        for mode in self.modes:
            vis_mode = []
            ys_vis_gt = data_dict[mode][:num_vis].cpu()
            ys_vis_contexts = self._get_contexts_image(
                ys_support_dict[mode][:num_vis], ids_restore_dict[mode][:num_vis], ms_support_dict[mode][:num_vis]
            ).cpu()
            ys_vis_recon = ys_recon_dict[mode][:num_vis].cpu()
            ys_vis_recon_init = ys_recon_init_dict[mode][:num_vis].cpu()

            ys_vis_contexts = rearrange(
                ys_vis_contexts, "b (h w) d -> b d h w", h=ys_vis_gt.shape[2], w=ys_vis_gt.shape[3]
            )
            ys_vis_recon = rearrange(ys_vis_recon, "b (h w) d -> b d h w", h=ys_vis_gt.shape[2], w=ys_vis_gt.shape[3])
            ys_vis_recon_init = rearrange(
                ys_vis_recon_init, "b (h w) d -> b d h w", h=ys_vis_gt.shape[2], w=ys_vis_gt.shape[3]
            )

            vis_mode = rearrange(
                torch.stack([ys_vis_gt, ys_vis_contexts, ys_vis_recon_init, ys_vis_recon]),
                "stack b d h w -> b d (stack h) w",
            )

            vis.append(vis_mode)

        vis = rearrange(torch.stack(vis), "stack b d h w -> b d h (stack w)").cpu()
        vis.clamp_(0, 1)
        vis = make_grid(vis, normalize=False)
        vis = torchvision.transforms.ToPILImage()(vis)

        return vis

    def _get_contexts_image(self, ys_support, ids_restore, ms_support, mask_value=0):
        B, N, Dy = ys_support.shape
        mask_token = torch.full((1, 1, Dy), mask_value).to(ys_support)
        ys_support = ys_support.masked_fill(ms_support.unsqueeze(-1).bool(), 0)
        ys_support = torch.cat([ys_support, mask_token.repeat(B, ids_restore.shape[1] - N, 1)], dim=1)
        ys_support = torch.gather(ys_support, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, Dy))

        return ys_support
