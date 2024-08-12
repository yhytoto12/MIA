from mia.conversion import Converter
from mia.models.models import MultimodalMetaModel
from mia.training import BaseTrainer

import torch
import matplotlib
from matplotlib import pyplot as plt
from typing import Dict


class SyntheticTrainer(BaseTrainer):
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
        matplotlib.rcParams.update({"font.size": 6.0})
        fig, axes = plt.subplots(num_vis, self.num_modes, figsize=(2 * self.num_modes, 2 * num_vis))
        fig.suptitle(f"Step: {self.step}")
        for j, mode in enumerate(self.modes):
            xs_vis_gt = xs_query_dict[mode][:num_vis, :, 0].cpu().numpy()  # (bsz, N, 1) -> (B, N)
            ys_vis_gt = ys_query_dict[mode][:num_vis, :, 0].cpu().numpy()  # (bsz, N, 1) -> (B, N)

            ys_vis_recon_init = ys_recon_init_dict[mode][:, :, 0].cpu().numpy()  # (B, N, 1) -> (B, N)
            ys_vis_recon = ys_recon_dict[mode][:, :, 0].cpu().numpy()  # (B, N, 1) -> (B, N)
            xs_vis_support = xs_support_dict[mode][:num_vis, :, 0].cpu().numpy()  # (bsz, N, 1) -> (B, N)
            ys_vis_support = ys_support_dict[mode][:num_vis, :, 0].cpu().numpy()  # (bsz, N, 1) -> (B, N)
            ms_vis_support = ms_support_dict[mode][:num_vis, :].bool().cpu().numpy()  # (bsz, N) -> (B, N)

            for i in range(num_vis):
                ax = axes[i, j] if self.num_modes > 1 else axes[i]
                ax.plot(xs_vis_gt[i], ys_vis_gt[i], label="gt", zorder=1, color="gray", linewidth=1.5, alpha=0.5)
                ax.plot(
                    xs_vis_gt[i], ys_vis_recon_init[i], label="init", zorder=2, color="blue", linewidth=1.5, alpha=0.5
                )
                ax.plot(xs_vis_gt[i], ys_vis_recon[i], label="recon", zorder=3, color="green", linewidth=1.5, alpha=0.5)
                ax.scatter(
                    xs_vis_support[i, ~ms_vis_support[i]],
                    ys_vis_support[i, ~ms_vis_support[i]],
                    label="context",
                    color="red",
                    zorder=4,
                    s=15,
                    marker="x",
                )
                ax.set_title(mode)

        plt.tight_layout()
        return fig
