from mia.conversion import Converter
import mia.losses as losses
import mia.metalearning as metalearning
from mia.models.models import MultimodalMetaModel

import torch
import wandb
from tqdm import tqdm
from pathlib import Path
from matplotlib import pyplot as plt
import time
from typing import Dict
from torch.distributions.beta import Beta

from mia.utils import sample_nmr, generate_random_masks, to_device


class BaseTrainer:
    def __init__(
        self,
        meta_model: MultimodalMetaModel,
        converter: Dict[str, Converter],
        args,
        train_dataset,
        test_dataset,
        model_path="",
    ):
        self.meta_model = meta_model
        self.converter = converter
        self.args = args
        self.modes = args.modes
        self.num_modes = len(args.modes)
        self.sample_modes = args.sample_modes
        self.num_sample_modes = len(args.sample_modes)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self._process_datasets()

        if self.train_dataset is not None:
            inr_params = self.meta_model.get_inr_params()
            self.outer_optimizer_inr = torch.optim.Adam(
                [{"params": inr_params.values(), "lr": args.outer_lr}],
                lr=args.outer_lr,
            )

            enc_params = self.meta_model.get_non_inr_params()
            logvar_params = self.meta_model.get_logvars()

            if hasattr(self.meta_model, "log_grad_scaler"):
                self.outer_optimizer_enc = torch.optim.Adam(
                    [
                        {
                            "params": enc_params.values(),
                            "lr": args.outer_lr * args.encoder_lr_ratio,
                        },
                        {
                            "params": logvar_params.values(),
                            "lr": args.outer_lr * args.logvar_lr_ratio,
                        },
                        {
                            "params": self.meta_model.log_grad_scaler.parameters(),
                            "lr": args.outer_lr * args.grad_encoder_grad_scaler_lr_ratio,
                        },
                    ],
                    lr=args.outer_lr,
                )
            else:
                self.outer_optimizer_enc = torch.optim.Adam(
                    [
                        {
                            "params": enc_params.values(),
                            "lr": args.outer_lr * args.encoder_lr_ratio,
                        },
                        {
                            "params": logvar_params.values(),
                            "lr": args.outer_lr * args.logvar_lr_ratio,
                        },
                    ],
                    lr=args.outer_lr,
                )

        self.model_path = model_path
        self.step = 0

        self.stime = time.time()

    def load_pretrained(
        self,
        states=None,
        step=None,
        filename=None,
        model_path=None,
        strict=False,
        load_step=True,
        load_optimizer=False,
        load_inr_only=False,
    ):
        if states is None:
            if step is not None:
                model_path = self.args.log_dir / "ckpt" / f"{step:010d}.pt"

            if filename is not None:
                model_path = self.args.log_dir / "ckpt" / f"{filename}"

            states = torch.load(model_path, map_location="cpu")

        if load_inr_only:
            states["state_dict"] = {k: v for k, v in states["state_dict"].items() if "grad" not in k}

        self.meta_model.load_state_dict(states["state_dict"], strict=strict)

        if load_step:
            self.step = states["step"]

        if load_optimizer:
            self.outer_optimizer_inr.load_state_dict(states["state_dict_optim_inr"])
            self.outer_optimizer_enc.load_state_dict(states["state_dict_optim_enc"])

    def _process_datasets(self):
        """Create dataloaders for datasets based on self.args."""
        self.train_dataloader = (
            torch.utils.data.DataLoader(
                self.train_dataset,
                shuffle=True,
                batch_size=self.args.batch_size,
                num_workers=self.args.num_workers,
                pin_memory=self.args.num_workers > 0,
                drop_last=True,
            )
            if self.train_dataset is not None
            else None
        )

        self.test_dataloader = (
            torch.utils.data.DataLoader(
                self.test_dataset,
                shuffle=False,
                batch_size=self.args.validation_batch_size,
                num_workers=self.args.num_workers,
            )
            if self.test_dataset is not None
            else None
        )

    @torch.no_grad()
    def create_supports_and_querys(self, data_dict, distM_dict):
        xs_support_dict, ys_support_dict, ms_support_dict = {}, {}, {}
        ids_restore_dict, ids_sample_dict = {}, {}
        xs_query_dict, ys_query_dict, ms_query_dict = {}, {}, {}

        Rmin, Rmax = self.args.Rmin, self.args.Rmax

        for mode in self.sample_modes:
            data = data_dict[mode].to(self.args.device)
            x_full, y_full = self.converter[mode].to_coordinates_and_features(data)
            x_full, y_full = x_full.flatten(1, -2), y_full.flatten(1, -2)
            batch_size, n = x_full.shape[:2]

            # Query set is always full coordinates
            xs_query_dict[mode], ys_query_dict[mode] = x_full, y_full
            ms_query_dict[mode] = torch.zeros_like(xs_query_dict[mode][..., 0])

            num_sample_min = int(n * Rmin[mode])
            num_sample_max = int(n * Rmax[mode])

            (
                xs_support_dict[mode],
                ys_support_dict[mode],
                ids_restore_dict[mode],
                ids_sample_dict[mode],
            ) = sample_nmr(x_full, y_full, num_sample_max)
            ms_support_dict[mode] = generate_random_masks(
                xs_support_dict[mode],
                num_sample_min=num_sample_min,
                num_sample_max=num_sample_max,
                dist=distM_dict[mode],
            )

        for mode in self.sample_modes:
            if mode not in self.modes:
                ms_support_dict.pop(mode)
                xs_support_dict.pop(mode)
                ys_support_dict.pop(mode)
                xs_query_dict.pop(mode)
                ys_query_dict.pop(mode)

        return (
            batch_size,
            xs_support_dict,
            ys_support_dict,
            ms_support_dict,
            ids_restore_dict,
            ids_sample_dict,
            xs_query_dict,
            ys_query_dict,
            ms_query_dict,
        )

    @torch.no_grad()
    def _create_element_mask(self, xs_query_dict, ms_support_dict, ids_sample_dict):
        element_mask_dict = {}
        for mode in self.modes:
            element_mask_dict[mode] = torch.zeros_like(xs_query_dict[mode][..., 0])
            element_mask_dict[mode] = element_mask_dict[mode].scatter(
                1, ids_sample_dict[mode], 1 - ms_support_dict[mode]
            )

        return element_mask_dict

    @torch.no_grad()
    def visualize(self, *args, **kwargs):
        NotImplemented

    def train_epoch(self):
        self.meta_model.train()
        inner_steps = self.args.inner_steps

        distM_dict = {mode: Beta(self.args.Ma[mode], self.args.Mb[mode]) for mode in self.sample_modes}

        """ Train model for a single epoch """
        for data_dict, _ in tqdm(self.train_dataloader):
            self.step += 1

            """ 1. Preprocessing """
            (
                batch_size,
                xs_support_dict,
                ys_support_dict,
                ms_support_dict,
                ids_restore_dict,
                _,
                xs_query_dict,
                ys_query_dict,
                _,
            ) = self.create_supports_and_querys(data_dict, distM_dict)

            num_vis = min(batch_size, self.args.num_vis)

            """ 2. Inner Loop Adaptation """
            loss_dict = {}
            metric_dict = {}
            ys_recon_dict = {}

            inner_lr_dict = self.meta_model.meta_lr
            latent_init_dict = self.meta_model.init_latent(batch_size)

            for inner_step in range(inner_steps + 1):
                if inner_step == 0:
                    latent_dict = {mode: latent_init_dict[mode] for mode in self.modes}
                else:
                    latent_dict, _ = metalearning.inner_loss_step(
                        meta_model=self.meta_model,
                        modes=self.modes,
                        xs_support_dict=xs_support_dict,
                        ys_support_dict=ys_support_dict,
                        ms_support_dict=ms_support_dict,
                        inner_lr_dict=inner_lr_dict,
                        latent_dict=latent_dict,
                        is_train=True,
                    )

                if inner_step == 0 or inner_step in self.args.outer_steps:
                    if inner_step not in loss_dict:
                        loss_dict[inner_step] = {}
                        metric_dict[inner_step] = {}

                    for mode in self.modes:
                        with torch.set_grad_enabled(inner_step in self.args.outer_steps):
                            ys_recon = self.meta_model.modulated_forward_single(
                                xs_query_dict[mode], latent_dict[mode], mode
                            )

                            per_example_loss = losses.batch_loss_fn(ys_recon, ys_query_dict[mode])
                            per_example_metric = losses.batch_metric_fn(ys_recon, ys_query_dict[mode])

                            loss_dict[inner_step][mode] = per_example_loss.mean()
                            metric_dict[inner_step][mode] = per_example_metric.mean()

                        if self.args.use_wandb and self.step % self.args.log_image_interval == 0:
                            if inner_step not in ys_recon_dict:
                                ys_recon_dict[inner_step] = {}

                            ys_recon_dict[inner_step][mode] = ys_recon[:num_vis].detach().cpu()

            """ 3. Outer update """
            loss = 0.0
            for inner_step in loss_dict:
                for mode in self.modes:
                    if self.args.loss_weight_mode == "none":
                        loss += loss_dict[inner_step][mode]
                    elif self.args.loss_weight_mode == "uncertainty":
                        logvar_mode = self.meta_model.logvars[mode].squeeze()
                        loss += loss_dict[inner_step][mode] / (2 * logvar_mode.exp()) + logvar_mode / 2
                    else:
                        raise NotImplementedError

            total_loss = loss

            self.outer_optimizer_inr.zero_grad()
            self.outer_optimizer_enc.zero_grad()

            total_loss.backward(create_graph=False)

            self.outer_optimizer_inr.step()
            self.outer_optimizer_enc.step()

            for mode in self.modes:
                self.meta_model.meta_lr[mode].data.clamp_(0.0, self.args.meta_sgd_lr_max)

            if self.step % self.args.log_interval == 0:
                print(f"Step {self.step}, Total Loss {total_loss:.5f}")
                for mode in self.modes:
                    print(
                        f"{mode:>10s}: (0-step) Loss {loss_dict[0][mode]:.5f}, Metric ({losses.metric_name[mode]}) {metric_dict[0][mode]:.5f}"
                    )
                    print(
                        f"{mode:>10s}: ({inner_steps}-step) Loss {loss_dict[inner_steps][mode]:.5f}, Metric ({losses.metric_name[mode]}) {metric_dict[inner_steps][mode]:.5f}"
                    )

            # log_meta = f'Rmin:{self.args.Rmin}-Rmax:{self.args.Rmax}'
            if self.args.use_wandb and self.step % self.args.log_interval == 0:
                log_dict = {}

                for mode in self.modes:
                    log_dict[f"inner_lr-{mode}"] = self.meta_model.meta_lr[mode].mean().item()

                log_dict[f"train-loss-avg"] = loss.item()

                for inner_step in loss_dict:
                    for mode in self.modes:
                        log_dict[f"train-loss-{mode}-in_step:{inner_step}"] = loss_dict[inner_step][mode].item()
                        log_dict[f"train-metric-{mode}-in_step:{inner_step}"] = metric_dict[inner_step][mode].item()

                if self.args.loss_weight_mode == "uncertainty":
                    for mode in self.modes:
                        log_dict[f"train-logvar-{mode}"] = self.meta_model.logvars[mode].item()

                if hasattr(self.meta_model, "log_grad_scaler"):
                    for mode in self.modes:
                        log_dict[f"train-log_grad_scaler-{mode}"] = self.meta_model.log_grad_scaler[mode].item()

                wandb.log(log_dict, step=self.step)

            if self.args.use_wandb and self.step % self.args.log_image_interval == 0:
                log_dict = {}

                vis = self.visualize(
                    num_vis=num_vis,
                    xs_support_dict=xs_support_dict,
                    ys_support_dict=ys_support_dict,
                    ms_support_dict=ms_support_dict,
                    ids_restore_dict=ids_restore_dict,
                    xs_query_dict=xs_query_dict,
                    ys_query_dict=ys_query_dict,
                    ys_recon_dict=ys_recon_dict[inner_steps],
                    ys_recon_init_dict=ys_recon_dict[0],
                    data_dict=data_dict,
                )

                if vis is None:
                    pass

                elif isinstance(vis, dict):
                    for mode in self.modes:
                        log_dict[f"train-recon-{mode}"] = wandb.Image(vis[mode], caption=f"mode:{mode}")
                else:
                    log_dict[f"train-recon"] = wandb.Image(vis, caption=f"modes:{self.modes}")

                wandb.log(log_dict, step=self.step)

                plt.close()

            if self.step % self.args.validate_every == 0:
                torch.cuda.empty_cache()
                for valid_inner_steps in self.args.validation_inner_steps:
                    self.validation(valid_inner_steps)
                    torch.cuda.empty_cache()

                model_path = self.args.log_dir / "ckpt" / f"{self.step:010d}.pt"
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "args": self.args,
                        "state_dict": self.meta_model.state_dict(),
                        "state_dict_optim_inr": self.outer_optimizer_inr.state_dict(),
                        "state_dict_optim_enc": self.outer_optimizer_enc.state_dict(),
                        "step": self.step,
                    },
                    model_path,
                )

            if time.time() - self.stime > 1200:
                print(f"Step: {self.step} -- Save model to {self.model_path}")
                torch.save(
                    {
                        "args": self.args,
                        "state_dict": self.meta_model.state_dict(),
                        "state_dict_optim_inr": self.outer_optimizer_inr.state_dict(),
                        "state_dict_optim_enc": self.outer_optimizer_enc.state_dict(),
                        "step": self.step,
                    },
                    self.model_path,
                )
                self.stime = time.time()

    @torch.no_grad()
    def validation(self, inner_steps):
        print(f"\nValidation, Step {self.step}:")

        loss_dict = {}
        metric_dict = {}
        for k in ["query", "support", "non-support"]:
            loss_dict[k] = {mode: {} for mode in self.modes}
            metric_dict[k] = {mode: {} for mode in self.modes}

        N_dict = {mode: {} for mode in self.modes}

        ys_recon_dict = {}
        ys_recon_init_dict = {}

        distM_dict = {mode: Beta(self.args.Ma[mode], self.args.Mb[mode]) for mode in self.modes}

        for data_dict, info_dict in tqdm(self.test_dataloader):
            if "xs_support_dict" in data_dict:
                # Meta test set
                data_dict = to_device(data_dict, self.args.device)
                batch_size = data_dict[self.modes[0]].shape[0]
                xs_support_dict = data_dict.pop("xs_support_dict")
                ys_support_dict = data_dict.pop("ys_support_dict")
                ms_support_dict = data_dict.pop("ms_support_dict")
                xs_query_dict = data_dict.pop("xs_query_dict")
                ys_query_dict = data_dict.pop("ys_query_dict")
                ids_restore_dict = data_dict.pop("ids_restore_dict")
                ids_sample_dict = data_dict.pop("ids_sample_dict")
                if self.args.dataset_config["name"] in ["synthetic"]:
                    info_dict = data_dict.pop("info_dict")

            else:
                # Non-meta test set
                """1. Preprocessing"""
                (
                    batch_size,
                    xs_support_dict,
                    ys_support_dict,
                    ms_support_dict,
                    ids_restore_dict,
                    ids_sample_dict,
                    xs_query_dict,
                    ys_query_dict,
                    _,
                    _,
                    _,
                    _,
                ) = self.create_supports_and_querys(data_dict, distM_dict)

            element_mask_dict = self._create_element_mask(xs_query_dict, ms_support_dict, ids_sample_dict)

            num_vis = min(self.args.num_vis, batch_size)

            """ 2. Inner Loop Adaptation """
            inner_lr_dict = self.meta_model.meta_lr
            latent_init_dict = self.meta_model.init_latent(batch_size)
            latent_dict = {mode: v.requires_grad_(True) for mode, v in latent_init_dict.items()}

            for inner_step in range(inner_steps + 1):
                if inner_step > 0:
                    latent_dict, _ = metalearning.inner_loss_step(
                        meta_model=self.meta_model,
                        modes=self.modes,
                        xs_support_dict=xs_support_dict,
                        ys_support_dict=ys_support_dict,
                        ms_support_dict=ms_support_dict,
                        inner_lr_dict=inner_lr_dict,
                        latent_dict=latent_dict,
                        is_train=False,
                    )

            for mode in self.modes:
                ys_recon = self.meta_model.modulated_forward_single(xs_query_dict[mode], latent_dict[mode], mode)

                per_example_loss, per_example_metric = {}, {}

                if isinstance(info_dict, dict):
                    scale = info_dict[mode]["a"].to(self.args.device).view(-1, 1, 1) ** 2
                else:
                    scale = 1.0

                (
                    per_example_loss["query"],
                    per_example_loss["support"],
                    per_example_loss["non-support"],
                ) = losses.batch_loss_fn_with_element_mask(
                    ys_recon,
                    ys_query_dict[mode],
                    norm=scale,
                    element_mask=element_mask_dict[mode].bool(),
                )
                (
                    per_example_metric["query"],
                    per_example_metric["support"],
                    per_example_metric["non-support"],
                ) = losses.batch_metric_fn_with_element_mask(
                    ys_recon,
                    ys_query_dict[mode],
                    norm=scale,
                    element_mask=element_mask_dict[mode].bool(),
                )

                per_example_Rs = element_mask_dict[mode].mean(dim=1)

                for Rrange in self.args.Rrange_lists[mode]:
                    Rmin, Rmax = Rrange
                    log_name = f"Rmin:{Rmin:.3f}-Rmax:{Rmax:.3f}"

                    mask = (per_example_Rs <= Rmax) & (Rmin <= per_example_Rs)

                    for k in ["query", "support", "non-support"]:
                        if log_name not in metric_dict[k][mode]:
                            metric_dict[k][mode][log_name] = 0
                            loss_dict[k][mode][log_name] = 0

                        metric_dict[k][mode][log_name] += per_example_metric[k][mask].nansum().item()
                        loss_dict[k][mode][log_name] += per_example_loss[k][mask].nansum().item()

                    if log_name not in N_dict[mode]:
                        N_dict[mode][log_name] = 0

                    N_dict[mode][log_name] += mask.float().sum().item()

                log_name = "avg"
                for k in ["query", "support", "non-support"]:
                    if log_name not in metric_dict[k][mode]:
                        metric_dict[k][mode][log_name] = 0
                        loss_dict[k][mode][log_name] = 0

                    metric_dict[k][mode][log_name] += per_example_metric[k].nansum().item()
                    loss_dict[k][mode][log_name] += per_example_loss[k].nansum().item()

                if log_name not in N_dict[mode]:
                    N_dict[mode][log_name] = 0

                N_dict[mode][log_name] += batch_size

                if self.args.use_wandb:
                    ys_recon_dict[mode] = ys_recon[:num_vis].detach().cpu()

                    ys_recon_init = self.meta_model.modulated_forward_single(
                        xs_query_dict[mode][:num_vis], latent_init_dict[mode][:num_vis], mode
                    )
                    ys_recon_init_dict[mode] = ys_recon_init.detach().cpu()

        for k in ["query", "support", "non-support"]:
            for mode in self.modes:
                for Rrange in self.args.Rrange_lists[mode]:
                    Rmin, Rmax = Rrange
                    log_name = f"Rmin:{Rmin:.3f}-Rmax:{Rmax:.3f}"

                    metric_dict[k][mode][log_name] /= N_dict[mode][log_name] + 1e-12
                    loss_dict[k][mode][log_name] /= N_dict[mode][log_name] + 1e-12

                log_name = "avg"
                metric_dict[k][mode][log_name] /= N_dict[mode][log_name] + 1e-12
                loss_dict[k][mode][log_name] /= N_dict[mode][log_name] + 1e-12

        """ Logging """
        log_dict = {}

        print(f"Inner steps {inner_steps}")
        for mode in self.modes:
            for Rrange in self.args.Rrange_lists[mode]:
                Rmin, Rmax = Rrange
                log_name = f"Rmin:{Rmin:.3f}-Rmax:{Rmax:.3f}"
                print(
                    f"{mode:>10s}: {log_name} Loss {loss_dict['query'][mode][log_name]:.5f}, Metric ({losses.metric_name[mode]}) {metric_dict['query'][mode][log_name]:.5f}"
                )

        for k in ["support", "non-support"]:
            for mode in self.modes:
                for log_name in metric_dict[k][mode]:
                    log_dict[f"val-loss-{mode}-{log_name}-{k}"] = loss_dict[k][mode][log_name]
                    log_dict[f"val-metric-{mode}-{log_name}-{k}"] = metric_dict[k][mode][log_name]

        for mode in self.modes:
            for log_name in metric_dict["query"][mode]:
                log_dict[f"val-loss-{mode}-{log_name}"] = loss_dict["query"][mode][log_name]
                log_dict[f"val-metric-{mode}-{log_name}"] = metric_dict["query"][mode][log_name]

        if self.args.use_wandb and wandb.run:
            # Visualize samples
            vis = self.visualize(
                num_vis=num_vis,
                xs_support_dict=xs_support_dict,
                ys_support_dict=ys_support_dict,
                ms_support_dict=ms_support_dict,
                ids_restore_dict=ids_restore_dict,
                xs_query_dict=xs_query_dict,
                ys_query_dict=ys_query_dict,
                ys_recon_dict=ys_recon_dict,
                ys_recon_init_dict=ys_recon_init_dict,
                data_dict=data_dict,
            )

            if vis is None:
                pass

            elif isinstance(vis, dict):
                for mode in self.modes:
                    log_dict[f"val-recon-{mode}"] = wandb.Image(vis[mode], caption=f"mode:{mode}")
            else:
                log_dict[f"val-recon"] = wandb.Image(vis, caption=f"modes:{self.modes}")

            wandb.log(log_dict)

        plt.close()

        return log_dict

    @torch.no_grad()
    def extract_meta_test_set(self, dataloader):
        savepath = Path(f"{self.args.dataset_config['meta_test_path']}/nr:{self.args.validation_repeat}.pt")
        savepath.parent.mkdir(parents=True, exist_ok=True)
        print(savepath)

        distM_dict = {mode: Beta(self.args.Ma[mode], self.args.Mb[mode]) for mode in self.modes}

        xs_support_dict = {mode: [] for mode in self.sample_modes}
        ys_support_dict = {mode: [] for mode in self.sample_modes}
        ms_support_dict = {mode: [] for mode in self.sample_modes}

        xs_query_dict = {mode: [] for mode in self.sample_modes}
        ys_query_dict = {mode: [] for mode in self.sample_modes}

        ids_sample_dict = {mode: [] for mode in self.sample_modes}
        ids_restore_dict = {mode: [] for mode in self.sample_modes}

        info_dict = {mode: {"a": []} for mode in self.sample_modes}

        for data_dict, info_dict_ in tqdm(dataloader):
            meta_test_set_info = self.create_supports_and_querys(data_dict, distM_dict)
            batch_size = meta_test_set_info[0]

            for mode in self.sample_modes:
                xs_support_dict[mode] += [to_device(meta_test_set_info[1][mode][i], "cpu") for i in range(batch_size)]
                ys_support_dict[mode] += [to_device(meta_test_set_info[2][mode][i], "cpu") for i in range(batch_size)]
                ms_support_dict[mode] += [to_device(meta_test_set_info[3][mode][i], "cpu") for i in range(batch_size)]
                ids_restore_dict[mode] += [to_device(meta_test_set_info[4][mode][i], "cpu") for i in range(batch_size)]
                ids_sample_dict[mode] += [to_device(meta_test_set_info[5][mode][i], "cpu") for i in range(batch_size)]
                xs_query_dict[mode] += [to_device(meta_test_set_info[6][mode][i], "cpu") for i in range(batch_size)]
                ys_query_dict[mode] += [to_device(meta_test_set_info[7][mode][i], "cpu") for i in range(batch_size)]

                if self.args.dataset_config["name"] in ["synthetic"]:
                    info_dict[mode]["a"] += [to_device(info_dict_[mode]["a"][i], "cpu") for i in range(batch_size)]

        print("Done")

        extracted_data = {
            "xs_support_dict": xs_support_dict,
            "ys_support_dict": ys_support_dict,
            "ms_support_dict": ms_support_dict,
            "xs_query_dict": xs_query_dict,
            "ys_query_dict": ys_query_dict,
            "ids_sample_dict": ids_sample_dict,
            "ids_restore_dict": ids_restore_dict,
        }
        if self.args.dataset_config["name"] in ["synthetic"]:
            extracted_data["info_dict"] = info_dict
        extracted_data = to_device(extracted_data, "cpu")
        torch.save(extracted_data, savepath)

    @torch.no_grad()
    def extract_meta_test_qual_set(self, dataloader):
        Rs = "_".join([str(int(v)) for v in [self.args.Ma[self.sample_modes[0]], self.args.Mb[self.sample_modes[0]]]])
        savepath = Path(
            f"{self.args.dataset_config['meta_test_path']}/qual-Rs:{Rs}-nr:{self.args.validation_repeat}.pt"
        )
        savepath.parent.mkdir(parents=True, exist_ok=True)
        print(savepath)

        distM_dict = {mode: Beta(self.args.Ma[mode], self.args.Mb[mode]) for mode in self.modes}

        xs_support_dict = {mode: [] for mode in self.sample_modes}
        ys_support_dict = {mode: [] for mode in self.sample_modes}
        ms_support_dict = {mode: [] for mode in self.sample_modes}

        xs_query_dict = {mode: [] for mode in self.sample_modes}
        ys_query_dict = {mode: [] for mode in self.sample_modes}

        ids_restore_dict = {mode: [] for mode in self.sample_modes}

        info_dict = {mode: {"a": []} for mode in self.sample_modes}

        for data_dict, info_dict_ in tqdm(dataloader):
            meta_test_set_info = self.create_supports_and_querys(data_dict, distM_dict)
            batch_size = meta_test_set_info[0]

            for mode in self.sample_modes:
                xs_support_dict[mode] += [meta_test_set_info[1][mode][i] for i in range(batch_size)]
                ys_support_dict[mode] += [meta_test_set_info[2][mode][i] for i in range(batch_size)]
                ms_support_dict[mode] += [meta_test_set_info[3][mode][i] for i in range(batch_size)]
                ids_restore_dict[mode] += [meta_test_set_info[4][mode][i] for i in range(batch_size)]

                xs_query_dict[mode] += [meta_test_set_info[6][mode][i] for i in range(batch_size)]
                ys_query_dict[mode] += [meta_test_set_info[7][mode][i] for i in range(batch_size)]

                if self.args.dataset_config["name"] in ["synthetic"]:
                    info_dict[mode]["a"] += [info_dict_[mode]["a"][i] for i in range(batch_size)]

        extracted_data = {
            "xs_support_dict": xs_support_dict,
            "ys_support_dict": ys_support_dict,
            "ms_support_dict": ms_support_dict,
            "xs_query_dict": xs_query_dict,
            "ys_query_dict": ys_query_dict,
            "ids_restore_dict": ids_restore_dict,
        }
        if self.args.dataset_config["name"] in ["synthetic"]:
            extracted_data["info_dict"] = info_dict
        extracted_data = to_device(extracted_data, "cpu")
        torch.save(extracted_data, savepath)
