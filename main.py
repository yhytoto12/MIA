import argparse
import helpers
import os
import pprint
import torch
import wandb
import ipdb

from pathlib import Path
from str2bool import str2bool
from omegaconf import OmegaConf
import numpy as np
import random

torch.backends.cudnn.benchmark = True


def add_arguments(parser: argparse.ArgumentParser):
    # INR Model arguments
    # -------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--num_layers", help="Number of layers in base network.", type=int, default=5)
    parser.add_argument("--dim_hidden", help="Dimension of hidden layers of base network.", type=int, default=32)
    parser.add_argument("--w0", help="w0 parameter from SIREN.", type=float, default=30.0)
    parser.add_argument("--w0_initial", help="initial w0 parameter from SIREN.", type=float, default=30.0)
    parser.add_argument("--ff_dim", type=int, default=0)
    parser.add_argument("--sigma", type=float, default=10.0)
    parser.add_argument("--inr_type", type=str, choices=["basic", "siren", "rffn", "ffn", "composer"], default="basic")

    # Modulation arguments
    # -------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--modulate_first", type=str2bool, default=False)
    parser.add_argument("--modulate_scale", help="Whether to modulate scale.", type=int, default=0)
    parser.add_argument("--modulate_shift", help="Whether to modulate shift.", type=int, default=1)
    parser.add_argument("--latent_spatial_shapes", type=str, default="1")
    parser.add_argument("--latent_dims", type=str, default="32")

    # Datasets arguments
    # -------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--dataset_config_path", default="configs/celeba.yaml", type=str)
    parser.add_argument("--num_data_ratio", default=1.0, type=float)
    parser.add_argument("--modes", default="rgb")
    parser.add_argument("--sample_modes", default="rgb,normal,sketch,semseg")
    parser.add_argument("--use_meta_test_set", type=str2bool, default=False)
    parser.add_argument("--extract_meta_test_set", type=str2bool, default=False)
    parser.add_argument("--extract_meta_test_qual_set", type=str2bool, default=False)
    parser.add_argument("--uncorrelated", type=str2bool, default=False)
    parser.add_argument("--uncorrlated_ratio", type=float, default=1.0)

    # Training arguments
    # -------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--seed", help="Random seed. If set to -1, seed is chosen at random.", type=int, default=-1)
    parser.add_argument("--outer_lr", help="Learning rate for the outer loop.", type=float, default=3e-6)
    parser.add_argument("--encoder_lr_ratio", help="Learning rate for the outer loop.", type=float, default=1.0)

    parser.add_argument("--inner_lr", help="Learning rate for the inner loop.", type=float, default=1e-2)
    parser.add_argument("--inner_steps", help="Number of inner loop steps.", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--validation_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=-1)
    parser.add_argument("--num_workers", help="Number of workers for dataloader.", type=int, default=4)
    parser.add_argument(
        "--validate_every", help="Run validation every {validate_every} iterations.", type=int, default=2000
    )
    parser.add_argument("--validation_repeat", help="replicate validation dataset.", type=int, default=1)
    parser.add_argument(
        "--validation_inner_steps", help="List of inner steps to use for validation.", type=str, default="3"
    )
    parser.add_argument("--outer_steps", type=str, default="3")
    parser.add_argument("--use_meta_sgd", type=str2bool, default=True)
    parser.add_argument("--meta_sgd_lr_max", type=float, default=5.0)
    parser.add_argument("--meta_sgd_lr_init", type=float, default=1.0)

    # Wandb arguments
    # -------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--use_wandb", type=str2bool, default=False)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--wandb_project_name", type=str, default="debug")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument(
        "--wandb_job_type", help="Wandb job type. This is useful for grouping runs together.", type=str, default=None
    )
    parser.add_argument("--wandb_run_id", help="", type=str, default=None)
    parser.add_argument("--wandb_tags", help="Wandb tags", type=str, default=None)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--log_image_interval", type=int, default=200)
    parser.add_argument("--num_vis", type=int, default=8)

    # Grad Encoder arguments
    # -------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--grad_encoder_type", type=str, default="transformer")
    parser.add_argument("--grad_encoder_dim", type=int, default=192)
    parser.add_argument("--grad_encoder_um_depth", type=int, default=0)
    parser.add_argument("--grad_encoder_mm_depth", type=int, default=0)
    parser.add_argument("--grad_encoder_heads", type=int, default=3)
    parser.add_argument("--grad_encoder_mlp_ratio", type=float, default=1)
    parser.add_argument("--grad_encoder_dropout", type=float, default=0.00)
    parser.add_argument("--grad_encoder_pos_embed_type", type=str, default="learned")
    parser.add_argument("--grad_encoder_use_latent", type=str2bool, default=True)
    parser.add_argument("--grad_encoder_use_grad", type=str2bool, default=True)

    parser.add_argument("--grad_encoder_use_fuser", type=str2bool, default=True)
    parser.add_argument("--grad_encoder_depth_fuser", type=int, default=2)

    parser.add_argument("--grad_encoder_projection_mlp_depth", type=int, default=2)

    parser.add_argument("--grad_encoder_use_grad_scaler", type=str2bool, default=False)
    parser.add_argument("--grad_encoder_grad_scaler_init", type=float, default=1.0)
    parser.add_argument("--grad_encoder_grad_scaler_learnable", type=str2bool, default=False)
    parser.add_argument("--grad_encoder_grad_scaler_lr_ratio", type=float, default=1.0)

    parser.add_argument("--grad_scale_pos", type=str, default="pre")

    # ALFA arguments
    parser.add_argument("--use_alfa", type=str2bool, default=False)
    parser.add_argument("--dim_alfa", type=int, default=-1)
    parser.add_argument("--depth_alfa", type=int, default=3)

    # GAP arguments
    parser.add_argument("--use_gap", type=str2bool, default=False)
    parser.add_argument("--use_gap_approx", type=str2bool, default=False)

    # Misc arguments
    # -------------------------------------------------------------------------------------------------------------------
    parser.add_argument("--inr_ckpt_idx", type=int, default=-1)

    parser.add_argument("--loss_weight_mode", type=str, default="none", choices=["none", "uncertainty"])
    parser.add_argument("--logvar_lr_ratio", type=float, default=1.0)
    parser.add_argument("--logvar_init", type=str, default="0")

    parser.add_argument("--Rmin", default=1.0)
    parser.add_argument("--Rmax", default=1.0)
    parser.add_argument("--Ma", default=1.0)
    parser.add_argument("--Mb", default=1.0)
    parser.add_argument("--Rrange_lists", type=str, default="0.0-1.0")


# -------------------------------------------------------------------------------------------------------------------
def compute_num_params(model, text=True):
    import numpy as np

    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return "{:.4f}M".format(tot / 1e6)
        elif tot >= 1e3:
            return "{:.4f}K".format(tot / 1e3)
        else:
            return str(tot)
    else:
        return tot


# -------------------------------------------------------------------------------------------------------------------
def main(args):
    args.wandb_run_id = None
    ckpt_dir = args.log_dir / "ckpt"
    ckpt_paths = sorted(ckpt_dir.glob("**/*.pt"), reverse=True)
    ckpt = None

    if args.use_wandb:
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[0]
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
                print(f"Load ckpt from {ckpt['step']}")
            except:
                pass
            else:
                if "wandb_run_id" in ckpt["args"]:
                    args.wandb_run_id = ckpt["args"].wandb_run_id

        if args.wandb_run_id is None:
            wandb_resume = None
        else:
            wandb_resume = "must"

        # Initialize wandb experiment
        if args.name is not None:
            os.environ["WANDB_NAME"] = args.name

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            job_type=args.wandb_job_type,
            tags=args.wandb_tags,
            config=args,
            id=args.wandb_run_id,
            resume=wandb_resume,
        )
        args.wandb_run_id = wandb.run.id

        # Save ENV variables
        with (Path(wandb.run.dir) / "env.txt").open("wt") as f:
            pprint.pprint(dict(os.environ), f)

        # Define path where model will be saved
        model_path = Path(wandb.run.dir) / "model.pt"
    else:
        model_path = ""

    # Optionally set random seed
    if args.seed != -1:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Build datasets, converters and model
    train_dataset, test_dataset, converter = helpers.get_datasets_and_converter(args)
    model = helpers.get_model(args)

    print(model)
    print(args)
    print()
    print(f"# parameters: {compute_num_params(model)}")

    # Optionally save model
    if args.use_wandb:
        torch.save({"args": args, "state_dict": model.state_dict(), "step": 0}, model_path)
        wandb.save(str(model_path.absolute()), base_path=wandb.run.dir, policy="live")
        wandb.log({"num_params": compute_num_params(model, text=False)}, step=0)

    # Initialize trainer and start training
    if args.train_dataset == "synthetic":
        from mia.training_synthetic import SyntheticTrainer as Trainer
    elif args.train_dataset == "celeba":
        from mia.training_celeba import CelebATrainer as Trainer
    elif args.train_dataset in "avmnist":
        from mia.training_avmnist import AVMNISTTrainer as Trainer
    elif args.train_dataset == "era5":
        from mia.training_era5 import ERA5Trainer as Trainer
    else:
        raise NotImplementedError

    trainer = Trainer(
        func_rep=model,
        converter=converter,
        args=args,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        model_path=model_path,
    )

    if args.extract_meta_test_set:
        print("extracting meta test sets")
        trainer.extract_meta_test_set(trainer.test_dataloader)
        exit()

    if args.extract_meta_test_qual_set:
        print("extracting meta test qual sets")
        trainer.extract_meta_test_qual_set(trainer.test_dataloader)
        exit()

    if ckpt is None:
        if len(ckpt_paths) > 0:
            ckpt_path = ckpt_paths[0]
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu")
            except:
                ckpt = None

    if ckpt is not None:
        # load pretrained weights
        print(f"Load pretrained weights from {ckpt_path}")
        trainer.load_pretrained(states=ckpt, strict=True, load_step=True, load_optimizer=True)

    if args.num_epochs is None or args.num_epochs == -1:
        args.num_epochs = float("inf")

    epoch = 0
    out = False
    while not out:
        try:
            print(f"\nEpoch {epoch + 1}:")
            trainer.train_epoch()
            epoch += 1

            if epoch >= args.num_epochs:
                out = True

        except KeyboardInterrupt:
            ipdb.set_trace(context=25)
            print("Interrupted...")


# -------------------------------------------------------------------------------------------------------------------


def to_dict(x, modes, cast_fn):
    if isinstance(x, cast_fn):
        res_dict = {mode: x for mode in modes}
    elif isinstance(x, str):
        str_list = x.split(",")
        if len(str_list) == 1:
            res_dict = {mode: cast_fn(str_list[0]) for mode in modes}
        elif len(str_list) == len(modes):
            res_dict = {mode: cast_fn(v) for mode, v in zip(modes, str_list)}
    else:
        raise
    return res_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    add_arguments(parser)
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.modes = args.modes.split(",")
    if args.sample_modes is None:
        args.sample_modes = args.modes
    else:
        args.sample_modes = args.sample_modes.split(",")
    args.log_dir = Path(args.log_dir) / args.name
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.validation_inner_steps = [int(val) for val in args.validation_inner_steps.split(",")]
    args.outer_steps = [int(val) for val in args.outer_steps.split(",")]

    # read dataset config files
    dataset_config = OmegaConf.load(args.dataset_config_path)
    args.train_dataset = dataset_config.name
    args.test_dataset = dataset_config.name
    args.dataset_config = dataset_config

    args.latent_spatial_shapes = to_dict(args.latent_spatial_shapes, args.sample_modes, int)
    args.latent_dims = to_dict(args.latent_dims, args.sample_modes, int)
    args.Rmin = to_dict(args.Rmin, args.sample_modes, float)
    args.Rmax = to_dict(args.Rmax, args.sample_modes, float)
    args.Ma = to_dict(args.Ma, args.sample_modes, float)
    args.Mb = to_dict(args.Mb, args.sample_modes, float)
    args.logvar_init = to_dict(args.logvar_init, args.sample_modes, float)

    Rrange_lists = args.Rrange_lists.split(",")
    if len(Rrange_lists) == 1:
        args.Rrange_lists = {}
        for mode in args.modes:
            v = [float(v) for v in Rrange_lists[0].split("-")]
            n = len(v)
            args.Rrange_lists[mode] = [(v[i], v[i + 1]) for i in range(n - 1)]
    elif len(Rrange_lists) == len(args.sample_modes):
        args.Rrange_lists = {}
        for j, mode in enumerate(args.sample_modes):
            v = [float(v) for v in Rrange_lists[j].split("-")]
            n = len(v)
            args.Rrange_lists[mode] = [(v[i], v[i + 1]) for i in range(n - 1)]
    else:
        raise ValueError

    if args.wandb_tags is None or args.wandb_tags == "":
        args.wandb_tags = None
    else:
        args.wandb_tags = args.wandb_tags.split(",")

    main(args)
