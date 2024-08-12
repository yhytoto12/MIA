import sys
import argparse
import subprocess as sp
from pprint import pprint

# Common Parameters
# -------------------------------------------------------------
default_config = dict(
    use_wandb=False,
    wandb_project_name="project_name",
    wandb_entity="entity_name",
    wandb_job_type="job_type",
    dataset_config_path="configs/synthetic.yaml",
    sample_modes="sine,tanh,gaussian,relu",
    mode="sine,tanh,gaussian,relu",
    Rmin=0.01,
    Rmax=0.1,
    Ma=1,
    Mb=1,
    Rrange_lists="0.01-0.02-0.05-0.1",
    num_epochs=16000,
    batch_size=64,
    validation_batch_size=20,
    num_vis=8,
    validate_every=2000,
    log_interval=400,
    log_image_interval=1000,
    validation_repeat=10,
    dim_hidden=128,
    num_layers=5,
    ff_dim=128,
    sigma=0,  # use 0 for synthetic experiments
    inr_type="composer",
    meta_sgd_lr_init=1.0,
    meta_sgd_lr_max=5.0,
    inner_steps=3,
    outer_lr=1e-4,
    loss_weight_mode="uncertainty",
    logvar_lr_ratio=1,
    use_meta_test_set=True,
    seed=1234,
    latent_spatial_shapes="8",
    latent_dims="128",
    use_alfa=False,
    use_gap=False,
    use_gap_approx=False,
    grad_scale_pos="pre",
    grad_encoder_type="transformer",
    grad_encoder_use_fuser=False,
    grad_encoder_um_depth=0,
    grad_encoder_mm_depth=0,
    grad_encoder_projection_mlp_depth=0,
    grad_encoder_use_grad_scaler=False,
    grad_encoder_grad_scaler_learnable=False,
    grad_encoder_grad_scaler_init=1,
)

# Specific Parameters
# -------------------------------------------------------------
experiments = {
    "CAVIA": dict(
        wandb_tags="CAVIA",
        use_meta_sgd=False,
    ),
    "MetaSGD": dict(
        wandb_tags="MetaSGD",
        use_meta_sgd=True,
    ),
    "ALFA": dict(
        wandb_tags="ALFA",
        use_meta_sgd=True,
        use_alfa=True,
        dim_alfa=-1,
        depth_alfa=2,
    ),
    "GAP": dict(
        wandb_tags="GAP",
        use_gap=True,
        use_gap_approx=True,
        grad_scale_pos="post",
    ),
    "MIA": dict(
        wandb_tags="MIA",
        use_meta_sgd=False,
        grad_encoder_use_fuser=True,
        grad_encoder_um_depth=1,
        grad_encoder_mm_depth=1,
        grad_encoder_projection_mlp_depth=2,
        grad_encoder_use_grad_scaler=True,
        grad_encoder_grad_scaler_learnable=True,
        grad_encoder_grad_scaler_init=10,
        grad_encoder_dim=192,
        grad_encoder_heads=3,
        grad_encoder_mlp_ratio=1,
    ),
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Run with `python run_synthetic.py CAVIA``
    parser.add_argument(
        "exp_name", type=str, help="Experiment to run (CAVIA, MetaSGD, ALFA, GAP, MIA)", choices=experiments.keys()
    )
    args = parser.parse_args()

    if args.exp_name not in experiments:
        print(f"Experiment {args.exp_name} not found")
        sys.exit(1)

    exp_params = experiments[args.exp_name]
    config = {**default_config, **exp_params}

    print(f"Running experiment {args.exp_name}")
    pprint(config)

    # Run the experiment
    cmd = " ".join(
        [
            "python main.py",
            "--name {}".format(args.exp_name),
        ]
        + [f"--{k} {v}" for k, v in config.items()]
    )

    sp.run(cmd.split())
