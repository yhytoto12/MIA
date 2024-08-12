import mia.conversion as conversion
from data import era5, image, synthetic, audiovisual
from PIL import Image
from torchvision import transforms
import yaml
from pathlib import Path

from mia.models.models import MultimodalMetaModel


def get_dataset_root(dataset_name: str):
    """Returns path to data based on dataset_paths.yaml file."""
    with open(r"data/dataset_paths.yaml") as f:
        dataset_paths = yaml.safe_load(f)

    return Path(dataset_paths[dataset_name])


def get_datasets_and_converter(args):
    if "celeba" in (args.train_dataset, args.test_dataset):
        return get_datasets_and_converter_celeba(args)
    elif "synthetic" in (args.train_dataset, args.test_dataset):
        return get_datasets_and_converter_synthetic(args)
    elif "avmnist" in (args.train_dataset, args.test_dataset):
        return get_datasets_and_converter_avmnist(args)
    elif "era5" in (args.train_dataset, args.test_dataset):
        return get_datasets_and_converter_era5(args)
    else:
        raise NotImplementedError


def get_datasets_and_converter_era5(args):
    """Returns train and test datasets as well as appropriate data converters.

    Args:
        args: Arguments parsed from input.
    """

    converter = conversion.Converter("era5")
    converter = {mode: converter for mode in args.sample_modes}

    transforms_dict = {
        "temperature": lambda x: x,
        "pressure": lambda x: x,
        "humidity": lambda x: x,
    }

    train_dataset = era5.ERA5(
        root=args.dataset_config.path,
        train=True,
        transform_dict=transforms_dict,
        modes=args.sample_modes,
        ratio=args.num_data_ratio,
    )

    if args.use_meta_test_set:
        test_dataset = era5.ERA5Meta(
            root=args.dataset_config.path,
            modes=args.sample_modes,
            ratio=1.0,
            repeat_factor=args.validation_repeat,
        )
    else:
        test_dataset = era5.ERA5(
            root=args.dataset_config.path,
            train=False,
            transform_dict=transforms_dict,
            modes=args.sample_modes,
            ratio=1.0,
            repeat_factor=args.validation_repeat,
        )

    print(f"Num train data: {len(train_dataset)}")
    print(f"Num valid data: {len(test_dataset)}")
    return train_dataset, test_dataset, converter


def get_datasets_and_converter_avmnist(args):
    """Returns train and test datasets as well as appropriate data converters.

    Args:
        args: Arguments parsed from input.
    """
    import warnings

    warnings.warn("dataset is hard-coded")

    converter = {
        "rgb": conversion.Converter("image"),
        "wav": conversion.Converter("audio"),
    }

    train_dataset = audiovisual.AVMNIST(
        root=args.dataset_config.path,
        train=True,
        configs=args.dataset_config.modality,
    )

    if args.use_meta_test_set:
        # raise NotImplementedError
        test_dataset = audiovisual.AVMNISTMeta(
            root=args.dataset_config.path,
            configs=args.dataset_config.modality,
            repeat_factor=args.validation_repeat,
        )
    else:
        test_dataset = audiovisual.AVMNIST(
            root=args.dataset_config.path,
            train=False,
            configs=args.dataset_config.modality,
            repeat_factor=args.validation_repeat,
        )

    print(f"Num train data: {len(train_dataset)}")
    print(f"Num valid data: {len(test_dataset)}")
    return train_dataset, test_dataset, converter


def get_datasets_and_converter_synthetic(args):
    """Returns train and test datasets as well as appropriate data converters.

    Args:
        args: Arguments parsed from input.
    """
    converter = conversion.Converter("synthetic")
    converter = {mode: converter for mode in args.sample_modes}

    train_dataset = synthetic.Synthetic(
        root=args.dataset_config.path,
        train=True,
        modes=args.sample_modes,
        ratio=args.num_data_ratio,
        use_jitter=args.dataset_config["use_jitter"],
    )

    if args.use_meta_test_set:
        test_dataset = synthetic.SyntheticMeta(
            root=args.dataset_config.path,
            modes=args.sample_modes,
            ratio=1.0,
            use_jitter=args.dataset_config["use_jitter"],
            repeat_factor=args.validation_repeat,
        )
    else:
        test_dataset = synthetic.Synthetic(
            root=args.dataset_config.path,
            train=False,
            modes=args.sample_modes,
            ratio=1.0,
            use_jitter=args.dataset_config["use_jitter"],
            repeat_factor=args.validation_repeat,
        )

    print(f"Num train data: {len(train_dataset)}")
    print(f"Num valid data: {len(test_dataset)}")
    return train_dataset, test_dataset, converter


def get_datasets_and_converter_celeba(args):
    """Returns train and test datasets as well as appropriate data converters.

    Args:
        args: Arguments parsed from input.
    """

    converter = conversion.Converter("image")
    converter = {mode: converter for mode in args.sample_modes}

    size = args.dataset_config["img_size"]
    transforms_dict = {
        "rgb": transforms.Compose(
            [transforms.Resize(size, interpolation=Image.BICUBIC), transforms.CenterCrop(size), transforms.ToTensor()]
        ),
        "sketch": transforms.Compose(
            [transforms.Resize(size, interpolation=Image.BILINEAR), transforms.CenterCrop(size), transforms.ToTensor()]
        ),
        "normal": transforms.Compose(
            [transforms.Resize(size, interpolation=Image.NEAREST), transforms.CenterCrop(size), transforms.ToTensor()]
        ),
        "semseg": transforms.Compose(
            [
                transforms.Resize(size, interpolation=Image.NEAREST),
                transforms.CenterCrop(size),
                transforms.PILToTensor(),
            ]
        ),
    }

    train_dataset = image.CelebA(
        root=args.dataset_config.path,
        train=True,
        transform_dict=transforms_dict,
        modes=args.sample_modes,
        ratio=args.num_data_ratio,
        size=size,
    )

    if args.use_meta_test_set:
        test_dataset = image.CelebAMeta(
            root=args.dataset_config.path,
            modes=args.sample_modes,
            ratio=1.0,
            size=size,
            repeat_factor=args.validation_repeat,
        )
    else:
        test_dataset = image.CelebA(
            root=args.dataset_config.path,
            train=False,
            transform_dict=transforms_dict,
            modes=args.sample_modes,
            ratio=1.0,
            size=size,
            repeat_factor=args.validation_repeat,
        )

    print(f"Num train data: {len(train_dataset)}")
    print(f"Num valid data: {len(test_dataset)}")
    return train_dataset, test_dataset, converter


def get_model(args):
    return MultimodalMetaModel(
        args=args,
        modes=args.modes,
        latent_spatial_shapes=args.latent_spatial_shapes,
        latent_dims=args.latent_dims,
        inr_dict={
            "dim_hidden": args.dim_hidden,
            "num_layers": args.num_layers,
            "inr_type": args.inr_type,
            "modulate_scale": args.modulate_scale,
            "modulate_shift": args.modulate_shift,
            "modulate_first": args.modulate_first,
            "w0": args.w0,
            "w0_initial": args.w0_initial,
            "ff_dim": args.ff_dim,
            "sigma": args.sigma,
        },
        grad_encoder_dict={
            "type": args.grad_encoder_type,
            "dim": args.grad_encoder_dim,
            "um_depth": args.grad_encoder_um_depth,
            "mm_depth": args.grad_encoder_mm_depth,
            "heads": args.grad_encoder_heads,
            "dim_head": args.grad_encoder_dim // args.grad_encoder_heads,
            "mlp_ratio": args.grad_encoder_mlp_ratio,
            "dropout": args.grad_encoder_dropout,
            "pos_embed_type": args.grad_encoder_pos_embed_type,
            "use_grad": args.grad_encoder_use_grad,
            "use_latent": args.grad_encoder_use_latent,
            #
            "use_fuser": args.grad_encoder_use_fuser,
            "depth_fuser": args.grad_encoder_depth_fuser,
            "use_grad_scaler": args.grad_encoder_use_grad_scaler,
            "grad_scaler_learnable": args.grad_encoder_grad_scaler_learnable,
            "grad_scaler_init": args.grad_encoder_grad_scaler_init,
            "projection_mlp_depth": args.grad_encoder_projection_mlp_depth,
        },
        meta_sgd_dict={
            "use_meta_sgd": args.use_meta_sgd,
            "inner_lr_init": args.meta_sgd_lr_init,
        },
    ).to(args.device)
