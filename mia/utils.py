import torch
import torch.nn.functional as F
import numpy as np

from torch.distributions.beta import Beta

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from einops import repeat


def get_output_dims(modes):
    dims_out = {}
    for mode in modes:
        if mode in ["rgb", "curvature", "normal", "reshading", "sketch"]:
            dims_out[mode] = 3
        elif mode in ["depth"]:
            dims_out[mode] = 1
        elif mode in ["semseg"]:
            dims_out[mode] = 19
        else:
            dims_out[mode] = 1

    return dims_out


def get_input_dims(modes):
    dims_out = {}
    for mode in modes:
        if mode in ["rgb", "curvature", "normal", "reshading", "depth", "sketch", "semseg"]:
            dims_out[mode] = 2
        elif mode in ["temperature", "pressure", "precipitation", "humidity"]:
            dims_out[mode] = 2
        else:
            dims_out[mode] = 1

    return dims_out


def get_out_bias(mode):
    if mode in ["rgb", "curvature", "normal", "reshading", "depth", "sketch"]:
        return 0.5
    elif mode in ["temperature", "pressure", "precipitation", "humidity"]:
        return 0.5
    else:
        return 0.0


def get_input_range(mode):
    if mode in ["sine", "tanh", "gaussian", "relu", "wav"]:
        return 1.0
    else:
        return 1.0


def plot_wav(wavs, sample_rate, resize=None, max_size=None, dpi=None):
    # wavs: (B, length, 1) or (B, length)

    if isinstance(wavs, torch.Tensor):
        wavs = wavs.numpy()

    assert isinstance(wavs, np.ndarray)

    if wavs.ndim == 2:
        wavs = wavs[..., None]

    batch_size, num_frames, num_channels = wavs.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    image_arrays = []
    for wav in wavs:
        fig, axes = plt.subplots(num_channels, 1, dpi=dpi)
        if num_channels == 1:
            axes = [axes]
        for c in range(num_channels):
            axes[c].plot(time_axis, wav[:, c], linewidth=1)
            axes[c].grid(True)
            if num_channels > 1:
                axes[c].set_ylabel(f"Channel {c+1}")

            # Hide the axes
            # axes[c].axis('off')

        # Remove marginal paddings
        fig.set_tight_layout(True)
        # fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # Convert figure to numpy array
        canvas = FigureCanvas(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        image_array = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        image_array = image_array.reshape(int(height), int(width), 3)
        image_array = torch.from_numpy(image_array).permute(2, 0, 1)
        image_arrays += [image_array]

        # Close fig to avoid memory leak
        plt.close(fig)

    image_arrays = torch.stack(image_arrays).float().div(255)

    if resize:
        image_arrays = F.interpolate(image_arrays, resize)

    if max_size and max(image_arrays.shape[-2:]) > max_size:
        scale_factor = max_size / max(image_arrays.shape[-2:])
        image_arrays = F.interpolate(image_arrays, scale_factor=scale_factor)

    return image_arrays


def sample_nmr(xs: torch.Tensor, ys: torch.Tensor, num_samples: int):
    B, N, Dx = xs.shape
    Dy = ys.shape[-1]
    device = xs.device

    noise = torch.rand(B, N, device=device)
    ids_sample = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_sample, dim=1)
    ids_sample = ids_sample[:, :num_samples]

    x_nmr = torch.gather(xs, dim=1, index=ids_sample.unsqueeze(-1).repeat(1, 1, Dx))
    y_nmr = torch.gather(ys, dim=1, index=ids_sample.unsqueeze(-1).repeat(1, 1, Dy))

    # 1 if sampled otherwise 0.
    ms_sample = torch.zeros_like(xs[..., 0])
    ms_sample = ms_sample.scatter(1, ids_sample, 1 - ms_sample)

    return x_nmr, y_nmr, ids_restore, ids_sample


def generate_random_masks(
    input_tokens,
    num_sample_min=0,
    num_sample_max=None,
    dist=Beta(1, 1),
):
    B, N = input_tokens.shape[:2]
    num_sample_min = max(0, num_sample_min)
    num_sample_max = min(N, N if num_sample_max is None else num_sample_max)

    if num_sample_min == num_sample_max:
        cutoff = repeat(torch.tensor([num_sample_max + 1]), "1 -> B 1", B=B)
    else:
        cutoff = dist.sample((B, 1)) * (num_sample_max + 1 - num_sample_min) + num_sample_min

    mask = (repeat(torch.arange(1, N + 1), "N -> B N", B=B) > cutoff).long()
    mask = mask.to(input_tokens)
    return mask


def to_device(data, device):
    """
    Load data with arbitrary structure on device.
    from MTP
    """

    def to_device_wrapper(data):
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, tuple):
            return tuple(map(to_device_wrapper, data))
        elif isinstance(data, list):
            return list(map(to_device_wrapper, data))
        elif isinstance(data, dict):
            return {key: to_device_wrapper(data[key]) for key in data}
        else:
            raise NotImplementedError

    return to_device_wrapper(data)
