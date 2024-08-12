import torch
from einops import rearrange

mse_fn = torch.nn.MSELoss()
per_element_mse_fn = torch.nn.MSELoss(reduction="none")


def per_element_loss_fn(pred, target, element_mask=None):
    # pred shape (batch_size, num_elements, Dy)
    # target shape (batch_size, num_elements, Dy)
    per_element_loss = per_element_mse_fn(pred, target).mean(dim=-1)

    if element_mask is not None:
        per_element_loss[element_mask] = 0

    return per_element_loss


metric_name = {
    # For 2D
    "rgb": "psnr",
    "normal": "psnr",
    "sketch": "psnr",
    # For Synthetic
    "sine": "psnr",
    "tanh": "psnr",
    "gaussian": "psnr",
    "relu": "psnr",
    # For Audio
    "wav": "psnr",
    # For ERA5
    "temperature": "psnr",
    "pressure": "psnr",
    "precipitation": "psnr",
    "humidity": "psnr",
}

loss_fn = {
    # For 2D
    "rgb": mse_fn,
    "depth": mse_fn,
    "normal": mse_fn,
    "sketch": mse_fn,
    # For Synthetic
    "sine": mse_fn,
    "tanh": mse_fn,
    "gaussian": mse_fn,
    "relu": mse_fn,
    # For Audio
    "wav": mse_fn,
    # For ERA5
    "temperature": mse_fn,
    "pressure": mse_fn,
    "humidity": mse_fn,
}


def batch_loss_fn(pred, target, norm=1.0):
    """Compute Loss between two batches of precictions and targets
    while preserving the batch dimension (per batch element loss)
    Args:
        pred (torch.Tensor): Shape (batch_size, N, Dy)
        target (torch.Tensor): Shape (batch_size, N, 1)
    Returns:
        Loss Tensor of shape (batch_size, )
    """
    per_element_loss = per_element_mse_fn(pred, target).div(norm)
    batch_loss = per_element_loss.view(pred.shape[0], -1).mean(dim=1)

    return batch_loss


def batch_mse_fn(x1, x2):
    """Computes MSE between two batches of signals while preserving the batch
    dimension (per batch element MSE).
    Args:
        x1 (torch.Tensor): Shape (batch_size, *).
        x2 (torch.Tensor): Shape (batch_size, *).
    Returns:
        MSE tensor of shape (batch_size,).
    """
    # Shape (batch_size, *)
    per_element_mse = per_element_mse_fn(x1, x2)
    # Shape (batch_size,)
    return per_element_mse.view(x1.shape[0], -1).mean(dim=1)


@torch.no_grad()
def batch_metric_fn(pred, target, norm=1.0, eps=1e-9):
    """Compute Pre-defined Metric between two batches of predictions and targets
    while preserving the batch dimension (per batch element metric)
    Args:
        pred (torch.Tensor): Shape (batch_size, L, Dy)
        target (torch.Tensor): Shape (batch_size, L, Dy)
    Returns:
        Metric Tensor of shape (batch_size,)
    """
    # Use PSNR metric
    # pred shape (batch_size, N, Dy)
    # target shape (batch_size, N, Dy)
    peak = 1.0
    noise = (pred - target).pow(2).div(norm).mean(1, keepdim=True)  # (batch_size, 1, Dy)
    # batchwise_mse = noise.mean([1, 2])
    batchwise_channelwise_psnr = 10 * torch.log10(peak / (noise + eps))  # (batch_size, 1, Dy)
    batchwise_psnr = batchwise_channelwise_psnr.mean([1, 2])

    return batchwise_psnr


@torch.no_grad()
def batch_loss_fn_with_element_mask(pred, target, element_mask, norm=1.0, eps=1e-9):
    """_summary_

    Args:
        pred (torch.Tensor): Shape (batch_size, num_points, Dy)
        target (torch.Tensor): Shape (batch_size, num_points, Dy)
        mode (str): name of modality
        element_mask (torch.Tensor): Shape (batch_size, num_points)
        eps (_type_, optional): eps. Defaults to 1e-9.

    Returns:
        Tuple(torch.Tensor, torch.Tensor, torch.Tensor): (query loss, support loss, non-support loss)
    """
    per_element_loss = per_element_mse_fn(pred, target).div(norm).mean(dim=-1)

    per_element_loss_support = per_element_loss.masked_fill(~element_mask, torch.nan)
    per_element_loss_non_support = per_element_loss.masked_fill(element_mask, torch.nan)

    return (
        per_element_loss.mean(dim=1),
        per_element_loss_support.nanmean(dim=1),
        per_element_loss_non_support.nanmean(dim=1),
    )


@torch.no_grad()
def batch_metric_fn_with_element_mask(pred, target, element_mask, norm=1.0, eps=1e-9):
    """
    Args:
        pred (torch.Tensor): Shape (batch_size, num_points, Dy)
        target (torch.Tensor): Shape (batch_size, num_points, Dy)
        mode (str): name of modality
        element_mask (torch.Tensor): Shape (batch_size, num_points)
        eps (_type_, optional): eps. Defaults to 1e-9.

    Returns:
        Tuple(torch.Tensor, torch.Tensor, torch.Tensor): (query metric, support metric, non-support metric)
    """
    peak = 1.0
    noise = (pred - target).pow(2).div(norm)  # (batch_size, N, Dy)
    noise_support = noise.masked_fill(~element_mask.unsqueeze(-1), torch.nan).nanmean(dim=1, keepdim=True)
    noise_non_support = noise.masked_fill(element_mask.unsqueeze(-1), torch.nan).nanmean(dim=1, keepdim=True)
    noise = noise.mean(dim=1, keepdim=True)

    batchwise_channelwise_psnr = 10 * torch.log10(peak / (noise + eps))  # (batch_size, 1, Dy)
    batchwise_channelwise_psnr_support = 10 * torch.log10(peak / (noise_support + eps))  # (batch_size, 1, Dy)
    batchwise_channelwise_psnr_non_support = 10 * torch.log10(peak / (noise_non_support + eps))  # (batch_size, 1, Dy)
    batchwise_psnr = batchwise_channelwise_psnr.mean([1, 2])
    batchwise_psnr_support = batchwise_channelwise_psnr_support.mean([1, 2])
    batchwise_psnr_non_support = batchwise_channelwise_psnr_non_support.mean([1, 2])

    return batchwise_psnr, batchwise_psnr_support, batchwise_psnr_non_support
