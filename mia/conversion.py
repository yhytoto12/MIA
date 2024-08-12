import torch


class Converter:
    """Class that converts data to coordinates and features and back.

    Args:
        data_type (string): One of 'image', 'mri', 'era5' or 'audio'.

    Notes:
        For images, MRI, ERA5 and audio we assume the coordinates are fixed so we don't
        recalculate them at every conversion.
    """

    def __init__(self, data_type="image"):
        assert data_type in ("image", "era5", "audio", "synthetic")
        self.data_type = data_type
        self.coordinates = None

    def to_coordinates_and_features(self, data):
        """
        Args:
            data (torch.Tensor):
        """
        if self.data_type in ("audio", "synthetic"):
            # If first conversion, calculate coordinates, otherwise reuse
            if self.coordinates == None:
                # Data has shape ({batch_size,} channels, width)
                self.coordinates = shape2coordinates(data.shape[-1:]).to(data.device)

            # If data has 3 dimensions, it is batched
            if data.ndim == 3:
                coordinates = repeat_coordinates(self.coordinates, data.shape[0])
                features = data2features(data, batched=True)
            else:
                coordinates = self.coordinates
                features = data2features(data, batched=False)
            return coordinates, features

        elif self.data_type in ("image", "era5"):
            # If first conversion, calculate coordinates, otherwise reuse
            if self.coordinates == None:
                self.coordinates = shape2coordinates(data.shape[-2:]).to(data.device)

            # If data has 4 dimensions, it is batched
            if data.ndim == 4:
                coordinates = repeat_coordinates(self.coordinates, data.shape[0])
                features = data2features(data, batched=True)
            else:
                coordinates = self.coordinates
                features = data2features(data, batched=False)
            return coordinates, features

    def to_data(self, coordinates, features):
        """
        Args:
            coordinates (torch.Tensor): Unused for 'era5', 'image', 'mri' and 'audio'.
            features (torch.Tensor):
        """
        if self.data_type == "audio":
            return features2data(features, batched=features.ndim == 3)
        elif self.data_type in ("image", "era5"):
            return features2data(features, batched=features.ndim == 4)


def data2features(data: torch.Tensor, batched: bool = False) -> torch.Tensor:
    """Converts an audio sample, image or volume to a features tensor of shape
    ({batch,} {depth x height} x width}, channel).

    Args:
        data (torch.Tensor): Shape (batch_size, channels, *) if batched is True
            or (channels, *) if batched is False, where * refers to any spatial
            dimensions, e.g. (H, W).
        batched (bool): If True, considers first dimension as batch dimension.

    Returns:
        torch.Tensor: of shape (batch_size, *, channels) or (*, channels).
    """
    # Move channels dimension to last axis
    if batched:
        return torch.moveaxis(data, 1, -1)
    else:
        return torch.moveaxis(data, 0, -1)


def features2data(features, batched=False):
    """Inverse function of data2features."""
    # Move channels dimension to first non batch axis
    if batched:
        return torch.moveaxis(features, -1, 1)
    else:
        return torch.moveaxis(features, -1, 0)

def shape2coordinates(spatial_shape: torch.Size, batch_size: int = 0):
    """Converts a shape tuple to a tensor of coordinates.

    Args:
        spatial_shape (tuple of ints): Tuple describing shape of data. For
            example (height, width) or (depth, height, width).
        batch_size (int): If not zero, repeats the coordinate tensor to create
            a batch of coordinates.

    Notes:
        The coordinate tensor will have coordinates lying in [0, 1] regardless
        of the input shape. Be careful if you have inputs that have very non
        square shapes, e.g. (4, 128) as each coordinate grid might then need to
        be scaled differently.
    """
    coords = []
    for i in range(len(spatial_shape)):
        coords.append(torch.linspace(-1.0, 1.0, spatial_shape[i]))
    # Tensor will have shape (*spatial_shape, len(spatial_shape))
    return torch.stack(torch.meshgrid(*coords, indexing="ij"), dim=-1)


def repeat_coordinates(coordinates, batch_size):
    """Repeats the coordinate tensor to create a batch of coordinates.

    Args:
        coordinates (torch.Tensor): Shape (*spatial_shape, len(spatial_shape)).
        batch_size (int): If not zero, repeats the coordinate tensor to create
            a batch of coordinates.
    """
    if batch_size:
        ones_like_shape = (1,) * coordinates.ndim
        return coordinates.unsqueeze(0).repeat(batch_size, *ones_like_shape)
    else:
        return coordinates


if __name__ == "__main__":
    converter = Converter("image")
    imgs = torch.rand(8, 3, 32, 32)
    coords, feats = converter.to_coordinates_and_features(imgs)
    print(coords.shape)
    print(feats.shape)
    print(coords[0, :4, :4])
    coords, feats = converter.to_coordinates_and_features(imgs[0])
    print(coords.shape)
    print(feats.shape)

    converter = Converter("era5")
    temperatures = torch.rand(5, 1, 46, 90)
    coords, feats = converter.to_coordinates_and_features(temperatures)
    print(coords.shape)
    print(feats.shape)
    rec_temps = converter.to_data(coords, feats)
    print((rec_temps - temperatures).abs().sum())

    converter = Converter("audio")
    audio = torch.rand(8, 1, 1000)
    coords, feats = converter.to_coordinates_and_features(audio)
    print(coords.shape)
    print(feats.shape)
