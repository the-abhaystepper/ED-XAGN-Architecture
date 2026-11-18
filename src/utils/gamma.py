import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def gamma_correction(image: torch.Tensor, gamma: float = 1.2, A: float = 1.0) -> torch.Tensor:
    return A * torch.pow(image.clamp(min=0.0), gamma)


def imagenet_normalize(image: torch.Tensor) -> torch.Tensor:
    view_shape = (1, -1, 1, 1) if image.ndim == 4 else (-1, 1, 1)
    mean = image.new_tensor(IMAGENET_MEAN).view(*view_shape)
    std = image.new_tensor(IMAGENET_STD).view(*view_shape)
    return (image - mean) / std
