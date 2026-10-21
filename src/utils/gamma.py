import torch
import torch.nn.functional as F


def gamma_correction(image: torch.Tensor, gamma: float = 1.2, A: float = 1.0) -> torch.Tensor:
    return A * torch.pow(image, gamma)
