import torch
import torch.nn.functional as F


def gamma_correction(image: torch.Tensor, gamma: float = 1.2, A: float = 1.0) -> torch.Tensor:
    """Apply gamma correction: I_norm = A * I_raw^gamma.

    Args:
        image: Input tensor of shape (B, C, H, W) with values in [0, 1].
        gamma: Gamma value (default 1.2).
        A: Scaling factor (default 1.0).

    Returns:
        Gamma-corrected tensor.
    """
    return A * torch.pow(image, gamma)
