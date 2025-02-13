import torch
from torch import Tensor

prototype_init_modes = ["NORMAL", "SHIFTED_NORMAL", "UNIFORM"]


def init_prototypes(
    num_prototypes: int,
    num_features: int,
    init_mode: str = "SHIFTED_NORMAL",
    data_dimensions: int = 2
) -> Tensor:
    r"""Creates a tensor of prototypes.

    Args:
        num_prototypes (int): Number of prototypes.
        num_features (int): Size of each prototype.
        init_mode (str, optional): Initialisation mode. Default: SHIFTED_NORMAL = N(0.5, 0.1).
        data_dimensions (int, optional): Dimensionality of the input data. Default: 2 (2D images)
    """
    data = (1,)*data_dimensions

    if init_mode not in prototype_init_modes:
        raise ValueError(f"Unknown prototype initialisation mode {init_mode}.")
    if init_mode == "UNIFORM":
        return torch.rand((num_prototypes, num_features)+data, requires_grad=True)
    elif init_mode == "NORMAL":
        return torch.randn((num_prototypes, num_features)+data, requires_grad=True)
    else:  # SHIFTED_NORMAL
        prototypes = torch.randn((num_prototypes, num_features)+data, requires_grad=True)
        torch.nn.init.normal_(prototypes, mean=0.5, std=0.1)
        return prototypes
