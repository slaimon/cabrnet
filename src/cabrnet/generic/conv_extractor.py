import os
import warnings
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as torch_models
from loguru import logger
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from cabrnet.utils.init import layer_init_functions

warnings.filterwarnings("ignore")


class ConvExtractor(nn.Module):
    """Class representing the feature extractor.

    Attributes:
        arch_name: Architecture name.
        weights: Weights for the network.
        layer: Layer to inspect.
        convnet: Graph module that represents the intermediate nodes from the given model.
        add_on: Add-on layers configuration.
        output_channels: Number of output channels of the feature extractor
    """

    def __init__(
        self, arch: str, weights: str | None, layer: str, add_on: dict[str, dict], seed: int | None = None
    ) -> None:
        """Initialize ConvExtractor.

        Args:
            arch: Architecture name.
            weights: Weights for the network.
            layer: Layer to inspect.
            add_on: Add-on layers configuration.
            seed: Random seed (used only to resynchronise random number generators in compatibility tests)
        """
        super(ConvExtractor, self).__init__()
        assert arch.lower() in torch_models.list_models(), f"Unsupported model architecture: {arch}"

        if weights is None:
            logger.warning(f"Random initialisation of feature extractor with architecture {arch}")
            model = torch_models.get_model(arch)
        elif os.path.isfile(weights):
            logger.info(f"Loading state dict for feature extractor: {weights}")
            loaded_weights = torch.load(weights, map_location="cpu")
            model = torch_models.get_model(arch)
            if isinstance(loaded_weights, dict):
                model.load_state_dict(loaded_weights)
            elif isinstance(loaded_weights, nn.Module):
                model.load_state_dict(loaded_weights.state_dict(), strict=False)
            else:
                raise ValueError(f"Unsupported weights type: {type(loaded_weights)}")
        elif hasattr(torch_models.get_model_weights(arch), weights):
            logger.info(f"Loading pytorch weights: {weights}")
            loaded_weights = getattr(torch_models.get_model_weights(arch), weights)
            model = torch_models.get_model(arch, weights=loaded_weights)
        else:
            raise ValueError(f"Cannot load weights {weights} for model of type {arch}. Possible typo or missing file.")

        if seed is not None:
            # Reset random generator (compatibility tests only)
            torch.manual_seed(seed)

        self.arch_name = arch.lower()
        self.weights = weights
        self.layer = layer
        try:
            self.convnet = create_feature_extractor(model=model, return_nodes={layer: "convnet"})
        except ValueError as e:
            logger.error(f"Could not create feature extractor. Possible layer names: {get_graph_node_names(model)}")
            logger.error(f"See model architecture below")
            logger.info(model)
            raise e
        # Dummy inference to recover number of output channels from the feature extractor
        self.convnet.eval()
        in_channels = self.convnet(torch.zeros((1, 3, 224, 224)))["convnet"].size(1)
        self.add_on, self.output_channels = self.create_add_on(config=add_on, in_channels=in_channels)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Define the computation performed at every call.

        Args:
            x: Tensor to run the model on.

        Returns:
            The tensor resulting from the inference of the model.
        """
        x = self.convnet(x)
        if isinstance(x, dict):
            # Output of a create_feature_extractor
            x = x["convnet"]  # type: ignore
        x = self.add_on(x)
        return x

    @staticmethod
    def create_add_on(config: dict[str, dict], in_channels: int) -> Tuple[nn.Sequential, int]:
        """Build add-on layers based on configuration.

        Args:
            config: Add-on layers configuration.
            in_channels: Number of input channels (as given by feature extractor).

        Returns:
            Module containing all add-on layers

        Raises:
            ValueError when configuration is invalid
        """
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        init_mode = None
        for idx, (key, val) in enumerate(config.items()):
            if key == "init_mode":
                # Extract initialisation mode
                if val not in layer_init_functions:
                    raise ValueError(f"Unsupported add_on layers initialisation mode {val}")
                init_mode = val
                continue
            if not hasattr(nn, val["type"]):
                raise ValueError(f"Module {val['type']} not found in torch.nn")
            params = val.get("params")
            if params is not None:
                if val["type"] == "Conv2d":
                    # Check or update in_channels
                    if params.get("in_channels") is None:
                        params["in_channels"] = in_channels
                    elif params["in_channels"] != in_channels:
                        raise ValueError(
                            f"Invalid number of input channels for layer {key}. "
                            f"Should be {in_channels} but {params['in_channels']} was given."
                        )
                    in_channels = params["out_channels"]
                layer_module = getattr(nn, val["type"])(**params)
            else:
                layer_module = getattr(nn, val["type"])()
            layers[key] = layer_module
        add_on = nn.Sequential(layers)

        # Apply initialisation function (if any)
        if init_mode:
            add_on.apply(layer_init_functions[init_mode])

        return add_on, in_channels

    @staticmethod
    def build_from_dict(config: dict[str, dict], seed: int | None = None) -> nn.Module:
        """
        Builds a ConvExtractor from a configuration dictionary.
        Args:
            config: Configuration dictionary
            seed: Random seed (used only to resynchronise random number generators in compatibility tests)
        Returns:
            ConvExtractor

        Raises:
            ValueError when configuration is invalid
        """
        for mandatory_key in ["backbone", "add_on"]:
            if mandatory_key not in config:
                raise ValueError(f"Missing mandatory key {mandatory_key} in extractor configuration")
        for mandatory_key in ["arch", "weights", "layer"]:
            if mandatory_key not in config["backbone"]:
                raise ValueError(f"Missing mandatory key {mandatory_key} in backbone configuration")
        backbone = config["backbone"]
        add_on = config["add_on"]
        return ConvExtractor(
            arch=backbone["arch"], weights=backbone["weights"], layer=backbone["layer"], add_on=add_on, seed=seed
        )
