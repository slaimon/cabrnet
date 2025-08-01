"""This file holds all the necessary functions to create datasets and dataloaders from configuration files."""

import argparse
import copy
import importlib
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor
import torchvision.transforms
from cabrnet.core.utils.parser import load_config
from loguru import logger
from torch.utils.data import DataLoader, Dataset, Subset


class VisionDatasetSubset(Subset):
    r"""Overwrites the Subset class so that it exposes all properties of a VisionDataset."""

    @property
    def transform(self) -> Any:
        r"""Returns the 'transform' function of the original dataset."""
        return getattr(self.dataset, "transform", None)

    def target_transform(self) -> Any:
        r"""Returns the 'target_transform' function of the original dataset."""
        return getattr(self.dataset, "target_transform", None)

    def transforms(self) -> Any:
        r"""Returns the 'transforms' function of the original dataset."""
        return getattr(self.dataset, "transforms", None)


class DatasetManager:
    r"""Class for handling datasets in CaBRNet."""

    DEFAULT_DATASET_CONFIG: str = "dataset.yml"

    @staticmethod
    def create_parser(
        parser: argparse.ArgumentParser | None = None, mandatory_config: bool = False
    ) -> argparse.ArgumentParser:
        r"""Creates the argument parser for CaBRNet datasets.

        Args:
            parser (ArgumentParser, optional): Existing parser (if any). Default: None.
            mandatory_config (bool, optional): If True, makes the configuration mandatory. Default: False.

        Returns:
            The parser itself.
        """
        if parser is None:
            parser = argparse.ArgumentParser(description="Load datasets.")
        parser.add_argument(
            "-d",
            "--dataset",
            required=mandatory_config,
            metavar="/path/to/file.yml",
            help="path to the dataset config",
        )
        parser.add_argument(
            "--sampling-ratio",
            type=int,
            required=False,
            default=1,
            metavar="ratio",
            help="data sampling ratio (e.g. 5 means only one image in five is used). Default: 1",
        )
        return parser

    # Common torchvision datasets use one of the following keywords to indicate data transformations.
    TRANSFORM_FIELDS = ["transform", "target_transform"]

    # Type of transformation compositions
    TRANSFORM_COMPOSITIONS = ["Compose", "RandomOrder", "RandomChoice"]

    @staticmethod
    def get_transform(trans_config: dict[str, Any]) -> Callable:
        r"""Builds a data transformation from a dictionary.

        Args:
            trans_config (dictionary): Transformation configuration.

        Returns:
            Transformation.

        Raises:
            ValueError whenever the configuration is incorrect.
        """
        if "type" not in trans_config:
            raise ValueError("Missing transformation type.")
        # Load custom module if necessary
        module = importlib.import_module(trans_config["module"]) if "module" in trans_config else torchvision.transforms
        # Check for recursive types
        if trans_config["type"] in DatasetManager.TRANSFORM_COMPOSITIONS:
            if "transforms" not in trans_config:
                raise ValueError(f"Missing <transforms> field for operation {trans_config['type']}.")
            ops = [
                DatasetManager.get_transform(trans_config["transforms"][op_name])
                for op_name in trans_config["transforms"]
            ]
            return getattr(module, trans_config["type"])(ops)

        if "params" in trans_config:
            return getattr(module, trans_config["type"])(**trans_config["params"])
        return getattr(module, trans_config["type"])()

    @staticmethod
    def get_datasets(
        config: str | dict[str, Any], sampling_ratio: int = 1, load_segmentation: bool = False, set_names: list[str] = None
    ) -> dict[str, dict[str, Dataset | int | bool]]:
        r"""Loads datasets from a configuration file.

        Args:
            config (str, dict): Path to configuration file, or configuration dictionary.
            sampling_ratio (int, optional): Sampling ratio (e.g. 5 means only one image in five is used). Default: 1.
            load_segmentation (bool, optional): If True, loads segmentation datasets if available. Default: False.
            set_names (list[str], optional): Only load the specified splits. Default: None (load everything)

        Returns:
            Dictionary of datasets with their respective batch size and shuffle property.

        Raises:
            ValueError whenever a dataset could not be loaded.
        """
        if not isinstance(config, (str, dict)):
            raise ValueError(f"Unsupported configuration format: {type(config)}")
        if isinstance(config, str):
            config = load_config(config)
        if sampling_ratio > 1:
            logger.warning(f"{'=' * 20} SAMPLING RATIO > 1: PROCESSING 1/{sampling_ratio} IMAGES {'=' * 20}")
        datasets: dict[str, dict[str, Dataset | int | bool]] = {}

        # Configuration should include at least train and projection sets
        mandatory_sets = ["train_set", "projection_set", "test_set"]
        for dataset_name in mandatory_sets:
            if dataset_name not in config:
                logger.error(f"Missing configuration for {dataset_name}.")

        set_names = set_names if set_names is not None else config

        for dataset_name in set_names:
            dataset: dict[str, Dataset | int | bool] = {}
            logger.info(f"Loading dataset {dataset_name}")
            dconfig = config[dataset_name]
            for key in ["name", "module", "params", "batch_size", "shuffle"]:
                if key not in dconfig:
                    raise ValueError(f"Missing dataset {key} information")

            params = copy.copy(dconfig["params"])
            for field in params:
                if field in DatasetManager.TRANSFORM_FIELDS:
                    # Replace configuration with actual transformation function
                    ops = [DatasetManager.get_transform(params[field][op_name]) for op_name in params[field]]
                    ops = torchvision.transforms.Compose(ops) if len(ops) > 1 else ops[0]
                    logger.debug(f"{field}: {ops}")
                    params[field] = ops
                else:
                    # Add parameter as is
                    params[field] = params[field]

            batch_size: int = dconfig["batch_size"]
            if batch_size < 1:
                raise ValueError(f"Invalid batch size: {batch_size}.")
            shuffle: bool = dconfig["shuffle"]

            # Load dataset
            module = importlib.import_module(dconfig["module"])
            dataset["dataset"] = getattr(module, dconfig["name"])(**params)
            if "transform" in params:
                # Remove image preprocessing to recover raw images
                params["transform"] = None
            dataset["raw_dataset"] = getattr(module, dconfig["name"])(**params)
            if load_segmentation:
                try:
                    params["root"] += "_seg"
                    dataset["seg_dataset"] = getattr(module, dconfig["name"])(**params)
                except FileNotFoundError:
                    logger.warning(f"Segmentation set unavailable for dataset {dataset_name}")

            if sampling_ratio > 1:
                # Apply data sub-selection
                selected_indices = [idx for idx in range(len(dataset["dataset"]))][::sampling_ratio]
                for key in ["dataset", "raw_dataset", "seg_dataset"]:
                    if dataset.get(key) is not None:
                        dset = dataset[key]
                        if not isinstance(dset, Dataset):
                            raise TypeError(f"{dataset[key]} should be a dataset, but is of type {type(dataset[key])}")
                        dataset[key] = VisionDatasetSubset(dset, selected_indices)

            dataset["batch_size"] = batch_size
            dataset["shuffle"] = shuffle
            # Recover optional number of workers
            dataset["num_workers"] = dconfig.get("num_workers", 0)
            datasets[dataset_name] = dataset
        return datasets

    @staticmethod
    def get_dataloaders(
        config: str | dict[str, Any], sampling_ratio: int = 1, load_segmentation: bool = False, set_names: list[str] = None
    ) -> dict[str, DataLoader]:
        r"""Creates dataloaders from a configuration file.

        Args:
            config (str, dict): Path to configuration file, or configuration dictionary.
            sampling_ratio (int, optional): Sampling ratio (e.g. 5 means only one image in five is used). Default: 1.
            load_segmentation (bool, optional): If True, loads segmentation datasets if available. Default: False.
            set_names (list[str], optional): Only load the specified splits. Default: None (load everything)

        Returns:
            Dictionary of dataloaders.

        Raises:
            ValueError whenever a dataset could not be loaded or a parameter is invalid.
        """
        datasets = DatasetManager.get_datasets(
            config=config, sampling_ratio=sampling_ratio, load_segmentation=load_segmentation, set_names=set_names
        )
        dataloaders: dict[str, DataLoader] = {}

        def _safe_item_load(item, t):
            if not isinstance(item, t):
                raise TypeError(f"{item} is of type {type(item)} but should be of type {t}.")
            return item

        for dataset_name in datasets:
            dataset = _safe_item_load(datasets[dataset_name]["dataset"], Dataset)
            raw_dataset = _safe_item_load(datasets[dataset_name]["raw_dataset"], Dataset)
            batch_size = _safe_item_load(datasets[dataset_name]["batch_size"], int)
            shuffle = _safe_item_load(datasets[dataset_name]["shuffle"], bool)
            num_workers = _safe_item_load(datasets[dataset_name]["num_workers"], int)
            dataloaders[dataset_name] = DataLoader(
                dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
            )
            dataloaders[dataset_name + "_raw"] = DataLoader(
                dataset=raw_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
            )
            if load_segmentation:
                try:
                    seg_dataset = _safe_item_load(datasets[dataset_name]["seg_dataset"], Dataset)
                    dataloaders[dataset_name + "_seg"] = DataLoader(
                        dataset=seg_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
                    )
                except KeyError:
                    pass
        return dataloaders

    @staticmethod
    def get_dataset_transform(config: str | dict[str, Any], dataset: str = "test_set") -> Callable | None:
        r"""Returns the transform function associated with a given dataset.

        Args:
            config (str | dict): Path to configuration file, or configuration dictionary.
            dataset (str, optional): Name of target dataset. Default: test_set.

        Returns:
            Transform function (if any).

        Raises:
            ValueError whenever the configuration is incorrect.
        """
        if isinstance(config, str):
            config = load_config(config)
        if dataset not in config:
            raise ValueError(f"Missing configuration for dataset {dataset} in {config}.")
        if "params" not in config[dataset]:
            raise ValueError(f"Missing parameters for dataset {dataset}.")
        if "transform" not in config[dataset]["params"]:
            return None
        transform_config = config[dataset]["params"]["transform"]

        ops = [DatasetManager.get_transform(transform_config[op_name]) for op_name in transform_config]
        ops = torchvision.transforms.Compose(ops) if len(ops) > 1 else ops[0]
        logger.debug(f"Transform function for dataset {dataset}: {ops}")
        return ops


def batch_mixup(data: Tensor, labels: Tensor, alpha: float = 0) -> tuple[Tensor, Tensor, float]:
    r"""Performs a random mix between elements inside a batch of data.
    Adapted from https://github.com/gmum/ProtoPool/.

    Args:
        data (tensor): Batch data. Shape (N x D).
        labels (tensor): Batch labels. Shape (N x L).
        alpha (float, optional): Parameter of the Beta-distribution controlling the percentage of mix. Default: 0.

    Returns:
        Modified batch data. Shape (N x A).
        Modified batch labels. Shape (N x L).
        Percentage of mix.
    """
    percentage = np.random.beta(a=alpha, b=alpha) if alpha > 0 else 1.0
    # Draws a random permutation of indices inside the batch
    index = torch.randperm(data.size(0), dtype=torch.long, device=data.device)
    return percentage * data + (1 - percentage) * data[index], labels[index], percentage
