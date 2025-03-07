from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import time

from cabrnet.archs.generic.decision import CaBRNetClassifier
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.optimizers import OptimizerManager


class Baseline(CaBRNet):
    r"""Baseline model that returns the extracted "features" as the model output.

    Attributes:
        extractor: Model used to extract convolutional features from the input image.
        classifier: Empty shell.
    """

    def __init__(self, extractor: nn.Module, classifier: CaBRNetClassifier, **kwargs):
        r"""Builds a Baseline.

        Args:
            extractor (Module): Feature extractor.
            classifier (CaBRNetClassifier): Classification based on extracted features.
        """
        super(Baseline, self).__init__(extractor, classifier, **kwargs)

    def loss(self, model_output: Any, label: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict[str, float]]:
        r"""Loss function.

        Args:
            model_output (Any): Model output, in this case a tuple containing the prediction and the minimum distances.
            label (tensor): Batch labels.

        Returns:
            Loss tensor and batch statistics.
        """
        # Cross-entropy loss
        cross_entropy = torch.nn.functional.cross_entropy(model_output, label)
        batch_accuracy = torch.sum(torch.eq(torch.argmax(model_output, dim=1), label)).item() / len(label)
        return cross_entropy, {"loss": cross_entropy.item(), "accuracy": batch_accuracy}

    def train_epoch(
        self,
        dataloaders: dict[str, DataLoader],
        optimizer_mngr: OptimizerManager,
        device: str | torch.device = "cuda:0",
        tqdm_position: int = 0,
        epoch_idx: int = 0,
        verbose: bool = False,
        max_batches: int | None = None,
        **kwargs,
    ) -> dict[str, float]:
        r"""Trains a ProtoPNet model for one epoch, performing prototype projection and fine-tuning if necessary.

        Args:
            dataloaders (dictionary): Dictionary of dataloaders.
            optimizer_mngr (OptimizerManager): Optimizer manager.
            device (str | device, optional): Hardware device. Default: cuda:0.
            tqdm_position (int, optional): Position of the progress bar. Default: 0.
            epoch_idx (int, optional): Epoch index. Default: 0.
            verbose (bool, optional): Display progress bar. Default: False.
            max_batches (int, optional): Max number of batches (early stop for small compatibility tests).
                Default: None.

        Returns:
            Dictionary containing learning statistics.
        """
        self.train()
        self.to(device)

        # Training stats
        train_info = {}
        nb_inputs = 0

        # Capture data fetch time relative to total batch time to ensure that there is no bottleneck here
        total_batch_time = 0.0
        total_data_time = 0.0

        # Use training dataloader
        train_loader = dataloaders["train_set"]

        # Show progress on progress bar if needed
        train_iter = tqdm(
            enumerate(train_loader),
            desc=f"Training epoch {epoch_idx}",
            total=len(train_loader),
            leave=False,
            position=tqdm_position,
            disable=not verbose,
        )
        ref_time = time.time()
        batch_idx = 0
        for batch_idx, (xs, ys) in train_iter:
            data_time = time.time() - ref_time
            nb_inputs += xs.size(0)

            # Reset gradients and map the data on the target device
            optimizer_mngr.zero_grad()
            xs, ys = xs.to(device), ys.to(device)

            # Perform inference and compute loss
            ys_pred = self.forward(xs)
            batch_loss, batch_stats = self.loss(ys_pred, ys)

            # Compute the gradient and update parameters
            batch_loss.backward()
            if isinstance(optimizer_mngr, OptimizerManager):
                optimizer_mngr.optimizer_step(epoch=epoch_idx)
            else:  # Simple optimizer
                optimizer_mngr.step()

            # Update progress bar
            batch_accuracy = batch_stats["accuracy"]
            batch_time = time.time() - ref_time
            postfix_str = (
                f"Batch [{batch_idx + 1}/{len(train_loader)}], "
                f"Batch loss: {batch_loss.item():.3f}, Acc: {batch_accuracy:.3f}, "
                f"Batch time: {batch_time:.3f}s (data: {data_time:.3f})"
            )
            train_iter.set_postfix_str(postfix_str)

            # Update all metrics
            if not train_info:
                train_info = batch_stats
            else:
                for key, value in batch_stats.items():
                    train_info[key] += value * xs.size(0)
            total_batch_time += batch_time
            total_data_time += data_time
            ref_time = time.time()

        # Clean gradients after last batch
        optimizer_mngr.zero_grad()

        train_info = {f"{key}/train": value / nb_inputs for key, value in train_info.items()}

        # Update batch_num with effective value
        batch_num = batch_idx + 1
        train_info.update(
            {
                "time/batch": total_batch_time / batch_num,
                "time/data": total_data_time / batch_num,
            }
        )

        if dataloaders.get("validation_set"):
            eval_info = self.evaluate(dataloaders["validation_set"], device)
            train_info.update({f"{key}/val": value for key, value in eval_info.items()})

        return train_info
