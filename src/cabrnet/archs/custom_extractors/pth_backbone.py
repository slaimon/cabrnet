from __future__ import annotations
from pathlib import PurePath
from typing import Callable, Dict, Optional, Tuple, Union

from loguru import logger
import numpy as np
import torch
from torchvision.models._api import register_model

import onnx
import onnx2torch as ot

class GraphModule(torch.nn.Module):

    def __init__(self, path:str):
        super().__init__()
        path = PurePath(path)
        onnx_model = onnx.load(path)
        onnx.checker.check_model(onnx_model)
        self.model = ot.convert(onnx_model)
        logger.info(f"Successfully loaded model from {path}.")

    def forward(self, x):
        return self.model(x)


@register_model()
def graph_module(pth_path: str) -> GraphModule:
    return GraphModule(pth_path)
