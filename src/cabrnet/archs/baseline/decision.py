from __future__ import annotations

from torch import Tensor
from cabrnet.archs.generic.decision import CaBRNetClassifier


class DummyClassifier(CaBRNetClassifier):
    r"""Dummy classifier."""

    def __init__(
        self,
        num_classes: int,
        num_features: int,
        proto_init_mode: str = "SHIFTED_NORMAL",
        **kwargs,
    ) -> None:
        r"""Initializes a ProtoPool classifier.

        Args:
            num_classes (int): Number of classes.
            num_features (int): Number of features (size of each prototype).
            proto_init_mode (str, optional): Init mode for prototypes. Default: UNIFORM.
        """
        super().__init__(num_classes=num_classes, num_features=num_features, proto_init_mode=proto_init_mode)

    def prototype_is_active(self, proto_id: int) -> bool:
        r"""Is the prototype *proto_idx* active or disabled?

        Args:
            proto_id (int): Prototype index.
        """
        return False

    def forward(self, features: Tensor, **kwargs) -> Tensor:
        r"""Returns the features unchanged.

        Args:
            features (tensor): Input features.
        """
        return features
