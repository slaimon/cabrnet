from __future__ import annotations
from pathlib import PurePath
from typing import Callable, Dict, Optional, Tuple, Union

from loguru import logger
import torch
import torch.nn as nn
from torchvision.models._api import register_model


class Matlab3DResNet(nn.Module):
    def __init__(self):
        super(Matlab3DResNet, self).__init__()
        # Initial layers (data normalization is assumed to be handled externally)
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)
        self.bn_conv1 = nn.BatchNorm3d(64)
        self.conv1_relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # ----- Residual Block: res2a -----
        self.res2a_branch2a = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2a_branch2a = nn.BatchNorm3d(64)
        self.res2a_branch2a_relu = nn.ReLU(inplace=True)
        self.res2a_branch2b = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2a_branch2b = nn.BatchNorm3d(64)
        self.res2a_relu = nn.ReLU(inplace=True)

        # ----- Residual Block: res2b -----
        self.res2b_branch2a = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2b_branch2a = nn.BatchNorm3d(64)
        self.res2b_branch2a_relu = nn.ReLU(inplace=True)
        self.res2b_branch2b = nn.Conv3d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2b_branch2b = nn.BatchNorm3d(64)
        self.res2b_relu = nn.ReLU(inplace=True)

        # ----- Residual Block: res3a (with downsampling) -----
        self.res3a_branch2a = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn3a_branch2a = nn.BatchNorm3d(128)
        self.res3a_branch2a_relu = nn.ReLU(inplace=True)
        self.res3a_branch2b = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3a_branch2b = nn.BatchNorm3d(128)
        self.res3a_branch1 = nn.Conv3d(64, 128, kernel_size=1, stride=2, bias=True)
        self.bn3a_branch1 = nn.BatchNorm3d(128)
        self.res3a_relu = nn.ReLU(inplace=True)

        # ----- Residual Block: res3b -----
        self.res3b_branch2a = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3b_branch2a = nn.BatchNorm3d(128)
        self.res3b_branch2a_relu = nn.ReLU(inplace=True)
        self.res3b_branch2b = nn.Conv3d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn3b_branch2b = nn.BatchNorm3d(128)
        self.res3b_relu = nn.ReLU(inplace=True)

        # ----- Residual Block: res4a (with downsampling) -----
        self.res4a_branch1 = nn.Conv3d(128, 256, kernel_size=1, stride=2, bias=True)
        self.bn4a_branch1 = nn.BatchNorm3d(256)
        self.res4a_branch2a = nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn4a_branch2a = nn.BatchNorm3d(256)
        self.res4a_branch2a_relu = nn.ReLU(inplace=True)
        self.res4a_branch2b = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4a_branch2b = nn.BatchNorm3d(256)
        self.res4a_relu = nn.ReLU(inplace=True)

        # ----- Residual Block: res4b -----
        self.res4b_branch2a = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4b_branch2a = nn.BatchNorm3d(256)
        self.res4b_branch2a_relu = nn.ReLU(inplace=True)
        self.res4b_branch2b = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn4b_branch2b = nn.BatchNorm3d(256)
        self.res4b_relu = nn.ReLU(inplace=True)

        # ----- Residual Block: res5a (with downsampling) -----
        self.res5a_branch1 = nn.Conv3d(256, 512, kernel_size=1, stride=2, bias=True)
        self.bn5a_branch1 = nn.BatchNorm3d(512)
        self.res5a_branch2a = nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn5a_branch2a = nn.BatchNorm3d(512)
        self.res5a_branch2a_relu = nn.ReLU(inplace=True)
        self.res5a_branch2b = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5a_branch2b = nn.BatchNorm3d(512)
        self.res5a_relu = nn.ReLU(inplace=True)

        # ----- Residual Block: res5b -----
        self.res5b_branch2a = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5b_branch2a = nn.BatchNorm3d(512)
        self.res5b_branch2a_relu = nn.ReLU(inplace=True)
        self.res5b_branch2b = nn.Conv3d(512, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn5b_branch2b = nn.BatchNorm3d(512)
        self.res5b_relu = nn.ReLU(inplace=True)

        # ----- Final Layers -----
        self.pool5 = nn.AdaptiveAvgPool3d(1)  # global average pooling (outputs 1x1x1 per channel)
        self.pool5_drop_7x7_s1 = nn.Dropout(p=0.4)
        self.new_fc = nn.Linear(512, 2)
        self.prob = nn.Softmax(dim=1)

    def forward(self, x):
        # x: assumed shape (N, 1, 224, 224, 224)
        # --- Initial layers ---
        x = self.conv1(x)
        x = self.bn_conv1(x)
        x = self.conv1_relu(x)
        x = self.pool1(x)

        # --- res2a ---
        identity = x  # from pool1
        out = self.res2a_branch2a(x)
        out = self.bn2a_branch2a(out)
        out = self.res2a_branch2a_relu(out)
        out = self.res2a_branch2b(out)
        out = self.bn2a_branch2b(out)
        x = self.res2a_relu(out + identity)

        # --- res2b ---
        identity = x  # from res2a_relu
        out = self.res2b_branch2a(x)
        out = self.bn2b_branch2a(out)
        out = self.res2b_branch2a_relu(out)
        out = self.res2b_branch2b(out)
        out = self.bn2b_branch2b(out)
        x = self.res2b_relu(out + identity)

        # --- res3a (with downsampling) ---
        out2 = self.res3a_branch2a(x)
        out2 = self.bn3a_branch2a(out2)
        out2 = self.res3a_branch2a_relu(out2)
        out2 = self.res3a_branch2b(out2)
        out2 = self.bn3a_branch2b(out2)
        out1 = self.res3a_branch1(x)
        out1 = self.bn3a_branch1(out1)
        x = self.res3a_relu(out2 + out1)

        # --- res3b ---
        identity = x  # from res3a_relu
        out = self.res3b_branch2a(x)
        out = self.bn3b_branch2a(out)
        out = self.res3b_branch2a_relu(out)
        out = self.res3b_branch2b(out)
        out = self.bn3b_branch2b(out)
        x = self.res3b_relu(out + identity)

        # --- res4a (with downsampling) ---
        out2 = self.res4a_branch2a(x)
        out2 = self.bn4a_branch2a(out2)
        out2 = self.res4a_branch2a_relu(out2)
        out2 = self.res4a_branch2b(out2)
        out2 = self.bn4a_branch2b(out2)
        out1 = self.res4a_branch1(x)
        out1 = self.bn4a_branch1(out1)
        x = self.res4a_relu(out1 + out2)

        # --- res4b ---
        identity = x  # from res4a_relu
        out = self.res4b_branch2a(x)
        out = self.bn4b_branch2a(out)
        out = self.res4b_branch2a_relu(out)
        out = self.res4b_branch2b(out)
        out = self.bn4b_branch2b(out)
        x = self.res4b_relu(out + identity)

        # --- res5a (with downsampling) ---
        out2 = self.res5a_branch2a(x)
        out2 = self.bn5a_branch2a(out2)
        out2 = self.res5a_branch2a_relu(out2)
        out2 = self.res5a_branch2b(out2)
        out2 = self.bn5a_branch2b(out2)
        out1 = self.res5a_branch1(x)
        out1 = self.bn5a_branch1(out1)
        x = self.res5a_relu(out1 + out2)

        # --- res5b ---
        identity = x  # from res5a_relu
        out = self.res5b_branch2a(x)
        out = self.bn5b_branch2a(out)
        out = self.res5b_branch2a_relu(out)
        out = self.res5b_branch2b(out)
        out = self.bn5b_branch2b(out)
        x = self.res5b_relu(out + identity)

        # --- Final layers ---
        x = self.pool5(x)  # shape: (N, 512, 1, 1, 1)
        x = x.view(x.size(0), -1)  # flatten to (N, 512)
        x = self.pool5_drop_7x7_s1(x)
        x = self.new_fc(x)
        x = self.prob(x)
        return x


@register_model()
def matlab_3d_resnet(pth_path: str) -> Matlab3DResNet:
    model = Matlab3DResNet()
    state = torch.load(pth_path)
    model.load_state_dict(state)
    logger.info(f"successfully loaded model parameters from {pth_path}")
    return model
