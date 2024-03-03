from importlib import resources

import torch
import numpy as np
from torch import nn, Tensor
from einops import rearrange
from torchvision.transforms import v2 as transforms
from kornia.utils import image_to_tensor


class SegmentationDino(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        self.dino.eval()
        self.head = nn.Linear(384, 1)

        if pretrained:
            with resources.path(
                "coin_ai.fg_segmentation", "segmentation_linear_24_03_03.pt"
            ) as path:
                self.head.load_state_dict(torch.load(path, map_location="cpu"))

    def forward(self, x: Tensor | np.ndarray) -> Tensor:
        if isinstance(x, np.ndarray):
            x = image_to_tensor(x, keepdim=False)

        b, c, h, w = x.shape
        assert (c, h, w) == (3, 518, 518)

        if x.dtype == torch.uint8:
            x = transforms.ToDtype(torch.float32, scale=True)(x)
        x = transforms.Grayscale(num_output_channels=3)(x)
        x = self.dino.forward_features(x)["x_norm_patchtokens"]
        x = rearrange(x, "b (h w) c -> b h w c", h=37, w=37)
        return self.head(x)

    def segment(self, x: Tensor) -> Tensor:
        return self.forward(x).squeeze(-1) > 0
