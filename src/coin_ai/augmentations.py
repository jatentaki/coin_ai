import torch
from torch import nn, Tensor


class CircleCrop(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

        grid_x, grid_y = torch.meshgrid(
            torch.linspace(-1, 1, size), torch.linspace(-1, 1, size)
        )
        self.mask = (grid_x**2 + grid_y**2) <= 1

    def forward(self, x: Tensor) -> Tensor:
        h, w = x.shape[-2:]
        assert h == w == self.size

        if torch.is_floating_point(x):
            fill = torch.ones_like(x)
        else:
            fill = torch.full_like(x, fill_value=torch.iinfo(x.dtype).max)

        if self.mask.device != x.device:
            self.mask = self.mask.to(x.device)

        return torch.where(self.mask, x, fill)
