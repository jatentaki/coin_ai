from __future__ import annotations

from typing import NamedTuple, Callable
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Matches:
    pts_1: Tensor  # [B, N, 2]
    pts_2: Tensor  # [B, N, 2]

    scores: Tensor  # [B, N]
    weights: Tensor  # [B, N]

    def __repr__(self):
        return (
            "Matches("
            + ", ".join([f"{k}={tuple(v.shape)}" for k, v in self.__dict__.items()])
            + ")"
        )

    def __post_init__(self):
        assert self.pts_1.shape == self.pts_2.shape, (
            self.pts_1.shape,
            self.pts_2.shape,
        )
        assert self.pts_1.shape[:-1] == self.scores.shape, (
            self.pts_1.shape,
            self.scores.shape,
        )
        assert self.scores.shape == self.weights.shape

    @property
    def B(self) -> int:
        return self.pts_1.shape[0]

    def apply(self, f: Callable) -> Matches:
        return Matches(**{k: f(v) for k, v in self.__dict__.items()})

    def to(self, *args, **kwargs) -> Matches:
        return self.apply(lambda t: t.to(*args, **kwargs))

    def clip_by_score(self, n: int) -> Matches:
        _weights, indices = self.weights.topk(n, dim=1, largest=True)
        do_slice = lambda t: t[torch.arange(self.B)[:, None], indices]
        return self.apply(do_slice)


class Batch(NamedTuple):
    image1: Tensor
    image2: Tensor

    features_1: Tensor
    features_2: Tensor

    H_12: Tensor
    H_21: Tensor

    def __repr__(self):
        return (
            "Batch("
            + ", ".join([f"{k}={tuple(v.shape)}" for k, v in self._asdict().items()])
            + ")"
        )

    def apply(self, f: Callable) -> Batch:
        return Batch(*[f(v) for v in self])

    def to(self, *args, **kwargs) -> Batch:
        return self.apply(lambda t: t.to(*args, **kwargs))

    def flip(self) -> Batch:
        return Batch(
            image1=self.image2,
            image2=self.image1,
            features_1=self.features_2,
            features_2=self.features_1,
            H_12=self.H_21,
            H_21=self.H_12,
        )
