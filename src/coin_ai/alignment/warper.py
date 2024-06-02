from __future__ import annotations

from dataclasses import dataclass, replace, fields

import torch
import matplotlib.pyplot as plt
import kornia.geometry as KG
import torch.nn.functional as F
from einops import rearrange
from tqdm.auto import tqdm
from torch import nn, Tensor
from matplotlib.figure import Figure


from coin_ai.alignment.infra import DenseDino, ScoringModel
from coin_ai.alignment.types import Matches


@dataclass
class WarpingState:
    image_raw_1: Tensor
    image_raw_2: Tensor
    H_12: Tensor

    @staticmethod
    def from_images(image1: Tensor, image2: Tensor) -> WarpingState:
        H = (
            torch.eye(3, device=image1.device)
            .unsqueeze(0)
            .repeat(image1.shape[0], 1, 1)
        )

        return WarpingState(
            image_raw_1=image1,
            image_raw_2=image2,
            H_12=H,
        )

    def to(self, *args, **kwargs) -> WarpingState:
        return replace(
            self,
            **{f.name: getattr(self, f.name).to(*args, **kwargs) for f in fields(self)},
        )

    def image_warp_1(self) -> Tensor:
        return KG.homography_warp(
            self.image_raw_1,
            self.H_12,
            dsize=(518, 518),
            normalized_homography=False,
            padding_mode="border",
        )

    def adjust_homography(self, H_12: Tensor) -> WarpingState:
        return replace(self, H_12=H_12 @ self.H_12)

    def plot(self) -> Figure:
        fig, (a1, a2) = plt.subplots(1, 2)
        a1.imshow(self.image_warp_1()[0].permute(1, 2, 0).numpy())
        a2.imshow(self.image_raw_2[0].permute(1, 2, 0).numpy())

        return fig

    def to_frame(self) -> Tensor:
        frame_bchw = torch.cat([self.image_warp_1(), self.image_raw_2], dim=-1)
        return frame_bchw.permute(0, 2, 3, 1)


def mask_lookup(mask: Tensor, endpts: Tensor) -> Tensor:
    mask = rearrange(mask, "b h w 1 -> b 1 h w").float()
    endpts = rearrange(endpts, "b n t -> b 1 n t")
    lookup = F.grid_sample(mask, endpts)
    return rearrange(lookup, "b 1 1 n -> b n").bool()


def weight_mask(matches: Matches, mask_1: Tensor, mask_2: Tensor) -> Tensor:
    is_masked_1 = mask_lookup(mask_1, matches.pts_1)
    is_masked_2 = mask_lookup(mask_2, matches.pts_2)
    return is_masked_1 & is_masked_2


@dataclass
class AlignmentArtifacts:
    H_12_final: Tensor
    states: list[WarpingState]


class Aligner(nn.Module):
    def __init__(
        self, matcher: ScoringModel, device: torch.device = torch.device("cpu")
    ):
        super().__init__()
        self.device = device

        self.dense_dino = DenseDino().to(self.device)
        self.matcher = matcher.to(self.device)

    def align(
        self, image1: Tensor, image2: Tensor, n_steps: int = 10, alpha: float = 0.3
    ) -> AlignmentArtifacts:
        states = [
            WarpingState.from_images(
                self.dense_dino.preprocess(image1),
                self.dense_dino.preprocess(image2),
            ).to(self.device),
        ]

        with torch.no_grad():
            features_2 = self.dense_dino(states[-1].image_raw_2)

        for _ in tqdm(range(n_steps)):
            image_warp_1 = states[-1].image_warp_1()

            with torch.no_grad():
                features_1 = self.dense_dino(image_warp_1)

            half_size = (518 + 1) / 2
            with torch.no_grad():
                matches = self.matcher.forward(features_1, features_2)

            to_denorm = lambda x: x * half_size + half_size
            endpts_1 = to_denorm(matches.pts_1)
            endpts_2 = to_denorm(matches.pts_2)

            H_relative = KG.homography.find_homography_dlt(
                endpts_1,
                endpts_2,
                matches.weights,
            )

            H_update = (
                torch.eye(3, device=H_relative.device).unsqueeze(0) * (1 - alpha)
                + H_relative * alpha
            )
            states.append(states[-1].adjust_homography(H_update))

        return AlignmentArtifacts(
            H_12_final=states[-1].H_12,
            states=states,
        )
