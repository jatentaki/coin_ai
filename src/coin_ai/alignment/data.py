from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import torch
import imageio
import networkx as nx
import kornia.geometry as KG
from torch import Tensor
from einops import rearrange, repeat


@dataclass
class HPathPair:
    path1: str
    path2: str
    H_12: Tensor

    def flip(self) -> HPathPair:
        return HPathPair(self.path2, self.path1, self.H_12.inverse())

    @staticmethod
    def _load_image(path: str) -> Tensor:
        return torch.tensor(imageio.imread(path)).float() / 255.0

    def get_image1(self) -> Tensor:
        return self._load_image(self.path1)

    def get_image2(self) -> Tensor:
        return self._load_image(self.path2)


class HomographyBatch(NamedTuple):
    images: Tensor  # [2, B, C, H, W]
    H_12: Tensor  # [B, 3, 3]

    def __repr__(self) -> str:
        return f"HomographyBatch(images={tuple(self.images.shape)}, H_12={tuple(self.H_12.shape)})"

    @staticmethod
    def from_path_pairs(pairs: list[HPathPair]) -> HomographyBatch:
        images = [torch.stack([p.get_image1(), p.get_image2()], dim=0) for p in pairs]
        H_12s = [pair.H_12 for pair in pairs]

        images = torch.stack(images, dim=1)
        images = rearrange(images, "t b h w c -> t b c h w")
        H_12s = torch.stack(H_12s)

        return HomographyBatch(images=images, H_12=H_12s)

    def slice(self, start: int | None, end: int | None) -> HomographyBatch:
        images = self.images[:, start:end]
        H_12 = self.H_12[start:end]

        return HomographyBatch(images=images, H_12=H_12)

    @property
    def B(self) -> int:
        return self.H_12.shape[0]

    @property
    def corners(self) -> Tensor:
        h, w = self.images.shape[-2:]

        return torch.tensor([[0, 0], [w, 0], [w, h], [0, h]], dtype=torch.float32)

    def get_alignment_transform(self) -> Tensor:
        return torch.stack(
            [
                self.H_12,
                repeat(torch.eye(3), "i j -> b i j", b=self.B),
            ]
        )

    def build_augmentation(self) -> AugmentationBuilder:
        return AugmentationBuilder(
            batch=self,
            transform=repeat(
                torch.eye(3, device=self.images.device), "i j -> c b i j", c=2, b=self.B
            ),
            target_size=self.images.shape[-2:],
        )

    def to(self, *args, **kwargs) -> HomographyBatch:
        return HomographyBatch(
            images=self.images.to(*args, **kwargs),
            H_12=self.H_12.to(*args, **kwargs),
        )


class AugmentationBuilder(NamedTuple):
    batch: HomographyBatch
    transform: Tensor  # [2, B, 3, 3]
    target_size: tuple[int, int]

    def resize(self, target_size: tuple[int, int]) -> AugmentationBuilder:
        h, w = target_size
        resize_transform = KG.get_perspective_transform(
            self.batch.corners.unsqueeze(0),
            torch.tensor(
                [[0, 0], [w, 0], [w, h], [0, h]], dtype=torch.float32
            ).unsqueeze(0),
        )
        resize_transform = repeat(resize_transform, "B i j -> C B i j", C=2)

        return AugmentationBuilder(
            batch=self.batch,
            transform=resize_transform @ self.transform,
            target_size=target_size,
        )

    def random_h_4_point(self, scale: float = 0.125) -> Tensor:
        corners = repeat(self.batch.corners, "i j -> b i j", b=self.batch.B)

        scale_px = torch.tensor([self.target_size]).reshape(1, 1, 2) * scale
        corners_1 = corners + torch.randn_like(corners) * scale_px
        corners_2 = corners + torch.randn_like(corners) * scale_px
        transform_1 = KG.get_perspective_transform(corners_2, corners_1)
        transform_2 = KG.get_perspective_transform(corners_1, corners_2)

        return torch.stack([transform_2, transform_1], dim=0)

    def apply(self, transform: Tensor) -> AugmentationBuilder:
        assert transform.shape == self.transform.shape
        t_1, t_2 = transform
        new_H_12 = t_2 @ self.batch.H_12 @ torch.inverse(t_1)
        return AugmentationBuilder(
            batch=self.batch._replace(H_12=new_H_12),
            transform=transform @ self.transform,
            target_size=self.target_size,
        )

    def build(self) -> HomographyBatch:
        warp_fn = partial(
            KG.warp_perspective, dsize=self.target_size, padding_mode="border"
        )
        return HomographyBatch(
            images=torch.vmap(warp_fn, in_dims=(0, 0))(
                self.batch.images, self.transform
            ),
            H_12=self.batch.H_12,
        )

    def to(self, *args, **kwargs) -> AugmentationBuilder:
        return AugmentationBuilder(
            batch=self.batch.to(*args, **kwargs),
            transform=self.transform.to(*args, **kwargs),
            target_size=self.target_size,
        )


class HPairDataset:
    def __init__(
        self,
        csv_path: str,
        augmentation: Callable[[HomographyBatch], HomographyBatch],
        skip_identity: bool = False,
        infer: bool = True,
    ):
        self.base_pairs = self.parse_homography_csv(csv_path)

        if not infer:
            self.inferred_pairs = self.base_pairs
        else:
            self.inferred_pairs = self.infer_homographies(
                self.base_pairs, skip_identity
            )
        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.inferred_pairs)

    def __getitem__(self, idx: int) -> HomographyBatch:
        batch = HomographyBatch.from_path_pairs([self.inferred_pairs[idx]])
        batch = self.augmentation(batch)
        return batch

    @staticmethod
    def collate_fn(batch: list[HomographyBatch]) -> HomographyBatch:
        images = torch.cat([b.images for b in batch], dim=1)
        H_12 = torch.cat([b.H_12 for b in batch])

        return HomographyBatch(images=images, H_12=H_12)

    @staticmethod
    def parse_homography_csv(source_path: str) -> list[HPathPair]:
        HEADERS = [
            "img1",
            "h11",
            "h12",
            "h13",
            "h21",
            "h22",
            "h23",
            "h31",
            "h32",
            "h33",
        ]

        if not os.path.exists(source_path):
            raise ValueError(f"File {source_path} does not exist")

        pairs = []
        with open(source_path, "r") as f:
            reader = csv.reader(f)
            headers = next(reader)
            assert headers[: len(HEADERS)] == HEADERS

            for img1, img2, *floats in reader:
                H_floats = [float(f) for f in floats[:9]]
                source_dir = os.path.dirname(source_path)
                path1 = os.path.join(source_dir, img1)
                path2 = os.path.join(source_dir, img2)
                H_opencv = torch.tensor(H_floats, dtype=torch.float32).reshape(3, 3)
                pairs.append(
                    HPathPair(
                        path1,
                        path2,
                        H_opencv.inverse(),
                    )
                )

        return pairs

    @staticmethod
    def infer_homographies(
        hpairs: list[HPathPair], skip_identity: bool = False
    ) -> list[HPathPair]:
        g = nx.DiGraph()
        for pair in hpairs + [pair.flip() for pair in hpairs]:
            g.add_edge(pair.path1, pair.path2, H_matrix=pair.H_12)

        inferred_pairs = []
        for source, paths in nx.all_pairs_shortest_path(g):
            for target, path in paths.items():
                if skip_identity and source == target:
                    continue
                path_graph = nx.path_graph(path)

                total_h_matrix = torch.eye(3)
                for s, t in path_graph.edges():
                    h_matrix = g.edges[s, t]["H_matrix"]
                    total_h_matrix = total_h_matrix @ h_matrix

                inferred_pairs.append(
                    HPathPair(
                        target,
                        source,
                        total_h_matrix,
                    )
                )

        return inferred_pairs
