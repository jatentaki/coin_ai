from __future__ import annotations

import csv
import glob
import os
from dataclasses import dataclass
from functools import partial
from typing import Callable, NamedTuple

import torch
import imageio
import networkx as nx
import kornia.geometry as KG
import kornia.color as KC
from torch import Tensor
from einops import rearrange, repeat
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset

FLIP_X_MATRIX = torch.tensor(
    [
        [-1, 0, 518],
        [0, 1, 0],
        [0, 0, 1],
    ],
    dtype=torch.float32,
)

FLIP_MATRICES = torch.stack(
    [
        torch.eye(3, dtype=torch.float32),
        FLIP_X_MATRIX,
    ]
)

ROT_90_MATRIX = torch.tensor(
    [
        [0, -1, 518],
        [1, 0, 0],
        [0, 0, 1],
    ],
    dtype=torch.float32,
)

ROT_180_MATRIX = ROT_90_MATRIX @ ROT_90_MATRIX
ROT_270_MATRIX = ROT_180_MATRIX @ ROT_90_MATRIX

ROT_MATRICES = torch.stack(
    [
        torch.eye(3, dtype=torch.float32),
        ROT_90_MATRIX,
        ROT_180_MATRIX,
        ROT_270_MATRIX,
    ]
)


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

    def flip(self) -> HomographyBatch:
        return HomographyBatch(
            images=self.images.flip((0,)),
            H_12=self.H_12.inverse(),
        )

    @staticmethod
    def collate_fn(batch: list[HomographyBatch]) -> HomographyBatch:
        images = torch.cat([b.images for b in batch], dim=1)
        H_12 = torch.cat([b.H_12 for b in batch])

        return HomographyBatch(images=images, H_12=H_12)


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

    def random_rotate_transform(self) -> torch.Tensor:
        index = torch.randint(4, (self.batch.B,))
        return repeat(ROT_MATRICES[index], "b i j -> 2 b i j")

    def random_flip_transform(self) -> torch.Tensor:
        index = torch.randint(2, (self.batch.B,))
        return repeat(FLIP_MATRICES[index], "b i j -> 2 b i j")

    def apply(self, transform: Tensor) -> AugmentationBuilder:
        assert transform.shape == self.transform.shape, (
            transform.shape,
            self.transform.shape,
        )
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

    @staticmethod
    def collate_fn(batch: list[AugmentationBuilder]) -> AugmentationBuilder:
        return AugmentationBuilder(
            batch=HomographyBatch.collate_fn([b.batch for b in batch]),
            transform=torch.cat([b.transform for b in batch], dim=1),
            target_size=batch[0].target_size,
        )


class HPairDataset:
    def __init__(
        self,
        csv_path: str,
        augmentation: Callable[[HomographyBatch], AugmentationBuilder],
        skip_identity: bool = True,
        infer: bool = True,
        ordered_pairs: bool = True,
        replicate: int = 1,
    ):
        self.base_pairs = self.parse_homography_csv(csv_path)
        self.replicate = replicate

        if not infer:
            self.inferred_pairs = self.base_pairs
        else:
            self.inferred_pairs = self.infer_homographies(
                self.base_pairs, skip_identity
            )

        if ordered_pairs:
            self.inferred_pairs = [
                pair for pair in self.inferred_pairs if pair.path1 >= pair.path2
            ]

        self.augmentation = augmentation

    def __len__(self) -> int:
        return len(self.inferred_pairs) * self.replicate

    def __getitem__(self, idx: int) -> AugmentationBuilder:
        idx = idx // self.replicate
        batch = HomographyBatch.from_path_pairs([self.inferred_pairs[idx]])
        return self.augmentation(batch)

    collate_fn = AugmentationBuilder.collate_fn

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


def assemble_datasets(
    path: str,
    augmentation: Callable[
        [
            HomographyBatch,
        ],
        HomographyBatch,
    ],
    infer: bool = True,
    skip_identity: bool = True,
    ordered_pairs: bool = True,
    replicate: int = 1,
) -> ConcatDataset:
    paths = glob.glob(f"{path}/**/homographies.csv", recursive=True)

    datasets = [
        HPairDataset(
            path,
            augmentation=augmentation,
            skip_identity=skip_identity,
            infer=infer,
            ordered_pairs=ordered_pairs,
            replicate=replicate,
        )
        for path in paths
    ]
    return ConcatDataset(datasets)


def train_augmentation(batch: HomographyBatch) -> AugmentationBuilder:
    builder = batch.build_augmentation()
    alignment = batch.get_alignment_transform()

    rotation = builder.random_rotate_transform()
    flip = builder.random_flip_transform()
    distortion = builder.random_h_4_point(scale=0.05)

    batch = batch._replace(
        images=KC.rgb_to_grayscale(batch.images).repeat(1, 1, 3, 1, 1)
    )

    return builder.apply(alignment).apply(rotation).apply(flip).apply(distortion)


def val_augmentation(batch: HomographyBatch) -> AugmentationBuilder:
    batch = batch._replace(
        images=KC.rgb_to_grayscale(batch.images).repeat(1, 1, 3, 1, 1)
    )

    return batch.build_augmentation()


class CoinDataModule(LightningDataModule):
    def __init__(
        self,
        train_root: str,
        val_root: str,
        batch_size: int,
        num_workers: int = 0,
        train_aug: Callable[[HomographyBatch], HomographyBatch] = train_augmentation,
        val_aug: Callable[[HomographyBatch], HomographyBatch] = val_augmentation,
    ):
        super().__init__()
        self.train_root = train_root
        self.val_root = val_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_aug = train_aug
        self.val_aug = val_aug

    def setup(self, stage: str):
        if stage == "fit" or stage is None:
            self.train_dataset = assemble_datasets(
                self.train_root,
                augmentation=self.train_aug,
                replicate=8,
            )
            self.val_dataset = assemble_datasets(
                self.val_root,
                augmentation=self.val_aug,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=HPairDataset.collate_fn,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=HPairDataset.collate_fn,
        )
