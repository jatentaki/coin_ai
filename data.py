import os
import csv
from typing import Callable
from dataclasses import dataclass

import torch
from torchvision import io
from torch import Tensor
from torch.utils.data import IterableDataset, get_worker_info


@dataclass
class BaseSymmetryAugmentation:
    name: str

    def __repr__(self):
        return self.name

    def __call__(self, image: Tensor) -> Tensor:
        raise NotImplementedError


class Identity(BaseSymmetryAugmentation):
    def __init__(self):
        super().__init__("id")

    def __call__(self, image: Tensor) -> Tensor:
        return image


class Rotation(BaseSymmetryAugmentation):
    def __init__(self, k: int):
        super().__init__(f"rot{k * 90}")
        self.k = k

    def __call__(self, image: Tensor) -> Tensor:
        return torch.rot90(image, k=self.k, dims=(-2, -1))


class FlipX(BaseSymmetryAugmentation):
    def __init__(self):
        super().__init__("flip_x")

    def __call__(self, image: Tensor) -> Tensor:
        return torch.flip(image, (-1,))


@dataclass
class CombinedSymmetryAugmentation:
    fn1: BaseSymmetryAugmentation
    fn2: BaseSymmetryAugmentation

    def __repr__(self):
        return f"{self.fn1} -> {self.fn2}"

    def __call__(self, image: Tensor) -> Tensor:
        return self.fn2(self.fn1(image))


identity = Identity()

rotations = [
    identity,
    Rotation(1),
    Rotation(2),
    Rotation(3),
]

flips = [
    identity,
    FlipX(),
]


class CoinType:
    def __init__(self, path: str, symmetry_augmentation: CombinedSymmetryAugmentation):
        self.path = path
        self.symmetry_augmentation = symmetry_augmentation
        self.images = [
            f for f in os.listdir(path) if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"{self.path}, {self.symmetry_augmentation} ({len(self.images)} images)"

    def get_item(
        self,
        idx: int,
        transform: Callable[
            [
                Tensor,
            ],
            Tensor,
        ] = identity,
    ) -> Tensor:
        path = os.path.join(self.path, self.images[idx])
        image = io.read_image(path, io.image.ImageReadMode.RGB)
        image = transform(image)
        return self.symmetry_augmentation(image)


def build_coin_types(root: str) -> list[CoinType]:
    with open(f"{root}/symmetries.csv", "r") as f:
        reader = csv.reader(f)
        next(reader)
        rows = list(reader)

    coin_types = []
    for relative_dir, symmetry in rows:
        path = os.path.join(root, relative_dir)
        if symmetry == "rot90":
            allowed_rotations = [rotations[0]]
        elif symmetry == "rot180":
            allowed_rotations = [rotations[0], rotations[1]]
        elif symmetry == "none":
            allowed_rotations = rotations
        else:
            raise ValueError(f"Unknown symmetry {symmetry}")

        augmentations = [
            CombinedSymmetryAugmentation(r, f) for r in allowed_rotations for f in flips
        ]

        coin_types.extend([CoinType(path, a) for a in augmentations])

    return coin_types


class CoinDataset(IterableDataset):
    def __init__(
        self,
        coin_types: list[CoinType],
        batch_size: int,
        min_in_class: int = 4,
        n_batches: int = 1_000,
        augmentation: Callable[[Tensor], Tensor] = identity,
        seed_init: int = 42,
    ):
        super().__init__()
        self.coin_types = coin_types
        self.batch_size = batch_size
        self.min_in_class = min_in_class
        self.n_batches = n_batches
        self.type_sizes = torch.tensor(
            [len(coin_type) for coin_type in coin_types], dtype=torch.int64
        )
        self.cum_type_sizes = self.type_sizes.cumsum(0)
        self.augmentation = augmentation
        self.seed_init = seed_init

        assert (
            self.batch_size >= self.min_in_class
        ), "Batch size must be greater or equal to min_in_class"

    def length(self):
        return self.type_sizes.sum().item()
    
    def get_nth_item(self, n: int) -> tuple[Tensor, int]:
        type_index = torch.searchsorted(self.cum_type_sizes, n, right=True)
        coin_index = n - self.cum_type_sizes[type_index]
        image = self.coin_types[type_index].get_item(coin_index, self.augmentation)
        return image, type_index

    def __iter__(self):
        worker_info = get_worker_info()
        rng = torch.Generator()
        if worker_info is not None:
            rng.manual_seed(worker_info.id + self.seed_init)
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            rng.manual_seed(self.seed_init)
            worker_id = 0
            num_workers = 1

        n_batches_in_worker = self.n_batches // num_workers
        if worker_id < self.n_batches % num_workers:
            n_batches_in_worker += 1

        for _ in range(n_batches_in_worker):
            yield self._prepare_batch(rng)

    def _prepare_batch(self, rng: torch.Generator) -> tuple[Tensor, Tensor]:
        n_types_to_include = min(
            self.batch_size // self.min_in_class, len(self.coin_types)
        )
        types_to_include = torch.randperm(len(self.coin_types), generator=rng)[
            :n_types_to_include
        ]
        n_per_type = torch.zeros(len(self.coin_types), dtype=torch.int64)
        n_per_type[types_to_include] = torch.minimum(
            torch.tensor(self.min_in_class, dtype=torch.int64),
            self.type_sizes[types_to_include],
        )
        n_to_fill = self.batch_size - n_per_type.sum()

        while n_to_fill > 0:
            idx = torch.randint(len(self.coin_types), generator=rng, size=(1,))
            if n_per_type[idx] < self.type_sizes[idx]:
                n_per_type[idx] += 1
                n_to_fill -= 1

        images, labels = [], []
        for type_id, n_in_type in zip(types_to_include, n_per_type[types_to_include]):
            coin_type = self.coin_types[type_id]
            image_ids = torch.randperm(len(coin_type), generator=rng)[:n_in_type]

            for image_id in image_ids:
                images.append(coin_type.get_item(image_id, self.augmentation))
                labels.append(type_id)

        return torch.stack(images), torch.tensor(labels, dtype=torch.int64)

    @staticmethod
    def collate_fn(batch):
        ((images, labels),) = batch
        return images, labels
