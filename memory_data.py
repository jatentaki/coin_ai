import os
from typing import Callable

import torch
import numpy as np
from kornia import augmentation
from torchvision.transforms import v2 as transforms
from torchvision import io
from torch import Tensor


def identity(x: Tensor) -> Tensor:
    return x


EXTENSIONS = (".jpg", ".jpeg", ".png")


def is_image_path(f: str) -> bool:
    return f.lower().endswith(EXTENSIONS)


class CoinType:
    def __init__(self, path: str):
        self.path = path
        self.images = [f for f in os.listdir(path) if is_image_path(f)]

    def __len__(self):
        return len(self.images)

    def __repr__(self):
        return f"{self.path}, ({len(self.images)} images)"

    def get_item(
        self,
        idx: int,
    ) -> Tensor:
        path = os.path.join(self.path, self.images[idx])
        image = io.read_image(path, io.image.ImageReadMode.RGB)
        return image


def build_coin_types(root: str) -> list[CoinType]:
    coin_types = []
    for dir, _, files in os.walk(root):
        if not any(is_image_path(f) for f in files):
            continue

        coin_types.append(CoinType(dir))

    return coin_types


class InMemoryCoins:
    def __init__(self, coin_types: list[CoinType], device=torch.device("cpu")):
        self.coin_types = coin_types
        self.type_sizes = torch.tensor(
            [len(coin_type) for coin_type in coin_types],
            dtype=torch.int64,
        )
        self.type_ix_starts = torch.cat(
            [torch.tensor([0], dtype=torch.int64), self.type_sizes.cumsum(0)]
        )
        self.device = device

        images = []
        for coin_type in coin_types:
            for i in range(len(coin_type)):
                images.append(coin_type.get_item(i))

        self.images = torch.stack(images).to(device)

        assert self.images.shape[0] == self.type_sizes.sum().item()
    
    def to_device(self, device: torch.device):
        self.images = self.images.to(device)
        self.device = device
        return self

    def __len__(self):
        return self.images.shape[0]

    def get(self, coin_type: Tensor, coin_index: Tensor) -> Tensor:
        type_index = coin_type
        image_index = coin_index

        assert (image_index < self.type_sizes[type_index]).all(), "Index out of bounds"

        slab_idx = self.type_ix_starts[type_index] + image_index
        slab_idx = slab_idx.to(self.images.device)
        return self.images[slab_idx]

class Generator:
    def __init__(self, seed: int):
        self.state = np.random.default_rng(seed)
    
    def randperm(self, n: int, k: int | None = None) -> torch.Tensor:
        perm = torch.from_numpy(self.state.permutation(n))
        if k is not None:
            perm = perm[:k]
        return perm
    
    def randint(self, low: int, high: int, size: int) -> torch.Tensor:
        return torch.from_numpy(self.state.integers(low, high, size=size))
    
    def get_state(self) -> dict:
        return self.state.bit_generator.state

class CoinIndexGenerator:
    def __init__(
        self,
        coin_types: list[CoinType],
        batch_size: int,
        max_in_class: int = 16,
    ):
        super().__init__()
        self.coin_types = coin_types
        self.batch_size = batch_size
        self.max_in_class = max_in_class
        self.type_sizes = torch.tensor(
            [len(coin_type) for coin_type in coin_types], dtype=torch.int64
        )

    def iterate(self, rng_seed: int = 42, n_batches: int = 1_000):
        rng = Generator(rng_seed)

        for _ in range(n_batches):
            types, counts = self._sample_types_and_counts(rng)
            type_indices, image_indices = self._sample_image_indices(types, counts, rng)
            yield type_indices, image_indices

    def _sample_types_and_counts(self, rng: Generator) -> tuple[Tensor, Tensor]:
        types_perm = rng.randperm(len(self.coin_types))
        type_sizes = self.type_sizes[types_perm]
        type_counts = torch.minimum(
            type_sizes, torch.tensor(self.max_in_class, dtype=torch.int64)
        )

        cum_counts = type_counts.cumsum(0)
        n_types = torch.searchsorted(cum_counts, self.batch_size, right=True) + 1
        chosen_types = types_perm[:n_types]
        chosen_counts = type_counts[:n_types]
        excess_count = chosen_counts.sum() - self.batch_size
        assert excess_count >= 0
        chosen_counts[-1] -= excess_count

        return chosen_types, chosen_counts

    def _sample_image_indices(
        self, types: Tensor, counts: Tensor, rng: Generator
    ) -> tuple[Tensor, Tensor]:
        type_ids, image_ids = [], []
        for type_id, n_in_type in zip(types, counts):
            n_max_in_type = len(self.coin_types[type_id])
            assert n_in_type <= n_max_in_type
            image_indices = rng.randperm(n_max_in_type, n_in_type)
            type_ids.append(torch.full((n_in_type,), type_id, dtype=torch.int64))
            image_ids.append(image_indices)

        return torch.cat(type_ids), torch.cat(image_ids)


class InMemoryCoinDataset:
    def __init__(
        self,
        coin_types: list[CoinType],
        batch_size: int,
        max_in_class: int = 4,
        augmentation: Callable[[Tensor], Tensor] = identity,
    ):
        super().__init__()
        self.coin_types = coin_types
        self.in_memory_slab = InMemoryCoins(coin_types)
        self.max_in_class = max_in_class
        self.batch_size = batch_size
        self.augmentation = augmentation

    def n_types(self):
        return len(self.coin_types)

    def iterate(self, rng_seed: int = 42, n_batches: int = 1_000, device: torch.device = torch.device("cpu")):
        index_generator = CoinIndexGenerator(
            self.coin_types,
            self.batch_size,
            max_in_class=self.max_in_class,
        )

        slab = self.in_memory_slab.to_device(device)

        for type_index, image_index in index_generator.iterate(rng_seed, n_batches):
            images = slab.get(type_index, image_index)
            images = self.augmentation(images)
            yield images, type_index


class FlipAdapter:
    def __init__(
        self,
        coin_dataset: InMemoryCoinDataset,
    ):
        self.coin_dataset = coin_dataset

    def n_types(self):
        return 2 * self.coin_dataset.n_types()

    def iterate(self, rng_seed: int = 42, n_batches: int = 1_000, device: torch.device = torch.device("cpu")):
        index_generator = CoinIndexGenerator(
            self.coin_dataset.coin_types,
            self.coin_dataset.batch_size // 2,
            max_in_class=self.coin_dataset.max_in_class,
        )

        slab = self.coin_dataset.in_memory_slab.to_device(device)

        rng = Generator(rng_seed)

        for _ in range(n_batches):
            types, counts = index_generator._sample_types_and_counts(rng)

            base_indices, base_image_indices = index_generator._sample_image_indices(types, counts, rng)
            flip_indices, flip_image_indices = index_generator._sample_image_indices(types, counts, rng)

            base_images = slab.get(base_indices, base_image_indices)
            flip_images = slab.get(flip_indices, flip_image_indices)

            flip_images = torch.flip(flip_images, (-1, ))
            flip_indices += self.coin_dataset.n_types()

            images = torch.cat([base_images, flip_images])
            type_index = torch.cat([base_indices, flip_indices])

            images = self.coin_dataset.augmentation(images)
            yield images, type_index

train_augmentation = augmentation.container.AugmentationSequential(
    transforms.ToDtype(torch.float32, scale=True),
    augmentation.RandomRotation(360, p=1.0),
    augmentation.RandomPerspective(distortion_scale=0.25),
    augmentation.RandomResizedCrop((224, 224), same_on_batch=False, scale=(0.75, 1.0)),
    augmentation.ColorJiggle(0.2, 0.2, 0.2),
    augmentation.RandomGrayscale(p=1.0),
)

val_augmentation = augmentation.container.AugmentationSequential(
    transforms.ToDtype(torch.float32, scale=True),
    augmentation.Resize((224, 224)),
    augmentation.RandomGrayscale(p=1.0),
)