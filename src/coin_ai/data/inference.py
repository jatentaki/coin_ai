import os
from torchvision import io
from torch import Tensor
from torchvision.transforms import functional as F


class ResizeAndKeepRatio:
    def __init__(self, target_size: int):
        self.target_size = target_size

    def __call__(self, image):
        # Original dimensions
        _, orig_width, orig_height = image.shape

        # Determine the longer side and compute the new size
        scale = self.target_size / max(orig_width, orig_height)
        new_width, new_height = int(orig_width * scale), int(orig_height * scale)

        # Resize the image
        resized_image = F.resize(image, (new_height, new_width))

        # Compute padding to make the image square
        padding_left = (self.target_size - new_width) // 2
        padding_top = (self.target_size - new_height) // 2
        padding_right = self.target_size - new_width - padding_left
        padding_bottom = self.target_size - new_height - padding_top

        # Pad the resized image
        padded_image = F.pad(
            resized_image,
            (padding_left, padding_top, padding_right, padding_bottom),
            fill=0,
        )

        return padded_image


class InferenceImageDataset:
    def __init__(self, root: str, transform: callable):
        self.root = root
        self.image_relative_paths = []
        for dir, _, files in os.walk(root):
            dir = os.path.relpath(dir, root)
            for file in files:
                if file.endswith(".png"):
                    self.image_relative_paths.append(os.path.join(dir, file))

        self.transform = transform

    def __len__(self):
        return len(self.image_relative_paths)

    def load_by_relative_path(self, relative_path: str) -> Tensor:
        path = os.path.join(self.root, relative_path)
        image = io.read_image(path, io.ImageReadMode.RGB)
        return self.transform(image).squeeze(0)

    def __getitem__(self, idx) -> tuple[Tensor, str]:
        relative_path = self.image_relative_paths[idx]
        return self.load_by_relative_path(relative_path), relative_path
