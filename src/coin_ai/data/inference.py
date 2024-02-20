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
        self.images = [f for f in os.listdir(root) if f.endswith(".png")]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def load_by_name(self, name: str) -> Tensor:
        path = os.path.join(self.root, name)
        image = io.read_image(path, io.ImageReadMode.RGB)
        return self.transform(image).squeeze(0)

    def __getitem__(self, idx) -> tuple[Tensor, str]:
        name = self.images[idx]
        return self.load_by_name(name), name
