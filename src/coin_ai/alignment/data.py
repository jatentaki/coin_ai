from __future__ import annotations

import csv
import os
from dataclasses import dataclass

import imageio
import numpy as np
import cv2


@dataclass
class HPair:
    WARP_KWARGS = dict(dsize=(518, 518), borderMode=cv2.BORDER_REPLICATE)

    image1: np.ndarray
    image2: np.ndarray
    H_12: np.ndarray

    @staticmethod
    def from_paths(path1: str, path2: str, H_12: np.ndarray) -> HPair:
        return HPair(
            imageio.imread(path1),
            imageio.imread(path2),
            H_12,
        )

    @property
    def H_21(self) -> np.ndarray:
        return np.linalg.inv(self.H_12)

    def image1_warped(self) -> np.ndarray:
        return cv2.warpPerspective(self.image1, self.H_12, **self.WARP_KWARGS)

    def image2_warped(self) -> np.ndarray:
        return cv2.warpPerspective(self.image2, self.H_21, **self.WARP_KWARGS)


def parse_homography_csv(source_path: str) -> list[HPair]:
    HEADERS = ["img1", "h11", "h12", "h13", "h21", "h22", "h23", "h31", "h32", "h33"]

    if not os.path.exists(source_path):
        raise ValueError(f"File {source_path} does not exist")

    pairs = []
    with open(source_path, "r") as f:
        reader = csv.reader(f)
        headers = next(reader)
        assert headers[: len(HEADERS)] == HEADERS

        for img1, img2, *floats in reader:
            H_floats = floats[:9]
            source_dir = os.path.dirname(source_path)
            path1 = os.path.join(source_dir, img1)
            path2 = os.path.join(source_dir, img2)
            H_opencv = np.array(H_floats, dtype=np.float32).reshape(3, 3)
            pairs.append(
                HPair.from_paths(
                    path1,
                    path2,
                    H_opencv,
                )
            )

    return pairs


def recursive_parse_homography_csv(root_path: str) -> list[HPair]:
    parsed = []

    for root, _, files in os.walk(root_path):
        for file in files:
            if file == "homographies.csv":
                parsed.extend(parse_homography_csv(os.path.join(root, file)))

    return parsed
