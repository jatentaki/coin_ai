from __future__ import annotations

import torch
import lightning as pl
import kornia.geometry as KG
import matplotlib.pyplot as plt
from torch import Tensor

from coin_ai.alignment.data import HomographyBatch, AugmentationBuilder
from coin_ai.alignment.hformer import HFormer, HCorrespondences


def plot_predictions(
    batch: HomographyBatch, correspondences: HCorrespondences
) -> list[plt.Figure]:
    corners_a = correspondences.corners_a.cpu()
    corners_b = correspondences.corners_b.cpu()
    corners_b_gt = KG.linalg.transform_points(batch.H_12.to("cpu"), corners_a)

    rep = [0, 1, 2, 3, 0]
    figures = []
    for s in range(batch.B):
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        a1.imshow(batch.images[s, 0].permute(1, 2, 0).cpu().numpy())
        a2.imshow(batch.images[s, 1].permute(1, 2, 0).cpu().numpy())
        a1.plot(corners_a[s, :, 0][rep], corners_a[s, :, 1][rep], "r--")
        a2.plot(corners_b[s, :, 0][rep], corners_b[s, :, 1][rep], "r--")
        a2.plot(corners_b_gt[s, :, 0][rep], corners_b_gt[s, :, 1][rep], "g--")
        a1.grid(True)
        a2.grid(True)

        figures.append(fig)

    return figures


class CoinLearner(pl.LightningModule):
    def __init__(self, model: HFormer, lr: float = 1e-4):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, batch: HomographyBatch) -> Tensor:
        return self.model(batch.images)

    def build_batch(self, batch: AugmentationBuilder) -> HomographyBatch:
        return batch.to("cpu").build().to(self.device)

    def training_step(self, raw_batch: AugmentationBuilder, batch_idx: int) -> Tensor:
        batch = self.build_batch(raw_batch)
        loss = self.model.loss(batch)
        self.log("train_loss", loss, batch_size=batch.B)
        return loss

    def validation_step(self, raw_batch: AugmentationBuilder, batch_idx: int) -> Tensor:
        batch = self.build_batch(raw_batch)
        loss = self.model.loss(batch)
        self.log("val_loss", loss, batch_size=batch.B)

        correspondences = self(batch)
        figures = plot_predictions(batch, correspondences)

        for i, fig in enumerate(figures):
            self.logger.experiment.add_figure(
                f"val_{batch_idx}_{i}", fig, self.current_epoch
            )

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
