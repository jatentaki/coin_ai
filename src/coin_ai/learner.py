import torch
import lightning.pytorch as pl
from torch import nn, Tensor


class LightningLearner(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        metric_fn: nn.Module,
        train_metric_fn: nn.Module | None = None,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn
        self.train_metric_fn = train_metric_fn

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        embeddings = self(images)
        loss = self.loss_fn(embeddings, labels)
        self.log("train/loss", loss)

        if self.train_metric_fn is not None:
            train_metrics = self.train_metric_fn(embeddings, labels)
            for k, v in train_metrics.items():
                self.log(f"train/{k}", v)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        images, labels = batch
        embeddings = self(images)
        metrics = self.metric_fn(embeddings, labels)
        for k, v in metrics.items():
            self.log(f"val/{k}", v)

    def configure_optimizers(self, weight_decay: float = 1e-2, lr: float = 1e-4):
        decay_params = [
            p
            for n, p in self.model.named_parameters()
            if "weight" in n and p.requires_grad
        ]
        non_decay_params = [
            p
            for n, p in self.model.named_parameters()
            if "weight" not in n and p.requires_grad
        ]

        optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": non_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
        )

        return optimizer
