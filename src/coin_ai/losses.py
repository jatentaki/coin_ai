import torch
from torch import nn, Tensor


class SigmoidLoss(nn.Module):
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        similarity = torch.einsum("ic,jc->ij", embeddings, embeddings)
        gt_similarity = labels.unsqueeze(0) == labels.unsqueeze(1)
        # ignore matching self-self
        weight = 1 - torch.eye(
            gt_similarity.size(0), dtype=torch.float, device=gt_similarity.device
        )

        return nn.functional.binary_cross_entropy_with_logits(
            similarity, gt_similarity.float(), weight=weight, reduction="mean"
        )

    def similarity(self, embeddings: Tensor) -> Tensor:
        return torch.einsum("ic,jc->ij", embeddings, embeddings).sigmoid()


class DotProductLoss(nn.Module):
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        similarity = self.similarity(embeddings)
        gt_similarity = labels.unsqueeze(0) == labels.unsqueeze(1)
        # ignore matching self-self
        weight = (~gt_similarity).float() * 2 - 1
        weight[
            torch.eye(gt_similarity.size(0), dtype=bool, device=gt_similarity.device)
        ] = 0

        return (weight * similarity).mean()

    def similarity(self, embeddings: Tensor) -> Tensor:
        embeddings = nn.functional.normalize(embeddings, dim=-1, p=2)
        return torch.einsum("ic,jc->ij", embeddings, embeddings)


class MarginLoss(nn.Module):
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        similarity = self.similarity(embeddings)
        gt_similarity = labels.unsqueeze(0) == labels.unsqueeze(1)

        is_valid_positive = gt_similarity.clone()
        is_valid_positive.fill_diagonal_(0)
        is_valid_negative = (~gt_similarity).clone()
        is_valid_negative.fill_diagonal_(0)

        positive = torch.where(
            is_valid_positive,
            similarity,
            torch.full_like(similarity, fill_value=float("-inf")),
        ).amax(dim=-1)
        negative = torch.where(
            is_valid_negative,
            similarity,
            torch.full_like(similarity, fill_value=float("-inf")),
        ).amax(dim=-1)

        # in case there are no valid positive or negative pairs (e.g. in case of single class in batch)
        positive = torch.where(
            torch.isfinite(positive), positive, torch.zeros_like(positive)
        )
        negative = torch.where(
            torch.isfinite(negative), negative, torch.zeros_like(negative)
        )

        return torch.relu(negative - positive + 0.5).mean()

    def similarity(
        self, embeddings: Tensor, other_embeddings: Tensor | None = None
    ) -> Tensor:
        embeddings = nn.functional.normalize(embeddings, dim=-1, p=2)
        if other_embeddings is None:
            other_embeddings = embeddings
        else:
            other_embeddings = nn.functional.normalize(other_embeddings, dim=-1, p=2)
        return torch.einsum("ic,jc->ij", embeddings, other_embeddings)


class CDistMarginLoss(MarginLoss):
    def similarity(
        self, embeddings: Tensor, other_embeddings: Tensor | None = None
    ) -> Tensor:
        if other_embeddings is None:
            other_embeddings = embeddings
        return -torch.cdist(embeddings, other_embeddings, p=2)
