import torch
from torch import nn, Tensor
import torch.nn.functional as F


class SigmoidLoss(nn.Module):
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        similarity = torch.einsum("ic,jc->ij", embeddings, embeddings)
        gt_similarity = labels.unsqueeze(0) == labels.unsqueeze(1)
        # ignore matching self-self
        weight = 1 - torch.eye(
            gt_similarity.size(0), dtype=torch.float, device=gt_similarity.device
        )

        return F.binary_cross_entropy_with_logits(
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
        embeddings = F.normalize(embeddings, dim=-1, p=2)
        return torch.einsum("ic,jc->ij", embeddings, embeddings)


class MarginLoss(nn.Module):
    def __init__(self, margin: float = 0.5, negative_weight: float = 1.0):
        super().__init__()

        assert margin >= 0, "Margin must be non-negative"
        assert negative_weight > 0, "Negative weight must be positive"

        self.margin = margin
        self.negative_weight = negative_weight

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

        pre_relu = self.negative_weight * negative - positive + self.margin
        losses = F.relu(pre_relu)

        return losses.mean()

    def similarity(
        self, embeddings: Tensor, other_embeddings: Tensor | None = None
    ) -> Tensor:
        embeddings = F.normalize(embeddings, dim=-1, p=2)
        if other_embeddings is None:
            other_embeddings = embeddings
        else:
            other_embeddings = F.normalize(other_embeddings, dim=-1, p=2)
        return torch.einsum("ic,jc->ij", embeddings, other_embeddings)


class CDistMarginLoss(MarginLoss):
    def similarity(
        self, embeddings: Tensor, other_embeddings: Tensor | None = None
    ) -> Tensor:
        if other_embeddings is None:
            other_embeddings = embeddings
        return -torch.cdist(embeddings, other_embeddings, p=2)


class CosineEmbeddingLoss(nn.Module):
    def __init__(self, margin: float = 0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        similarity = self.similarity(embeddings)
        gt_similarity = labels.unsqueeze(0) == labels.unsqueeze(1)

        positive_mask = gt_similarity.to(similarity.dtype)
        positive_mask.fill_diagonal_(0)
        negative_mask = (~gt_similarity).to(similarity.dtype)
        negative_mask.fill_diagonal_(0)

        positive_term = 1 - similarity
        negative_term = F.relu(similarity - self.margin)

        return (positive_term * positive_mask + negative_term * negative_mask).mean()

    def similarity(
        self, embeddings: Tensor, other_embeddings: Tensor | None = None
    ) -> Tensor:
        if other_embeddings is None:
            other_embeddings = embeddings
        return F.cosine_similarity(
            embeddings.unsqueeze(1), other_embeddings.unsqueeze(0), dim=-1
        )


class DiversityLoss(nn.Module):
    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        del labels
        return self.similarity(embeddings).mean()

    def similarity(
        self, embeddings: Tensor, other_embeddings: Tensor | None = None
    ) -> Tensor:
        embeddings = F.normalize(embeddings, dim=-1, p=2)
        if other_embeddings is None:
            other_embeddings = embeddings
        else:
            other_embeddings = F.normalize(other_embeddings, dim=-1, p=2)
        return torch.einsum("ic,jc->ij", embeddings, other_embeddings)


class ExhaustiveMarginLoss(nn.Module):
    def __init__(self, margin: float = 0.5, negative_weight: float = 1.0):
        super().__init__()

        assert margin >= 0, "Margin must be non-negative"
        assert negative_weight > 0, "Negative weight must be positive"

        self.margin = margin
        self.negative_weight = negative_weight

    def forward(self, embeddings: Tensor, labels: Tensor) -> Tensor:
        similarity = self.similarity(embeddings)
        gt_similarity = labels.unsqueeze(0) == labels.unsqueeze(1)

        is_valid_positive = gt_similarity.clone()
        is_valid_positive.fill_diagonal_(0)
        is_valid_negative = (~gt_similarity).clone()
        is_valid_negative.fill_diagonal_(0)

        positive_dist = similarity.unsqueeze(0)
        negative_dist = similarity.unsqueeze(1)

        positive_mask = is_valid_positive.unsqueeze(0)
        negative_mask = is_valid_negative.unsqueeze(1)

        pre_relu = self.negative_weight * negative_dist - positive_dist + self.margin
        mask = positive_mask & negative_mask & pre_relu.ge(0)

        return pre_relu[mask].mean()

    def similarity(
        self, embeddings: Tensor, other_embeddings: Tensor | None = None
    ) -> Tensor:
        embeddings = F.normalize(embeddings, dim=-1, p=2)
        if other_embeddings is None:
            other_embeddings = embeddings
        else:
            other_embeddings = F.normalize(other_embeddings, dim=-1, p=2)
        return torch.einsum("ic,jc->ij", embeddings, other_embeddings)
