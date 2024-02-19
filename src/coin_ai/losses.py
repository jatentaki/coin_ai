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

        positive_mask = gt_similarity & ~torch.eye(
            gt_similarity.size(0), dtype=bool, device=gt_similarity.device
        )
        negative_mask = ~gt_similarity & ~torch.eye(
            gt_similarity.size(0), dtype=bool, device=gt_similarity.device
        )

        positive = torch.where(positive_mask, similarity, torch.full_like(similarity, fill_value=float("-inf"))).amin(dim=-1)
        negative = torch.where(negative_mask, similarity, torch.full_like(similarity, fill_value=float("-inf"))).amin(dim=-1)

        return torch.relu(negative - positive + 0.5).mean()

    def similarity(self, embeddings: Tensor) -> Tensor:
        embeddings = nn.functional.normalize(embeddings, dim=-1, p=2)
        return torch.einsum("ic,jc->ij", embeddings, embeddings)