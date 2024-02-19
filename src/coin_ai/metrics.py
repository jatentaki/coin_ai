from typing import Callable

import torch
from torch import nn, Tensor

class AccuracyMetric(nn.Module):
    def __init__(self, similarity: Callable[[Tensor], Tensor]):
        super().__init__()
        self.similarity = similarity

    def forward(self, embeddings: Tensor, labels: Tensor) -> dict[str, Tensor]:
        similarity = self.similarity(embeddings)
        pad_values = torch.full(
            (similarity.size(0),),
            fill_value=-float("inf"),
            dtype=similarity.dtype,
            device=similarity.device,
        )
        similarity = torch.diagonal_scatter(similarity, pad_values)
        gt_similarity = labels.unsqueeze(0) == labels.unsqueeze(1)

        _values, indices = similarity.topk(10, dim=1)
        is_correct_at_k = gt_similarity[
            torch.arange(similarity.size(0)).unsqueeze(1), indices
        ]
        is_correct_up_to_k = is_correct_at_k.cumsum(dim=1).bool()

        accuracy_at_k = is_correct_up_to_k.float().mean(dim=0).cpu()

        return {f'acc_at_{k+1}': accuracy_at_k[k].item() for k in range(10)}