from abc import abstractmethod
from math import sqrt

import torch
import kornia.geometry as KG
import torch.nn.functional as F
from torch import Tensor
from einops import repeat, rearrange

from coin_ai.alignment.infra import (
    EmbeddingHead,
    Matches,
    Batch,
    get_grid,
    ScoringModel,
)


class DenseScore(ScoringModel):
    def __init__(
        self,
        embedder: EmbeddingHead,
        mean_pos_features: Tensor,
        error_hinge: float = 0.1,
    ):
        super().__init__()
        self.embedder = embedder
        self.error_hinge = error_hinge
        self.register_buffer("mean_pos_features", mean_pos_features)

    @abstractmethod
    def similarity_to_scores(self, scores: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplemented

    def similarity(self, feat_1: Tensor, feat_2: Tensor) -> Tensor:
        emb_1 = self.embedder(feat_1 - self.mean_pos_features)
        emb_2 = self.embedder(feat_2 - self.mean_pos_features)
        sim = torch.einsum("b i j c, b k l c -> b i j k l", emb_1, emb_2) / sqrt(
            emb_1.shape[-1]
        )
        return rearrange(sim, "b i j k l -> b (i j) (k l)")

    def forward(self, feat_1: Tensor, feat_2: Tensor) -> Matches:
        sim = self.similarity(feat_1, feat_2)
        scores, weights = self.similarity_to_scores(sim)

        grid_2d = get_grid(feat_1)

        pts_1, pts_2 = torch.broadcast_tensors(
            grid_2d[:, :, None, None, :],
            grid_2d[None, None, :, :, :],
        )

        pts_1 = repeat(pts_1, "i j k l t -> b (i j k l) t", b=feat_1.shape[0])
        pts_2 = repeat(pts_2, "i j k l t -> b (i j k l) t", b=feat_1.shape[0])

        scores_flat = rearrange(scores, "b ij kl -> b (ij kl)")
        weights_flat = rearrange(weights, "b ij kl -> b (ij kl)")

        return Matches(pts_1, pts_2, scores=scores_flat, weights=weights_flat)

    def find_error(self, matches: Matches, batch: Batch) -> Tensor:
        matches = self.forward(batch.features_1, batch.features_2)
        H_12_norm = KG.normalize_homography(batch.H_12, (518, 518), (518, 518))
        endpts_1_transformed = KG.transform_points(H_12_norm, matches.pts_1)
        error = torch.linalg.norm(endpts_1_transformed - matches.pts_2, dim=-1)

        return error

    @abstractmethod
    def loss(self, batch: Batch) -> tuple[Tensor, dict[str, Tensor]]:
        raise NotImplemented


class DoubleSoftmaxScore(DenseScore):
    def similarity_to_scores(self, scores: Tensor) -> tuple[Tensor, Tensor]:
        assert scores.ndim == 3
        scores = scores
        log_scores = F.log_softmax(scores, dim=-1) + F.log_softmax(scores, dim=-2)
        weights = log_scores.exp()

        return log_scores, weights

    def loss(self, batch: Batch) -> tuple[Tensor, dict[str, Tensor]]:
        matches = self.forward(batch.features_1, batch.features_2)
        error = self.find_error(matches, batch)

        with torch.no_grad():
            n_matches = matches.weights.sum()
            exp_error = torch.einsum("bn,bn->", matches.weights, error) / n_matches

        reward_per_choice = self.error_hinge - error
        # mean_reward = (matches.weights * reward_per_choice).mean()
        mean_reward = torch.einsum(
            "bn, bn -> b", matches.weights, reward_per_choice
        ).mean() / (37**2)
        loss = -mean_reward

        return loss, {
            "loss": loss.item(),
            "exp_error": exp_error.item(),
            "n_matches": n_matches.item(),
        }


class SigmoidScore(DenseScore):
    def similarity_to_scores(self, scores: Tensor) -> tuple[Tensor, Tensor]:
        assert scores.ndim == 3
        log_scores = scores
        weights = log_scores.sigmoid()

        return log_scores, weights

    def loss(self, batch: Batch) -> tuple[Tensor, dict[str, Tensor]]:
        matches = self.forward(batch.features_1, batch.features_2)
        error = self.find_error(matches, batch)

        n_matches = matches.weights.sum()

        exp_error = (matches.weights * error).sum() / n_matches
        loss = exp_error - 1e-7 * n_matches

        return loss, {
            "loss": loss.item(),
            "exp_error": exp_error.item(),
            "n_matches": n_matches.item(),
        }


class GlobalSoftmaxScore(DenseScore):
    def similarity_to_scores(self, scores: Tensor) -> tuple[Tensor, Tensor]:
        assert scores.ndim == 3
        log_scores_flat = F.log_softmax(
            rearrange(scores, "b ij kl -> b (ij kl)"), dim=-1
        )
        log_scores = rearrange(
            log_scores_flat,
            "b (ij kl) -> b ij kl",
            ij=scores.shape[1],
            kl=scores.shape[2],
        )
        weights = log_scores.exp()

        return log_scores, weights

    def loss(self, batch: Batch) -> tuple[Tensor, dict[str, Tensor]]:
        matches = self.forward(batch.features_1, batch.features_2)
        error = self.find_error(matches, batch)

        exp_error = (matches.weights * error).sum(dim=-1).mean()
        entropy = -(matches.weights * matches.scores).sum(dim=-1).mean()
        loss = exp_error - 1e-1 * entropy

        return loss, {
            "loss": loss.item(),
            "exp_error": exp_error.item(),
            "entropy": entropy.item(),
        }


class BCEScore(DenseScore):
    def similarity_to_scores(self, scores: Tensor) -> tuple[Tensor, Tensor]:
        assert scores.ndim == 3
        log_scores = scores
        weights = log_scores.sigmoid()

        return log_scores, weights

    def loss(self, batch: Batch) -> tuple[Tensor, dict[str, Tensor]]:
        matches = self.forward(batch.features_1, batch.features_2)
        error = self.find_error(matches, batch)

        with torch.no_grad():
            exp_error = (matches.weights * error).sum() / matches.weights.sum()
            n_matches = matches.weights.sum()

        is_correct = (error < self.error_hinge).float()
        loss = F.binary_cross_entropy_with_logits(matches.scores, is_correct)

        return loss, {
            "loss": loss.item(),
            "exp_error": exp_error.item(),
            "n_matches": n_matches.item(),
        }


class DotProductScore(DenseScore):
    def similarity_to_scores(self, scores: Tensor) -> tuple[Tensor, Tensor]:
        assert scores.ndim == 3
        log_scores = scores
        weights = (scores + 1) / 2

        return log_scores, weights

    def loss(self, batch: Batch) -> tuple[Tensor, dict[str, Tensor]]:
        matches = self.forward(batch.features_1, batch.features_2)
        error = self.find_error(matches, batch)

        with torch.no_grad():
            exp_error = (matches.weights * error).sum() / matches.weights.sum()
            n_matches = matches.weights.sum()

        reward_per_choice = self.error_hinge - error
        mean_reward = ((matches.weights - 0.5) * reward_per_choice).mean()
        loss = -mean_reward

        return loss, {
            "loss": loss.item(),
            "exp_error": exp_error.item(),
            "n_matches": n_matches.item(),
        }
