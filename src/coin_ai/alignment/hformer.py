from dataclasses import dataclass

from coin_ai.alignment.data import HomographyBatch
import torch
import kornia.geometry as KG
from torch import nn, Tensor
from einops import rearrange, repeat

from coin_ai.alignment.infra import DenseDino


@dataclass
class HCorrespondences:
    corners_a: Tensor  # B x 4 x 2
    corners_b: Tensor  # B x 4 x 2

    def implied_homography(self) -> Tensor:
        return KG.get_perspective_transform(self.corners_a.cpu(), self.corners_b.cpu())


class QueryInit(nn.Module):
    def __init__(self, d_memory: int, d_target: int):
        super().__init__()
        self.n_heads = 8
        self.q = nn.Parameter(torch.randn(self.n_heads, 4, d_memory))
        self.v = nn.Linear(d_memory, d_target)
        self.o = nn.Linear(d_target, d_target)

    def forward(self, src_features: Tensor) -> Tensor:
        b, n, c = src_features.shape
        q = repeat(self.q, "h q c -> b h q c", b=b)
        k = rearrange(src_features, "b n c -> b 1 n c")  # full attention
        v = rearrange(self.v(src_features), "b n (h c) -> b h n c", h=self.n_heads)
        queries = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.o(rearrange(queries, "b h n c -> b n (h c)"))


class MLP(nn.Module):
    def __init__(self, d_io: int, d_ff: int | None = None):
        super().__init__()

        if d_ff is None:
            d_ff = 4 * d_io

        self.fc1 = nn.Linear(d_io, d_ff)
        self.fc2 = nn.Linear(d_ff, d_io)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(nn.functional.gelu(self.fc1(x)))


class CrossAttention(nn.Module):
    def __init__(self, d_src: int, d_tgt: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.q = nn.Linear(d_tgt, d_tgt)
        self.kv = nn.Linear(d_src, 2 * d_tgt)
        self.o = nn.Linear(d_tgt, d_tgt)

    def forward(self, target: Tensor, memory: Tensor) -> Tensor:
        b, q, c = target.shape

        q = rearrange(self.q(target), "b q (h c) -> b h q c", h=self.n_heads)
        k, v = rearrange(
            self.kv(memory), "b n (t h c) -> t b h n c", h=self.n_heads, t=2
        )

        out = nn.functional.scaled_dot_product_attention(q, k, v)
        return self.o(rearrange(out, "b h n c -> b n (h c)"))


class CrossAttentionBlock(nn.Module):
    def __init__(
        self, d_memory: int, d_target: int, n_heads: int, d_ff: int | None = None
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_target)
        self.cross_attention = CrossAttention(
            d_src=d_memory, d_tgt=d_target, n_heads=n_heads
        )
        self.norm2 = nn.LayerNorm(d_target)
        self.mlp = MLP(d_target, d_ff)

    def forward(self, queries: Tensor, memory: Tensor) -> Tensor:
        q = queries
        q = q + self.cross_attention(self.norm1(q), memory)
        q = q + self.mlp(self.norm2(q))
        return q


def homography_loss(
    homography_batch: HomographyBatch,
    predictions: HCorrespondences,
) -> Tensor:
    corners_b_gt = KG.linalg.transform_points(
        homography_batch.H_12, predictions.corners_a
    )

    pred_corners_b = predictions.corners_b / homography_batch.images.shape[-1]
    corners_b_gt = corners_b_gt / homography_batch.images.shape[-1]

    return nn.functional.mse_loss(pred_corners_b, corners_b_gt)


class HFormer(nn.Module):
    def __init__(
        self, n_layers: int = 3, d_target: int = 128, deformation_scale: float = 0.5
    ):
        super().__init__()

        self.dino = DenseDino()
        self.dino.requires_grad_(False)
        d_memory = 384
        self.n_heads = 8
        self.query_init = QueryInit(d_memory=d_memory, d_target=d_target)
        self.attn_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    d_memory=d_memory, d_target=d_target, n_heads=self.n_heads
                )
                for _ in range(2 * n_layers - 1)
            ]
        )

        self.final_norm = nn.LayerNorm(d_target)
        self.xy_head = nn.Linear(d_target, 2, bias=False)

        self.register_buffer(
            "corners_a",
            torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
            ).reshape(1, 4, 2)
            * 0.5
            + 0.25,
        )
        self.deformation_scale = deformation_scale

    def forward_from_features(
        self, src_feat: Tensor, dst_feat: Tensor, image_shape: tuple[int, int]
    ) -> HCorrespondences:
        b = src_feat.shape[0]

        q = self.query_init(src_feat)

        for i, block in enumerate(self.attn_blocks):
            if i % 2 == 0:
                memory = dst_feat
            else:
                memory = src_feat

            q = block(q, memory)

        offset = torch.tanh(self.xy_head(self.final_norm(q)))

        corners_b = self.corners_a + self.deformation_scale * offset

        scale = torch.tensor(
            image_shape,
            dtype=torch.float32,
            device=offset.device,
        )
        return HCorrespondences(
            corners_a=self.corners_a.repeat(b, 1, 1) * scale,
            corners_b=corners_b * scale,
        )

    def forward(self, images: Tensor) -> HCorrespondences:
        src_feat, dst_feat = self._get_features(images)
        return self.forward_from_features(src_feat, dst_feat, images.shape[-2:])

    def _get_features(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """
        Images: 2 x B x C x H x W

        Returns:
            src_feat: B x N x C
            dst_feat: B x N x C
        """
        t, b, c, h, w = images.shape
        assert t == 2
        assert c == 3

        images_flat = rearrange(images, "t b c h w -> (t b) c h w")
        with torch.no_grad():
            features_flat: Tensor = self.dino(images_flat)
        src_feat, dst_feat = rearrange(
            features_flat, "(t b) h w c -> t b (h w) c", b=b, t=2
        )
        return src_feat, dst_feat

    def loss(self, homography_batch: HomographyBatch) -> Tensor:
        feat_a, feat_b = self._get_features(homography_batch.images)
        loss_ab = homography_loss(
            homography_batch,
            self.forward_from_features(
                feat_a, feat_b, homography_batch.images.shape[-2:]
            ),
        )
        loss_ba = homography_loss(
            homography_batch.flip(),
            self.forward_from_features(
                feat_b, feat_a, homography_batch.images.shape[-2:]
            ),
        )

        return (loss_ab + loss_ba) / 2


class MeanLearner(nn.Module):
    def __init__(self):
        super().__init__()

        self.register_buffer(
            "corners_a",
            torch.tensor(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
            ).reshape(1, 4, 2)
            * 0.5
            + 0.25,
        )

        self.offsets = nn.Parameter(torch.zeros_like(self.corners_a))

    def forward(self, images: Tensor) -> HCorrespondences:
        b = images.shape[1]

        offset = torch.tanh(self.offsets).repeat(b, 1, 1)

        corners_b = self.corners_a + offset

        scale = torch.tensor(
            [images.shape[-1], images.shape[-2]],
            dtype=torch.float32,
            device=images.device,
        )
        return HCorrespondences(
            corners_a=self.corners_a.repeat(b, 1, 1) * scale,
            corners_b=corners_b * scale,
        )
