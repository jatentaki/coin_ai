import torch
from torch import nn, Tensor
from einops import rearrange, repeat


def make_whitening_linear(embs: Tensor, out_dim: int | None = None) -> nn.Linear:
    if out_dim is None:
        out_dim = embs.shape[-1]

    linear = nn.Linear(embs.shape[-1], out_dim, bias=True)
    with torch.no_grad():
        embs_demean = embs - embs.mean(dim=0)
        _u, s, vT = torch.linalg.svd(embs_demean, full_matrices=False)
        weight = vT[:out_dim]
        # unclear why we don't need to multiply by the inverse of the square root of s
        # weight = (vT[:out_dim].T @ torch.diag(1 / s[:out_dim].sqrt())).T
        bias = torch.nn.functional.linear(embs, weight, bias=None).mean(dim=0)
        linear.weight = nn.Parameter(weight)
        linear.bias = nn.Parameter(-bias)

    return linear


class AttentionReadout(nn.Module):
    def __init__(
        self,
        out_dim: int = 32,
        head_dim: int = 16,
        n_head: int = 4,
        dino_dim: int = 384,
    ):
        super().__init__()
        assert out_dim % n_head == 0

        c = head_dim * n_head
        self.n_head = n_head
        self.query = nn.Parameter(torch.randn(n_head, head_dim))
        self.key = nn.Linear(dino_dim, c, bias=False)
        self.value = nn.Linear(dino_dim, c, bias=False)
        self.proj = nn.Linear(c, out_dim)

    def forward(self, dino_output: dict[str, Tensor]) -> Tensor:
        tokens = dino_output["x_norm_patchtokens"]

        h = self.n_head

        keys = rearrange(self.key(tokens), "b n (h c) -> b h n c", h=h)
        values = rearrange(self.value(tokens), "b n (h c) -> b h n c", h=h)
        query = repeat(self.query, "h c -> b h 1 c", b=keys.shape[0])

        attn = nn.functional.scaled_dot_product_attention(
            query,
            keys,
            values,
        )
        attn = rearrange(attn, "b h 1 c -> b (h c)")

        return self.proj(attn)


class MLP(nn.Sequential):
    def __init__(self, dim: int):
        in_proj = nn.Linear(dim, 4 * dim, bias=False)
        out_proj = nn.Linear(4 * dim, dim, bias=False)
        activation = nn.GELU()
        super().__init__(in_proj, activation, out_proj)


class AttentionReadoutLayer(nn.Module):
    def __init__(self, n_head: int = 8, dino_dim: int = 384):
        super().__init__()
        self.mha_norm = nn.LayerNorm(dino_dim, elementwise_affine=False)
        self.mha = nn.MultiheadAttention(dino_dim, n_head, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dino_dim, elementwise_affine=False)
        self.mlp = MLP(dino_dim)

        with torch.no_grad():
            self.mlp[-1].weight *= 0.1
            self.mha.out_proj.weight *= 0.1

    def forward(self, patches: Tensor, query: Tensor) -> Tensor:
        query = (
            query
            + self.mha(self.mha_norm(query), patches, patches, need_weights=False)[0]
        )
        query = query + self.mlp(self.mlp_norm(query))
        return query


class MultilayerAttentionReadout(nn.Module):
    def __init__(
        self, out_dim: int = 32, n_head: int = 8, dino_dim: int = 384, n_layers: int = 2
    ):
        super().__init__()

        self.query = nn.Parameter(torch.randn(1, 1, dino_dim))
        self.layers = nn.ModuleList(
            [AttentionReadoutLayer(n_head, dino_dim) for _ in range(n_layers)]
        )
        self.out_norm = nn.LayerNorm(dino_dim, elementwise_affine=False)
        self.proj = nn.Linear(dino_dim, out_dim)

    def forward(self, dino_output: dict[str, Tensor]) -> Tensor:
        tokens = dino_output["x_norm_patchtokens"]

        query = self.query.repeat(tokens.size(0), 1, 1)

        for layer in self.layers:
            query = layer(tokens, query)

        return self.proj(self.out_norm(query.squeeze(1)))


class TransformerReadout(nn.Module):
    def __init__(self, out_dim: int = 32, n_head: int = 8, n_layers: int = 2):
        super().__init__()

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=384, nhead=n_head, batch_first=True, norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )
        self.query_token = nn.Parameter(torch.randn(1, 1, 384))
        self.norm = nn.LayerNorm(384, elementwise_affine=False)
        self.proj = nn.Linear(384, out_dim)

    def forward(self, dino_output: dict[str, Tensor]) -> Tensor:
        memory = dino_output["x_norm_patchtokens"]
        target = torch.cat(
            [
                self.query_token.repeat(memory.shape[0], 1, 1),
                dino_output["x_norm_clstoken"].unsqueeze(1),
                dino_output["x_norm_regtokens"],
            ],
            dim=1,
        )

        decoded = self.transformer_decoder(target, memory)

        return self.proj(self.norm(decoded[:, 0]))


class DinoWithHead(nn.Module):
    def __init__(self, head: nn.Module):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14_reg")
        self.dino.requires_grad_(False)
        self.head = head

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            dino_output = self.dino.forward_features(x)
        return self.head(dino_output)


class WhiteningWrapper(nn.Module):
    def __init__(
        self, inner_head: nn.Module, dino_dim: int = 384, learnable: bool = False
    ):
        super().__init__()
        self.inner_head = inner_head
        self.whitening = None
        self.learnable = learnable
        self.dino_dim = dino_dim

    def forward(self, dino_output: dict[str, Tensor]) -> Tensor:
        patches = dino_output["x_norm_patchtokens"]
        if self.whitening is None:
            print(f"CREATING WHITENING (learnable = {self.learnable})")
            self.whitening = make_whitening_linear(
                patches.flatten(0, -2), self.dino_dim
            )
            if not self.learnable:
                self.whitening.requires_grad_(False)
        whitened_patches = self.whitening(patches)
        dino_output["x_norm_patchtokens"] = whitened_patches
        return self.inner_head(dino_output)
