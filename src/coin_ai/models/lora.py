import torch
from torch import nn, Tensor
import loralib as lora


class LoraLinear(lora.Linear):
    def __repr__(self):
        r = self.lora_A.shape[0]
        bias = self.bias is not None
        n_trainable = self.lora_A.numel() + self.lora_B.numel()
        return (
            f"LoraLinear(in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={bias}, r={r}, alpha={self.lora_alpha}, n_trainable={n_trainable})"
        )


def swap_linear_for_lora(
    module: nn.Module, r: int, lora_alpha: float = 1.0, skip_mlp: bool = False
) -> nn.Module:
    for name, child in module.named_children():
        if skip_mlp and 'mlp' in name:
            continue

        if isinstance(child, nn.Linear):
            lora_linear = LoraLinear(
                child.in_features,
                child.out_features,
                r=r,
                lora_alpha=lora_alpha,
                bias=child.bias is not None,
            )
            lora_linear.load_state_dict(child.state_dict(), strict=False)
            setattr(module, name, lora_linear)
        else:
            swap_linear_for_lora(child, r, lora_alpha=lora_alpha, skip_mlp=skip_mlp)


class LoraDino(nn.Module):
    def __init__(
        self,
        r: int,
        out_dim: int,
        model: str = "dinov2_vits14_reg",
        n_lora_layers: int = 1,
        lora_alpha: float = 1.0,
        skip_mlp: bool = False,
    ):
        super().__init__()
        self.dino = torch.hub.load("facebookresearch/dinov2", model)
        lora_layers = self.dino.blocks[-n_lora_layers:]
        swap_linear_for_lora(lora_layers, r, lora_alpha=lora_alpha, skip_mlp=skip_mlp)
        dino_out_dim = self.dino.norm.bias.shape[0]
        lora.mark_only_lora_as_trainable(self)
        self.head = nn.Linear(dino_out_dim, out_dim)
        self.skip_mlp_lora = skip_mlp
    
    def extra_repr(self) -> str:
        return f'skip_mlp_lora={self.skip_mlp_lora}'

    def n_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dino(x)
        x = self.head(x)
        return x
