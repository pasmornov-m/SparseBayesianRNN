import torch


def alpha_regf(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * torch.log1p(torch.exp(-a)) - (
        0.03 + 1.0 / (1.0 + torch.exp(-(1.5 * (a + 1.3)))) * 0.64
    )

def hard_sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(0.2 * x + 0.5, min=0.0, max=1.0)

def safe_torch_log(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return torch.log(x + eps)

def clip_func(x, to=8):
        return torch.clamp(x, min=-to, max=to)