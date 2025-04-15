# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang
from torch import nn
from torch.nn import functional as F
from openfold.model.primitives import Linear, LayerNorm

class SwiGLU(nn.Module):
    def forward(
        self,
        x
    ):

        x, gates = x.chunk(2, dim = -1)
        return F.silu(gates) * x

class Transition(nn.Module):
    def __init__(
        self,
        *,
        dim,
        expansion_factor = 2
    ):
        super().__init__()
        dim_inner = int(dim * expansion_factor)

        self.ff = nn.Sequential(
            Linear(dim, dim_inner * 2, bias = False),
            SwiGLU(),
            Linear(dim_inner, dim, bias = False, init='final'),
        )

    def forward(
        self,
        x
    ):

        return self.ff(x)

class PreLayerNorm(nn.Module):
    def __init__(
        self,
        fn,
        *,
        dim,
    ):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(
        self,
        x,
        **kwargs
    ):

        x = self.norm(x)
        return self.fn(x, **kwargs)
