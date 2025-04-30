# started from code from https://github.com/lucidrains/alphafold3-pytorch, MIT License, Copyright (c) 2024 Phil Wang
import math
import torch
from torch import nn
from openfold.model.primitives import Linear, LayerNorm
from openfold.utils.tensor_utils import add
from AF2Dock.model.primitives import LayerNormNoBias
from AF2Dock.model.transition import Transition, PreLayerNorm

class FourierEmbedding(nn.Module):
    """ Algorithm 22 """

    def __init__(self, dim):
        super().__init__()
        self.proj = Linear(1, dim)
        nn.init.normal_(self.proj.weight, mean = 0.0, std = 1.0)
        nn.init.normal_(self.proj.bias, mean = 0.0, std = 1.0)
        self.proj.requires_grad_(False)

    def forward(
        self,
        times,
    ):
        times = times.unsqueeze(-1) #rearrange(times, 'b -> b 1')
        rand_proj = self.proj(times)
        return torch.cos(2 * math.pi * rand_proj)

class PairConditioning(nn.Module):

    def __init__(
        self,
        dim_pair = 64,
        dim_fourier = 256,
        # num_transitions = 2,
        # transition_expansion_factor = 2,
    ):
        super().__init__()

        self.dim_pair = dim_pair

        self.norm_pair = LayerNorm(dim_pair)

        self.fourier_embed = FourierEmbedding(dim_fourier)
        self.norm_fourier = LayerNorm(dim_fourier)
        self.fourier_to_pair = Linear(dim_fourier, dim_pair, bias=False)

        # transitions = nn.ModuleList([])
        # for _ in range(num_transitions):
        #     transition = PreLayerNorm(Transition(dim = dim_pair, expansion_factor = transition_expansion_factor), dim = dim_pair)
        #     transitions.append(transition)

        # self.transitions = transitions

    def forward(
        self,
        times,
        pair_cond,
        # chunk_size = None,
        inplace_safe = False
    ):

        pair_cond = self.norm_pair(pair_cond)

        fourier_embed = self.fourier_embed(times)

        normed_fourier = self.norm_fourier(fourier_embed)

        fourier_to_pair = self.fourier_to_pair(normed_fourier)

        pair_cond = add(pair_cond, fourier_to_pair[..., None, None, :], inplace_safe) #rearrange(fourier_to_single, 'b d -> b 1 d') + single_repr

        # for transition in self.transitions:
        #     pair_cond = add(pair_cond, transition(pair_cond, chunk_size=chunk_size))

        return pair_cond

class AdaptiveLayerNorm(nn.Module):
    """ Algorithm 26 """

    def __init__(
        self,
        dim,
        dim_cond
    ):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine = False)
        self.norm_cond = LayerNormNoBias(dim_cond)

        self.to_gamma = nn.Sequential(
            Linear(dim_cond, dim),
            nn.Sigmoid()
        )

        self.to_beta = Linear(dim_cond, dim, bias = False)

    def forward(
        self,
        x,
        cond
    ):

        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)

        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        return normed * gamma + beta

class ConditionWrapper(nn.Module):
    """ Algorithm 25 """

    def __init__(
        self,
        fn,
        dim,
        dim_cond,
        adaln_zero_bias_init_value = -2.
    ):
        super().__init__()
        self.fn = fn
        self.adaptive_norm = AdaptiveLayerNorm(dim = dim, dim_cond = dim_cond)

        adaln_zero_gamma_linear = Linear(dim_cond, dim)
        nn.init.zeros_(adaln_zero_gamma_linear.weight)
        nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value)

        self.to_adaln_zero_gamma = nn.Sequential(
            adaln_zero_gamma_linear,
            nn.Sigmoid()
        )

    def forward(
        self,
        x,
        cond,
        **kwargs
    ):
        x = self.adaptive_norm(x, cond = cond)

        out = self.fn(x, **kwargs)

        tuple_output = isinstance(out, tuple)

        if tuple_output:
            out, *rest = out

        gamma = self.to_adaln_zero_gamma(cond)
        out = out * gamma

        if tuple_output:
            out = (out, *rest)

        return out
