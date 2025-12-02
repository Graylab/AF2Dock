# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License

# Copyright (c) 2024 Phil Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from typing import Optional
import torch
from torch import nn
from torch.nn import functional as F
from openfold.model.primitives import Linear, LayerNorm
from openfold.utils.chunk_utils import chunk_layer

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

    def _transition(self, z):
        return self.ff(z)

    @torch.jit.ignore
    def _chunk(self,
        z: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        return chunk_layer(
            self._transition,
            {"z": z},
            chunk_size=chunk_size,
            no_batch_dims=len(z.shape[:-2]),
        )

    def forward(self, 
        z: torch.Tensor, 
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            z:
                [*, N_res, N_res, C_z] pair embedding
        Returns:
            [*, N_res, N_res, C_z] pair embedding update
        """

        if chunk_size is not None:
            z = self._chunk(z, chunk_size)
        else:
            z = self._transition(z=z)

        return z

class PreLayerNorm(nn.Module):
    def __init__(
        self,
        fn,
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
