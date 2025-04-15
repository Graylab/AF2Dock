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
import importlib
import torch
import torch.nn as nn

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed

class LayerNormNoBias(nn.Module):
    def __init__(self, c_in, eps=1e-5):
        super(LayerNormNoBias, self).__init__()
        
        self.c_in = (c_in,)
        self.eps = eps

        self.weight = nn.Parameter(torch.ones(c_in))

    def forward(self, x): 
        d = x.dtype
        deepspeed_is_initialized = (
            deepspeed_is_installed and 
            deepspeed.comm.comm.is_initialized()
        )
        if d is torch.bfloat16 and not deepspeed_is_initialized:
            with torch.cuda.amp.autocast(enabled=False):
                out = nn.functional.layer_norm(
                    x, 
                    self.c_in, 
                    self.weight.to(dtype=d), 
                    None, 
                    self.eps
                )
        else:
            out = nn.functional.layer_norm(
                x,
                self.c_in,
                self.weight,
                None,
                self.eps,
            )

        return out
