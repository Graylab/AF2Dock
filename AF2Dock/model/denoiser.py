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

from functools import partial

import torch
import torch.nn as nn
from typing import Optional

from openfold.utils import all_atom_multimer
from openfold.utils.feats import (
    pseudo_beta_fn,
    dgram_from_positions,
)
from openfold.model.primitives import Linear, LayerNorm
from openfold.utils import geometry
from openfold.utils.tensor_utils import add, tensor_tree_map

from openfold.model.dropout import (
    DropoutRowwise,
    DropoutColumnwise,
)
from openfold.model.triangular_multiplicative_update import (
    TriangleMultiplicationOutgoing,
    TriangleMultiplicationIncoming,
    FusedTriangleMultiplicationOutgoing,
    FusedTriangleMultiplicationIncoming
)
from openfold.model.template import TemplatePairStack
from openfold.utils.checkpointing import checkpoint_blocks
from openfold.utils.chunk_utils import ChunkSizeTuner

from AF2Dock.model.triangular import (
    ConditionedTriangleAttentionStartingNode,
    ConditionedTriangleAttentionEndingNode,
)
from AF2Dock.model.transition import Transition
from AF2Dock.model.conditioning import ConditionWrapper, PairConditioning

class RigidDenoiserStackBlock(nn.Module):
    def __init__(
        self,
        c_r: int,
        c_cond: int,
        c_hidden_tri_att: int,
        c_hidden_tri_mul: int,
        no_heads: int,
        pair_transition_n: int,
        dropout_rate: float,
        fuse_projection_weights: bool,
        inf: float,
        **kwargs,
    ):
        super(RigidDenoiserStackBlock, self).__init__()

        self.c_r = c_r
        self.c_cond = c_cond
        self.c_hidden_tri_att = c_hidden_tri_att
        self.c_hidden_tri_mul = c_hidden_tri_mul
        self.no_heads = no_heads
        self.pair_transition_n = pair_transition_n
        self.dropout_rate = dropout_rate
        self.inf = inf

        self.dropout_row = DropoutRowwise(self.dropout_rate)
        self.dropout_col = DropoutColumnwise(self.dropout_rate)

        self.tri_att_start = ConditionedTriangleAttentionStartingNode(
            self.c_r,
            self.c_cond,
            self.c_hidden_tri_att,
            self.no_heads,
            inf=inf,
        )
        self.tri_att_end = ConditionedTriangleAttentionEndingNode(
            self.c_r,
            self.c_cond,
            self.c_hidden_tri_att,
            self.no_heads,
            inf=inf,
        )

        if fuse_projection_weights:
            self.tri_mul_out = FusedTriangleMultiplicationOutgoing(
                self.c_r,
                self.c_hidden_tri_mul,
            )
            self.tri_mul_in = FusedTriangleMultiplicationIncoming(
                self.c_r,
                self.c_hidden_tri_mul,
            )
        else:
            self.tri_mul_out = TriangleMultiplicationOutgoing(
                self.c_r,
                self.c_hidden_tri_mul,
            )
            self.tri_mul_in = TriangleMultiplicationIncoming(
                self.c_r,
                self.c_hidden_tri_mul,
            )

        transition = Transition(
            self.c_r,
            self.pair_transition_n,
        )
        self.conditioned_transition = ConditionWrapper(
            transition,
            self.c_r,
            self.c_cond,
            adaln_zero_bias_init_value = -2.
        )

    def tri_att_start_end(self,
                          single: torch.Tensor,
                          cond: torch.Tensor,
                          padding_mask: torch.Tensor,
                          inter_chain_mask: torch.Tensor,
                          _attn_chunk_size: Optional[int],
                          use_deepspeed_evo_attention: bool,
                          use_lma: bool,
                          inplace_safe: bool):
        single = add(single,
                     self.dropout_row(
                         self.tri_att_start(
                             single,
                             cond,
                             mask=padding_mask,
                             chunk_size=_attn_chunk_size,
                             use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                             use_lma=use_lma,
                             inplace_safe=inplace_safe,
                         )
                     ) * inter_chain_mask[..., None],
                     inplace_safe,
                     )

        single = add(single,
                     self.dropout_col(
                         self.tri_att_end(
                             single,
                             cond,
                             mask=padding_mask,
                             chunk_size=_attn_chunk_size,
                             use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                             use_lma=use_lma,
                             inplace_safe=inplace_safe,
                         )
                     ) * inter_chain_mask[..., None],
                     inplace_safe,
                     )

        return single

    def tri_mul_out_in(self,
                       single: torch.Tensor,
                       padding_mask: torch.Tensor,
                       inter_chain_mask: torch.Tensor,
                       inplace_safe: bool):
        tmu_update = self.tri_mul_out(
            single,
            mask=padding_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=False,
        )
        tmu_update = tmu_update * inter_chain_mask[..., None]
        if not inplace_safe:
            single = single + self.dropout_row(tmu_update)
        else:
            single += tmu_update

        del tmu_update

        tmu_update = self.tri_mul_in(
            single,
            mask=padding_mask,
            inplace_safe=inplace_safe,
            _add_with_inplace=False,
        )
        tmu_update = tmu_update * inter_chain_mask[..., None]
        if not inplace_safe:
            single = single + self.dropout_row(tmu_update)
        else:
            single += tmu_update

        del tmu_update

        return single

    def forward(self,
                z: torch.Tensor,
                cond: torch.Tensor,
                padding_mask: torch.Tensor,
                inter_chain_mask: torch.Tensor,
                chunk_size: Optional[int] = None,
                use_deepspeed_evo_attention: bool = False,
                use_lma: bool = False,
                inplace_safe: bool = False,
                _attn_chunk_size: Optional[int] = None,
                ):
        if _attn_chunk_size is None:
            _attn_chunk_size = chunk_size

        single = z

        single = self.tri_att_start_end(single=self.tri_mul_out_in(single=single,
                                                                   padding_mask=padding_mask,
                                                                   inter_chain_mask=inter_chain_mask,
                                                                   inplace_safe=inplace_safe),
                                        cond=cond,
                                        padding_mask=padding_mask,
                                        inter_chain_mask=inter_chain_mask,
                                        _attn_chunk_size=_attn_chunk_size,
                                        use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                                        use_lma=use_lma,
                                        inplace_safe=inplace_safe)

        single = add(single,
                     self.conditioned_transition(
                         single,
                         cond,
                         chunk_size=chunk_size,
                         ) * inter_chain_mask[..., None],
                     inplace_safe,
                    )

        return single


class RigidDenoiserStack(nn.Module):
    """
    Implements Algorithm 16.
    """

    def __init__(
        self,
        c_r,
        c_cond,
        c_hidden_tri_att,
        c_hidden_tri_mul,
        no_blocks,
        no_heads,
        pair_transition_n,
        dropout_rate,
        fuse_projection_weights,
        blocks_per_ckpt,
        tune_chunk_size: bool = False,
        inf=1e9,
        **kwargs,
    ):
        """
        Args:
            c_t:
                Template embedding channel dimension
            c_hidden_tri_att:
                Per-head hidden dimension for triangular attention
            c_hidden_tri_att:
                Hidden dimension for triangular multiplication
            no_blocks:
                Number of blocks in the stack
            pair_transition_n:
                Scale of pair transition (Alg. 15) hidden dimension
            dropout_rate:
                Dropout rate used throughout the stack
            blocks_per_ckpt:
                Number of blocks per activation checkpoint. None disables
                activation checkpointing
        """
        super(RigidDenoiserStack, self).__init__()

        self.blocks_per_ckpt = blocks_per_ckpt

        self.blocks = nn.ModuleList()
        for _ in range(no_blocks):
            block = RigidDenoiserStackBlock(
                c_r=c_r,
                c_cond=c_cond,
                c_hidden_tri_att=c_hidden_tri_att,
                c_hidden_tri_mul=c_hidden_tri_mul,
                no_heads=no_heads,
                pair_transition_n=pair_transition_n,
                dropout_rate=dropout_rate,
                fuse_projection_weights=fuse_projection_weights,
                inf=inf,
            )
            self.blocks.append(block)

        self.layer_norm = LayerNorm(c_r)

        self.tune_chunk_size = tune_chunk_size
        self.chunk_size_tuner = None
        if tune_chunk_size:
            self.chunk_size_tuner = ChunkSizeTuner()

    def forward(
        self,
        t: torch.tensor,
        cond: torch.tensor,
        padding_mask: torch.tensor,
        inter_chain_mask: torch.tensor,
        chunk_size: int,
        use_deepspeed_evo_attention: bool = False,
        use_lma: bool = False,
        inplace_safe: bool = False,
    ):
        """
        Args:
            t:
                [*, N_templ, N_res, N_res, C_t] template embedding
            mask:
                [*, N_templ, N_res, N_res] mask
        Returns:
            [*, N_templ, N_res, N_res, C_t] template embedding update
        """

        blocks = [
            partial(
                b,
                cond=cond,
                padding_mask=padding_mask,
                inter_chain_mask=inter_chain_mask,
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
            )
            for b in self.blocks
        ]

        if chunk_size is not None and self.chunk_size_tuner is not None:
            assert (not self.training)
            tuned_chunk_size = self.chunk_size_tuner.tune_chunk_size(
                representative_fn=blocks[0],
                args=(t.clone(),),
                min_chunk_size=chunk_size,
            )
            blocks = [
                partial(b,
                        chunk_size=tuned_chunk_size,
                        _attn_chunk_size=max(chunk_size, tuned_chunk_size // 4),
                        ) for b in blocks
            ]

        t, = checkpoint_blocks(
            blocks=blocks,
            args=(t,),
            blocks_per_ckpt=self.blocks_per_ckpt if self.training else None,
        )

        t = self.layer_norm(t)

        return t

class TemplatePairEmbedderMultimer(nn.Module):
    def __init__(self,
        c_in: int,
        c_out: int,
        c_dgram: int,
        c_aatype: int,
        c_esm: int,
    ):
        super(TemplatePairEmbedderMultimer, self).__init__()

        self.dgram_linear = Linear(c_dgram, c_out, init='relu')
        self.aatype_linear_1 = Linear(c_aatype, c_out, init='relu')
        self.aatype_linear_2 = Linear(c_aatype, c_out, init='relu')
        self.query_embedding_layer_norm = LayerNorm(c_in)
        self.query_embedding_linear = Linear(c_in, c_out, init='relu')
        
        self.pseudo_beta_mask_linear = Linear(1, c_out, init='relu')
        self.x_linear = Linear(1, c_out, init='relu')
        self.y_linear = Linear(1, c_out, init='relu')
        self.z_linear = Linear(1, c_out, init='relu')
        self.backbone_mask_linear = Linear(1, c_out, init='relu')
        self.esm_embedding_linear_1 = Linear(c_esm, c_out, init='relu')
        self.esm_embedding_linear_2 = Linear(c_esm, c_out, init='relu')

    def forward(self,
        template_dgram: torch.Tensor,
        aatype_one_hot: torch.Tensor,
        query_embedding: torch.Tensor,
        pseudo_beta_mask: torch.Tensor,
        backbone_mask: torch.Tensor,
        unit_vector: geometry.Vec3Array,
        esm_embedding: torch.Tensor,
        inplace_safe: bool = False,
    ) -> torch.Tensor:
        act = 0.

        pseudo_beta_mask_2d = (
            pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]
        )
        template_dgram *= pseudo_beta_mask_2d[..., None]
        act = add(act, self.dgram_linear(template_dgram), inplace_safe)
        act = add(act, self.pseudo_beta_mask_linear(pseudo_beta_mask_2d[..., None]), inplace_safe)
       
        aatype_one_hot = aatype_one_hot.to(template_dgram.dtype)
        act = add(act, self.aatype_linear_1(aatype_one_hot[..., None, :, :]), inplace_safe)
        act = add(act, self.aatype_linear_2(aatype_one_hot[..., None, :]), inplace_safe)

        backbone_mask_2d = (
            backbone_mask[..., None] * backbone_mask[..., None, :]
        )
        x, y, z = [(coord * backbone_mask_2d).to(dtype=query_embedding.dtype) for coord in unit_vector]
        act = add(act, self.x_linear(x[..., None]), inplace_safe)
        act = add(act, self.y_linear(y[..., None]), inplace_safe)
        act = add(act, self.z_linear(z[..., None]), inplace_safe)
       
        act = add(act, self.backbone_mask_linear(backbone_mask_2d[..., None].to(dtype=query_embedding.dtype)), inplace_safe)

        act = add(act, self.esm_embedding_linear_1(esm_embedding[..., None, :, :]), inplace_safe)
        act = add(act, self.esm_embedding_linear_2(esm_embedding[..., None, :]), inplace_safe)

        query_embedding = self.query_embedding_layer_norm(query_embedding)
        act = add(act, self.query_embedding_linear(query_embedding), inplace_safe)

        return act

class RigidDenoiser(nn.Module):
    def __init__(self, config):
        super(RigidDenoiser, self).__init__()
        
        self.config = config
        self.template_pair_embedder = TemplatePairEmbedderMultimer(
            **config["template_pair_embedder"],
        )
        self.template_pair_stack = TemplatePairStack(
            **config["template_pair_stack"],
        )
        self.pair_conditioning = PairConditioning(
            **config["pair_conditioning"],
        )
        self.pair_conditioning_stack = TemplatePairStack(
            **config["template_pair_stack"],
        )
        self.rigid_denoiser_stack = RigidDenoiserStack(
            **config["rigid_denoiser_stack"],
        )

        self.linear_tp = Linear(config.c_t, config.c_r)

        self.linear_cond = Linear(config.c_t, config.c_t)

        self.linear_final = Linear(config.c_r, config.c_z, init='final')
    
    def forward(self, 
        batch, 
        z, 
        padding_mask, 
        templ_dim,
        chunk_size,
        inter_chain_mask,
        _mask_trans=True,
        use_deepspeed_evo_attention=False,
        use_lma=False,
        inplace_safe=False
    ):
        full_pair_update = 0.0
        
        n_templ = batch["template_aatype"].shape[templ_dim]
        esm_embedding = batch.pop("esm_embedding")
        for i in range(n_templ):
            idx = batch["template_aatype"].new_tensor(i)
            single_template_feats = tensor_tree_map(
                lambda t: torch.index_select(t, templ_dim, idx),
                batch,
            )
            times = single_template_feats['times']

            template_positions, pseudo_beta_mask = pseudo_beta_fn(
                single_template_feats["template_aatype"],
                single_template_feats["template_all_atom_positions"],
                single_template_feats["template_all_atom_mask"])

            template_dgram = dgram_from_positions(
                template_positions,
                inf=self.config.inf,
                **self.config.distogram,
            )

            aatype_one_hot = torch.nn.functional.one_hot(
                single_template_feats["template_aatype"], 22,
            )
            
            raw_atom_pos = single_template_feats["template_all_atom_positions"]

            # Vec3Arrays are required to be float32
            atom_pos = geometry.Vec3Array.from_array(raw_atom_pos.to(dtype=torch.float32))

            rigid, backbone_mask = all_atom_multimer.make_backbone_affine(
                atom_pos,
                single_template_feats["template_all_atom_mask"],
                single_template_feats["template_aatype"],
            )
            points = rigid.translation
            rigid_vec = rigid[..., None].inverse().apply_to_point(points)
            unit_vector = rigid_vec.normalized()

            pair_act = self.template_pair_embedder(
                template_dgram,
                aatype_one_hot,
                z,
                pseudo_beta_mask,
                backbone_mask,
                unit_vector,
                esm_embedding,
                inplace_safe,
            )
            tp, cond = pair_act.chunk(2, dim=-1)

            # [*, S_t, N, N, C_z]
            tp = self.template_pair_stack(
                tp,
                padding_mask.unsqueeze(-3).to(dtype=z.dtype), 
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            # [*, N, N, C_z]
            tp = torch.nn.functional.relu(tp)
            tp = self.linear_tp(tp)

            cond = self.pair_conditioning(
                times,
                cond,
                inplace_safe=inplace_safe,
            )
            cond = self.pair_conditioning_stack(
                cond,
                padding_mask.unsqueeze(-3).to(dtype=z.dtype),
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
                _mask_trans=_mask_trans,
            )
            # [*, N, N, C_z]
            cond = torch.nn.functional.relu(cond)
            cond = self.linear_cond(cond)
            
            tp = self.rigid_denoiser_stack(
                tp,
                cond,
                padding_mask=padding_mask.unsqueeze(-3).to(dtype=z.dtype),
                inter_chain_mask=inter_chain_mask.unsqueeze(-3).to(dtype=z.dtype),
                chunk_size=chunk_size,
                use_deepspeed_evo_attention=use_deepspeed_evo_attention,
                use_lma=use_lma,
                inplace_safe=inplace_safe,
            )

            del cond
        
            tp = tp.squeeze(templ_dim)
            
            full_pair_update = add(full_pair_update, tp, inplace_safe)
            
            del tp
        
        full_pair_update = full_pair_update / n_templ
        
        full_pair_update = self.linear_final(full_pair_update)

        return full_pair_update
