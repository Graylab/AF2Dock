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

import torch
import torch.nn as nn

from openfold.utils.feats import (
    atom14_to_atom37,
)
from openfold.model.embedders import (
    InputEmbedderMultimer,
    TemplateEmbedderMultimer,
)
from openfold.model.evoformer import EvoformerStack
from openfold.model.heads import AuxiliaryHeads
from openfold.model.structure_module import StructureModule
from openfold.utils.feats import (
    atom14_to_atom37,
)
from openfold.utils.tensor_utils import (
    add,
    tensor_tree_map,
)

class AF2Dock(nn.Module):

    def __init__(self, config):
        """
        Args:
            config:
                A dict-like config object (like the one in config.py)
        """
        super().__init__()

        self.globals = config.globals
        self.config = config.model

        # Main trunk + structure module
        self.input_embedder = InputEmbedderMultimer(
            **self.config["input_embedder"]
        )

        self.template_embedder = TemplateEmbedderMultimer(
            self.template_config,
        )

        self.evoformer = EvoformerStack(
            **self.config["evoformer_stack"],
        )

        self.structure_module = StructureModule(
            is_multimer=self.globals.is_multimer,
            **self.config["structure_module"],
        )
        self.aux_heads = AuxiliaryHeads(
            self.config["heads"],
        )

    def embed_templates(self, batch, feats, z, pair_mask, templ_dim, inplace_safe):
        asym_id = feats["asym_id"]
        interchain_mask_2d = (
            asym_id[..., None] != asym_id[..., None, :]
        )
        template_pair_embed = self.template_embedder(
            batch,
            z,
            pair_mask.to(dtype=z.dtype),
            templ_dim,
            chunk_size=self.globals.chunk_size,
            interchain_mask_2d=interchain_mask_2d,
            use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
            use_lma=self.globals.use_lma,
            inplace_safe=inplace_safe,
            _mask_trans=self.config._mask_trans
        )

        return template_pair_embed

    def iteration(self, feats):
        # Primary output dictionary
        outputs = {}

        # This needs to be done manually for DeepSpeed's sake
        dtype = next(self.parameters()).dtype
        for k in feats:
            if feats[k].dtype == torch.float32:
                feats[k] = feats[k].to(dtype=dtype)

        # Grab some data about the input
        batch_dims = feats["target_feat"].shape[:-2]
        no_batch_dims = len(batch_dims)
        n_seq = feats["msa_feat"].shape[-3]

        # Controls whether the model uses in-place operations throughout
        # The dual condition accounts for activation checkpoints
        inplace_safe = not (self.training or torch.is_grad_enabled())

        # Prep some features
        seq_mask = feats["seq_mask"]
        pair_mask = seq_mask[..., None] * seq_mask[..., None, :]
        msa_mask = feats["msa_mask"]

        # Initialize the MSA and pair representations
        # m: [*, S_c, N, C_m]
        # z: [*, N, N, C_z]
        m, z = self.input_embedder(feats)

        # Embed the templates + merge with MSA/pair embeddings
        template_feats = {
            k: v for k, v in feats.items() if k.startswith("template_")
        }
        template_feats["esm_embedding"] = feats["esm_embedding"]
        template_feats["times"] = feats["t"]

        template_pair_embed = self.embed_templates(
            template_feats,
            feats,
            z,
            pair_mask.to(dtype=z.dtype),
            no_batch_dims,
            inplace_safe=inplace_safe,
        )

        # [*, N, N, C_z]
        z = add(z,
                template_pair_embed,
                inplace_safe,
                )

        # Run MSA + pair embeddings through the trunk of the network
        # m: [*, S, N, C_m]
        # z: [*, N, N, C_z]
        # s: [*, N, C_s]          
        if self.globals.offload_inference:
            input_tensors = [m, z]
            del m, z
            m, z, s = self.evoformer._forward_offload(
                input_tensors,
                msa_mask=msa_mask.to(dtype=input_tensors[0].dtype),
                pair_mask=pair_mask.to(dtype=input_tensors[1].dtype),
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_lma=self.globals.use_lma,
                _mask_trans=self.config._mask_trans,
            )

            del input_tensors
        else:
            m, z, s = self.evoformer(
                m,
                z,
                msa_mask=msa_mask.to(dtype=m.dtype),
                pair_mask=pair_mask.to(dtype=z.dtype),
                chunk_size=self.globals.chunk_size,
                use_deepspeed_evo_attention=self.globals.use_deepspeed_evo_attention,
                use_lma=self.globals.use_lma,
                use_flash=self.globals.use_flash,
                inplace_safe=inplace_safe,
                _mask_trans=self.config._mask_trans,
            )

        outputs["msa"] = m[..., :n_seq, :, :]
        outputs["pair"] = z
        outputs["single"] = s

        del z

        # Predict 3D structure
        outputs["sm"] = self.structure_module(
            outputs,
            feats["aatype"],
            mask=feats["seq_mask"].to(dtype=s.dtype),
            inplace_safe=inplace_safe,
            _offload_inference=self.globals.offload_inference,
        )
        outputs["final_atom_positions"] = atom14_to_atom37(
            outputs["sm"]["positions"][-1], feats
        )
        outputs["final_atom_mask"] = feats["atom37_atom_exists"]
        outputs["final_affine_tensor"] = outputs["sm"]["frames"][-1]

        return outputs

    def forward(self, batch):
        """
        Args:
            batch:
                Dictionary of arguments outlined in Algorithm 2. Keys must
                include the official names of the features in the
                supplement subsection 1.2.9.

                The final dimension of each input must have length equal to
                the number of recycling iterations.

                Features (without the recycling dimension):

                    "aatype" ([*, N_res]):
                        Contrary to the supplement, this tensor of residue
                        indices is not one-hot.
                    "target_feat" ([*, N_res, C_tf])
                        One-hot encoding of the target sequence. C_tf is
                        config.model.input_embedder.tf_dim.
                    "residue_index" ([*, N_res])
                        Tensor whose final dimension consists of
                        consecutive indices from 0 to N_res.
                    "msa_feat" ([*, N_seq, N_res, C_msa])
                        MSA features, constructed as in the supplement.
                        C_msa is config.model.input_embedder.msa_dim.
                    "seq_mask" ([*, N_res])
                        1-D sequence mask
                    "msa_mask" ([*, N_seq, N_res])
                        MSA mask
                    "pair_mask" ([*, N_res, N_res])
                        2-D pair mask
                    "extra_msa_mask" ([*, N_extra, N_res])
                        Extra MSA mask
                    "template_mask" ([*, N_templ])
                        Template mask (on the level of templates, not
                        residues)
                    "template_aatype" ([*, N_templ, N_res])
                        Tensor of template residue indices (indices greater
                        than 19 are clamped to 20 (Unknown))
                    "template_all_atom_positions"
                        ([*, N_templ, N_res, 37, 3])
                        Template atom coordinates in atom37 format
                    "template_all_atom_mask" ([*, N_templ, N_res, 37])
                        Template atom coordinate mask
                    "template_pseudo_beta" ([*, N_templ, N_res, 3])
                        Positions of template carbon "pseudo-beta" atoms
                        (i.e. C_beta for all residues but glycine, for
                        for which C_alpha is used instead)
                    "template_pseudo_beta_mask" ([*, N_templ, N_res])
                        Pseudo-beta mask
        """
        is_grad_enabled = torch.is_grad_enabled()

        fetch_cur_batch = lambda t: t[..., 0]
        feats = tensor_tree_map(fetch_cur_batch, batch)

        with torch.set_grad_enabled(is_grad_enabled):
            # Sidestep AMP bug (PyTorch issue #65766)
            if torch.is_autocast_enabled():
                torch.clear_autocast_cache()

            # Run the next iteration of the model
            outputs = self.iteration(feats)

        outputs["num_recycles"] = torch.tensor(1, device=feats["aatype"].device)

        if "asym_id" in batch:
            outputs["asym_id"] = feats["asym_id"]

        # Run auxiliary heads
        outputs.update(self.aux_heads(outputs))

        return outputs
