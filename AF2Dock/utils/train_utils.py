# Copyright 2021 AlQuraishi Laboratory
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

import numpy as np
from functools import partial
import numpy as np
from openfold.utils import import_weights
from openfold.utils.import_weights import Param, ParamType

def generate_translation_dict(model):
    #######################
    # Some templates
    #######################
    LinearWeight = lambda l: (Param(l, param_type=ParamType.LinearWeight))
    LinearBias = lambda l: (Param(l))
    LinearWeightMHA = lambda l: (Param(l, param_type=ParamType.LinearWeightMHA))
    LinearBiasMHA = lambda b: (Param(b, param_type=ParamType.LinearBiasMHA))
    LinearWeightOPM = lambda l: (Param(l, param_type=ParamType.LinearWeightOPM))
    LinearWeightSwap = lambda l: (Param(l, param_type=ParamType.LinearWeight, swap=True))
    LinearBiasSwap = lambda l: (Param(l, swap=True))

    LinearParams = lambda l: {
        "weights": LinearWeight(l.weight),
        "bias": LinearBias(l.bias),
    }

    LinearParamsMHA = lambda l: {
        "weights": LinearWeightMHA(l.weight),
        "bias": LinearBiasMHA(l.bias),
    }

    LinearParamsSwap = lambda l: {
        "weights": LinearWeightSwap(l.weight),
        "bias": LinearBiasSwap(l.bias),
    }

    LayerNormParams = lambda l: {
        "scale": Param(l.weight),
        "offset": Param(l.bias),
    }

    AttentionParams = lambda att: {
        "query_w": LinearWeightMHA(att.linear_q.weight),
        "key_w": LinearWeightMHA(att.linear_k.weight),
        "value_w": LinearWeightMHA(att.linear_v.weight),
        "output_w": Param(
            att.linear_o.weight,
            param_type=ParamType.LinearMHAOutputWeight,
        ),
        "output_b": LinearBias(att.linear_o.bias),
    }

    AttentionGatedParams = lambda att: dict(
        **AttentionParams(att),
        **{
            "gating_w": LinearWeightMHA(att.linear_g.weight),
            "gating_b": LinearBiasMHA(att.linear_g.bias),
        },
    )

    TriAttParams = lambda tri_att: {
        "query_norm": LayerNormParams(tri_att.layer_norm),
        "feat_2d_weights": LinearWeight(tri_att.linear.weight),
        "attention": AttentionGatedParams(tri_att.mha),
    }

    def TriMulOutParams(tri_mul, outgoing=True):
        lin_param_type = LinearParams if outgoing else LinearParamsSwap
        d = {
            "left_norm_input": LayerNormParams(tri_mul.layer_norm_in),
            "projection": lin_param_type(tri_mul.linear_ab_p),
            "gate": lin_param_type(tri_mul.linear_ab_g),
            "center_norm": LayerNormParams(tri_mul.layer_norm_out),
        }

        d.update({
            "output_projection": LinearParams(tri_mul.linear_z),
            "gating_linear": LinearParams(tri_mul.linear_g),
        })

        return d

    TriMulInParams = partial(TriMulOutParams, outgoing=False)

    PairTransitionParams = lambda pt: {
        "input_layer_norm": LayerNormParams(pt.layer_norm),
        "transition1": LinearParams(pt.linear_1),
        "transition2": LinearParams(pt.linear_2),
    }

    MSAAttParams = lambda matt: {
        "query_norm": LayerNormParams(matt.layer_norm_m),
        "attention": AttentionGatedParams(matt.mha),
    }

    MSAColAttParams = lambda matt: {
        "query_norm": LayerNormParams(matt._msa_att.layer_norm_m),
        "attention": AttentionGatedParams(matt._msa_att.mha),
    }

    MSAAttPairBiasParams = lambda matt: dict(
        **MSAAttParams(matt),
        **{
            "feat_2d_norm": LayerNormParams(matt.layer_norm_z),
            "feat_2d_weights": LinearWeight(matt.linear_z.weight),
        },
    )

    PointProjectionParams = lambda pp: {
        "point_projection": LinearParamsMHA(
            pp.linear,
        ),
    }

    IPAParamsMultimer = lambda ipa: {
        "q_scalar_projection": {
            "weights": LinearWeightMHA(
                ipa.linear_q.weight,
            ),
        },
        "k_scalar_projection": {
            "weights": LinearWeightMHA(
                ipa.linear_k.weight,
            ),
        },
        "v_scalar_projection": {
            "weights": LinearWeightMHA(
                ipa.linear_v.weight,
            ),
        },
        "q_point_projection": PointProjectionParams(
            ipa.linear_q_points
        ),
        "k_point_projection": PointProjectionParams(
            ipa.linear_k_points
        ),
        "v_point_projection": PointProjectionParams(
            ipa.linear_v_points
        ),
        "trainable_point_weights": Param(
            param=ipa.head_weights, param_type=ParamType.Other
        ),
        "attention_2d": LinearParams(ipa.linear_b),
        "output_projection": LinearParams(ipa.linear_out),
    }

    MSATransitionParams = lambda m: {
        "input_layer_norm": LayerNormParams(m.layer_norm),
        "transition1": LinearParams(m.linear_1),
        "transition2": LinearParams(m.linear_2),
    }

    OuterProductMeanParams = lambda o: {
        "layer_norm_input": LayerNormParams(o.layer_norm),
        "left_projection": LinearParams(o.linear_1),
        "right_projection": LinearParams(o.linear_2),
        "output_w": LinearWeightOPM(o.linear_out.weight),
        "output_b": LinearBias(o.linear_out.bias),
    }

    def EvoformerBlockParams(b):
        col_att_name = "msa_column_attention"
        msa_col_att_params = MSAColAttParams(b.msa_att_col)

        d = {
            "msa_row_attention_with_pair_bias": MSAAttPairBiasParams(
                b.msa_att_row
            ),
            col_att_name: msa_col_att_params,
            "msa_transition": MSATransitionParams(b.msa_transition),
            "outer_product_mean": 
                OuterProductMeanParams(b.outer_product_mean),
            "triangle_multiplication_outgoing": 
                TriMulOutParams(b.pair_stack.tri_mul_out),
            "triangle_multiplication_incoming": 
                TriMulInParams(b.pair_stack.tri_mul_in),
            "triangle_attention_starting_node": 
                TriAttParams(b.pair_stack.tri_att_start),
            "triangle_attention_ending_node": 
                TriAttParams(b.pair_stack.tri_att_end),
            "pair_transition": 
                PairTransitionParams(b.pair_stack.pair_transition),
        }

        return d

    def FoldIterationParams(sm):
        d = {
            "invariant_point_attention": 
                IPAParamsMultimer(sm.ipa),
            "attention_layer_norm": LayerNormParams(sm.layer_norm_ipa),
            "transition": LinearParams(sm.transition.layers[0].linear_1),
            "transition_1": LinearParams(sm.transition.layers[0].linear_2),
            "transition_2": LinearParams(sm.transition.layers[0].linear_3),
            "transition_layer_norm": LayerNormParams(sm.transition.layer_norm),
            "affine_update": LinearParams(sm.bb_update.linear),
            "rigid_sidechain": {
                "input_projection": LinearParams(sm.angle_resnet.linear_in),
                "input_projection_1": 
                    LinearParams(sm.angle_resnet.linear_initial),
                "resblock1": LinearParams(sm.angle_resnet.layers[0].linear_1),
                "resblock2": LinearParams(sm.angle_resnet.layers[0].linear_2),
                "resblock1_1": 
                    LinearParams(sm.angle_resnet.layers[1].linear_1),
                "resblock2_1": 
                    LinearParams(sm.angle_resnet.layers[1].linear_2),
                "unnormalized_angles": 
                    LinearParams(sm.angle_resnet.linear_out),
            },
        }

        d.pop("affine_update")
        d["quat_rigid"] = {
            "rigid": LinearParams(
                sm.bb_update.linear
            )
        }

        return d

    ############################
    # translations dict overflow
    ############################
    evo_blocks = model.evoformer.blocks
    evo_blocks_params = import_weights.stacked([EvoformerBlockParams(b) for b in evo_blocks])

    translations = {
        "evoformer": {
            "preprocess_1d": LinearParams(model.input_embedder.linear_tf_m),
            "preprocess_msa": LinearParams(model.input_embedder.linear_msa_m),
            "left_single": LinearParams(model.input_embedder.linear_tf_z_i),
            "right_single": LinearParams(model.input_embedder.linear_tf_z_j),
            "~_relative_encoding": {
                "position_activations": LinearParams(
                    model.input_embedder.linear_relpos
                ),
            },
            "evoformer_iteration": evo_blocks_params,
            "single_activations": LinearParams(model.evoformer.linear),
        },
        "structure_module": {
            "single_layer_norm": LayerNormParams(
                model.structure_module.layer_norm_s
            ),
            "initial_projection": LinearParams(
                model.structure_module.linear_in
            ),
            "pair_layer_norm": LayerNormParams(
                model.structure_module.layer_norm_z
            ),
            "fold_iteration": FoldIterationParams(model.structure_module),
        },
        "predicted_lddt_head": {
            "input_layer_norm": LayerNormParams(
                model.aux_heads.plddt.layer_norm
            ),
            "act_0": LinearParams(model.aux_heads.plddt.linear_1),
            "act_1": LinearParams(model.aux_heads.plddt.linear_2),
            "logits": LinearParams(model.aux_heads.plddt.linear_3),
        },
        "distogram_head": {
            "half_logits": LinearParams(model.aux_heads.distogram.linear),
        },
        "experimentally_resolved_head": {
            "logits": LinearParams(
                model.aux_heads.experimentally_resolved.linear
            ),
        },
    }

    translations["predicted_aligned_error_head"] = {
        "logits": LinearParams(model.aux_heads.tm.linear)
    }

    return translations

def get_flattened_translations_dict(model):
    translations = generate_translation_dict(model)
    
    # Flatten keys and insert missing key prefixes
    flat = import_weights.process_translation_dict(translations)
    
    return flat

def import_jax_weights_(model, npz_path):
    data = np.load(npz_path)
    flat = get_flattened_translations_dict(model)

    # Sanity check
    keys = list(data.keys())
    flat_keys = list(flat.keys())
    incorrect = [k for k in flat_keys if k not in keys]
    missing = [k for k in keys if k not in flat_keys]
    # print(f"Incorrect: {incorrect}")
    # print(f"Missing: {missing}")

    assert len(incorrect) == 0
    # assert(sorted(list(flat.keys())) == sorted(list(data.keys())))

    # Set weights
    import_weights.assign(flat, data)
