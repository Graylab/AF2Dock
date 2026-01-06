import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import random
import json
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import torch
torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if (
    torch_major_version > 1 or
    (torch_major_version == 1 and torch_minor_version >= 12)
):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)

from openfold.np import residue_constants
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.data import feature_pipeline

from esm.models.esmc import ESMC

from AF2Dock.config import model_config
from AF2Dock.model.model import AF2Dock
from AF2Dock.utils import data_utils, inference_utils

def main(args):
    # Create the output directory
    config = model_config(
        long_sequence_inference=args.long_sequence_inference,
        use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
        use_cuequivariance_attention=args.use_cuequivariance_attention,
        use_cuequivariance_multiplicative_update=args.use_cuequivariance_multiplicative_update,
        precision=args.precision,
        trt_mode=args.trt_mode,
        trt_engine_dir=args.trt_engine_dir,
        trt_num_profiles=args.trt_num_profiles,
        trt_optimization_level=args.trt_optimization_level,
        trt_max_sequence_len=args.trt_max_sequence_len,
        )

    if args.no_esm:
        config.model.pair_denoiser.use_esm = False

    if args.experiment_config_json:
        with open(args.experiment_config_json, 'r') as f:
            custom_config_dict = json.load(f)
        config.update_from_flattened_dict(custom_config_dict)
    
    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2 ** 32)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
    if not output_dir_base.exists():
        output_dir_base.mkdir(exist_ok=True)
    
    # Loading the model
    model = AF2Dock(config)
    model = inference_utils.prepare_model(model, args.checkpoint_path, args.model_device)
    
    if config.model.pair_denoiser.use_esm:
        esm_client = ESMC.from_pretrained("esmc_600m")
    else:
        esm_client = None
    
    ca_idx = residue_constants.atom_order["CA"]
    
    if args.input_csv is not None:
        predict_targets = pd.read_csv(args.input_csv)
    elif args.rec_struc_path is not None and args.lig_struc_path is not None:
        if args.data_id is None:
            data_id = f"{args.rec_struc_path.stem}_{args.lig_struc_path.stem}"
        else:
            data_id = args.data_id
        predict_targets = {
            'id': [data_id],
            'rec': [str(args.rec_struc_path)],
            'lig': [str(args.lig_struc_path)],
        }
        if args.rec_seq_path is not None:
            predict_targets['rec_seq'] = [str(args.rec_seq_path)]
        if args.lig_seq_path is not None:
            predict_targets['lig_seq'] = [str(args.lig_seq_path)]
        predict_targets = pd.DataFrame(predict_targets)
    else:
        raise ValueError("Either input_csv or both rec_struc_path and lig_struc_path must be provided.")

    feat_pipeline = feature_pipeline.FeaturePipeline(config.data)
    
    for data_idx in tqdm(range(args.data_starting_index, len(predict_targets))):
        target_row = predict_targets.iloc[data_idx]
        data, ini_struct_feats_dict, original_asym_id, original_residue_index = inference_utils.load_data(target_row, 
                                                                                                           config, 
                                                                                                           esm_client=esm_client, 
                                                                                                           device=args.model_device,
                                                                                                           plddt_cutoff=args.input_plddt_cutoff,
                                                                                                           min_res_num=args.min_res_num,
                                                                                                           min_res_ratio=args.min_res_ratio)

        data_id = target_row['id']
        
        out_dir_data = output_dir_base / f'{data_idx}_{data_id}'
        if not out_dir_data.exists():
            out_dir_data.mkdir(exist_ok=True)

        metrics_file = out_dir_data / f'{data_id}_s{args.sample_starting_index}_{args.sample_starting_index + args.num_samples - 1}_iptm.csv'
        if metrics_file.exists() and not args.overwrite_existing:
            metrics = pd.read_csv(metrics_file).to_dict(orient='list')
        else:
            metrics = {'sample_idx': [], 'iptm': []}

        for sample_idx in tqdm(range(args.sample_starting_index, args.sample_starting_index + args.num_samples)):
            if not args.overwrite_existing:
                if sample_idx in metrics['sample_idx']:
                    logger.info(f"Results exist for {data_id} sample {sample_idx}, skipping")
                    continue
            
            # process feature each time for msa random subsampling and masking
            batch = feat_pipeline.process_features(
                data, mode='predict', is_multimer=True
            )
            # Zero out extra MSA features for when using MSA inputs
            batch['extra_msa'] = torch.zeros_like(batch['extra_msa'])
            batch['extra_msa_mask'] = torch.zeros_like(batch['extra_msa_mask'])
            batch['extra_deletion_matrix'] = torch.zeros_like(batch['extra_deletion_matrix'])
            
            is_homomer = 2 in batch['sym_id']
            batch = tensor_tree_map(lambda x: x.unsqueeze(0).to(args.model_device), batch)
            
            curr_atom_pos = []
            atom_masks = []
            for part in ['rec', 'lig']:
                part_0_all_atom_mask = ini_struct_feats_dict[part]["ini_all_atom_mask"][None, ...]
                part_0_all_atom_positions = ini_struct_feats_dict[part]["ini_all_atom_positions"][None, ...]
                part_com = np.mean(part_0_all_atom_positions[..., ca_idx, :], axis=-2)
                part_0_all_atom_positions = part_0_all_atom_positions - part_com[:, None, None, :]
                if part == 'lig':
                    tr_0, rot_0 = data_utils.get_rigid_body_noise_at_0(tr_sigma=config.data.rigid_body.tr_sigma,
                                                                       num_struct_batch=1,
                                                                       rot_prior='uniform')
                    part_0_all_atom_positions = data_utils.apply_rigid_body_transform_atom37(part_0_all_atom_positions,
                                                                                             part_0_all_atom_mask,
                                                                                             ca_idx,
                                                                                             tr_0,
                                                                                             rot_0)
                curr_atom_pos.append(torch.tensor(part_0_all_atom_positions[0]))
                atom_masks.append(torch.tensor(part_0_all_atom_mask[0]))
            
            template_all_atom_mask = torch.cat(atom_masks, dim=-2)
            assert template_all_atom_mask.shape[-2] == batch['template_all_atom_mask'].shape[-3]
            batch['template_all_atom_mask'] = template_all_atom_mask[None, None, ...][..., None].clone().expand(
                *([-1] * (len(batch['template_all_atom_mask'].shape) - 1) + [batch['template_all_atom_mask'].size(dim=-1)])).to(
                batch['template_all_atom_mask'].dtype).to(batch['template_all_atom_mask'].device)
            
            total_steps =  args.num_steps + args.additional_refine_steps
            interpolation_steps = args.num_steps
            
            time_indexes = np.arange(0, total_steps)
            if args.merge_first_n_steps > 0:
                time_indexes = np.insert(time_indexes[args.merge_first_n_steps:], 0, 0)
            
            for tidx, time_index in enumerate(time_indexes):
                t = min(time_index / interpolation_steps, 1.0)
                if tidx == 0 and args.merge_first_n_steps > 0:
                    s = t + args.merge_first_n_steps / interpolation_steps
                else:
                    s = min(t + 1 / interpolation_steps, 1.0)
                
                batch['t'] = batch['t'].new_tensor(np.array([[[[t]]]]))
                
                template_all_atom_pos = torch.cat(curr_atom_pos, dim=-3)
                assert template_all_atom_pos.shape[-3] == batch['template_all_atom_positions'].shape[-4]
                batch['template_all_atom_positions'] = template_all_atom_pos[None, None, ...][..., None].clone().expand(
                    *([-1] * (len(batch['template_all_atom_positions'].shape) - 1) + [batch['template_all_atom_positions'].size(dim=-1)])).to(
                    batch['template_all_atom_positions'].dtype).to(batch['template_all_atom_positions'].device)
                
                if args.use_cuequivariance_attention:
                    # cuequivariance attention does not support batch dimension due to https://github.com/aqlaboratory/openfold/blob/be2ec1841f16c966c65ae0e7599ebbadc725757d/openfold/model/msa.py#L285
                    batch = tensor_tree_map(lambda x: x.squeeze(0), batch)
                
                out = model(batch)
                out = tensor_tree_map(lambda x: x.cpu(), out)
                
                if args.use_cuequivariance_attention:
                    out = tensor_tree_map(lambda x: x.unsqueeze(0), out)
                    batch = tensor_tree_map(lambda x: x.unsqueeze(0), batch)
                
                if tidx < len(time_indexes) - 1:
                    curr_atom_pos = inference_utils.update_pose(batch, out, atom_masks, curr_atom_pos, s, t, ca_idx, is_homomer)
                    
                    if args.save_intermediate_template or args.save_intermediate_pred or tidx == 0:
                        inference_utils.write_output(batch, out, out_dir_data, f'{data_id}_s{sample_idx}_t{time_index}', out_pred=args.save_intermediate_pred, 
                                                     out_conf=args.save_intermediate_conf, out_template=args.save_intermediate_template or tidx == 0,
                                                     original_residue_index=original_residue_index, original_asym_id=original_asym_id)

            inference_utils.write_output(batch, out, out_dir_data, f'{data_id}_s{sample_idx}', out_pred=True, out_conf=True, out_template=args.save_intermediate_template,
                                         original_residue_index=original_residue_index, original_asym_id=original_asym_id)
            
            metrics['sample_idx'].append(sample_idx)
            metrics['iptm'].append(out['iptm_score'].item())
            pd.DataFrame(metrics).sort_values('iptm', ascending=False).to_csv(metrics_file, index=False)

        batch = tensor_tree_map(lambda x: x.cpu(), batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", type=Path,
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--input_csv", type=Path, default=None,
        help="""Input csv file containing rec lig structures and sequences, 
        have priority over individually supplied input files.""",
    )
    parser.add_argument(
        "--data_id", type=str, default=None,
        help="""Data ID for output prefix"""
    )
    parser.add_argument(
        "--rec_struc_path", type=Path, default=None,
        help="""Path to the receptor structure file""",
    )
    parser.add_argument(
        "--lig_struc_path", type=Path, default=None,
        help="""Path to the ligand structure file""",
    )
    parser.add_argument(
        "--rec_seq_path", type=Path, default=None,
        help="""Path to the receptor sequence a3m file with resolved sequence 
        aligned to full sequence for each chain. If not supplied,
        the sequence will be extracted from the structure file""",
    )
    parser.add_argument(
        "--lig_seq_path", type=Path, default=None,
        help="""Path to the ligand sequence a3m file with resolved sequence
        aligned to full sequence for each chain. If not supplied,
        the sequence will be extracted from the structure file""",
    )
    parser.add_argument(
        "--num_samples", type=int, default=40,
        help="""Number of samples to generate per target"""
    )
    parser.add_argument(
        "--num_steps", type=int, default=10,
        help="""Number of steps for the denoising process"""
    )
    parser.add_argument(
        "--merge_first_n_steps", type=int, default=0,
        help="""Merge the first N time steps"""
    )
    parser.add_argument(
        "--additional_refine_steps", type=int, default=0,
        help="""Number of additional refine steps at t=1"""
    )
    parser.add_argument(
        "--model_device", type=str, default="cuda",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--checkpoint_path", type=Path, default=None,
        help="""Path to AF2Dock checkpoint. Can be either a DeepSpeed 
             checkpoint directory or a .pt file"""
    )
    parser.add_argument(
        "--save_intermediate_template", action="store_true", default=False,
        help="""Whether to save intermediate templates"""
    )
    parser.add_argument(
        "--save_intermediate_pred", action="store_true", default=False,
        help="""Whether to save intermediate predictions"""
    )
    parser.add_argument(
        "--save_intermediate_conf", action="store_true", default=False,
        help="""Whether to save intermediate confidence scores"""
    )
    parser.add_argument(
        "--data_starting_index", type=int, default=0,
        help="""Starting index for the test set. Used to skip the first N
             targets in the test set"""
    )
    parser.add_argument(
        "--sample_starting_index", type=int, default=0,
        help="""Starting index for samples. Used to skip the first N
             samples for each target"""
    )
    parser.add_argument(
        "--overwrite_existing", action="store_true", default=False,
        help="""Whether to overwrite existing results. If False,
             will skip the prediction if the results files already exists"""
    )
    parser.add_argument(
        "--input_plddt_cutoff", type=float, default=None,
        help="""Only use when using AF predicted structures as input. B factor of input structures will be treated as plddt
             and residues with plddt < cutoff will be masked off."""
    )
    parser.add_argument(
        "--min_res_num", type=int, default=0,
        help="""When filtering by plddt, minimum number of residues to keep. If lower, all residues will be kept."""
    )
    parser.add_argument(
        "--min_res_ratio", type=float, default=0.8,
        help="""When filtering by plddt, minimum ratio of residues to keep. If lower, all residues will be kept."""
    )
    parser.add_argument(
        "--data_random_seed", type=int, default=None
    )
    parser.add_argument(
        "--long_sequence_inference", action="store_true", default=False,
        help="""enable options to reduce memory usage at the cost of speed, helps longer sequences fit into GPU memory, see the openfold README for details"""
    )
    parser.add_argument(
        "--use_deepspeed_evoformer_attention", action="store_true", default=False, 
        help="Whether to use the DeepSpeed evoformer attention layer. Must have deepspeed installed in the environment.",
    )
    parser.add_argument(
        "--use_cuequivariance_attention", action="store_true", default=False,
        help="""Use cuEquivariance kernels for attention computation."""
    )
    parser.add_argument(
        "--use_cuequivariance_multiplicative_update", action="store_true", default=False,
        help="""Use cuEquivariance kernels for triangular multiplicative update computation."""
    )
    parser.add_argument(
        "--trt_mode", type=str, default=None,
        help="build = Build engine; run = Run engine; None = Disable TRT"
    )
    parser.add_argument(
        "--trt_engine_dir", type=str, default=None,
        help="Absolute path to directory containing .onnx and .plan files"
    )
    parser.add_argument(
        "--precision", type=str, default="tf32",
        help="tf32 | fp32 | fp16 | bf16"
    )
    parser.add_argument(
        "--trt_max_sequence_len", type=int, default=640,
        help="Maximum sequence length supported by TRT, default=640"
    )
    parser.add_argument(
        "--trt_num_profiles", type=int, default=1,
        help="1 = Single profile[50-800]; 2 = [50-200][200-800]; 4 = [50-100]; [100-200]; [200-400]; [400-800]"
    )
    parser.add_argument(
        "--trt_optimization_level", type=int, default=3,
        help="Allowed values: 0 to 5"
    )
    parser.add_argument(
        "--no-esm", action="store_true", default=False,
        help="""Do not use ESM inputs. Only use with model weights trained without ESM."""
    )
    parser.add_argument(
        "--experiment_config_json", default="", help="Path to a json file with custom config values to overwrite config setting",
    )
    args = parser.parse_args()

    main(args)
