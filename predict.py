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

from esm.models.esmc import ESMC

from AF2Dock.config import model_config
from AF2Dock.data.datamodule import AF2DockDataModule
from AF2Dock.model.model import AF2Dock
from AF2Dock.utils import data_utils, inference_utils

def main(args):
    # Create the output directory
    config = model_config(
        long_sequence_inference=args.long_sequence_inference,
        use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
        )

    if args.experiment_config_json:
        with open(args.experiment_config_json, 'r') as f:
            custom_config_dict = json.load(f)
        config.update_from_flattened_dict(custom_config_dict)
    
    config.data.data_module.num_workers = args.num_workers
    
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
    
    for data_idx in tqdm(range(args.data_starting_index, len(predict_targets))):
        target_row = predict_targets.iloc[data_idx]
        batch, ini_struct_feats_dict, original_asym_id, original_residue_index = inference_utils.load_data(target_row, 
                                                                                                           config, 
                                                                                                           esm_client, 
                                                                                                           args.model_device)
        
        data_id = target_row['id']
        is_homomer = 2 in batch['sym_id']
        batch = tensor_tree_map(lambda x: x.unsqueeze(0).to(args.model_device), batch)
        
        out_dir_data = output_dir_base / f'{data_idx}_{data_id}'
        if not out_dir_data.exists():
            out_dir_data.mkdir(exist_ok=True)

        metrics = {'sample_idx': [], 'iptm': []}
        
        for sample_idx in tqdm(range(args.sample_starting_index, args.sample_starting_index + args.num_samples)):
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
            batch['template_all_atom_mask'] = template_all_atom_mask[None, None, ...][..., None].clone().to(
                batch['template_all_atom_mask'].dtype).to(batch['template_all_atom_mask'].device)
            
            total_steps =  args.num_steps
            
            for time_idx in range(total_steps):
                t = time_idx / total_steps
                s = t + 1 / total_steps
                
                batch['t'] = batch['t'].new_tensor(np.array([[[[t]]]]))
                
                template_all_atom_pos = torch.cat(curr_atom_pos, dim=-3)
                assert template_all_atom_pos.shape[-3] == batch['template_all_atom_positions'].shape[-4]
                batch['template_all_atom_positions'] = template_all_atom_pos[None, None, ...][..., None].clone().to(
                    batch['template_all_atom_positions'].dtype).to(batch['template_all_atom_positions'].device)
                
                out = model(batch)
                out = tensor_tree_map(lambda x: x.cpu(), out)
                
                if time_idx < total_steps - 1:
                    curr_atom_pos = inference_utils.update_pose(batch, out, atom_masks, curr_atom_pos, s, t, ca_idx, is_homomer)
                    
                    if args.save_intermediate_template or args.save_intermediate_pred or time_idx == 0:
                        inference_utils.write_output(batch, out, out_dir_data, f'{data_id}_s{sample_idx}_t{time_idx}', out_pred=args.save_intermediate_pred, 
                                                     out_conf=args.save_intermediate_conf, out_template=args.save_intermediate_template or time_idx == 0,
                                                     residue_index=original_residue_index, asym_id=original_asym_id)

            inference_utils.write_output(batch, out, out_dir_data, f'{data_id}_s{sample_idx}', out_pred=True, out_conf=True, out_template=args.save_intermediate_template,
                                         residue_index=original_residue_index, asym_id=original_asym_id)
            
            metrics['sample_idx'].append(sample_idx)
            metrics['iptm'].append(out['iptm_score'].item())

        batch = tensor_tree_map(lambda x: x.cpu(), batch)
        metrics = pd.DataFrame(metrics)
        metrics = metrics.sort_values('iptm', ascending=False)
        metrics.to_csv(out_dir_data / f'{data_id}_s{args.sample_starting_index}_{args.sample_starting_index + args.num_samples - 1}_iptm.csv', index=False)

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
        "--data_random_seed", type=int, default=None
    )
    parser.add_argument(
        "--long_sequence_inference", action="store_true", default=False,
        help="""enable options to reduce memory usage at the cost of speed, helps longer sequences fit into GPU memory, see the README for details"""
    )
    parser.add_argument(
        "--experiment_config_json", default="", help="Path to a json file with custom config values to overwrite config setting",
    )
    parser.add_argument(
        "--use_deepspeed_evoformer_attention", action="store_true", default=False, 
        help="Whether to use the DeepSpeed evoformer attention layer. Must have deepspeed installed in the environment.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of workers for the dataloader.",
    )
    args = parser.parse_args()

    main(args)
