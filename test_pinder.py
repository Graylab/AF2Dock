import argparse
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import pickle
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

from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict
)

from openfold.np import protein, residue_constants
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils import multi_chain_permutation

from AF2Dock.config import model_config
from AF2Dock.data.datamodule import AF2DockDataModule
from AF2Dock.model.model import AF2Dock
from AF2Dock.utils import data_utils

def get_global_rigid_body_transform(denoised_atom_pos, curr_atom_pos, atom_masks, is_homomer):
    ca_idx = residue_constants.atom_order["CA"]
    if not is_homomer:
        global_r, global_x = multi_chain_permutation.get_optimal_transform(denoised_atom_pos[0][..., ca_idx, :],
                                                                           curr_atom_pos[0][..., ca_idx, :],
                                                                           atom_masks[0][..., ca_idx].to(torch.bool))
        swap = False
    else:
        r1, x1 = multi_chain_permutation.get_optimal_transform(denoised_atom_pos[0][..., ca_idx, :],
                                                               curr_atom_pos[0][..., ca_idx, :],
                                                               atom_masks[0][..., ca_idx].to(torch.bool))
        _, x1other = multi_chain_permutation.get_optimal_transform(denoised_atom_pos[1][..., ca_idx, :].to(r1.dtype) @ r1 + x1,
                                                               curr_atom_pos[1][..., ca_idx, :],
                                                               atom_masks[0][..., ca_idx].to(torch.bool))
        r2, x2 = multi_chain_permutation.get_optimal_transform(denoised_atom_pos[1][..., ca_idx, :],
                                                               curr_atom_pos[0][..., ca_idx, :],
                                                               atom_masks[0][..., ca_idx].to(torch.bool))
        _, x2other = multi_chain_permutation.get_optimal_transform(denoised_atom_pos[0][..., ca_idx, :].to(r2.dtype) @ r2 + x2,
                                                               curr_atom_pos[1][..., ca_idx, :],
                                                               atom_masks[0][..., ca_idx].to(torch.bool))
        if x1other.norm() < x2other.norm():
            global_r = r1
            global_x = x1
            swap = False
        else:
            global_r = r2
            global_x = x2
            swap = True
    
    return global_r, global_x, swap
    
def write_output(batch, out, outpath, outprefix, out_pred=True, out_conf=True, out_template=False):
    out_items_to_save = ['plddt', 'ptm_score', 'iptm_score',
                         'weighted_ptm_score', 'aligned_confidence_probs', 
                         'predicted_aligned_error', 'max_predicted_aligned_error']
    aatype = batch['aatype'][0][..., -1].clone().detach().cpu().numpy()
    residue_index = batch["residue_index"][0][..., -1].clone().detach().cpu().numpy() + 1
    chain_index = batch["asym_id"][0][..., -1].clone().detach().cpu().numpy() - 1
    
    if out_template:
        template_all_atom_pos_out = batch['template_all_atom_positions'].clone().detach().cpu().numpy()[0][0][..., -1]
        template_all_atom_mask_out = batch['template_all_atom_mask'].clone().detach().cpu().numpy()[0][0][..., -1]
        prot_template = protein.Protein(
            aatype=aatype,
            atom_positions=template_all_atom_pos_out,
            atom_mask=template_all_atom_mask_out,
            residue_index=residue_index,
            b_factors=np.zeros_like(template_all_atom_mask_out),
            chain_index=chain_index,
        )
        with open(outpath / (outprefix+'_template.pdb'), 'w') as fp:
            fp.write(protein.to_pdb(prot_template))
    
    if out_pred:
        out = tensor_tree_map(lambda x: np.array(x), out)
        all_atom_pos_out = out['final_atom_positions'][0]
        all_atom_mask_out = out['final_atom_mask'][0]
        b_factors_out = out['plddt'][0]
        b_factors_out = np.repeat(b_factors_out[..., None], residue_constants.atom_type_num, axis=-1)
        prot_pred = protein.Protein(
            aatype=aatype,
            atom_positions=all_atom_pos_out,
            atom_mask=all_atom_mask_out,
            residue_index=residue_index,
            b_factors=b_factors_out,
            chain_index=chain_index,
        )
        with open(outpath / (outprefix+'.pdb'), 'w') as fp:
            fp.write(protein.to_pdb(prot_pred))
        if out_conf:
            out = {k: v for k, v in out.items() if k in out_items_to_save}
            with open(outpath / (outprefix+'_out.pkl'), "wb") as fp:
                pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

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
    
    if args.pinder_test_type == 'holo':
        config.data.test.pinder_cate_prob = {"holo": 1.0, "apo": 0.0, "pred": 0.0,}
    elif args.pinder_test_type == 'apo':
        config.data.test.pinder_cate_prob = {"holo": 0.0, "apo": 1.0, "pred": 0.0,}
    elif args.pinder_test_type == 'predicted':
        config.data.test.pinder_cate_prob = {"holo": 0.0, "apo": 0.0, "pred": 1.0,}
    
    config.data.data_module.num_workers = args.num_workers
    
    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2 ** 32)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
    if not output_dir_base.exists():
        output_dir_base.mkdir()

    # Getting dataloader
    data_module = AF2DockDataModule(
        config=config.data, 
        training_mode=False,
        batch_seed=random_seed,
        cached_esm_embedding_folder=args.cached_esm_embedding_folder,
        test_split=args.pinder_test_split,
        test_type=args.pinder_test_type,
    )
    data_module.setup('test')
    dataloader = data_module.test_dataloader()
    
    # Loading the model
    model = AF2Dock(config)
    model = model.eval()
    checkpoint_path = args.checkpoint_path
    if checkpoint_path.is_dir():
        ckpt_path = checkpoint_path / f'{checkpoint_path.stem}.pt'
        if not ckpt_path.exists():
            convert_zero_checkpoint_to_fp32_state_dict(
                checkpoint_path,
                ckpt_path,
            )
        d = torch.load(ckpt_path)
        model.load_state_dict(d["ema"]["params"])
    else:
        ckpt_path = checkpoint_path
        d = torch.load(ckpt_path)

        if "ema" in d:
            d = d["ema"]["params"]
        model.load_state_dict(d)

    model = model.to(args.model_device)
    logger.info(
        f"Loaded parameters at {checkpoint_path}..."
    )
    ca_idx = residue_constants.atom_order["CA"]
    
    for data_idx, batch in tqdm(enumerate(dataloader)):
        gt_features = batch.pop("gt_features")
        data_id = dataloader.dataset.data_index.iloc[batch["batch_idx"].item()]['id']
        is_homomer = 2 in batch['sym_id']
        batch = tensor_tree_map(lambda x: x.to(args.model_device), batch)
        
        out_dir_data = output_dir_base / data_id
        if not out_dir_data.exists():
            out_dir_data.mkdir()

        for sample_idx in tqdm(range(args.num_samples)):
            curr_atom_pos = []
            atom_masks = []
            for part in ['rec', 'lig']:
                part_0_all_atom_mask = gt_features["ini_struct_feats"][part]["ini_all_atom_mask"][0].numpy()[None, ...]
                part_0_all_atom_positions = gt_features["ini_struct_feats"][part]["ini_all_atom_positions"][0].numpy()[None, ...]
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
                
                batch['t'] = batch['t'].new_tensor(np.array([[[t]]]))
                
                template_all_atom_pos = torch.cat(curr_atom_pos, dim=-3)
                assert template_all_atom_pos.shape[-3] == batch['template_all_atom_positions'].shape[-4]
                batch['template_all_atom_positions'] = template_all_atom_pos[None, None, ...][..., None].clone().to(
                    batch['template_all_atom_positions'].dtype).to(batch['template_all_atom_positions'].device)
                
                out = model(batch)
                out = tensor_tree_map(lambda x: x.cpu(), out)
                
                if time_idx < total_steps - 1:
                    split_idx = torch.searchsorted(batch['asym_id'].squeeze(), 2).item()
                    denoised_atom_pos = torch.split(out['final_atom_positions'][0], [split_idx, out['final_atom_positions'].shape[-3] - split_idx], dim=-3)
                    global_r, global_x, swap = get_global_rigid_body_transform(denoised_atom_pos, curr_atom_pos, atom_masks, is_homomer)
                    if swap:
                        denoised_atom_pos = [denoised_atom_pos[1], denoised_atom_pos[0]]
                    denoised_atom_pos = [denoised_atom_pos[0].to(global_r.dtype) @ global_r + global_x, denoised_atom_pos[1].to(global_r.dtype) @ global_r + global_x]
                    denoised_atom_pos = [denoised_atom_pos[0] * atom_masks[0][..., None], denoised_atom_pos[1] * atom_masks[1][..., None]]
                    lig_coms = [denoised_atom_pos[1][..., ca_idx, :].mean(dim=-2),
                                curr_atom_pos[1][..., ca_idx, :].mean(dim=-2)]
                    denoised_atom_pos[1] = denoised_atom_pos[1] - lig_coms[0][..., None, :]
                    curr_atom_pos[1] = curr_atom_pos[1] - lig_coms[1][..., None, :]
                    lig_x = lig_coms[1] - lig_coms[0]
                    lig_r, _ = multi_chain_permutation.get_optimal_transform(denoised_atom_pos[1][..., ca_idx, :],
                                                                             curr_atom_pos[1][..., ca_idx, :],
                                                                             atom_masks[1][..., ca_idx].to(torch.bool))
                    denoised_atom_pos[1] = denoised_atom_pos[1].to(lig_r.dtype) @ lig_r
                    updated_atom_pos = [denoised_atom_pos[0] * (s - t) / (1 - t) + curr_atom_pos[0] * (1 - s) / (1 - t), 
                                        denoised_atom_pos[1] * (s - t) / (1 - t) + curr_atom_pos[1] * (1 - s) / (1 - t)]
                    lig_r_t = - R.from_matrix(lig_r.numpy()).as_rotvec() * (s - t) / (1 - t)
                    lig_r_t = lig_r.new_tensor(R.from_rotvec(lig_r_t).as_matrix())
                    lig_x_t = - lig_x * (s - t) / (1 - t)
                    updated_atom_pos[1] = (updated_atom_pos[1] @ lig_r_t + lig_x_t + lig_coms[1][..., None, :]) * atom_masks[1][..., None]
                    curr_atom_pos = updated_atom_pos
                    
                    if args.save_intermediate_template or args.save_intermediate_pred:
                        write_output(batch, out, out_dir_data, f'{data_id}_s{sample_idx}_t{time_idx}', out_pred=args.save_intermediate_pred, out_conf=args.save_intermediate_conf, out_template=args.save_intermediate_template)
                    # if data_idx == 0:
                    #     write_output(batch, out, out_dir_data, f'{data_id}_s{sample_idx}_t{time_idx}', out_pred=True, out_conf=False, out_template=True)
                        
            write_output(batch, out, out_dir_data, f'{data_id}_s{sample_idx}', out_pred=True, out_conf=True, out_template=args.save_intermediate_template)
            # write_output(batch, out, out_dir_data, f'{data_id}_s{sample_idx}', out_pred=True, out_conf=True, out_template=data_idx == 0)
        
        batch = tensor_tree_map(lambda x: x.cpu(), batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", type=Path,
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--cached_esm_embedding_folder", type=str, default=None,
        help="Directory with cached ESM embeddings."
    )
    parser.add_argument(
        "--num_samples", type=int, default=40,
        help="""Number of samples to generate per target"""
    )
    parser.add_argument(
        "--num_steps", type=int, default=10,
        help="""Number of steps to the denoising process"""
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
        "--pinder_test_split", type=str, default='pinder_af2',
        choices=['pinder_af2', 'pinder_xl', 'pinder_s'],
    )
    parser.add_argument(
        "--pinder_test_type", type=str, default='holo',
        choices=['holo', 'apo', 'predicted'],
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
