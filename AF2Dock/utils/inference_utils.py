import pickle
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict
)

from openfold.np import protein, residue_constants
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils import multi_chain_permutation

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

def prepare_model(model, checkpoint_path, device):
    model = model.eval()
    
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

    model = model.to(device)
    logger.info(
        f"Loaded parameters at {checkpoint_path}..."
    )
    
    return model

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
    out_items_to_save = ['plddt', 'ptm_score', 'iptm_score', 'weighted_ptm_score', 
                         'predicted_aligned_error', 'max_predicted_aligned_error',
                         'final_atom_positions', 'final_atom_mask']
    aatype = batch['aatype'][0][..., -1].clone().detach().cpu().numpy()
    residue_index = batch["residue_index"][0][..., -1].clone().detach().cpu().numpy() + 1
    chain_index = batch["asym_id"][0][..., -1].clone().detach().cpu().numpy() - 1
    template_all_atom_mask_out = batch['template_all_atom_mask'].clone().detach().cpu().numpy()[0][0][..., -1]
    
    if out_template:
        template_all_atom_pos_out = batch['template_all_atom_positions'].clone().detach().cpu().numpy()[0][0][..., -1]
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
        prot_pred_masked = protein.Protein(
            aatype=aatype,
            atom_positions=all_atom_pos_out,
            atom_mask=template_all_atom_mask_out,
            residue_index=residue_index,
            b_factors=b_factors_out,
            chain_index=chain_index,
        )
        with open(outpath / (outprefix+'_masked.pdb'), 'w') as fp:
            fp.write(protein.to_pdb(prot_pred_masked))
        if out_conf:
            out = {k: v for k, v in out.items() if k in out_items_to_save}
            with open(outpath / (outprefix+'_out.pkl'), "wb") as fp:
                pickle.dump(out, fp, protocol=pickle.HIGHEST_PROTOCOL)

def update_pose(batch, out, atom_masks, curr_atom_pos, s, t, ca_idx, is_homomer):
    
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
    
    return updated_atom_pos
