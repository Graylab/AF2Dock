import pickle
import logging
import collections
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict
)

from biotite import structure as struc
from biotite.structure.io import pdb, pdbx

from openfold.data import parsers, feature_pipeline
from openfold.np import protein, residue_constants
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils import multi_chain_permutation

from AF2Dock.data import of_data
from AF2Dock.utils import data_utils

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
    
def write_output(batch, out, outpath, outprefix, out_pred=True, out_conf=True,
                 out_template=False, residue_index=None, asym_id=None):
    out_items_to_save = ['plddt', 'ptm_score', 'iptm_score', 'weighted_ptm_score', 
                         'predicted_aligned_error', 'max_predicted_aligned_error',
                         'final_atom_positions', 'final_atom_mask']
    aatype = batch['aatype'][0][..., -1].clone().detach().cpu().numpy()
    if residue_index is None:
        residue_index = batch["residue_index"][0][..., -1].clone().detach().cpu().numpy()
    residue_index = residue_index + 1
    if asym_id is None:
        asym_id = batch["asym_id"][0][..., -1].clone().detach().cpu().numpy()
    chain_index = asym_id - 1
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

def get_struc(part_struct_file):
    struct_file_type = part_struct_file.split('.')[-1]
    if struct_file_type == 'pdb':
        part_struc_pdb_file = pdb.PDBFile.read(part_struct_file)
        part_struc = pdb.get_structure(part_struc_pdb_file, model=1)
    elif struct_file_type == 'cif':
        part_struc_cif_file = pdbx.CIFFile.read(part_struct_file)
        part_struc = pdbx.get_structure(part_struc_cif_file, model=1)
    else:
        raise ValueError(f"Unsupported structure file type: {struct_file_type}")

    return part_struc

def get_seqs(target_row, part_struc, part, chains):
    part_seq_full_list = []
    part_resi_is_resolved_list = []
    
    if f'{part}_seq' in target_row:
        part_seq_file = target_row[f'{part}_seq']
        with open(part_seq_file, 'r') as f:
            part_seq_aln_str = f.read().strip()
        part_seqs_aln, part_seqs_aln_tags = parsers.parse_fasta(part_seq_aln_str)
        part_seqs_dict = {tag: seq for tag, seq in zip(part_seqs_aln_tags, part_seqs_aln)}
        
        for chain_id in chains:
            part_seq_full_list.append(part_seqs_dict[f'{chain_id}_full'])
            chain_resi_is_resolved = np.array([res != '-' for res in part_seqs_dict[chain_id]])
            part_resi_is_resolved_list.append(chain_resi_is_resolved)
    else:
        for chain_id in chains:
            chain_part_struc = part_struc[part_struc.chain_id == chain_id]
            _, res_names = struc.get_residues(chain_part_struc)
            chain_part_seq = []
            for res_name in res_names:
                olc = struc.info.one_letter_code(res_name)
                if olc is None or len(olc) != 1:
                    olc = 'X'
                chain_part_seq.append(olc)
            chain_part_seq = ''.join(chain_part_seq)
            part_seq_full_list.append(chain_part_seq)
            chain_resi_is_resolved = np.ones(len(chain_part_seq), dtype=bool)
    
    return part_seq_full_list, part_resi_is_resolved_list

def adjust_assembly_features(data, seq_dict, index_offset=200):

    seq_all_dict = {part:[seq for key, seq in seq_dict.items() if key.startswith(part)] for part in ['rec', 'lig']}
    seq_to_entity_id = {}
    for part, seqs in seq_all_dict.items():
      seq = ''.join(seqs)
      if seq not in seq_to_entity_id:
        seq_to_entity_id[seq] = len(seq_to_entity_id) + 1

    new_asym_id = []
    new_sym_id = []
    new_entity_id = []
    residue_index_offset_list = []
    
    entity_counter = collections.defaultdict(int)
    chain_id = 1
    for part in ['rec', 'lig']:
        seq = ''.join(seq_all_dict[part])
        entity_id = seq_to_entity_id[seq]
        entity_counter[entity_id] += 1
        sym_id = entity_counter[entity_id]
        
        seq_length = len(seq)
        new_asym_id.append((chain_id * np.ones(seq_length)).astype(np.int64))
        new_sym_id.append((sym_id * np.ones(seq_length)).astype(np.int64))
        new_entity_id.append((entity_id * np.ones(seq_length)).astype(np.int64))
        chain_id += 1

        part_chain_lengths = [len(chain_seq) for chain_seq in seq_all_dict[part]]
        part_chain_offset = [index_offset * idx for idx in range(len(part_chain_lengths))]
        part_residue_index_offset = np.concatenate([np.ones(length) * offset for length, offset in zip(part_chain_lengths, part_chain_offset)])
        residue_index_offset_list.append(part_residue_index_offset)
    
    data['asym_id'] =  data['asym_id'].new_tensor(np.concatenate(new_asym_id, axis=0)[..., None])
    data['sym_id'] =  data['sym_id'].new_tensor(np.concatenate(new_sym_id, axis=0)[..., None])
    data['entity_id'] =  data['entity_id'].new_tensor(np.concatenate(new_entity_id, axis=0)[..., None])
    
    residue_index_offset = data['residue_index'].new_tensor(np.concatenate(residue_index_offset_list, axis=0)[..., None])
    data['residue_index'] = data['residue_index'] + residue_index_offset

    return data

def load_data(target_row, config, esm_client=None, device='cuda'):
    data_pipeline = of_data.DataPipelineMultimer()
    feat_pipeline = feature_pipeline.FeaturePipeline(config)
    
    seq_dict = {}
    ini_struct_feats_dict = {}
    template_fests_dict = {}
    if config.model.pair_denoiser.use_esm:
        esm_embedding_dict = {}
    
    for part in ['rec', 'lig']:
        part_struct_file = target_row[part]
        part_struc = get_struc(part_struct_file)
        
        chains = struc.get_chains(part_struc)
        if len(chains) == 1 and chains[0] == '':
            part_struc.chain_id = ['A'] * len(part_struc.chain_id)
            chains = struc.get_chains(part_struc)

        part_seq_full_list, part_resi_is_resolved_list = get_seqs(target_row,
                                                                  part_struc,
                                                                  part,
                                                                  chains)
        
        part_ini_atom_positions_list = []
        part_ini_atom_mask_list = []
        for chain_idx, chain_id in enumerate(chains):
            chain_part_seq = part_seq_full_list[chain_idx]
            chain_part_resi_is_resolved = part_resi_is_resolved_list[chain_idx]
            chain_part_struc = part_struc[part_struc.chain_id == chain_id]
            assert len(chain_part_struc) == chain_part_resi_is_resolved.sum(), f"Length mismatch for {part} chain {chain_id}"
            part_ini_atom_positions_chain, part_ini_atom_mask_chain = of_data.get_atom_coords_pinder(chain_part_seq,
                                                                                                     chain_part_resi_is_resolved,
                                                                                                     chain_part_struc)
            part_ini_atom_positions_list.append(part_ini_atom_positions_chain)
            part_ini_atom_mask_list.append(part_ini_atom_mask_chain)
            
            part_ini_aatype = np.array(residue_constants.sequence_to_onehot(
                chain_part_seq, residue_constants.HHBLITS_AA_TO_ID
            ))
            seq_dict[f'{part}_{chain_id}'] = chain_part_seq
            template_fests_dict[f'{part}_{chain_id}'] = {
                "template_all_atom_positions": part_ini_atom_positions_chain[None, ...],
                "template_all_atom_mask": part_ini_atom_mask_chain[None, ...],
                "template_aatype": part_ini_aatype[None, ...],
            }
        
        ini_struct_feats_dict[part] = {
            "ini_all_atom_positions": np.concatenate(part_ini_atom_positions_list, axis=0),
            "ini_all_atom_mask": np.concatenate(part_ini_atom_mask_list, axis=0),
        }
        
        if config.model.pair_denoiser.use_esm:
            esm_client = esm_client.to(device)
            esm_embedding_list = []
            for part_seq in part_seq_full_list:
                chain_part_esm_embeddings = data_utils.get_esm_embeddings(part_seq, esm_client)
                chain_part_esm_embeddings = chain_part_esm_embeddings.cpu().numpy()
                chain_part_esm_embeddings = chain_part_esm_embeddings[1:-1] # remove BOS and EOS
                esm_embedding_list.append(chain_part_esm_embeddings)
            esm_client = esm_client.to('cpu')
            esm_embedding_dict[part] = np.concatenate(esm_embedding_list, axis=0)
    
    fasta_str = ''.join([f'>{tag}\n{seq}\n' for tag, seq in seq_dict.items()])
    
    data = data_pipeline.process_fasta(
        fasta_str,
        template_fests_dict,
        max_templates=1
    )
    
    if config.model.pair_denoiser.use_esm:
        data["esm_embedding"] = np.concatenate([esm_embedding_dict['rec'], esm_embedding_dict['lig']], axis=0)
    
    data = feat_pipeline.process_features(
        data, mode='predict', is_multimer=True
    )
    
    original_asym_id = data['asym_id'].clone().detach().cpu()
    original_residue_index = data['residue_index'].clone().detach().cpu()
    
    data = adjust_assembly_features(data, seq_dict)
    
    return data, ini_struct_feats_dict, original_asym_id, original_residue_index
