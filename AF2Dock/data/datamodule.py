import os
from pathlib import Path
from functools import partial
import json
import logging
from typing import Optional, Sequence, Any
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
import ml_collections as mlc
import torch
from openfold.data import (
    data_modules,
    feature_pipeline,
)
from openfold.np import residue_constants
from pinder.core.index.utils import PinderSystem, get_index, get_supplementary_data
from pinder.data.plot.performance import get_subsampled_train
from AF2Dock.data import of_data

def truncate_seq_to_resolved(full_seq, resi_resolved):
    l_index = resi_resolved.index(True)
    r_index = len(resi_resolved) - resi_resolved[::-1].index(True) - 1
    return full_seq[l_index:r_index + 1], resi_resolved[l_index:r_index + 1]

class SingleDataset(torch.utils.data.Dataset):
    def __init__(self,
                 cached_esm_embedding_folder,
                 config: mlc.ConfigDict,
                 mode: str = "train",
                 ):
        """
        This class check each individual PDB ID and return its chain(s) features/ground truth 
            Args:
                config:
                    A dataset config object. See openfold.config
                mode:
                    "train", "val", or "predict"
        """
        super(SingleDataset, self).__init__()

        self.config = config
        self.cached_esm_embedding_folder = Path(cached_esm_embedding_folder)
        self.mode = mode

        valid_modes = ["train", "eval", "predict"]
        if mode not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}')

        self.data_pipeline = of_data.DataPipelineMultimer()
        self.feature_pipeline = feature_pipeline.FeaturePipeline(config)

        if mode == "train":
            full_index = get_index()
            self.data_index = get_subsampled_train(full_index)
            self.entity_meta = get_supplementary_data("entity_metadata")
            self.chain_meta = get_supplementary_data("chain_metadata")
        elif mode == "eval":
            self.data_index = get_index().query("split == 'val'")
            self.entity_meta = get_supplementary_data("entity_metadata")
            self.chain_meta = get_supplementary_data("chain_metadata")
        elif mode == "predict":
            raise NotImplementedError("Predict mode not implemented yet")

    def struct_id_to_idx(self, struct_id):
        return self.data_index.index[self.data_index['id'] == struct_id].values[0]

    def idx_to_struct_id(self, idx):
        return self.data_index.iloc[idx]['id']
    
    def get_chain_all_atom_feats(self, resi_auth, seqres, atom_array):
        resi_resolved_full = [item != '' for item in resi_auth.split(',')]
        assert len(resi_resolved_full) == len(seqres)
        seq, resi_resolved = truncate_seq_to_resolved(seqres, resi_resolved_full)
        all_atom_positions, all_atom_mask = of_data.get_atom_coords_pinder(seq, resi_resolved, atom_array)
        return all_atom_positions, all_atom_mask, seq, resi_resolved

    def get_ini_struct_cate(self, cate_probs, cate_present):
        cate_names = list(cate_probs.keys())
        cate_probs_real = torch.tensor([cate_probs[key] if cate_present[key] else 0.0 for key in cate_names])
        cate = cate_names[torch.multinomial(cate_probs_real, 1).item()]
        return cate
    
    def get_map_by_uniprot(self, ini, holo, res_auth):
        ini_pdb_res_num, _ = ini.get_residues()
        ini_seqential_id_to_pdb = {idx: ini_pdb_res_num[idx] for idx in range(len(ini_pdb_res_num))}
        ini_uniprot_map = ini.resolved_pdb2uniprot
        ini_sequential_id_to_uniprot = {key: ini_uniprot_map[ini_seqential_id_to_pdb[key]] for key in ini_seqential_id_to_pdb if ini_seqential_id_to_pdb[key] in ini_uniprot_map}
        holo_seq_pos = res_auth.strip(',').split(',')
        holo_seq_seqential_id_to_pdb = {idx: holo_seq_pos[idx] for idx in range(len(holo_seq_pos)) if holo_seq_pos[idx] != ''}
        holo_uniprot_map = holo.resolved_pdb2uniprot
        holo_seq_sequential_id_to_uniprot = {key: holo_uniprot_map[holo_seq_seqential_id_to_pdb[key]] for key in holo_seq_seqential_id_to_pdb if holo_seq_seqential_id_to_pdb[key] in holo_uniprot_map}
        uniprot_to_holo_seq_sequential_id = {v: k for k, v in holo_seq_sequential_id_to_uniprot.items()}
        ini_seqential_id_to_holo_seq_sequential_id = {key: uniprot_to_holo_seq_sequential_id[ini_sequential_id_to_uniprot[key]] 
                                                      for key in ini_sequential_id_to_uniprot if ini_sequential_id_to_uniprot[key] in uniprot_to_holo_seq_sequential_id}
        return ini_seqential_id_to_holo_seq_sequential_id
    
    def get_rigid_body_noise_at_0(self, tr_sigma, rot_sigma):
        tr_0 = torch.normal(0, tr_sigma, (3,))
        rot_axis = torch.uniform(0, 1, (3,))
        rot_axis = rot_axis / torch.linalg.norm(rot_axis)
        rot_angle = torch.abs(torch.normal(0, rot_sigma)) % math.pi
        rot_0 = rot_angle * rot_axis
        return tr_0, rot_0
    
    def apply_rigid_body_noise(self, all_atom_positions, all_atom_mask, tr, rot):
        ca_idx = residue_constants.atom_order["CA"]
        com = np.mean(all_atom_positions[..., ca_idx, :], axis=-2)
        rot_t_mat = R.from_rotvec(rot).as_matrix()
        all_atom_positions = all_atom_positions - com
        all_atom_positions = np.einsum('...ij,kj->...ik', all_atom_positions, rot_t_mat)
        all_atom_positions = all_atom_positions + com + tr
        all_atom_positions = all_atom_positions * all_atom_mask
        return all_atom_positions
    
    def __getitem__(self, idx):
        struct_id = self.idx_to_struct_id(idx)
        index_entry = self.data_index.iloc[idx]

        if self.mode == 'train' or self.mode == 'eval':
            t = torch.rand(1) * (1. - 1e-5) + 1e-5

            ps = PinderSystem(struct_id)
            chain_meta_i = self.chain_meta.query(f"id == '{struct_id}'").iloc[0]
            cate_probs_ori = {'holo':0.6, 'apo':0.2, 'pred':0.2} #change to read from config

            all_atom_positions_dict = {}
            all_atom_mask_dict = {}
            seq_dict = {}
            struct_feats_at_t_dict = {}
            esm_embedding_dict = {}
            
            for part in ['rec', 'lig']:
                abbr = 'R' if part == 'rec' else 'L'
                part_id = index_entry[f'holo_{abbr}_pdb'].split('.pdb')[0]
                part_seqres = self.entity_meta.query(f"entry_id == '{part_id.split('_')[0]}' and chain == '{part_id.split('_')[2]}'").sequence.values[0]
                part_resi_auth = chain_meta_i[f"resi_auth_{abbr}"]
                part_all_atom_positions, part_all_atom_mask, part_seq, part_resi_resolved = self.get_chain_all_atom_feats(part_resi_auth,
                                                                                                                          part_seqres,
                                                                                                                          getattr(ps, f'native_{abbr}').atom_array)
                part_esm_embedding = torch.load(self.cached_esm_embedding_folder / f"{part_id}.pt")
                assert part_esm_embedding.shape[0] == len(part_seq)
                
                # Get the initial structure for the receptor and ligand, which are processed in data pipeline as templates
                part_cate = self.get_ini_struct_cate(cate_probs_ori, {cate: getattr(ps, f"{cate}_{abbr}") is not None for cate in cate_probs_ori.keys()})
                part_ini_struct = getattr(ps, f"{part_cate}_{abbr}")
                part_ini_struct, _, _ = part_ini_struct.superimpose(getattr(ps, f'native_{abbr}'))
                if part_cate != 'holo':
                    part_ini_to_holo_map = self.get_map_by_uniprot(part_ini_struct, getattr(ps, f'native_{abbr}'), part_resi_auth)

                    part_holo_ini_overlap_range = [min(list(part_ini_to_holo_map.values())), max(list(part_ini_to_holo_map.values()))]
                    part_all_atom_positions = part_all_atom_positions[part_holo_ini_overlap_range[0]:part_holo_ini_overlap_range[1] + 1]
                    part_all_atom_mask = part_all_atom_mask[part_holo_ini_overlap_range[0]:part_holo_ini_overlap_range[1] + 1]
                    part_seq = part_seq[part_holo_ini_overlap_range[0]:part_holo_ini_overlap_range[1] + 1]
                    part_resi_resolved = part_resi_resolved[part_holo_ini_overlap_range[0]:part_holo_ini_overlap_range[1] + 1]
                    part_esm_embedding = part_esm_embedding[part_holo_ini_overlap_range[0]:part_holo_ini_overlap_range[1] + 1]
                    
                    part_ini_resi_resolved = [True if (i + part_holo_ini_overlap_range[0]) in part_ini_to_holo_map.values() else False for i in range(len(part_seq))]
                    part_ini_overlapped_atom_array = part_ini_struct.atom_array[list(part_holo_ini_overlap_range.keys())]
                    part_ini_all_atom_positions, part_ini_all_atom_mask = of_data.get_atom_coords_pinder(part_seq, part_ini_resi_resolved, part_ini_overlapped_atom_array)
                else:
                    part_ini_all_atom_positions, part_ini_all_atom_mask = part_all_atom_mask, part_all_atom_positions
                part_ini_aatype = residue_constants.sequence_to_onehot(
                    part_seq, residue_constants.HHBLITS_AA_TO_ID
                )
                
                # Interpolate and add noise
                part_t_all_atom_mask = part_ini_all_atom_mask * part_all_atom_mask
                part_t_all_atom_positions = (part_ini_all_atom_positions * (1. - t.item()) + part_all_atom_positions * t.item()) * part_t_all_atom_mask
                if part == 'lig':
                    tr_0, rot_0 = self.get_rigid_body_noise_at_0(self.config.tr_sigma, self.config.rot_sigma)
                    tr_t = tr_0 * (1. - t.item())
                    rot_t = rot_0 * (1. - t.item())
                    part_t_all_atom_positions = self.apply_rigid_body_noise(part_t_all_atom_positions, part_t_all_atom_mask, tr_t.numpy(), rot_t.numpy())
                part_feats_at_t = {
                    "template_all_atom_positions": part_t_all_atom_positions,
                    "template_all_atom_mask": part_t_all_atom_mask,
                    "template_sequence": part_seq,
                    "template_aatype": part_ini_aatype,
                }

                all_atom_positions_dict[part] = part_all_atom_positions
                all_atom_mask_dict[part] = part_all_atom_mask
                seq_dict[part] = part_seq
                esm_embedding_dict[part] = part_esm_embedding
                struct_feats_at_t_dict[part] = part_feats_at_t

            fasta_str = f">rec\n{seq_dict['rec']}\n>lig\n{seq_dict['lig']}\n"

            data = self.data_pipeline.process_fasta_with_atom_pos(
                input_fasta_str=fasta_str,
                all_atom_positions_dict=all_atom_positions_dict,
                all_atom_mask_dict=all_atom_mask_dict,
                struct_feats_at_t_dict=struct_feats_at_t_dict,
            )

        else:
            raise NotImplementedError("Predict mode not implemented yet")

        # process all_chain_features
        data = self.feature_pipeline.process_features(data,
                                                      mode=self.mode,
                                                      is_multimer=True)

        # if it's inference mode, only need all_chain_features
        data["batch_idx"] = torch.tensor(
            [idx for _ in range(data["aatype"].shape[-1])],
            dtype=torch.int64,
            device=data["aatype"].device)
        
        data["esm_embedding"] = torch.cat([esm_embedding_dict['rec'], esm_embedding_dict['lig']], dim=0)
        if self.mode == 'train' or self.mode == 'eval':
            data["t"] = t
            data["tr_0"] = tr_0
            data["rot_0"] = rot_0
        
        return data

    def __len__(self):
        return len(self.data_index)

# class OpenFoldMultimerDataset(data_modules.OpenFoldDataset):
#     """
#     Create a torch Dataset object for multimer training and 
#     add filtering steps described in AlphaFold Multimer's paper:
#     https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf Supplementary section 7.1 
#     """

#     def __init__(self,
#                  datasets: Sequence[OpenFoldSingleMultimerDataset],
#                  probabilities: Sequence[float],
#                  epoch_len: int,
#                  generator: torch.Generator = None,
#                  _roll_at_init: bool = True
#                  ):
#         super(OpenFoldMultimerDataset, self).__init__(datasets=datasets,
#                                                       probabilities=probabilities,
#                                                       epoch_len=epoch_len,
#                                                       generator=generator,
#                                                       _roll_at_init=_roll_at_init)

#     @staticmethod
#     def deterministic_train_filter(
#         cache_entry: Any,
#         is_distillation: bool,
#         max_resolution: float = 9.,
#         max_single_aa_prop: float = 0.8,
#         minimum_number_of_residues: int = 200,
#         *args, **kwargs
#     ) -> bool:
#         """
#         Implement multimer training filtering criteria described in
#         https://www.biorxiv.org/content/10.1101/2021.10.04.463034v2.full.pdf Supplementary section 7.1
#         """
#         resolution = cache_entry.get("resolution", None)
#         seqs = cache_entry["seqs"]

#         return all([data_modules.resolution_filter(resolution=resolution,
#                                       max_resolution=max_resolution),
#                     data_modules.aa_count_filter(seqs=seqs,
#                                     max_single_aa_prop=max_single_aa_prop),
#                     (not is_distillation or data_modules.all_seq_len_filter(seqs=seqs,
#                                                                             minimum_number_of_residues=minimum_number_of_residues))])

#     @staticmethod
#     def get_stochastic_train_filter_prob(
#         cache_entry: Any,
#         *args, **kwargs
#     ) -> list:
#         # Stochastic filters
#         cluster_sizes = cache_entry.get("cluster_sizes")
#         if cluster_sizes is not None:
#             return [1 / c if c > 0 else 1 for c in cluster_sizes]

#         num_chains = len(cache_entry["chain_ids"])
#         return [1.] * num_chains

#     def looped_samples(self, dataset_idx):
#         max_cache_len = int(self.epoch_len * self.probabilities[dataset_idx])
#         dataset = self.datasets[dataset_idx]
#         is_distillation = dataset.treat_pdb_as_distillation
#         idx_iter = self.looped_shuffled_dataset_idx(len(dataset))
#         mmcif_data_cache = dataset.mmcif_data_cache
#         while True:
#             weights = []
#             idx = []
#             for _ in range(max_cache_len):
#                 candidate_idx = next(idx_iter)
#                 mmcif_id = dataset.idx_to_mmcif_id(candidate_idx)
#                 mmcif_data_cache_entry = mmcif_data_cache[mmcif_id]
#                 if not self.deterministic_train_filter(cache_entry=mmcif_data_cache_entry,
#                                                        is_distillation=is_distillation):
#                     continue

#                 chain_probs = self.get_stochastic_train_filter_prob(
#                     mmcif_data_cache_entry,
#                 )
#                 weights.extend([[1. - p, p] for p in chain_probs])
#                 idx.extend([candidate_idx] * len(chain_probs))

#             samples = torch.multinomial(
#                 torch.tensor(weights),
#                 num_samples=1,
#                 generator=self.generator,
#             )
#             samples = samples.squeeze()

#             cache = [i for i, s in zip(idx, samples) if s]

#             for datapoint_idx in cache:
#                 yield datapoint_idx

# class OpenFoldMultimerDataModule(data_modules.OpenFoldDataModule):
#     """
#     Create a datamodule specifically for multimer training

#     Compared to OpenFoldDataModule, OpenFoldMultimerDataModule
#     requires mmcif_data_cache_path which is the product of 
#     scripts/generate_mmcif_cache.py mmcif_data_cache_path should be 
#     a file that record what chain(s) each mmcif file has 
#     """

#     def __init__(self, config: mlc.ConfigDict,
#                  template_mmcif_dir: str, max_template_date: str,
#                  train_data_dir: Optional[str] = None,
#                  train_mmcif_data_cache_path: Optional[str] = None,
#                  val_mmcif_data_cache_path: Optional[str] = None,
#                  **kwargs):
#         super(OpenFoldMultimerDataModule, self).__init__(config,
#                                                          template_mmcif_dir,
#                                                          max_template_date,
#                                                          train_data_dir,
#                                                          **kwargs)

#         self.train_mmcif_data_cache_path = train_mmcif_data_cache_path
#         self.training_mode = self.train_data_dir is not None
#         self.val_mmcif_data_cache_path = val_mmcif_data_cache_path

#     def setup(self, setup=None):
#         # Most of the arguments are the same for the three datasets 
#         dataset_gen = partial(OpenFoldSingleMultimerDataset,
#                               template_mmcif_dir=self.template_mmcif_dir,
#                               max_template_date=self.max_template_date,
#                               config=self.config,
#                               kalign_binary_path=self.kalign_binary_path,
#                               template_release_dates_cache_path=self.template_release_dates_cache_path,
#                               obsolete_pdbs_file_path=self.obsolete_pdbs_file_path)

#         if self.training_mode:
#             train_dataset = dataset_gen(
#                 data_dir=self.train_data_dir,
#                 mmcif_data_cache_path=self.train_mmcif_data_cache_path,
#                 alignment_dir=self.train_alignment_dir,
#                 filter_path=self.train_filter_path,
#                 max_template_hits=self.config.train.max_template_hits,
#                 shuffle_top_k_prefiltered=self.config.train.shuffle_top_k_prefiltered,
#                 treat_pdb_as_distillation=False,
#                 mode="train",
#                 alignment_index=self.alignment_index,
#             )

#             distillation_dataset = None
#             if self.distillation_data_dir is not None:
#                 distillation_dataset = dataset_gen(
#                     data_dir=self.distillation_data_dir,
#                     alignment_dir=self.distillation_alignment_dir,
#                     filter_path=self.distillation_filter_path,
#                     max_template_hits=self.config.train.max_template_hits,
#                     treat_pdb_as_distillation=True,
#                     mode="train",
#                     alignment_index=self.distillation_alignment_index,
#                     _structure_index=self._distillation_structure_index,
#                 )

#                 d_prob = self.config.train.distillation_prob

#             if distillation_dataset is not None:
#                 datasets = [train_dataset, distillation_dataset]
#                 d_prob = self.config.train.distillation_prob
#                 probabilities = [1. - d_prob, d_prob]
#             else:
#                 datasets = [train_dataset]
#                 probabilities = [1.]

#             generator = None
#             if self.batch_seed is not None:
#                 generator = torch.Generator()
#                 generator = generator.manual_seed(self.batch_seed + 1)

#             self.train_dataset = OpenFoldMultimerDataset(
#                 datasets=datasets,
#                 probabilities=probabilities,
#                 epoch_len=self.train_epoch_len,
#                 generator=generator,
#                 _roll_at_init=True,
#             )

#             if self.val_data_dir is not None:
#                 self.eval_dataset = dataset_gen(
#                     data_dir=self.val_data_dir,
#                     alignment_dir=self.val_alignment_dir,
#                     mmcif_data_cache_path=self.val_mmcif_data_cache_path,
#                     filter_path=None,
#                     max_template_hits=self.config.eval.max_template_hits,
#                     mode="eval",
#                 )
#             else:
#                 self.eval_dataset = None
#         else:
#             self.predict_dataset = dataset_gen(
#                 data_dir=self.predict_data_dir,
#                 alignment_dir=self.predict_alignment_dir,
#                 filter_path=None,
#                 max_template_hits=self.config.predict.max_template_hits,
#                 mode="predict",
#             )
