from pathlib import Path
from functools import partial
import math
import numpy as np
import ml_collections as mlc
import torch
import pytorch_lightning as pl
from biotite.structure import get_residues
from openfold.data import (
    data_modules,
    feature_pipeline,
)
from openfold.np import residue_constants
from pinder.core import PinderSystem, get_index, get_supplementary_data, get_metadata
from pinder.data.plot.performance import get_subsampled_train

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from AF2Dock.data import of_data
from AF2Dock.utils import utils

class AF2DockDataset(torch.utils.data.Dataset):
    def __init__(self,
                 config: mlc.ConfigDict,
                 mode: str = "train",
                 cached_esm_embedding_folder: str = None,
                 ):
        """
        This class check each individual PDB ID and return its chain(s) features/ground truth 
            Args:
                config:
                    A dataset config object. See openfold.config
                mode:
                    "train", "val", or "predict"
        """
        super(AF2DockDataset, self).__init__()

        self.config = config
        if cached_esm_embedding_folder is not None:
            self.cached_esm_embedding_folder = Path(cached_esm_embedding_folder)
        else:
            if self.mode == "train" or self.mode == "eval":
                raise ValueError("cached_esm_embedding_folder must be provided for train and eval modes")
            self.cached_esm_embedding_folder = None
        self.mode = mode

        valid_modes = ["train", "eval", "predict"]
        if mode not in valid_modes:
            raise ValueError(f'mode must be one of {valid_modes}')

        self.data_pipeline = of_data.DataPipelineMultimer()
        self.feature_pipeline = feature_pipeline.FeaturePipeline(config)

        if mode == "train" or mode == "eval":
            full_index = get_index()
            entity_meta = get_supplementary_data("entity_metadata")
            chain_meta = get_supplementary_data("chain_metadata")
            if mode == "train":
                train_index = full_index.query("split == 'train'").copy().reset_index(drop=True)
                train_index = utils.prefilter(train_index,
                                              get_metadata(),
                                              entity_meta,
                                              chain_meta)
                self.data_index = get_subsampled_train(train_index)
            elif mode == "eval":
                self.data_index = get_index().query("split == 'val'").copy().reset_index(drop=True)
            entity_meta['part_id'] = entity_meta['entry_id'].astype(str) + '_' + entity_meta['chain'].astype(str)
            self.data_index['holo_R_id'] = self.data_index['holo_R_pdb'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[2])
            self.data_index['holo_L_id'] = self.data_index['holo_L_pdb'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[2])
            self.data_index = self.data_index.merge(entity_meta[['part_id', 'sequence']], 
                                                    left_on='holo_R_id',
                                                    right_on='part_id',
                                                    how='left').rename(columns={'sequence': 'seq_R'})
            self.data_index = self.data_index.merge(entity_meta[['part_id', 'sequence']],
                                                    left_on='holo_L_id',
                                                    right_on='part_id',
                                                    how='left').rename(columns={'sequence': 'seq_L'})
            self.data_index = self.data_index.merge(chain_meta[['id', 'resi_auth_R', 'resi_auth_L']],
                                                    on='id',
                                                    how='left')
            self.data_index = self.data_index.drop(columns=['part_id_x', 'part_id_y', 'holo_R_id', 'holo_L_id'])
        elif mode == "predict":
            raise NotImplementedError("Predict mode not implemented yet")

    def get_ini_struct_cate(self, cate_probs, cate_present):
        cate_names = list(cate_probs.keys())
        cate_probs_real = torch.tensor([cate_probs[key] if cate_present[key] else 0.0 for key in cate_names])
        cate = cate_names[torch.multinomial(cate_probs_real, 1).item()]
        return cate
    
    def get_map_by_uniprot(self, ini, holo, part_resi_split):
        ini_pdb_res_num, _ = get_residues(ini)
        ini_seqential_id_to_pdb = {idx: ini_pdb_res_num[idx] for idx in range(len(ini_pdb_res_num))}
        ini_uniprot_map = ini.resolved_pdb2uniprot
        ini_sequential_id_to_uniprot = {key: ini_uniprot_map[ini_seqential_id_to_pdb[key]] for key in ini_seqential_id_to_pdb if ini_seqential_id_to_pdb[key] in ini_uniprot_map}
        holo_seq_pos = ','.join(part_resi_split).strip(',').split(',')
        holo_seq_seqential_id_to_pdb = {idx: int(holo_seq_pos[idx]) for idx in range(len(holo_seq_pos)) if holo_seq_pos[idx] != ''}
        holo_uniprot_map = holo.resolved_pdb2uniprot
        holo_seq_sequential_id_to_uniprot = {key: holo_uniprot_map[holo_seq_seqential_id_to_pdb[key]] for key in holo_seq_seqential_id_to_pdb if holo_seq_seqential_id_to_pdb[key] in holo_uniprot_map}
        uniprot_to_holo_seq_sequential_id = {v: k for k, v in holo_seq_sequential_id_to_uniprot.items()}
        ini_seqential_id_to_holo_seq_sequential_id = {key: uniprot_to_holo_seq_sequential_id[ini_sequential_id_to_uniprot[key]] 
                                                      for key in ini_sequential_id_to_uniprot if ini_sequential_id_to_uniprot[key] in uniprot_to_holo_seq_sequential_id}
        return ini_seqential_id_to_holo_seq_sequential_id
    
    def get_rigid_body_noise_at_0(self, tr_sigma, rot_sigma):
         tr_0 = torch.normal(0, tr_sigma, (3,))
         rot_axis = torch.rand(3)
         rot_axis = rot_axis / torch.linalg.norm(rot_axis)
         rot_angle = torch.abs(torch.normal(0, rot_sigma)) % math.pi
         rot_0 = rot_angle * rot_axis
         return tr_0.numpy(), rot_0.numpy()
    
    def __getitem__(self, idx):
        index_entry = self.data_index.iloc[idx]
        struct_id = index_entry['id']

        if self.mode == 'train' or self.mode == 'eval':
            t = torch.rand(1).item() * (1. - 1e-5) + 1e-5

            ps = PinderSystem(struct_id)
            cate_probs_ori = dict(self.config.data[self.mode].pinder_cate_prob)

            all_atom_positions_dict = {}
            all_atom_mask_dict = {}
            seq_dict = {}
            struct_feats_at_t_dict = {}
            esm_embedding_dict = {}
            
            for part in ['rec', 'lig']:
                abbr = 'R' if part == 'rec' else 'L'
                full_n = 'receptor' if part == 'rec' else 'ligand'
                part_id = index_entry[f'holo_{abbr}_pdb'].split('.pdb')[0]
                part_seqres = index_entry[f'seq_{abbr}']
                part_resi_auth = index_entry[f"resi_auth_{abbr}"]
                part_resi_auth_split = part_resi_auth.split(',')
                part_resi_auth_resolved_num = len(part_resi_auth_split) - part_resi_auth_split.count('')
                struct_resi = getattr(ps, f'native_{abbr}').residues
                assert part_resi_auth_resolved_num == len(struct_resi), "Mismatched lengths between resi auth and structure"
                part_resi_split = []
                idx = 0
                for resi in part_resi_auth_split:
                    if resi != '':
                        part_resi_split.append(str(struct_resi[idx]))
                        idx += 1
                    else:
                        part_resi_split.append('')
                if len(part_resi_split) != len(part_seqres):
                    # e.g. 8hco chain G, fall back to sequence in structure
                    part_seqres, part_resi_split = utils.get_seq_from_atom_array(getattr(ps, f'native_{abbr}').atom_array)
                assert len(part_resi_split) == len(part_seqres), "Mismatch between resi split and seqres"
                part_seq, part_resi_resolved = utils.truncate_to_resolved(part_seqres, part_resi_split)
                part_all_atom_positions, part_all_atom_mask = of_data.get_atom_coords_pinder(part_seq,
                                                                                             part_resi_resolved,
                                                                                             getattr(ps, f'native_{abbr}').atom_array)
                part_esm_embedding = np.load(self.cached_esm_embedding_folder / f"{part_id}.npy")
                assert part_esm_embedding.shape[0] == len(part_seq), "Mismatch between ESM embedding and sequence length"
                
                # Get the initial structure for the receptor and ligand, which are processed in data pipeline as templates
                part_cate = self.get_ini_struct_cate(cate_probs_ori, {cate: getattr(ps, f"{cate}_{full_n}") is not None for cate in cate_probs_ori.keys()})
                if part_cate != 'holo':
                    part_ini_struct = getattr(ps, f"{part_cate}_{full_n}")
                    part_ini_struct, _, _ = part_ini_struct.superimpose(getattr(ps, f'native_{abbr}'))
                    part_holo_struct = getattr(ps, f'aligned_holo_{abbr}')
                    part_ini_to_holo_map = self.get_map_by_uniprot(part_ini_struct, part_holo_struct, part_resi_split)

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
                part_t_all_atom_positions = (part_ini_all_atom_positions * (1. - t) + part_all_atom_positions * t) * part_t_all_atom_mask
                if part == 'lig':
                    tr_0, rot_0 = self.get_rigid_body_noise_at_0(self.config.data.rigid_body.tr_sigma, self.config.data.rigid_body.rot_sigma)
                    tr_t = tr_0 * (1. - t)
                    rot_t = rot_0 * (1. - t)
                    part_t_all_atom_positions = utils.apply_rigid_body_transform_atom37(part_t_all_atom_positions,
                                                                                        part_t_all_atom_mask,
                                                                                        residue_constants.atom_order["CA"],
                                                                                        tr_t,
                                                                                        rot_t)
                part_feats_at_t = {
                    "template_all_atom_positions": part_t_all_atom_positions,
                    "template_all_atom_mask": part_t_all_atom_mask,
                    "template_sequence": part_seq.encode(),
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

            data["esm_embedding"] = np.concatenate([esm_embedding_dict['rec'], esm_embedding_dict['lig']], dim=0)
            data["t"] = t
            data["tr_0"] = tr_0
            data["rot_0"] = rot_0

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
        
        return data

    def __len__(self):
        return len(self.data_index)

class AF2DockDataModule(pl.LightningDataModule):
    def __init__(self, config: mlc.ConfigDict,
                 training_mode,
                 batch_seed,
                 cached_esm_embedding_folder: str = None,
                 **kwargs):
        super().__init__()

        self.config = config
        self.batch_seed = batch_seed
        self.training_mode = training_mode
        self.cached_esm_embedding_folder = cached_esm_embedding_folder

    def setup(self, setup=None):
        # Most of the arguments are the same for the three datasets 
        dataset_gen = partial(AF2DockDataset,
                              cached_esm_embedding_folder=self.cached_esm_embedding_folder,
                              config=self.config)

        if self.training_mode:
            self.train_dataset = dataset_gen(mode="train")
            self.eval_dataset = dataset_gen(mode="eval")
        else:
            raise NotImplementedError("Prediction mode is not implemented yet")

    def _gen_dataloader(self, stage=None):
        generator = None
        if self.batch_seed is not None:
            generator = torch.Generator()
            generator = generator.manual_seed(self.batch_seed)

        if stage == "train":
            dataset = self.train_dataset
        elif stage == "eval":
            dataset = self.eval_dataset
        elif stage == "predict":
            raise NotImplementedError("Predict mode is not implemented yet")
        else:
            raise ValueError("Invalid stage")

        batch_collator = data_modules.OpenFoldBatchCollator()

        dl = data_modules.OpenFoldDataLoader(
            dataset,
            config=self.config,
            stage=stage,
            generator=generator,
            batch_size=self.config.data_module.data_loaders.batch_size,
            num_workers=self.config.data_module.data_loaders.num_workers,
            collate_fn=batch_collator,
        )

        return dl

    def train_dataloader(self):
        return self._gen_dataloader("train")

    def val_dataloader(self):
        if self.eval_dataset is not None:
            return self._gen_dataloader("eval")
        return [] 

    def predict_dataloader(self):
        return self._gen_dataloader("predict")
