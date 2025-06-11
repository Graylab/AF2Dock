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

from typing import MutableMapping, Tuple
import numpy as np
from openfold.data import (
    parsers,
    msa_pairing,
    feature_processing_multimer,
    data_pipeline,
)
import openfold.np.residue_constants as residue_constants
FeatureDict = MutableMapping[str, np.ndarray]

def get_atom_coords_pinder(
    full_sequence: str,
    res_resolved: np.ndarray,
    resolved_atom_array: np.ndarray,
    _zero_center_positions: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    res_ids = []
    res_id_to_atom = {}
    for atom in resolved_atom_array:
        atom_res_id = str(atom.res_id) + atom.ins_code
        if atom_res_id in res_id_to_atom:
            res_id_to_atom[atom_res_id].append(atom)
        else:
            res_id_to_atom[atom_res_id] = [atom]
        if atom_res_id not in res_ids:
            res_ids.append(atom_res_id)
    
    resolved_to_full_seq = {}
    resolved_idx = 0
    for idx, resolved in enumerate(res_resolved):
        if resolved:
            resolved_to_full_seq[resolved_idx] = idx
            resolved_idx += 1

    assert len(full_sequence) == len(res_resolved)
    assert len(res_ids) == len(resolved_to_full_seq)
    # Extract the coordinates
    num_res = len(full_sequence)
    all_atom_positions = np.zeros(
        [num_res, residue_constants.atom_type_num, 3], dtype=np.float32
    )
    all_atom_mask = np.zeros(
        [num_res, residue_constants.atom_type_num], dtype=np.float32
    )
    for idx, res_id in enumerate(res_ids):
        res_index = resolved_to_full_seq[idx]
        pos = np.zeros([residue_constants.atom_type_num, 3], dtype=np.float32)
        mask = np.zeros([residue_constants.atom_type_num], dtype=np.float32)
        resi_atoms = res_id_to_atom[res_id]
        resi_name = resi_atoms[0].res_name
        for atom in resi_atoms:
            atom_name = atom.atom_name
            x, y, z = atom.coord
            if atom_name in residue_constants.atom_order.keys():
                pos[residue_constants.atom_order[atom_name]] = [x, y, z]
                mask[residue_constants.atom_order[atom_name]] = 1.0
            elif atom_name.upper() == "SE" and resi_name == "MSE":
                # Put the coords of the selenium atom in the sulphur column
                pos[residue_constants.atom_order["SD"]] = [x, y, z]
                mask[residue_constants.atom_order["SD"]] = 1.0

        # Fix naming errors in arginine residues where NH2 is incorrectly
        # assigned to be closer to CD than NH1
        cd = residue_constants.atom_order['CD']
        nh1 = residue_constants.atom_order['NH1']
        nh2 = residue_constants.atom_order['NH2']
        if(
            resi_name == 'ARG' and
            all(mask[atom_index] for atom_index in (cd, nh1, nh2)) and
            (np.linalg.norm(pos[nh1] - pos[cd]) > 
                np.linalg.norm(pos[nh2] - pos[cd]))
        ):
            pos[nh1], pos[nh2] = pos[nh2].copy(), pos[nh1].copy()
            mask[nh1], mask[nh2] = mask[nh2].copy(), mask[nh1].copy()

        all_atom_positions[res_index] = pos
        all_atom_mask[res_index] = mask

    if _zero_center_positions:
        binary_mask = all_atom_mask.astype(bool)
        translation_vec = all_atom_positions[binary_mask].mean(axis=0)
        all_atom_positions[binary_mask] -= translation_vec

    return all_atom_positions, all_atom_mask

def make_dummy_msa_obj(input_sequence, input_description) -> parsers.Msa:
    deletion_matrix = [[0 for _ in input_sequence]]
    return parsers.Msa(sequences=[input_sequence],
                       deletion_matrix=deletion_matrix,
                       descriptions=[input_description])

def make_dummy_msa_feats(input_sequence, input_description) -> FeatureDict:
    msa_data_obj = make_dummy_msa_obj(input_sequence, input_description)
    return data_pipeline.make_msa_features([msa_data_obj])

def merge_features(all_chain_features, max_templates):
    # from https://github.com/sokrypton/ColabFold/blob/b119520d8f43e1547e1c4352fd090c59a8dbb369/colabfold/batch.py#L913C1-L952C1
    feature_processing_multimer.process_unmerged_features(all_chain_features)
    np_chains_list = list(all_chain_features.values())
    # noinspection PyProtectedMember
    pair_msa_sequences = not feature_processing_multimer._is_homomer_or_monomer(np_chains_list)
    chains = list(np_chains_list)
    chain_keys = chains[0].keys()
    updated_chains = []
    for chain_num, chain in enumerate(chains):
        new_chain = {k: v for k, v in chain.items() if "_all_seq" not in k}
        for feature_name in chain_keys:
            if feature_name.endswith("_all_seq"):
                feats_padded = msa_pairing.pad_features(
                    chain[feature_name], feature_name
                )
                new_chain[feature_name] = feats_padded
        new_chain["num_alignments_all_seq"] = np.asarray(
            len(np_chains_list[chain_num]["msa_all_seq"])
        )
        updated_chains.append(new_chain)
    np_chains_list = updated_chains
    np_chains_list = feature_processing_multimer.crop_chains(
        np_chains_list,
        msa_crop_size=feature_processing_multimer.MSA_CROP_SIZE,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=max_templates,
    )
    # merge_chain_features crashes if there are additional features only present in one chain
    # remove all features that are not present in all chains
    common_features = set([*np_chains_list[0]]).intersection(*np_chains_list)
    np_chains_list = [
        {key: value for (key, value) in chain.items() if key in common_features}
        for chain in np_chains_list
    ]
    np_example = feature_processing_multimer.msa_pairing.merge_chain_features(
        np_chains_list=np_chains_list,
        pair_msa_sequences=pair_msa_sequences,
        max_templates=max_templates,
    )
    np_example = feature_processing_multimer.process_final(np_example)

    return np_example

class DataPipelineMultimer:
    """Assembles the input features."""

    def __init__(self):
        """Initializes the data pipeline."""
        pass

    def process_sequence(
        self,
        input_sequence: str,
        input_description: str,
    ) -> FeatureDict:
        num_res = len(input_sequence)

        sequence_features = data_pipeline.make_sequence_features(
            sequence=input_sequence,
            description=input_description,
            num_res=num_res,
        )

        msa_features = make_dummy_msa_feats(input_sequence, input_description)

        return {
            **sequence_features,
            **msa_features, 
        }

    def _process_single_chain(
            self,
            chain_id: str,
            sequence: str,
            description: str,
    ) -> FeatureDict:
        """Runs the pipeline on a single chain."""

        chain_features = self.process_sequence(
            input_sequence=sequence,
            input_description=description,
        )

        all_seq_msa_features = self._all_seq_msa_features_dummy(sequence, chain_id)
        chain_features.update(all_seq_msa_features)
        
        return chain_features

    @staticmethod
    def _all_seq_msa_features_dummy(sequence, chain_id):
        """Get dummy MSA features for single sequences"""
        msa_data_obj = make_dummy_msa_obj(sequence, chain_id)
        all_seq_features = data_pipeline.make_msa_features([msa_data_obj])
        valid_feats = msa_pairing.MSA_FEATURES
        feats = {
            f'{k}_all_seq': v for k, v in all_seq_features.items()
            if k in valid_feats
        }
        return feats

    def process_fasta(self,
                      input_fasta_str: str,
                      struct_feats_at_t_dict,
                      max_templates,
                      ) -> FeatureDict:
        """Creates features."""

        input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)

        all_chain_features = {}

        for desc, seq in zip(input_descs, input_seqs):
            chain_features = self._process_single_chain(
                chain_id=desc,
                sequence=seq,
                description=desc,
            )

            chain_features.update(struct_feats_at_t_dict[desc])

            chain_features = data_pipeline.convert_monomer_features(
                chain_features,
                chain_id=desc
            )
            all_chain_features[desc] = chain_features

        all_chain_features = data_pipeline.add_assembly_features(all_chain_features)

        np_example = merge_features(all_chain_features, max_templates)

        # Pad MSA to avoid zero-sized extra_msa.
        np_example = data_pipeline.pad_msa(np_example, 512)

        return np_example

    def process_fasta_with_atom_pos(
            self,
            input_fasta_str: str,
            all_atom_positions_dict,
            all_atom_mask_dict,
            struct_feats_at_t_dict,
            max_templates,
    ) -> FeatureDict:
        input_seqs, input_descs = parsers.parse_fasta(input_fasta_str)

        all_chain_features = {}

        for desc, seq in zip(input_descs, input_seqs):
            chain_features = self._process_single_chain(
                chain_id=desc,
                sequence=seq,
                description=desc,
            )

            chain_features.update(struct_feats_at_t_dict[desc])

            chain_features = data_pipeline.convert_monomer_features(
                chain_features,
                chain_id=desc
            )

            chain_features["all_atom_positions"] = all_atom_positions_dict[desc]
            chain_features["all_atom_mask"] = all_atom_mask_dict[desc]

            all_chain_features[desc] = chain_features

        all_chain_features = data_pipeline.add_assembly_features(all_chain_features)

        np_example = merge_features(all_chain_features, max_templates)

        # Pad MSA to avoid zero-sized extra_msa.
        np_example = data_pipeline.pad_msa(np_example, 512)

        return np_example
