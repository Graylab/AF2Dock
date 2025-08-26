import numpy as np
import torch
import math
import collections
import pandas as pd
from scipy.spatial.transform import Rotation as R
from biotite.structure import get_residues
from pinder.core.structure.atoms import resn2seq

def fix_resi_auth(resi_auth_split):
    #e.g. 4v88 chain BO
    resi_auth_split_fixed = []
    first_num = True
    last_int = 0
    for idx in range(len(resi_auth_split)):
        res_i = resi_auth_split[idx]
        if res_i == '':
            resi_auth_split_fixed.append(res_i)
        else:
            if first_num:
                resi_auth_split_fixed.append(res_i)
                first_num = False
                last_int = int(res_i)
            else:
                if int(res_i) > last_int:
                    resi_auth_split_fixed.append(res_i)
                    last_int = int(res_i)
    return resi_auth_split_fixed

def truncate_to_resolved(seqres, resi_split):
    resi_resolved_full = [item != '' for item in resi_split]
    l_index = resi_resolved_full.index(True)
    r_index = len(resi_resolved_full) - resi_resolved_full[::-1].index(True) - 1
    return seqres[l_index:r_index + 1], resi_resolved_full[l_index:r_index + 1]

def get_seq_from_atom_array(atom_array, fill_gaps=True):
    pdb_res_num, pdb_res_name = get_residues(atom_array)
    part_resi_split = [str(pdb_res_num[i]) for i in range(len(pdb_res_num))]
    pdb_res_one_letter = resn2seq(pdb_res_name)

    if fill_gaps:
        # fill the gap in res_num with X
        part_resi_split_with_gap = []
        pdb_res_one_letter_with_gap = []
        for idx, (res_name, res_num) in enumerate(zip(pdb_res_one_letter, part_resi_split)):
            if idx == 0:
                part_resi_split_with_gap.append(res_num)
                pdb_res_one_letter_with_gap.append(res_name)
            else:
                res_num_diff = int(res_num) - int(part_resi_split_with_gap[-1])
                if res_num_diff > 1:
                    for i in range(1, res_num_diff):
                        part_resi_split_with_gap.append('')
                        pdb_res_one_letter_with_gap.append('X')
                part_resi_split_with_gap.append(res_num)
                pdb_res_one_letter_with_gap.append(res_name)
        
        part_resi_split = part_resi_split_with_gap
        part_seqres = ''.join(pdb_res_one_letter_with_gap)
    else:
        part_seqres = pdb_res_one_letter

    return part_seqres, part_resi_split

def prefilter(train_index, index_meta, entity_meta, chain_meta, filter_by_date = True):
    # Remove entries with more than 1 consecutive 'X' in the sequence
    entity_meta['part_id'] = entity_meta['entry_id'].astype(str) + '_' + entity_meta['chain'].astype(str)
    rec_id = train_index['holo_R_pdb'].apply(lambda x: x.split('.pdb')[0])
    rec_id = rec_id.apply(lambda x: x.split('_')[0] + '_' + x.split('_')[2])
    rec_order_df = pd.DataFrame({'part_id': rec_id.tolist(), 'position': range(len(rec_id))})
    ordered_rec_data = pd.merge(rec_order_df, entity_meta, on='part_id').sort_values('position').drop('position', axis=1)
    lig_id = train_index['holo_L_pdb'].apply(lambda x: x.split('.pdb')[0])
    lig_id = lig_id.apply(lambda x: x.split('_')[0] + '_' + x.split('_')[2])
    lig_order_df = pd.DataFrame({'part_id': lig_id.tolist(), 'position': range(len(lig_id))})
    ordered_lig_data = pd.merge(lig_order_df, entity_meta, on='part_id').sort_values('position').drop('position', axis=1)
    rec_seq = ordered_rec_data.sequence
    lig_seq = ordered_lig_data.sequence
    rec_has_XX = rec_seq.apply(lambda x: 'XX' in x)
    lig_has_XX = lig_seq.apply(lambda x: 'XX' in x)
    entity_meta.drop('part_id', axis=1, inplace=True)

    # Remove entries with mismatched sequence length and auth resi number
    chain_meta_train = pd.merge(train_index, chain_meta[["id", 'resi_auth_R', 'resi_auth_L']], how='left', on='id')
    rec_seq_len = ordered_rec_data.sequence.apply(lambda x: len(x)).to_numpy()
    lig_seq_len = ordered_lig_data.sequence.apply(lambda x: len(x)).to_numpy()
    rec_resi_auth_len = chain_meta_train['resi_auth_R'].apply(lambda x: len(x.split(','))).to_numpy()
    lig_resi_auth_len = chain_meta_train['resi_auth_L'].apply(lambda x: len(x.split(','))).to_numpy()
    rec_len_mismatched = rec_seq_len != rec_resi_auth_len
    lig_len_mismatched = lig_seq_len != lig_resi_auth_len

    # Remove entries with buried_sasa less than 400 and optionally entries later than 2021-09-30
    train_with_meta = pd.merge(train_index, index_meta[["id",'buried_sasa', 'date']], how="left", on="id")
    low_buried_sasa = train_with_meta['buried_sasa'] < 400
    if filter_by_date:
        train_with_meta['date'] = pd.to_datetime(train_with_meta['date'])
        date_filter = train_with_meta['date'] > '2021-09-30'
    else:
        date_filter = np.array([False] * len(train_index))

    train_index = train_index[~rec_len_mismatched & ~lig_len_mismatched & ~rec_has_XX & ~lig_has_XX & ~low_buried_sasa & ~date_filter]
    train_index = train_index.reset_index(drop=True)

    return train_index

def get_subsampled_train_with_seq_cluster(split_index: pd.DataFrame, split_meta: pd.DataFrame) -> pd.DataFrame:
    # Modified from pinder.data.plot.performance.get_subsampled_train
    train = split_index.query("split == 'train'").reset_index(drop=True)
    train.loc[:, "apo_count"] = train[["apo_R", "apo_L"]].sum(axis=1).astype(int)
    train.loc[:, "pred_count"] = (
        train[["predicted_R", "predicted_L"]].sum(axis=1).astype(int)
    )
    train.loc[:, "apo_pred_available"] = -(
        (train.apo_count > 0) & (train.pred_count > 0)
    ).astype("int")
    train.loc[:, "apo_available"] = -(train.apo_count > 0).astype("int")
    train.loc[:, "pred_available"] = -(train.pred_count > 0).astype("int")
    # split_meta = get_metadata().copy()
    train = pd.merge(
        train,
        split_meta[
            [
                "id",
                "method",
                "num_atom_types",
                "max_var_1",
                "max_var_2",
                "length_resolved_1",
                "length_resolved_2",
                "resolution",
            ]
        ],
        how="inner",
        on="id",
    )
    train.loc[:, "is_xray"] = -np.array([("RAY" in x) for x in train["method"]]).astype(
        int
    )
    train["resolution"] = train["resolution"].astype(float)
    train = train[
        (
            train["num_atom_types"] >= 4
        )  # ensures that proteins contain full spectrum of atoms - this is not always the case for low quality structures
        & (
            train["max_var_1"] < 0.98
        )  # top 1 component of the PCA (of coords) should be lower than 0.98 to ignore low complexity elongated structures
        & (train["max_var_2"] < 0.98)
        & (train["length_resolved_1"] > 40)  # length filter for the resolved residues
        & (train["length_resolved_2"] > 40)
        & (train["resolution"] < 5.0)
    ]

    seq_cluster_R_num = train["seq_cluster_R"].apply(lambda x: int(x))
    seq_cluster_L_num = train["seq_cluster_L"].apply(lambda x: int(x))
    s_seq_cluster = np.minimum(seq_cluster_R_num, seq_cluster_L_num)
    l_seq_cluster = np.maximum(seq_cluster_R_num, seq_cluster_L_num)
    train["cluster_struct_seq"] = train["cluster_id"].astype(str) + "_" + s_seq_cluster.astype(str) + "_" + l_seq_cluster.astype(str)
    
    sample_ids = set(
        train.sort_values(
            ["apo_available", "pred_available", "is_xray", "resolution"], ascending=True
        )
        .groupby("cluster_struct_seq", as_index=False, observed=True)
        .head(2)
        .id
    )
    sampled_train = split_index[split_index["id"].isin(sample_ids)].reset_index(
        drop=True
    )
    return sampled_train

def get_subsampled_train_with_seq_cluster_three_chain(three_body_interactions, train_index, split_meta) -> pd.DataFrame:
    # Modified from pinder.data.plot.performance.get_subsampled_train
    train = three_body_interactions.copy()
    train['rec_pair_id'] =  train['chain_comb_0'].apply(lambda x: x.split(',')[0])
    train['lig_pair_id'] =  train['chain_comb_0'].apply(lambda x: x.split(',')[1].split(':')[0])
    train['lig_pair_part'] =  train['chain_comb_0'].apply(lambda x: x.split(':')[1])
    train['pair_id_1'] =  train['chain_comb_1'].apply(lambda x: x.split(',')[0])
    train['pair_id_2'] =  train['chain_comb_2'].apply(lambda x: x.split(',')[0])

    train = pd.merge(train, train_index[["id",
                                        "apo_R",
                                        "apo_L",
                                        "predicted_R",
                                        "predicted_L",
                                        "seq_cluster_R",
                                        "seq_cluster_L",
                                        "cluster_id_R",
                                        "cluster_id_L"
                                        ]],
                     left_on=['rec_pair_id'], right_on=['id'], 
                     how='left').rename(columns={"id_x": "id",
                                                 "apo_R": "apo_R_R",
                                                 "apo_L": "apo_R_L",
                                                 "predicted_R": "predicted_R_R",
                                                 "predicted_L": "predicted_R_L",
                                                 "seq_cluster_R": "seq_cluster_R_R",
                                                 "seq_cluster_L": "seq_cluster_R_L",
                                                 "cluster_id_R": "cluster_id_0_R",
                                                 "cluster_id_L": "cluster_id_0_L"}).drop(columns=['id_y'])

    train = pd.merge(train, train_index[["id",
                                        "cluster_id_R",
                                        "cluster_id_L"
                                        ]],
                     left_on=['pair_id_1'], right_on=['id'], 
                     how='left').rename(columns={"id_x": "id",
                                                 "cluster_id_R": "cluster_id_1_R",
                                                 "cluster_id_L": "cluster_id_1_L"}).drop(columns=['id_y'])

    train = pd.merge(train, train_index[["id",
                                        "cluster_id_R",
                                        "cluster_id_L"
                                        ]],
                     left_on=['pair_id_2'], right_on=['id'], 
                     how='left').rename(columns={"id_x": "id",
                                                 "cluster_id_R": "cluster_id_2_R",
                                                 "cluster_id_L": "cluster_id_2_L"}).drop(columns=['id_y'])

    train = pd.merge(train, train_index[["id",
                                        "apo_R",
                                        "apo_L",
                                        "predicted_R",
                                        "predicted_L",
                                        "seq_cluster_R",
                                        "seq_cluster_L"
                                        ]],
                     left_on=['lig_pair_id'], right_on=['id'],
                     how='left').rename(columns={"id_x": "id",
                                                 "apo_R": "apo_L_R",
                                                 "apo_L": "apo_L_L",
                                                 "predicted_R": "predicted_L_R",
                                                 "predicted_L": "predicted_L_L",
                                                 "seq_cluster_R": "seq_cluster_L_R",
                                                 "seq_cluster_L": "seq_cluster_L_L",}).drop(columns=['id_y'])

    # Select the appropriate columns based on lig_pair_part
    train['apo_L'] = train.apply(lambda row: row[f'apo_L_{row["lig_pair_part"]}'], axis=1)
    train['predicted_L'] = train.apply(lambda row: row[f'predicted_L_{row["lig_pair_part"]}'], axis=1)
    train['seq_cluster_L'] = train.apply(lambda row: row[f'seq_cluster_L_{row["lig_pair_part"]}'], axis=1)

    train.loc[:, "apo_count"] = train[["apo_R_R", "apo_R_L", "apo_L"]].sum(axis=1).astype(int)
    train.loc[:, "pred_count"] = (
        train[["predicted_R_R", "predicted_R_L", "predicted_L"]].sum(axis=1).astype(int)
    )
    train.loc[:, "apo_pred_available"] = -(
        (train.apo_count > 0) & (train.pred_count > 0)
    ).astype("int")
    train.loc[:, "apo_available"] = -(train.apo_count > 0).astype("int")
    train.loc[:, "pred_available"] = -(train.pred_count > 0).astype("int")
    # split_meta = get_metadata().copy()
    train = pd.merge(
        train,
        split_meta[
            [
                "id",
                "method",
                "resolution",
            ]
        ],
        how="left",
        left_on=['rec_pair_id'], right_on=['id'],
    ).rename(columns={"id_x": "id"}).drop(columns=['id_y'])

    train = pd.merge(
        train,
        split_meta[
            [
                "id",
                "num_atom_types",
                "max_var_1",
                "max_var_2",
                "length_resolved_1",
                "length_resolved_2",
            ]
        ],
        how="left",
        left_on=['rec_pair_id'], right_on=['id'], 
    ).rename(columns={"id_x": "id",
                        "num_atom_types": "num_atom_types_R",
                        "max_var_1": "max_var_R_1",
                        "max_var_2": "max_var_R_2",
                        "length_resolved_1": "length_resolved_R_1",
                        "length_resolved_2": "length_resolved_R_2"
                    }).drop(columns=['id_y'])

    train = pd.merge(
        train,
        split_meta[
            [
                "id",
                "num_atom_types",
                "max_var_1",
                "max_var_2",
                "length_resolved_1",
                "length_resolved_2",
            ]
        ],
        how="left",
        left_on=['lig_pair_id'], right_on=['id'], 
    ).rename(columns={"id_x": "id",
                        "num_atom_types": "num_atom_types_L",
                        "max_var_1": "max_var_L_1",
                        "max_var_2": "max_var_L_2",
                        "length_resolved_1": "length_resolved_L_1",
                        "length_resolved_2": "length_resolved_L_2"
                    }).drop(columns=['id_y'])

    translate_dict = {'R': '1', 'L': '2'}
    train['max_var_L'] = train.apply(lambda row: row[f'max_var_L_{translate_dict[row["lig_pair_part"]]}'], axis=1)
    train['length_resolved_L'] = train.apply(lambda row: row[f'length_resolved_L_{translate_dict[row["lig_pair_part"]]}'], axis=1)

    train.loc[:, "is_xray"] = -np.array([("RAY" in x) for x in train["method"]]).astype(
        int
    )
    train["resolution"] = train["resolution"].astype(float)
    train = train[
        (
            train["num_atom_types_R"] >= 4
        )  # ensures that proteins contain full spectrum of atoms - this is not always the case for low quality structures
        & (
            train["num_atom_types_L"] >= 4
        ) 
        & (
            train["max_var_R_1"] < 0.98
        )  # top 1 component of the PCA (of coords) should be lower than 0.98 to ignore low complexity elongated structures
        & (train["max_var_R_2"] < 0.98)
        & (train["max_var_L"] < 0.98)
        & (train["length_resolved_R_1"] > 40)  # length filter for the resolved residues
        & (train["length_resolved_R_2"] > 40)
        & (train["length_resolved_L"] > 40)
        & (train["resolution"] < 5.0)
    ]

    train["seq_cluster_R_R"] = train["seq_cluster_R_R"].apply(lambda x: int(x))
    train["seq_cluster_R_L"] = train["seq_cluster_R_L"].apply(lambda x: int(x))
    train["seq_cluster_L"] = train["seq_cluster_L"].apply(lambda x: int(x))
    train["cluster_seq"] = train.apply(lambda row: '_'.join(sorted([str(row['seq_cluster_R_R']),
                                                                    str(row['seq_cluster_R_L']),
                                                                    str(row['seq_cluster_L'])])), axis=1)
    
    for idx in range(3):
        for part in ['R', 'L']:
            train[f"cluster_id_{idx}_{part}"] = train[f"cluster_id_{idx}_{part}"].apply(lambda x: '-2' if x.split('_')[1] == 'p' else int(x.split('_')[1]))
    train['cluster_struct'] = train.apply(lambda row: '_'.join(sorted([str(row['cluster_id_0_R']),
                                                                    str(row['cluster_id_0_L']),
                                                                    str(row['cluster_id_1_R']),
                                                                    str(row['cluster_id_1_L']),
                                                                    str(row['cluster_id_2_R']),
                                                                    str(row['cluster_id_2_L'])])), axis=1)

    train["cluster_struct_seq"] = train["cluster_struct"].astype(str) + "_" + train["cluster_seq"].astype(str)

    three_body_interactions['resolution'] = train['resolution']
    sample_ids = set(
        train.sort_values(
            ["apo_available", "pred_available", "is_xray", "resolution"], ascending=True
        )
        .groupby("cluster_struct_seq", as_index=False, observed=True)
        .head(1)
        .id
    )
    sampled_train = three_body_interactions[three_body_interactions["id"].isin(sample_ids)].reset_index(
        drop=True
    )

    sampled_pair_ids = pd.concat([sampled_train['chain_comb_0'].apply(lambda x: x.split(',')[0]),
                              sampled_train['chain_comb_1'].apply(lambda x: x.split(',')[0]),
                              sampled_train['chain_comb_2'].apply(lambda x: x.split(',')[0])
                             ]).unique()
    
    sampled_pair = train_index[train_index['id'].isin(sampled_pair_ids)].reset_index(drop=True)

    return sampled_train, sampled_pair

def get_rigid_body_noise_at_0(tr_sigma, num_struct_batch, rot_prior='uniform', rot_sigma=0.8):
    tr_0 = torch.randn(num_struct_batch, 3) * tr_sigma
    tr_0 = tr_0.numpy()
    if rot_prior == 'gaussian':
        # approximating IGSO3 at small angle sigma
        rot_axis = torch.rand(num_struct_batch, 3)
        rot_axis = rot_axis / torch.linalg.norm(rot_axis, dim=-1, keepdim=True)
        rot_angle = torch.abs(torch.randn(num_struct_batch, 1) * rot_sigma) % math.pi
        rot_0 = rot_angle * rot_axis
        rot_0 = rot_0.numpy()
    elif rot_prior == 'uniform':
        rot_0 = R.random(num_struct_batch).as_rotvec().astype(np.float32)
    else:
        raise ValueError(f"Unknown rotation prior: {rot_prior}")
    return tr_0, rot_0

def apply_rigid_body_transform_atom37(all_atom_positions, all_atom_mask, ca_idx, tr, rot):
    com = np.mean(all_atom_positions[..., ca_idx, :], axis=-2)
    rot_t_mat = R.from_rotvec(rot).as_matrix().astype(np.float32)
    rot_t_mat = rot_t_mat[:, None, ...]
    all_atom_positions = all_atom_positions - com[:, None, None, :]
    all_atom_positions = np.einsum('...ij,...kj->...ik', all_atom_positions, rot_t_mat)
    all_atom_positions = all_atom_positions + com[:, None, None, :] + tr[:, None, None, ...]
    all_atom_positions = all_atom_positions * all_atom_mask[..., None]
    return all_atom_positions

def get_esm_embeddings(seq, client):
    from esm.sdk.api import ESMProtein, LogitsConfig
    
    protein = ESMProtein(sequence=seq)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    return logits_output.embeddings[0]

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
        part_chain_end_idx = np.insert(np.cumsum(part_chain_lengths)[:-1], 0, 0)
        part_chain_offset = [index_offset * idx + part_chain_end_idx[idx] for idx in range(len(part_chain_lengths))]
        part_residue_index_offset = np.concatenate([np.ones(length) * offset for length, offset in zip(part_chain_lengths, part_chain_offset)])
        residue_index_offset_list.append(part_residue_index_offset)
    
    data['asym_id'] =  np.concatenate(new_asym_id, axis=0)
    data['sym_id'] =  np.concatenate(new_sym_id, axis=0)
    data['entity_id'] =  np.concatenate(new_entity_id, axis=0)
    
    residue_index_offset = np.concatenate(residue_index_offset_list, axis=0)
    data['residue_index'] = data['residue_index'] + residue_index_offset

    return data

def add_meta_attributes(data_index, entity_meta, chain_meta, supp_meta, metadata):
    entity_meta['part_id'] = entity_meta['entry_id'].astype(str) + '_' + entity_meta['chain'].astype(str)
    data_index['holo_R_id'] = data_index['holo_R_pdb'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[2])
    data_index['holo_L_id'] = data_index['holo_L_pdb'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[2])
    data_index = data_index.merge(entity_meta[['part_id', 'sequence']], 
                                            left_on='holo_R_id',
                                            right_on='part_id',
                                            how='left').rename(columns={'sequence': 'seq_R'})
    data_index = data_index.merge(entity_meta[['part_id', 'sequence']],
                                            left_on='holo_L_id',
                                            right_on='part_id',
                                            how='left').rename(columns={'sequence': 'seq_L'})
    data_index = data_index.merge(chain_meta[['id', 'resi_auth_R', 'resi_auth_L']],
                                            on='id',
                                            how='left')
    data_index = data_index.merge(supp_meta[['id', 'chain_1_residues', 'chain_2_residues']],
                                            on='id',
                                            how='left').rename(columns={'chain_1_residues': 'chain_R_residues',
                                                                        'chain_2_residues': 'chain_L_residues'})
    data_index = data_index.merge(metadata[['id', 'resolution']],
                                            on='id',
                                            how='left')
    data_index = data_index.drop(columns=['part_id_x', 'part_id_y', 'holo_R_id', 'holo_L_id'])

    return data_index
