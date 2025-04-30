import numpy as np
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

def prefilter(train_index, index_meta, entity_meta, chain_meta):
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

    # Remove entries with buried_sasa less than 400
    train_with_meta = pd.merge(train_index, index_meta[["id",'buried_sasa']], how="left", on="id")
    low_buried_sasa = train_with_meta['buried_sasa'] < 400

    train_index = train_index[~rec_len_mismatched & ~lig_len_mismatched & ~rec_has_XX & ~lig_has_XX & ~low_buried_sasa]
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

def apply_rigid_body_transform_atom37(all_atom_positions, all_atom_mask, ca_idx, tr, rot):
    com = np.mean(all_atom_positions[..., ca_idx, :], axis=-2)
    rot_t_mat = R.from_rotvec(rot).as_matrix()
    all_atom_positions = all_atom_positions - com
    all_atom_positions = np.einsum('...ij,kj->...ik', all_atom_positions, rot_t_mat).astype(np.float32)
    all_atom_positions = all_atom_positions + com + tr
    all_atom_positions = all_atom_positions * all_atom_mask[..., None]
    return all_atom_positions
