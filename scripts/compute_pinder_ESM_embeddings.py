import logging
import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from tqdm import tqdm
from pinder.core import PinderSystem, get_index, get_supplementary_data, get_metadata

from esm.models.esmc import ESMC

import sys
sys.path.append(str(Path(__file__).parent.parent))
from AF2Dock.utils import data_utils

logger = logging.getLogger(__name__)

def add_args(parser):
    parser.add_argument(
        '-o',
        '--outdir',
        type=Path,
        help=""
    )
    parser.add_argument(
        '--skip-train-val',
        default=False,
        action='store_true',
        help=""
    )
    parser.add_argument(
        '--full-train',
        default=False,
        action='store_true',
        help=""
    )
    parser.add_argument(
        '--test-set',
        default='pinder_af2',
        choices=['pinder_af2', 'pinder_xl', 'pinder_s'],
        help="",
    )
    parser.add_argument(
        '-c',
        '--pinder-entity-seq-cluster-pkl',
        default=None,
        type=Path,
        help=""
    )
    parser.add_argument(
        '-t',
        '--three_body_interactions_pkl',
        default=None,
        type=Path,
        help=""
    )
    parser.add_argument(
        '--not-filter-train-by-date',
        default=False,
        action='store_true',
    )

def main(args):
    client = ESMC.from_pretrained("esmc_600m").to("cuda")
    entity_meta = get_supplementary_data("entity_metadata")
    chain_meta = get_supplementary_data("chain_metadata")
    full_index = get_index()
    metadata = get_metadata()

    indexes_to_compute = []
    if not args.skip_train_val:
        train_index = full_index.query("split == 'train'").copy().reset_index(drop=True)
        if not args.full_train:
            train_index = data_utils.prefilter(train_index,
                                               metadata,
                                               entity_meta,
                                               chain_meta,
                                               not args.not_filter_train_by_date)
            
            entity_seq_cluster = pd.read_pickle(args.pinder_entity_seq_cluster_pkl)
            train_index['holo_R_id'] = train_index['holo_R_pdb'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[2])
            train_index['holo_L_id'] = train_index['holo_L_pdb'].apply(lambda x: x.split('_')[0] + '_' + x.split('_')[2])
            train_index = train_index.merge(entity_seq_cluster[['part_id', 'seq_cluster_40']], 
                                            left_on='holo_R_id',
                                            right_on='part_id',
                                            how='left').rename(columns={'seq_cluster_40': 'seq_cluster_R'})
            train_index = train_index.merge(entity_seq_cluster[['part_id', 'seq_cluster_40']], 
                                            left_on='holo_L_id',
                                            right_on='part_id',
                                            how='left').rename(columns={'seq_cluster_40': 'seq_cluster_L'})
            train_index = train_index.drop(columns=['part_id_x', 'part_id_y', 'holo_R_id', 'holo_L_id'])
            two_chain_train_index = data_utils.get_subsampled_train_with_seq_cluster(train_index, metadata)

            indexes_to_compute.append(two_chain_train_index)

            three_body_interactions = pd.read_pickle(args.three_body_interactions_pkl)
            _, three_chain_train_pair_index = data_utils.get_subsampled_train_with_seq_cluster_three_chain(
                three_body_interactions,
                train_index,
                metadata
                )
            
            indexes_to_compute.append(three_chain_train_pair_index)
        else:
            indexes_to_compute.append(train_index)
        indexes_to_compute.append(full_index.query("split == 'val'").copy().reset_index(drop=True))
    
    test_index = full_index.query(f'{args.test_set} == True').copy().reset_index(drop=True)
    indexes_to_compute.append(test_index)

    entity_meta['part_id'] = entity_meta['entry_id'].astype(str) + '_' + entity_meta['chain'].astype(str)
    indexes_to_compute_with_addi_meta = []
    for data_index in indexes_to_compute:
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
        data_index = data_index.merge(chain_meta[['id','resi_auth_R', 'resi_auth_L']],
                                                on='id',
                                                how='left')
        data_index = data_index.drop(columns=['part_id_x', 'part_id_y', 'holo_R_id', 'holo_L_id'])
        indexes_to_compute_with_addi_meta.append(data_index)
    indexes_to_compute = indexes_to_compute_with_addi_meta

    sequences = {}
    for data_index in tqdm(indexes_to_compute):
        for idx in tqdm(range(len(data_index))):
            index_entry = data_index.iloc[idx]
            struct_id = index_entry['id']
            struct_seq = {}
            for part in ['rec', 'lig']:
                abbr = 'R' if part == 'rec' else 'L'
                part_id = index_entry[f'holo_{abbr}_pdb'].split('.pdb')[0]
                part_seqres = index_entry[f'seq_{abbr}']
                part_resi_auth = index_entry[f"resi_auth_{abbr}"]
                part_resi_auth_split = part_resi_auth.split(',')
                part_resi_split = part_resi_auth_split
                if len(part_resi_split) != len(part_seqres):
                    # e.g. 8hco chain G, fall back to sequence in structure
                    ps = PinderSystem(struct_id)
                    part_seqres, part_resi_split = data_utils.get_seq_from_atom_array(getattr(ps, f'native_{abbr}').atom_array)
                assert len(part_resi_split) == len(part_seqres), "Length mismatch between resi and seq"
                part_seq, _ = data_utils.truncate_to_resolved(part_seqres, part_resi_split)
                part_esm_embeddings = data_utils.get_esm_embeddings(part_seq, client)
                part_esm_embeddings = part_esm_embeddings.cpu().numpy()
                np.save(args.outdir / f"{part_id}.npy", part_esm_embeddings)
                struct_seq[part] = part_seq
            sequences[struct_id] = struct_seq
    with open(args.outdir / 'sequences.pkl', 'wb') as f:
        pickle.dump(sequences, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
