import logging
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from pinder.core import PinderSystem, get_index, get_supplementary_data, get_metadata
from pinder.data.plot.performance import get_subsampled_train

from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

import sys
sys.path.append(str(Path(__file__).parent.parent))
from AF2Dock.utils import utils

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

def get_esm_embeddings(seq, client):
    protein = ESMProtein(sequence=seq)
    protein_tensor = client.encode(protein)
    logits_output = client.logits(
    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
    )
    return logits_output.embeddings[0]

def main(args):
    client = ESMC.from_pretrained("esmc_600m").to("cuda")
    entity_meta = get_supplementary_data("entity_metadata")
    chain_meta = get_supplementary_data("chain_metadata")
    full_index = get_index()

    indexes_to_compute = []
    if not args.skip_train_val:
        if not args.full_train:
            train_indexes = get_subsampled_train(full_index)
            train_indexes = utils.further_filter(train_indexes,
                                                 get_metadata(),
                                                 entity_meta,
                                                 chain_meta)
            indexes_to_compute.append(train_indexes)
        else:
            train_indexes = full_index.query("split == 'train'")
            indexes_to_compute.append(train_indexes)
        indexes_to_compute.append(full_index.query("split == 'val'"))
    
    test_indexes = full_index.query(f'{args.test_set} == True')
    indexes_to_compute.append(test_indexes)

    for data_index in tqdm(indexes_to_compute):
        for idx in tqdm(range(len(data_index))):
            index_entry = data_index.iloc[idx]
            struct_id = index_entry['id']
            chain_meta_i = chain_meta.query(f"id == '{struct_id}'").iloc[0]
            for part in ['rec', 'lig']:
                abbr = 'R' if part == 'rec' else 'L'
                part_id = index_entry[f'holo_{abbr}_pdb'].split('.pdb')[0]
                part_seqres = entity_meta.query(f"entry_id == '{part_id.split('_')[0]}' and chain == '{part_id.split('_')[2]}'").sequence.values[0]
                part_resi_auth = chain_meta_i[f"resi_auth_{abbr}"]
                part_resi_auth_split = part_resi_auth.split(',')
                if len(part_resi_auth_split) != len(part_seqres):
                    part_resi_auth_split = utils.fix_resi_auth(part_resi_auth_split)
                    if len(part_resi_auth_split) != len(part_seqres):
                        # e.g. 8hco chain G, fall back to sequence in structure
                        ps = PinderSystem(struct_id)
                        part_seqres, part_resi_auth_split = utils.get_seq_from_atom_array(getattr(ps, f'native_{abbr}').atom_array)
                    assert len(part_resi_auth_split) == len(part_seqres)
                part_seq, _ = utils.truncate_to_resolved(part_seqres, part_resi_auth_split)
                part_esm_embeddings = get_esm_embeddings(part_seq, client)
                part_esm_embeddings = part_esm_embeddings.cpu().numpy()
                np.save(args.outdir / f"{part_id}.npy", part_esm_embeddings)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
