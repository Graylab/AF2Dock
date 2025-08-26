import argparse
from pathlib import Path

import pandas as pd
import tqdm
import networkx as nx
from pinder.core import get_index, get_supplementary_data, get_metadata

import sys
sys.path.append(str(Path(__file__).parent.parent))
from AF2Dock.utils import data_utils

def add_args(parser):
    parser.add_argument(
        '-o',
        '--outdir',
        type=Path,
        default=Path.cwd(),
        help=""
    )
    parser.add_argument(
        '--not-filter-train-by-date',
        default=False,
        action='store_true',
    )

def find_three_body_interactions(index):
    """
    Find three-body interactions within each PDB.
    A three-body interaction exists when three chains A, B, C all interact with each other
    (A-B, B-C, A-C interactions all exist).
    """
    
    # Group by PDB ID to analyze interactions within each structure
    three_body_interactions = []

    for pdb_id, group in tqdm.tqdm(index.groupby('pdb_id', observed=True)):

        G = nx.Graph()

        # Build the interaction graph
        for _, row in group.iterrows():
            entry_id = row['id']
            
            rec_node = row['id'].split('--')[0]
            lig_node = row['id'].split('--')[1]
            
            if rec_node not in G:
                G.add_node(rec_node, ori_pair = [entry_id])
            else:
                G.nodes[rec_node]['ori_pair'].append(entry_id)
            if lig_node not in G:
                G.add_node(lig_node, ori_pair = [entry_id])
            else:
                G.nodes[lig_node]['ori_pair'].append(entry_id)

            # Add edge between interacting chains
            G.add_edge(rec_node, lig_node)

        # Find all triangles (3-cliques) in the graph
        triangles = [clique for clique in nx.enumerate_all_cliques(G) if len(clique) == 3]
        
        # Check all possible three-chain combinations
        for triangle in triangles:
            
            all_node_chain_pairs = []
            for node in triangle:
                all_node_chain_pairs.extend(G.nodes[node]['ori_pair'])
            all_node_chain_pairs = set(all_node_chain_pairs)
            chain_triplet = []
            for chain_pair in all_node_chain_pairs:
                if all(chain in triangle for chain in chain_pair.split('--')):
                    chain_triplet.append(chain_pair)
            chain_triplet = sorted(list(set(chain_triplet)))
            assert len(chain_triplet) == 3

            unique_chains = sorted(triangle)
            entry_id = '--'.join(unique_chains)

            indi_chain_info = []
            for chain in unique_chains:
                for chain_pair_include in chain_triplet:
                    if chain in chain_pair_include:
                        break
                for chain_pair_exclude in chain_triplet:
                    if chain not in chain_pair_exclude:
                        break
                chain_pair_chains = chain_pair_include.split('--')
                chain_part = ['R', 'L'][chain_pair_chains.index(chain)]
                indi_chain_info.append(f'{chain_pair_exclude},{chain_pair_include}:{chain_part}')

            # Found a three-body interaction
            three_body_result = {
                'pdb_id': pdb_id,
                'id': entry_id,
                'chain_comb_0': indi_chain_info[0],
                'chain_comb_1': indi_chain_info[1],
                'chain_comb_2': indi_chain_info[2],
            }
            three_body_interactions.append(three_body_result)

    three_body_interactions = pd.DataFrame(three_body_interactions)
    
    return three_body_interactions

def main(args):
    index = get_index()
    entity_meta = get_supplementary_data("entity_metadata")
    chain_meta = get_supplementary_data("chain_metadata")
    metadata = get_metadata()
    
    train_index = index.query("split == 'train'").reset_index(drop=True)
    train_index = data_utils.prefilter(train_index,
                                        metadata,
                                        entity_meta,
                                        chain_meta,
                                        not args.not_filter_train_by_date)
    three_body_interactions = find_three_body_interactions(train_index)
    three_body_interactions.to_pickle(args.outdir / 'train_three_body_interactions.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_args(parser)
    main(parser.parse_args())
