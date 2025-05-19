# MIT License

# Copyright (c) 2025 Lab of Dr. Roland Dunbrack

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ipsae.py
# script for calculating the ipSAE score for scoring pairwise protein-protein interactions in AlphaFold2 and AlphaFold3 models
# https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1

# Also calculates:
#    pDockQ: Bryant, Pozotti, and Eloffson. https://www.nature.com/articles/s41467-022-28865-w
#    pDockQ2: Zhu, Shenoy, Kundrotas, Elofsson. https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714
#    LIS: Kim, Hu, Comjean, Rodiger, Mohr, Perrimon. https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1

# Roland Dunbrack
# Fox Chase Cancer Center
# version 3
# April 6, 2025
# MIT license: script can be modified and redistributed for non-commercial and commercial use, as long as this information is reproduced.

# includes support for Boltz1 structures and structures with nucleic acids

import math
import numpy as np
import pandas as pd

# Define the ptm and d0 functions
def ptm_func(x,d0):
    return 1.0/(1+(x/d0)**2.0)  
ptm_func_vec=np.vectorize(ptm_func)  # vector version

# Define the d0 functions for numbers and arrays; minimum value = 1.0; from Yang and Skolnick, PROTEINS: Structure, Function, and Bioinformatics 57:702–710 (2004)
def calc_d0(L,pair_type):
    L=float(L)
    if L<27: L=27
    min_value=1.0
    if pair_type=='nucleic_acid': min_value=2.0
    d0=1.24*(L-15)**(1.0/3.0) - 1.8
    return max(min_value, d0)

def calc_d0_array(L,pair_type):
    # Convert L to a NumPy array if it isn't already one (enables flexibility in input types)
    L = np.array(L, dtype=float)
    L = np.maximum(27,L)
    min_value=1.0

    if pair_type=='nucleic_acid': min_value=2.0

    # Calculate d0 using the vectorized operation
    return np.maximum(min_value, 1.24 * (L - 15) ** (1.0/3.0) - 1.8)

# Initializes a nested dictionary with all values set to 0
def init_chainpairdict_zeros(chainlist):
    return {chain1: {chain2: 0 for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}

# Initializes a nested dictionary with NumPy arrays of zeros of a specified size
def init_chainpairdict_npzeros(chainlist, arraysize):
    return {chain1: {chain2: np.zeros(arraysize) for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}

# Initializes a nested dictionary with empty sets.
def init_chainpairdict_set(chainlist):
    return {chain1: {chain2: set() for chain2 in chainlist if chain1 != chain2} for chain1 in chainlist}


def classify_chains(chains, residue_types):
    nuc_residue_set = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
    chain_types = {}
    
    # Get unique chains and iterate over them
    unique_chains = np.unique(chains)
    for chain in unique_chains:
        # Find indices where the current chain is located
        indices = np.where(chains == chain)[0]
        # Get the residues for these indices
        chain_residues = residue_types[indices]
        # Count nucleic acid residues
        nuc_count = sum(residue in nuc_residue_set for residue in chain_residues)
        
        # Determine if the chain is a nucleic acid or protein
        chain_types[chain] = 'nucleic_acid' if nuc_count > 0 else 'protein'
    
    return chain_types

def get_info_from_atom_array(atom_array):
    residues = []
    cb_residues = []
    chains = []
    for atom_num, atom in enumerate(atom_array):
        if atom.atom_name == "CA" or "C1" in atom.atom_name:
            residues.append({
                'atom_num': atom_num,
                'coor': atom.coord,
                'res': atom.res_name,
                'chainid': atom.chain_id,
                'resnum': atom.res_id,
                'residue': f"{atom.res_name:3}   {atom.chain_id:3} {atom.res_id:4}"
            })
            chains.append(atom.chain_id)

        if atom.atom_name == "CB" or "C3" in atom.atom_name or (atom.res_name=="GLY" and atom.atom_name=="CA"):
            cb_residues.append({
                'atom_num': atom_num,
                'coor': atom.coord,
                'res': atom.res_name,
                'chainid': atom.chain_id,
                'resnum': atom.res_id,
                'residue': f"{atom.res_name:3}   {atom.chain_id:3} {atom.res_id:4}"
            })

    return residues, cb_residues, np.array(chains)

def compute_metrices(data_id, atom_array, pae_matrix, plddt, iptm_af,
                     pae_cutoff=10, dist_cutoff=10):
    #numres, distances, residues, chains, unique_chains, chain_pair_type
    residues, cb_residues, chains = get_info_from_atom_array(atom_array)
    numres = len(residues)
    residue_types=np.array([res['res'] for res in residues])
    coordinates = np.array([res['coor']       for res in cb_residues])

    unique_chains = np.unique(chains)
    
    # chain types (nucleic acid (NA) or protein) and chain_pair_types ('nucleic_acid' if either chain is NA) for d0 calculation
    # arbitrarily setting d0 to 2.0 for NA/protein or NA/NA chain pairs (approximately 21 base pairs)
    chain_dict = classify_chains(chains, residue_types)
    chain_pair_type = init_chainpairdict_zeros(unique_chains)
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1==chain2: continue
            if chain_dict[chain1] == 'nucleic_acid' or chain_dict[chain2] == 'nucleic_acid':
                chain_pair_type[chain1][chain2]='nucleic_acid'
            else:
                chain_pair_type[chain1][chain2]='protein'
    distances = np.sqrt(((coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :])**2).sum(axis=2))
    
    cb_plddt = plddt

    iptm_d0chn_byres  = init_chainpairdict_npzeros(unique_chains, numres)
    ipsae_d0chn_byres = init_chainpairdict_npzeros(unique_chains, numres)
    ipsae_d0dom_byres = init_chainpairdict_npzeros(unique_chains, numres)
    ipsae_d0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

    iptm_d0chn_asym   = init_chainpairdict_zeros(unique_chains)
    ipsae_d0chn_asym  = init_chainpairdict_zeros(unique_chains)
    ipsae_d0dom_asym  = init_chainpairdict_zeros(unique_chains)
    ipsae_d0res_asym  = init_chainpairdict_zeros(unique_chains)

    iptm_d0chn_max    = init_chainpairdict_zeros(unique_chains)
    ipsae_d0chn_max   = init_chainpairdict_zeros(unique_chains)
    ipsae_d0dom_max   = init_chainpairdict_zeros(unique_chains)
    ipsae_d0res_max   = init_chainpairdict_zeros(unique_chains)

    iptm_d0chn_asymres   = init_chainpairdict_zeros(unique_chains)
    ipsae_d0chn_asymres  = init_chainpairdict_zeros(unique_chains)
    ipsae_d0dom_asymres  = init_chainpairdict_zeros(unique_chains)
    ipsae_d0res_asymres  = init_chainpairdict_zeros(unique_chains)

    iptm_d0chn_maxres    = init_chainpairdict_zeros(unique_chains)
    ipsae_d0chn_maxres   = init_chainpairdict_zeros(unique_chains)
    ipsae_d0dom_maxres   = init_chainpairdict_zeros(unique_chains)
    ipsae_d0res_maxres   = init_chainpairdict_zeros(unique_chains)

    n0chn       = init_chainpairdict_zeros(unique_chains)
    n0dom       = init_chainpairdict_zeros(unique_chains)
    n0dom_max   = init_chainpairdict_zeros(unique_chains)
    n0res       = init_chainpairdict_zeros(unique_chains)
    n0res_max   = init_chainpairdict_zeros(unique_chains)
    n0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

    d0chn       = init_chainpairdict_zeros(unique_chains)
    d0dom       = init_chainpairdict_zeros(unique_chains)
    d0dom_max   = init_chainpairdict_zeros(unique_chains)
    d0res       = init_chainpairdict_zeros(unique_chains)
    d0res_max   = init_chainpairdict_zeros(unique_chains)
    d0res_byres = init_chainpairdict_npzeros(unique_chains, numres)

    valid_pair_counts           = init_chainpairdict_zeros(unique_chains)
    dist_valid_pair_counts      = init_chainpairdict_zeros(unique_chains)
    unique_residues_chain1      = init_chainpairdict_set(unique_chains)
    unique_residues_chain2      = init_chainpairdict_set(unique_chains)
    dist_unique_residues_chain1 = init_chainpairdict_set(unique_chains)
    dist_unique_residues_chain2 = init_chainpairdict_set(unique_chains)
    pDockQ_unique_residues      = init_chainpairdict_set(unique_chains)

    pDockQ  = init_chainpairdict_zeros(unique_chains)
    pDockQ2 = init_chainpairdict_zeros(unique_chains)
    LIS     = init_chainpairdict_zeros(unique_chains)

    # pDockQ
    pDockQ_cutoff=8.0

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:    continue
            npairs=0
            for i in range(numres):
                if chains[i] != chain1:   continue
                valid_pairs = (chains==chain2) & (distances[i] <= pDockQ_cutoff)
                npairs += np.sum(valid_pairs)
                if valid_pairs.any():
                    pDockQ_unique_residues[chain1][chain2].add(i)
                    chain2residues=np.where(valid_pairs)[0]

                    for residue in chain2residues:
                        pDockQ_unique_residues[chain1][chain2].add(residue)
                        
            if npairs>0:
                nres=len(list(pDockQ_unique_residues[chain1][chain2]))
                mean_plddt= cb_plddt[ list(pDockQ_unique_residues[chain1][chain2])].mean()
                x=mean_plddt*math.log10(npairs)
                pDockQ[chain1][chain2]= 0.724 / (1 + math.exp(-0.052*(x-152.611)))+0.018
            else:
                mean_plddt=0.0
                x=0.0
                pDockQ[chain1][chain2]=0.0
                nres=0
            
    # pDockQ2

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            npairs=0
            sum=0.0
            for i in range(numres):
                if chains[i] != chain1:
                    continue
                valid_pairs = (chains==chain2) & (distances[i] <= pDockQ_cutoff)
                if valid_pairs.any():
                    npairs += np.sum(valid_pairs)
                    pae_list=pae_matrix[i][valid_pairs]
                    pae_list_ptm=ptm_func_vec(pae_list,10.0)
                    sum += pae_list_ptm.sum()
                
            if npairs>0:
                nres=len(list(pDockQ_unique_residues[chain1][chain2]))
                mean_plddt= cb_plddt[ list(pDockQ_unique_residues[chain1][chain2])].mean()
                mean_ptm = sum/npairs
                x=mean_plddt*mean_ptm
                pDockQ2[chain1][chain2]= 1.31 / (1 + math.exp(-0.075*(x-84.733)))+0.005
            else:
                mean_plddt=0.0
                x=0.0
                nres=0
                pDockQ2[chain1][chain2]=0.0
            
    # LIS

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1==chain2: continue
            
            mask = (chains[:, None] == chain1) & (chains[None, :] == chain2)  # Select residues for (chain1, chain2)
            selected_pae = pae_matrix[mask]  # Get PAE values for this pair
            
            if selected_pae.size > 0:  # Ensure we have values
                valid_pae = selected_pae[selected_pae <= 12]  # Apply the threshold
                if valid_pae.size > 0:
                    scores = (12 - valid_pae) / 12  # Compute scores
                    avg_score = np.mean(scores)  # Average score for (chain1, chain2)
                    LIS[chain1][chain2] = avg_score
                else:
                    LIS[chain1][chain2] = 0.0  # No valid values
            else:
                LIS[chain1][chain2]=0.0

    # calculate ipTM/ipSAE with and without PAE cutoff

    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            n0chn[chain1][chain2]=np.sum( chains==chain1) + np.sum(chains==chain2) # total number of residues in chain1 and chain2
            d0chn[chain1][chain2]=calc_d0(n0chn[chain1][chain2], chain_pair_type[chain1][chain2])
            ptm_matrix_d0chn=np.zeros((numres,numres))
            ptm_matrix_d0chn=ptm_func_vec(pae_matrix,d0chn[chain1][chain2])

            valid_pairs_iptm = (chains == chain2)
            valid_pairs_matrix = (chains == chain2) & (pae_matrix < pae_cutoff)

            for i in range(numres):


                if chains[i] != chain1:
                    continue

                valid_pairs_ipsae = valid_pairs_matrix[i]  # row for residue i of chain1
                iptm_d0chn_byres[chain1][chain2][i] =  ptm_matrix_d0chn[i, valid_pairs_iptm].mean() if valid_pairs_iptm.any() else 0.0
                ipsae_d0chn_byres[chain1][chain2][i] = ptm_matrix_d0chn[i, valid_pairs_ipsae].mean() if valid_pairs_ipsae.any() else 0.0

                # Track unique residues contributing to the IPSAE for chain1,chain2
                valid_pair_counts[chain1][chain2] += np.sum(valid_pairs_ipsae)
                if valid_pairs_ipsae.any():
                    iresnum=residues[i]['resnum']
                    unique_residues_chain1[chain1][chain2].add(iresnum)
                    for j in np.where(valid_pairs_ipsae)[0]:
                        jresnum=residues[j]['resnum']
                        unique_residues_chain2[chain1][chain2].add(jresnum)
                        
                # Track unique residues contributing to iptm in interface
                valid_pairs = (chains == chain2) & (pae_matrix[i] < pae_cutoff) & (distances[i] < dist_cutoff)
                dist_valid_pair_counts[chain1][chain2] += np.sum(valid_pairs)

                # Track unique residues contributing to the IPTM
                if valid_pairs.any():
                    iresnum=residues[i]['resnum']
                    dist_unique_residues_chain1[chain1][chain2].add(iresnum)
                    for j in np.where(valid_pairs)[0]:
                        jresnum=residues[j]['resnum']
                        dist_unique_residues_chain2[chain1][chain2].add(jresnum)

    # OUT2.write("i   AlignChn ScoredChain  AlignResNum  AlignResType  AlignRespLDDT      n0chn  n0dom  n0res    d0chn     d0dom     d0res   ipTM_pae  ipSAE_d0chn ipSAE_d0dom    ipSAE \n")
    metrics_byres_columns = [
        'i', 'AlignChn', 'ScoredChain', 'AlignResNum', 'AlignResType',
        'AlignRespLDDT', 'n0chn', 'n0dom', 'n0res', 'd0chn', 'd0dom',
        'd0res', 'ipTM_pae', 'ipSAE_d0chn', 'ipSAE_d0dom', 'ipSAE'
    ]
    metrics_byres_data = []
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue
            residues_1 = len(unique_residues_chain1[chain1][chain2])
            residues_2 = len(unique_residues_chain2[chain1][chain2])
            n0dom[chain1][chain2] = residues_1+residues_2
            d0dom[chain1][chain2] = calc_d0(n0dom[chain1][chain2], chain_pair_type[chain1][chain2])

            ptm_matrix_d0dom = np.zeros((numres,numres))
            ptm_matrix_d0dom = ptm_func_vec(pae_matrix,d0dom[chain1][chain2])

            valid_pairs_matrix = (chains == chain2) & (pae_matrix < pae_cutoff)

            # Assuming valid_pairs_matrix is already defined
            n0res_byres_all = np.sum(valid_pairs_matrix, axis=1)
            d0res_byres_all = calc_d0_array(n0res_byres_all, chain_pair_type[chain1][chain2])

            n0res_byres[chain1][chain2] = n0res_byres_all
            d0res_byres[chain1][chain2] = d0res_byres_all
            
            for i in range(numres):
                if chains[i] != chain1:
                    continue
                valid_pairs = valid_pairs_matrix[i]
                ipsae_d0dom_byres[chain1][chain2][i] = ptm_matrix_d0dom[i, valid_pairs].mean() if valid_pairs.any() else 0.0

                ptm_row_d0res=np.zeros((numres))
                ptm_row_d0res=ptm_func_vec(pae_matrix[i], d0res_byres[chain1][chain2][i])
                ipsae_d0res_byres[chain1][chain2][i] = ptm_row_d0res[valid_pairs].mean() if valid_pairs.any() else 0.0

                metrics_byres_data.append([
                    i+1, chain1, chain2, residues[i]['resnum'], residues[i]['res'],
                    plddt[i], int(n0chn[chain1][chain2]), int(n0dom[chain1][chain2]),
                    int(n0res_byres[chain1][chain2][i]), d0chn[chain1][chain2],
                    d0dom[chain1][chain2], d0res_byres[chain1][chain2][i],
                    iptm_d0chn_byres[chain1][chain2][i], ipsae_d0chn_byres[chain1][chain2][i],
                    ipsae_d0dom_byres[chain1][chain2][i], ipsae_d0res_byres[chain1][chain2][i]
                ])
    metrics_byres = pd.DataFrame(metrics_byres_data, columns=metrics_byres_columns)
    
    # Compute interchain ipTM and ipSAE for each chain pair
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 == chain2:
                continue

            interchain_values = iptm_d0chn_byres[chain1][chain2]
            max_index = np.argmax(interchain_values)
            iptm_d0chn_asym[chain1][chain2] = interchain_values[max_index]
            iptm_d0chn_asymres[chain1][chain2] = residues[max_index]['residue'] if max_index is not None else "None"

            interchain_values = ipsae_d0chn_byres[chain1][chain2]
            max_index = np.argmax(interchain_values)
            ipsae_d0chn_asym[chain1][chain2] = interchain_values[max_index]
            ipsae_d0chn_asymres[chain1][chain2] = residues[max_index]['residue'] if max_index is not None else "None"

            interchain_values = ipsae_d0dom_byres[chain1][chain2]
            max_index = np.argmax(interchain_values)
            ipsae_d0dom_asym[chain1][chain2] = interchain_values[max_index]
            ipsae_d0dom_asymres[chain1][chain2] = residues[max_index]['residue'] if max_index is not None else "None"

            interchain_values = ipsae_d0res_byres[chain1][chain2]
            max_index = np.argmax(interchain_values)
            ipsae_d0res_asym[chain1][chain2] = interchain_values[max_index]
            ipsae_d0res_asymres[chain1][chain2] = residues[max_index]['residue'] if max_index is not None else "None"
            n0res[chain1][chain2]=n0res_byres[chain1][chain2][max_index]
            d0res[chain1][chain2]=d0res_byres[chain1][chain2][max_index]

            # pick maximum value for each chain pair for each iptm/ipsae type
            if chain1 > chain2:
                maxvalue=max(iptm_d0chn_asym[chain1][chain2], iptm_d0chn_asym[chain2][chain1])
                if maxvalue==iptm_d0chn_asym[chain1][chain2]: maxres=iptm_d0chn_asymres[chain1][chain2]
                else: maxres=iptm_d0chn_asymres[chain2][chain1]
                iptm_d0chn_max[chain1][chain2]=maxvalue
                iptm_d0chn_maxres[chain1][chain2]=maxres
                iptm_d0chn_max[chain2][chain1]=maxvalue
                iptm_d0chn_maxres[chain2][chain1]=maxres

                maxvalue=max(ipsae_d0chn_asym[chain1][chain2], ipsae_d0chn_asym[chain2][chain1])
                if maxvalue==ipsae_d0chn_asym[chain1][chain2]: maxres=ipsae_d0chn_asymres[chain1][chain2]
                else: maxres=ipsae_d0chn_asymres[chain2][chain1]
                ipsae_d0chn_max[chain1][chain2]=maxvalue
                ipsae_d0chn_maxres[chain1][chain2]=maxres
                ipsae_d0chn_max[chain2][chain1]=maxvalue
                ipsae_d0chn_maxres[chain2][chain1]=maxres

                maxvalue=max(ipsae_d0dom_asym[chain1][chain2], ipsae_d0dom_asym[chain2][chain1])
                if maxvalue==ipsae_d0dom_asym[chain1][chain2]:
                    maxres=ipsae_d0dom_asymres[chain1][chain2]
                    maxn0=n0dom[chain1][chain2]
                    maxd0=d0dom[chain1][chain2]
                else:
                    maxres=ipsae_d0dom_asymres[chain2][chain1]
                    maxn0=n0dom[chain2][chain1]
                    maxd0=d0dom[chain2][chain1]
                ipsae_d0dom_max[chain1][chain2]=maxvalue
                ipsae_d0dom_maxres[chain1][chain2]=maxres
                ipsae_d0dom_max[chain2][chain1]=maxvalue
                ipsae_d0dom_maxres[chain2][chain1]=maxres
                n0dom_max[chain1][chain2]=maxn0
                n0dom_max[chain2][chain1]=maxn0
                d0dom_max[chain1][chain2]=maxd0
                d0dom_max[chain2][chain1]=maxd0

                maxvalue=max(ipsae_d0res_asym[chain1][chain2], ipsae_d0res_asym[chain2][chain1])
                if maxvalue==ipsae_d0res_asym[chain1][chain2]:
                    maxres=ipsae_d0res_asymres[chain1][chain2]
                    maxn0=n0res[chain1][chain2]
                    maxd0=d0res[chain1][chain2]
                else:
                    maxres=ipsae_d0res_asymres[chain2][chain1]
                    maxn0=n0res[chain2][chain1]
                    maxd0=d0res[chain2][chain1]
                ipsae_d0res_max[chain1][chain2]=maxvalue
                ipsae_d0res_maxres[chain1][chain2]=maxres
                ipsae_d0res_max[chain2][chain1]=maxvalue
                ipsae_d0res_maxres[chain2][chain1]=maxres
                n0res_max[chain1][chain2]=maxn0
                n0res_max[chain2][chain1]=maxn0
                d0res_max[chain1][chain2]=maxd0
                d0res_max[chain2][chain1]=maxd0

    chainpairs=set()
    for chain1 in unique_chains:
        for chain2 in unique_chains:
            if chain1 >= chain2: continue
            chainpairs.add(chain1 + "-" + chain2)

    metrics_columns= ['Chn1', 'Chn2', 'PAE', 'Dist', 'Type', 'ipSAE', 'ipSAE_d0chn', 'ipSAE_d0dom',
                      'ipTM_af', 'ipTM_d0chn', 'pDockQ', 'pDockQ2', 'LIS', 'n0res', 'n0chn',
                      'n0dom', 'd0res', 'd0chn', 'd0dom', 'nres1', 'nres2', 'dist1', 'dist2', 'Model']
    metrics_data = []
    for pair in sorted(chainpairs):
        (chain_a, chain_b) = pair.split("-")
        pair1 = (chain_a, chain_b)
        pair2 = (chain_b, chain_a)
        for pair in (pair1, pair2):
            chain1=pair[0]
            chain2=pair[1]

            residues_1 = len(unique_residues_chain1[chain1][chain2])
            residues_2 = len(unique_residues_chain2[chain1][chain2])
            dist_residues_1 = len(dist_unique_residues_chain1[chain1][chain2])
            dist_residues_2 = len(dist_unique_residues_chain2[chain1][chain2])

            metrics_data.append([
                chain1, chain2, pae_cutoff, dist_cutoff, "asym",
                ipsae_d0res_asym[chain1][chain2], ipsae_d0chn_asym[chain1][chain2],
                ipsae_d0dom_asym[chain1][chain2], iptm_af, iptm_d0chn_asym[chain1][chain2],
                pDockQ[chain1][chain2], pDockQ2[chain1][chain2], LIS[chain1][chain2],
                int(n0res[chain1][chain2]), int(n0chn[chain1][chain2]),
                int(n0dom[chain1][chain2]), d0res[chain1][chain2], d0chn[chain1][chain2],
                d0dom[chain1][chain2], residues_1, residues_2,
                dist_residues_1, dist_residues_2, data_id
            ])
            if chain1 > chain2:
                residues_1 = max(len(unique_residues_chain2[chain1][chain2]), len(unique_residues_chain1[chain2][chain1]))
                residues_2 = max(len(unique_residues_chain1[chain1][chain2]), len(unique_residues_chain2[chain2][chain1]))
                dist_residues_1 = max(len(dist_unique_residues_chain2[chain1][chain2]), len(dist_unique_residues_chain1[chain2][chain1]))
                dist_residues_2 = max(len(dist_unique_residues_chain1[chain1][chain2]), len(dist_unique_residues_chain2[chain2][chain1]))

                iptm_af_value=iptm_af
                pDockQ2_value=max(pDockQ2[chain1][chain2], pDockQ2[chain2][chain1])

                LIS_Score=(LIS[chain1][chain2]+LIS[chain2][chain1])/2.0
                metrics_data.append([
                    chain2, chain1, pae_cutoff, dist_cutoff, "max",
                    ipsae_d0res_max[chain1][chain2], ipsae_d0chn_max[chain1][chain2],
                    ipsae_d0dom_max[chain1][chain2], iptm_af_value, iptm_d0chn_max[chain1][chain2],
                    pDockQ[chain1][chain2], pDockQ2_value, LIS_Score,
                    int(n0res_max[chain1][chain2]), int(n0chn[chain1][chain2]),
                    int(n0dom_max[chain1][chain2]), d0res_max[chain1][chain2], d0chn[chain1][chain2],
                    d0dom_max[chain1][chain2], residues_1, residues_2,
                    dist_residues_1, dist_residues_2, data_id
                ])
    
    metrics = pd.DataFrame(metrics_data, columns=metrics_columns)
    
    return metrics, metrics_byres
