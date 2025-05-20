# MIT License

# Copyright (c) 2024 bjornwallner

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

import sys
import itertools
from functools import partial
import logging
from DockQ import DockQ

def compute_metrics(model, native, mapping=None, small_molecule=False, allowed_mismatches=0,
                    no_align=False, capri_peptide=False, n_cpu=1, max_chunk=1000):

    initial_mapping, model_chains, native_chains = DockQ.format_mapping(
        mapping, small_molecule
    )
    model_structure = DockQ.load_PDB(
        model, chains=model_chains, small_molecule=small_molecule
    )
    native_structure = DockQ.load_PDB(
        native, chains=native_chains, small_molecule=small_molecule
    )
    # check user-given chains are in the structures
    model_chains = [c.id for c in model_structure] if not model_chains else model_chains
    native_chains = (
        [c.id for c in native_structure] if not native_chains else native_chains
    )

    if len(model_chains) < 2 or len(native_chains) < 2:
        print("Need at least two chains in the two inputs\n")
        sys.exit()

    # permute chains and run on a for loop
    best_dockq = -1
    best_result = None
    best_mapping = None

    model_chains_to_combo = [
        mc for mc in model_chains if mc not in initial_mapping.values()
    ]
    native_chains_to_combo = [
        nc for nc in native_chains if nc not in initial_mapping.keys()
    ]

    chain_clusters, reverse_map = DockQ.group_chains(
        model_structure,
        native_structure,
        model_chains_to_combo,
        native_chains_to_combo,
        allowed_mismatches,
    )
    chain_maps = DockQ.get_all_chain_maps(
        chain_clusters,
        initial_mapping,
        reverse_map,
        model_chains_to_combo,
        native_chains_to_combo,
    )

    num_chain_combinations = DockQ.count_chain_combinations(chain_clusters)
    # copy iterator to use later
    chain_maps, chain_maps_ = itertools.tee(chain_maps)

    low_memory = num_chain_combinations > 100
    run_chain_map = partial(
        DockQ.run_on_all_native_interfaces,
        model_structure,
        native_structure,
        no_align=no_align,
        capri_peptide=capri_peptide,
        low_memory=low_memory,
    )

    if num_chain_combinations > 1:
        cpus = min(num_chain_combinations, n_cpu)
        chunk_size = min(max_chunk, max(1, num_chain_combinations // cpus))

        # for large num_chain_combinations it should be possible to divide the chain_maps in chunks
        result_this_mappings = DockQ.progress_map(
            run_chain_map,
            chain_maps,
            total=num_chain_combinations,
            n_cpu=cpus,
            chunk_size=chunk_size,
        )

        for chain_map, (result_this_mapping, total_dockq) in zip(
            chain_maps_, result_this_mappings
        ):

            if total_dockq > best_dockq:
                best_dockq = total_dockq
                best_result = result_this_mapping
                best_mapping = chain_map

        if low_memory:  # retrieve the full output by rerunning the best chain mapping
            best_result, total_dockq = DockQ.run_on_all_native_interfaces(
                model_structure,
                native_structure,
                chain_map=best_mapping,
                no_align=no_align,
                capri_peptide=capri_peptide,
                low_memory=False,
            )

    else:  # skip multi-threading for single jobs (skip the bar basically)
        best_mapping = next(chain_maps)
        best_result, best_dockq = run_chain_map(best_mapping)

    if not best_result:
        logging.error(
            "Could not find interfaces in the native model. Please double check the inputs or select different chains with the --mapping flag."
        )
        sys.exit(1)

    info = dict()
    info["model"] = model
    info["native"] = native
    info["best_dockq"] = best_dockq
    info["best_result"] = best_result
    info["GlobalDockQ"] = best_dockq / len(best_result)
    info["best_mapping"] = best_mapping
    info["best_mapping_str"] = f"{DockQ.format_mapping_string(best_mapping)}"

    return info
