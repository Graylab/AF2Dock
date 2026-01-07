# `AF2Dock`

Preprint: https://doi.org/10.1101/2025.11.28.691195

## Installation

First, install `openfold` following instructions on [this page](https://openfold.readthedocs.io/en/latest/Installation.html). Current repo is tested with [this version of openfold](https://github.com/aqlaboratory/openfold/tree/be2ec1841f16c966c65ae0e7599ebbadc725757d).

(Notes from Jan 2026: I needed to remove `flash-attn` from the `environment.yml` file to successfully set up the enviroment.)

Then, in the enviroment with `openfold`, install `AF2Dock` as follows:

```
git clone https://github.com/Graylab/AF2Dock.git
cd AF2Dock
pip install .
```

If you run into issue installing `fastpdb` (required by `pinder`), follow instructions on the `pinder` [repo page](https://github.com/pinder-org/pinder) to download the rust toolchain.

To use cuEquivariance, install additional packages as follows at the end:
```
pip install cuequivariance_ops_torch_cu12 cuequivariance_torch
```

(Notes from Jan 2026: Currently, `triangle_multiplicative_updates` from `cuEquivariance` does not work out of the box. It gives me error on typing, as `pytorch` 2.5 does not seem to support using `list` for typing. I needed to manually patch the `triangle_multiplicative_update.py` and `attention_pair_bias_torch.py` files in `cuequivariance_ops_torch` by replacing `list` typing with `List` and add `from typing import List` at the start of the files for it to work. [Related issue](https://github.com/NVIDIA/cuEquivariance/issues/229).)

## Inference

Model weights are uploaded to [Zenodo](https://doi.org/10.5281/zenodo.17782958). They can be downloaded with the `scripts/download_weights.py` script as follows:
```
python scripts/download_weights.py --model-name AF2Dock_base
```

Run prediction with a single set of input structures:
```
python predict.py \
    output_dir \
    --rec_struc_path path/to/receptor/structure \
    --lig_struc_path path/to/ligand/structure \
    --num_samples 40 \
    --num_steps 10 \
    --checkpoint_path path/to/model/weights
```

Or supply the input as a csv file (example in `data/example_input`):
```
python predict.py \
    output_dir \
    --input_csv path/to/csv/file \
    --num_samples 40 \
    --num_steps 10 \
    --checkpoint_path path/to/model/weights
```

When using input strucutures that have missing residues, an a3m file containing the alignment of resolved residues with the full sequence for each chain is required as an input. And example is available as `data/example_input/ab_8tbq_1_r_pred_wo_cdrh3.a3m`.

Code for computing success rates and bootstrapping are available as jupyter notebooks in `notebooks`. To run tests on the PINDER-AF2 benchmark, first cache the ESM embeddings using `scripts/compute_pinder_ESM_embeddings.py` and then run `scripts/test_pinder.py`. To run predictions and tests with single-sequence AF-M, use `scripts/predict_afm.py` and `scripts/test_pinder_afm.py`.

## Training

To train using the PINDER training set, first cache the ESM embeddings using `scripts/compute_pinder_ESM_embeddings.py` and obtain the three-body interactions using `scripts/find_train_three_body_interactions.py`.

Then train as follows (training from scratch in this case):
```
python train.py \
    output_dir \
    --cached_esm_embedding_folder path/to/cached/esm \
    --pinder_entity_seq_cluster_pkl data/pinder_entity_seq_cluster.pkl \
    --three_body_interactions_pkl path/to/three/body/results \
    --resume_from_jax_params path/to/afm/weights \
    --af_params freeze \
    --list_of_samples_to_exclude data/train_samples_to_exclude.txt \
    --gpus 4 \
    --num_nodes 1 \
    --precision bf16 \
    --max_epochs 100 \
    --log_every_n_steps 8 \
    --accumulate_grad_batches 16 \
    --val_check_interval 0.1 \
    --limit_val_batches 0.15 \
    --num_sanity_val_steps 2 \
```
Certain settings such as the percentages of holo/apo/predicted structures need to be specified in a json file with the `--experiment_config_json` argument. For instance, use the following json file for completely holo inputs:
```
{
    "data.train.pinder_cate_prob.holo": 1.0,
    "data.train.pinder_cate_prob.apo": 0.0,
    "data.train.pinder_cate_prob.pred": 0.0
}
```
