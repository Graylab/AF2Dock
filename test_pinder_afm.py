import argparse
import logging
import numpy as np
from pathlib import Path
import random
import json
from tqdm import tqdm

logging.basicConfig()
logger = logging.getLogger(__file__)
logger.setLevel(level=logging.INFO)

import torch
torch_versions = torch.__version__.split(".")
torch_major_version = int(torch_versions[0])
torch_minor_version = int(torch_versions[1])
if (
    torch_major_version > 1 or
    (torch_major_version == 1 and torch_minor_version >= 12)
):
    # Gives a large speedup on Ampere-class GPUs
    torch.set_float32_matmul_precision("high")

torch.set_grad_enabled(False)

from openfold.config import model_config as of_model_config
from openfold.np import residue_constants
from openfold.utils.tensor_utils import tensor_tree_map
from openfold.utils.import_weights import import_jax_weights_

from AF2Dock.config import model_config as AF2Dock_model_config
from AF2Dock.data.datamodule import AF2DockDataModule
from AF2Dock.model.model_alt import AlphaFoldUnmasked
from AF2Dock.utils import data_utils, inference_utils


def main(args):
    # Create the output directory
    of_config = of_model_config(
        name="model_1_multimer_v3",
        long_sequence_inference=args.long_sequence_inference,
        use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
        )
    of_config.data.predict.max_msa_clusters = 17
    of_config.data.predict.max_extra_msa = 17
    of_config.data.predict.max_templates = 1
    of_config.data.predict.masked_msa_replace_fraction = 0.0
    
    AF2Dock_config = AF2Dock_model_config(
        long_sequence_inference=args.long_sequence_inference,
        use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
        )

    if args.pinder_test_type == 'holo':
        AF2Dock_config.data.test.pinder_cate_prob = {"holo": 1.0, "apo": 0.0, "pred": 0.0,}
    elif args.pinder_test_type == 'apo':
        AF2Dock_config.data.test.pinder_cate_prob = {"holo": 0.0, "apo": 1.0, "pred": 0.0,}
    elif args.pinder_test_type == 'predicted':
        AF2Dock_config.data.test.pinder_cate_prob = {"holo": 0.0, "apo": 0.0, "pred": 1.0,}
    
    AF2Dock_config.data.data_module.num_workers = args.num_workers
    
    output_dir_base = args.output_dir
    random_seed = args.data_random_seed
    if random_seed is None:
        random_seed = random.randrange(2 ** 32)

    np.random.seed(random_seed)
    torch.manual_seed(random_seed + 1)
    if not output_dir_base.exists():
        output_dir_base.mkdir(exist_ok=True)

    # Getting dataloader
    data_module = AF2DockDataModule(
        config=AF2Dock_config.data, 
        training_mode=False,
        batch_seed=random_seed,
        test_split=args.pinder_test_split,
        test_type=args.pinder_test_type,
        test_starting_index=args.data_starting_index,
        test_len_threshold=args.test_len_threshold,
        test_longer_ones=args.test_longer_ones,
    )
    data_module.setup('test')
    
    # Loading the model
    if args.unmasked:
        models_to_evaluate = ['model_1_multimer_v3']
    else:
        models_to_evaluate = [f'model_{i}_multimer_v3' for i in range(1, 6)]
    
    ca_idx = residue_constants.atom_order["CA"]
    
    for model_name in models_to_evaluate:
        model = model = AlphaFoldUnmasked(of_config, unmasked=args.unmasked)
        model = model.eval()
        import_jax_weights_(
            model, args.jax_weights_path / f"params_{model_name}.npz", version=model_name
        )
        model = model.to(args.model_device)
        
        dataloader = data_module.test_dataloader()
        
        for data_idx, batch in tqdm(enumerate(dataloader)):
            data_idx = data_idx + args.data_starting_index
            gt_features = batch.pop("gt_features")
            data_id = dataloader.dataset.data_index.iloc[batch["batch_idx"].item()]['id']
            is_homomer = 2 in batch['sym_id']
            batch = tensor_tree_map(lambda x: x.to(args.model_device), batch)
            
            out_dir_data = output_dir_base / f'{data_idx}_{data_id}'
            if not out_dir_data.exists():
                out_dir_data.mkdir(exist_ok=True)
            
            for sample_idx in tqdm(range(args.sample_starting_index, args.sample_starting_index + args.num_samples)):
                curr_atom_pos = []
                atom_masks = []
                for part in ['rec', 'lig']:
                    part_0_all_atom_mask = gt_features["ini_struct_feats"][part]["ini_all_atom_mask"][0].numpy()[None, ...]
                    part_0_all_atom_positions = gt_features["ini_struct_feats"][part]["ini_all_atom_positions"][0].numpy()[None, ...]
                    part_com = np.mean(part_0_all_atom_positions[..., ca_idx, :], axis=-2)
                    part_0_all_atom_positions = part_0_all_atom_positions - part_com[:, None, None, :]
                    if part == 'lig':
                        tr_0, rot_0 = data_utils.get_rigid_body_noise_at_0(tr_sigma=AF2Dock_config.data.rigid_body.tr_sigma,
                                                                        num_struct_batch=1,
                                                                        rot_prior='uniform')
                        part_0_all_atom_positions = data_utils.apply_rigid_body_transform_atom37(part_0_all_atom_positions,
                                                                                                part_0_all_atom_mask,
                                                                                                ca_idx,
                                                                                                tr_0,
                                                                                                rot_0)
                    curr_atom_pos.append(torch.tensor(part_0_all_atom_positions[0]))
                    atom_masks.append(torch.tensor(part_0_all_atom_mask[0]))
                
                template_all_atom_mask = torch.cat(atom_masks, dim=-2)
                assert template_all_atom_mask.shape[-2] == batch['template_all_atom_mask'].shape[-3]
                batch['template_all_atom_mask'] = template_all_atom_mask[None, None, ...][..., None].clone().to(
                    batch['template_all_atom_mask'].dtype).to(batch['template_all_atom_mask'].device)
                
                total_steps =  args.num_steps
                
                for time_idx in range(total_steps):
                    t = time_idx / total_steps
                    s = t + 1 / total_steps
                
                    template_all_atom_pos = torch.cat(curr_atom_pos, dim=-3)
                    assert template_all_atom_pos.shape[-3] == batch['template_all_atom_positions'].shape[-4]
                    batch['template_all_atom_positions'] = template_all_atom_pos[None, None, ...][..., None].clone().to(
                        batch['template_all_atom_positions'].dtype).to(batch['template_all_atom_positions'].device)
                    
                    out = model(batch)
                    out = tensor_tree_map(lambda x: x.cpu(), out)
                    
                    if not args.interpolate or not args.unmasked:
                        break
                    else:
                        if time_idx < total_steps - 1:
                            curr_atom_pos = inference_utils.update_pose(batch, out, atom_masks, curr_atom_pos, s, t, ca_idx, is_homomer)     

                if args.unmasked:
                    inference_utils.write_output(batch, out, out_dir_data, f'{data_id}_s{sample_idx}', out_pred=True, out_conf=True, out_template=True)
                else:
                    inference_utils.write_output(batch, out, out_dir_data, f'{data_id}_{model_name}', out_pred=True, out_conf=True, out_template=True)
                    break

            batch = tensor_tree_map(lambda x: x.cpu(), batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", type=Path,
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--num_samples", type=int, default=40,
        help="""Number of samples to generate per target"""
    )
    parser.add_argument(
        "--num_steps", type=int, default=10,
        help="""Number of steps to the interpolation process"""
    )
    parser.add_argument(
        "--model_device", type=str, default="cuda",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--jax_weights_path", type=Path, default=None,
        help="""Path to directory with AlphaFold jax weights"""
    )
    parser.add_argument(
        "--unmasked", action="store_true", default=False,
        help="""Whether to use the unmasked version of AlphaFold."""
    )
    parser.add_argument(
        "--interpolate", action="store_true", default=False,
        help="""Whether to interpolate the template inputs for multiple steps"""
    )
    parser.add_argument(
        "--pinder_test_split", type=str, default='pinder_af2',
        choices=['pinder_af2', 'pinder_xl', 'pinder_s'],
    )
    parser.add_argument(
        "--pinder_test_type", type=str, default='holo',
        choices=['holo', 'apo', 'predicted'],
    )
    parser.add_argument(
        "--test_len_threshold", type=int, default=None,
        help="""Length threshold for the test set"""
    )
    parser.add_argument(
        "--test_longer_ones", action="store_true", default=False,
        help="""Whether to test the ones longer than the threshold, otherwise
             only the ones shorter than the threshold are tested, default is False"""
    )
    parser.add_argument(
        "--data_starting_index", type=int, default=0,
        help="""Starting index for the test set. Used to skip the first N
             targets in the test set"""
    )
    parser.add_argument(
        "--sample_starting_index", type=int, default=0,
        help="""Starting index for samples. Used to skip the first N
             samples for each target"""
    )
    parser.add_argument(
        "--data_random_seed", type=int, default=None
    )
    parser.add_argument(
        "--long_sequence_inference", action="store_true", default=False,
        help="""enable options to reduce memory usage at the cost of speed, helps longer sequences fit into GPU memory, see the README for details"""
    )
    parser.add_argument(
        "--use_deepspeed_evoformer_attention", action="store_true", default=False, 
        help="Whether to use the DeepSpeed evoformer attention layer. Must have deepspeed installed in the environment.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="Number of workers for the dataloader.",
    )
    args = parser.parse_args()

    main(args)
