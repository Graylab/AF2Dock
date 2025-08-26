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

from openfold.np import residue_constants
from openfold.utils.tensor_utils import tensor_tree_map

from AF2Dock.config import model_config
from AF2Dock.data.datamodule import AF2DockDataModule
from AF2Dock.model.model import AF2Dock
from AF2Dock.utils import data_utils, inference_utils


def main(args):
    # Create the output directory
    config = model_config(
        long_sequence_inference=args.long_sequence_inference,
        use_deepspeed_evoformer_attention=args.use_deepspeed_evoformer_attention,
        )

    if args.experiment_config_json:
        with open(args.experiment_config_json, 'r') as f:
            custom_config_dict = json.load(f)
        config.update_from_flattened_dict(custom_config_dict)
    
    if args.pinder_test_type == 'holo':
        config.data.test.pinder_cate_prob = {"holo": 1.0, "apo": 0.0, "pred": 0.0,}
    elif args.pinder_test_type == 'apo':
        config.data.test.pinder_cate_prob = {"holo": 0.0, "apo": 1.0, "pred": 0.0,}
    elif args.pinder_test_type == 'predicted':
        config.data.test.pinder_cate_prob = {"holo": 0.0, "apo": 0.0, "pred": 1.0,}
    
    config.data.data_module.num_workers = args.num_workers
    config.model.pair_denoiser.use_interchain_mask = args.use_interchain_mask
    
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
        config=config.data, 
        training_mode=False,
        batch_seed=random_seed,
        cached_esm_embedding_folder=args.cached_esm_embedding_folder,
        test_split=args.pinder_test_split,
        test_type=args.pinder_test_type,
        test_starting_index=args.data_starting_index,
        test_len_threshold=args.test_len_threshold,
        test_longer_ones=args.test_longer_ones,
    )
    data_module.setup('test')
    dataloader = data_module.test_dataloader()
    
    # Loading the model
    model = AF2Dock(config)
    model = inference_utils.prepare_model(model, args.checkpoint_path, args.model_device)

    ca_idx = residue_constants.atom_order["CA"]
    
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
                    tr_0, rot_0 = data_utils.get_rigid_body_noise_at_0(tr_sigma=config.data.rigid_body.tr_sigma,
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
                
                batch['t'] = batch['t'].new_tensor(np.array([[[[t]]]]))
                
                template_all_atom_pos = torch.cat(curr_atom_pos, dim=-3)
                assert template_all_atom_pos.shape[-3] == batch['template_all_atom_positions'].shape[-4]
                batch['template_all_atom_positions'] = template_all_atom_pos[None, None, ...][..., None].clone().to(
                    batch['template_all_atom_positions'].dtype).to(batch['template_all_atom_positions'].device)
                
                out = model(batch)
                out = tensor_tree_map(lambda x: x.cpu(), out)
                
                if time_idx < total_steps - 1:
                    curr_atom_pos = inference_utils.update_pose(batch, out, atom_masks, curr_atom_pos, s, t, ca_idx, is_homomer)
                    
                    if args.save_intermediate_template or args.save_intermediate_pred or time_idx == 0:
                        inference_utils.write_output(batch, out, out_dir_data, f'{data_id}_s{sample_idx}_t{time_idx}', out_pred=args.save_intermediate_pred, 
                                                     out_conf=args.save_intermediate_conf, out_template=args.save_intermediate_template or time_idx == 0)
 
            inference_utils.write_output(batch, out, out_dir_data, f'{data_id}_s{sample_idx}', out_pred=True, out_conf=True, out_template=args.save_intermediate_template)

        batch = tensor_tree_map(lambda x: x.cpu(), batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_dir", type=Path,
        help="""Name of the directory in which to output the prediction""",
    )
    parser.add_argument(
        "--cached_esm_embedding_folder", type=str, default=None,
        help="Directory with cached ESM embeddings."
    )
    parser.add_argument(
        "--num_samples", type=int, default=40,
        help="""Number of samples to generate per target"""
    )
    parser.add_argument(
        "--num_steps", type=int, default=10,
        help="""Number of steps to the denoising process"""
    )
    parser.add_argument(
        "--model_device", type=str, default="cuda",
        help="""Name of the device on which to run the model. Any valid torch
             device name is accepted (e.g. "cpu", "cuda:0")"""
    )
    parser.add_argument(
        "--use_interchain_mask", action="store_true", default=False,
        help="""Whether to use interchain mask in docking module for inference"""
    )
    parser.add_argument(
        "--checkpoint_path", type=Path, default=None,
        help="""Path to AF2Dock checkpoint. Can be either a DeepSpeed 
             checkpoint directory or a .pt file"""
    )
    parser.add_argument(
        "--save_intermediate_template", action="store_true", default=False,
        help="""Whether to save intermediate templates"""
    )
    parser.add_argument(
        "--save_intermediate_pred", action="store_true", default=False,
        help="""Whether to save intermediate predictions"""
    )
    parser.add_argument(
        "--save_intermediate_conf", action="store_true", default=False,
        help="""Whether to save intermediate confidence scores"""
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
        "--experiment_config_json", default="", help="Path to a json file with custom config values to overwrite config setting",
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
