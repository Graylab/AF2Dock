import numpy as np
from scipy.spatial.transform import Rotation as R

def truncate_to_resolved(seqres, resi_auth):
    resi_resolved_full = [item != '' for item in resi_auth.split(',')]
    assert len(resi_resolved_full) == len(seqres)
    l_index = resi_resolved_full.index(True)
    r_index = len(resi_resolved_full) - resi_resolved_full[::-1].index(True) - 1
    return seqres[l_index:r_index + 1], resi_resolved_full[l_index:r_index + 1]

def apply_rigid_body_transform_atom37(all_atom_positions, all_atom_mask, ca_idx, tr, rot):
    com = np.mean(all_atom_positions[..., ca_idx, :], axis=-2)
    rot_t_mat = R.from_rotvec(rot).as_matrix()
    all_atom_positions = all_atom_positions - com
    all_atom_positions = np.einsum('...ij,kj->...ik', all_atom_positions, rot_t_mat)
    all_atom_positions = all_atom_positions + com + tr
    all_atom_positions = all_atom_positions * all_atom_mask
    return all_atom_positions
