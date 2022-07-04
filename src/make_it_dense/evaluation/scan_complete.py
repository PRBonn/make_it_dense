import numpy as np
import torch
from torch.utils.data.dataloader import default_collate
from vdb_to_numpy import LeafNodeGrid
from vdbfusion import VDBVolume

from make_it_dense.utils import torch_to_vdb, vdb_to_torch


def run_tsdf(points, voxel_size, sdf_trunc, pose=np.eye(4)):
    """Also give a base grid to extract target coordinates."""
    tsdf_volume = VDBVolume(voxel_size, sdf_trunc)
    tsdf_volume.integrate(points, pose)
    # TODO: This is still a hack, run tsdf with a higher resolution just to obtain the target
    # cooridnates values
    base_volume = VDBVolume(4 * voxel_size, 4 * sdf_trunc)
    base_volume.integrate(points, pose)
    target_coords_ijk_a, _ = LeafNodeGrid(base_volume.tsdf).numpy()
    target_coords_xyz_a = base_volume.voxel_size * target_coords_ijk_a
    return tsdf_volume, target_coords_xyz_a


def get_input_tensors(grid, coords_xyz):
    inputs = []
    for origin_xyz in coords_xyz:
        input_dict = vdb_to_torch(grid, origin_xyz, shape=(32, 32, 32), empty_th=0.1)
        inputs.append(input_dict) if input_dict else None
    return default_collate(inputs)


def complete_vdb(in_grid, coords_xyz, model, cuda=False):
    """Receives an input scan(TSDF), in its VDB representation and returns a completed scan."""
    voxel_size = in_grid.transform.voxelSize()[0]
    sdf_trunc = in_grid.background
    inputs = get_input_tensors(in_grid, coords_xyz)

    # Normalize leaf nodes
    nodes_t = inputs["nodes"] / np.float32(sdf_trunc)

    # Move data to the GPU
    if cuda and torch.cuda.is_available():
        nodes_t = nodes_t.to("cuda")
        model = model.cuda()

    # Run the model
    out_t = model.predict_step(nodes_t)["out_tsdf_10"].squeeze(1)
    coords_ijk_t = inputs["origin"]
    return torch_to_vdb(
        coords_ijk_t=coords_ijk_t,
        leaf_nodes_t=out_t,
        voxel_size=voxel_size,
        sdf_trunc=sdf_trunc,
        normalize=True,
    )


def run_completion_pipeline(scan, voxel_size, voxel_trunc, model, cuda=False):
    tsdf_volume, coords_xyz = run_tsdf(scan, voxel_size, voxel_size * voxel_trunc)
    out_grid = complete_vdb(tsdf_volume.tsdf, coords_xyz, model, cuda)
    return out_grid, tsdf_volume
