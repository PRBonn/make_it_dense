import os
from typing import Tuple

import numpy as np
import pyopenvdb as vdb
import torch


def get_occ_percentage(nodes_a):
    return np.count_nonzero((np.abs(nodes_a) < 1.0)) / nodes_a.ravel().shape[0]


def get_shape(voxel_size: float, max_voxel_size: float = 0.4) -> Tuple:
    """Obtain volume shape from voxel size.

    These are some examples of the output of this function:

        voxel_size==0.4 -> (8,   8,  8)
        voxel_size==0.2 -> (16, 16, 16)
        voxel_size==0.1 -> (32, 32, 32)
    """
    nlog2 = round(np.log2(max_voxel_size / voxel_size))
    return 3 * (int(2 ** (nlog2 + 3)),)


def vdb_to_torch(grid, origin_xyz, shape: Tuple = None, empty_th: float = 0.0):
    shape = get_shape(grid.transform.voxelSize()[0]) if not shape else shape
    grid_origin = grid.transform.worldToIndexCellCentered(origin_xyz)
    x = np.empty(shape, dtype=np.float32)
    grid.copyToArray(x, grid_origin)
    if get_occ_percentage(x) <= empty_th:
        return None
    return {
        "nodes": torch.as_tensor(x).unsqueeze(0),
        "origin": torch.as_tensor(grid_origin, dtype=torch.int32),
    }


def write_vdb(grid, name=None, path="."):
    os.makedirs(path, exist_ok=True)
    grid.name = name
    grid_fn = os.path.join(path, name + ".vdb")
    vdb.write(grid_fn, grid)
    return grid_fn


def torch_to_vdb(coords_ijk_t, leaf_nodes_t, voxel_size, sdf_trunc, normalize=True):
    """Convert torch arrays to VDB grids."""
    if normalize:
        leaf_nodes_t = np.float32(sdf_trunc) * leaf_nodes_t
    vdb_grid = vdb.FloatGrid()
    vdb_grid.background = np.float32(sdf_trunc)
    vdb_grid.gridClass = vdb.GridClass.LEVEL_SET
    vdb_grid.transform = vdb.createLinearTransform(voxelSize=voxel_size)
    coords_ijk_a = coords_ijk_t.detach().cpu().numpy()
    leaf_nodes_a = leaf_nodes_t.detach().cpu().numpy()
    # Network predicts bigger values than sdf_trunc due the scaled tanh
    tolerance = np.float32(leaf_nodes_a.max() - sdf_trunc)
    for i, ijk in enumerate(coords_ijk_a):
        vdb_grid.copyFromArray(leaf_nodes_a[i], ijk, tolerance=tolerance)
    return vdb_grid


def write_to_vdb(
    coords_ijk_t,
    leaf_nodes_t,
    voxel_size,
    voxel_trunc,
    normalize=True,
    name=None,
    path=".",
):
    """Given a torch array, convert to its VDB representation and store to disk."""
    sdf_trunc = voxel_trunc * voxel_size
    grid = torch_to_vdb(coords_ijk_t, leaf_nodes_t, voxel_size, sdf_trunc, normalize)
    return write_vdb(grid, name, path)
