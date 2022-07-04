import os
from typing import Dict

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from vdb_to_numpy import LeafNodeGrid, normalize_grid
from vdbfusion import VDBVolume

from make_it_dense.dataset.kitti_sequence import KITTIOdometrySequence
from make_it_dense.utils.cache import get_cache, memoize
from make_it_dense.utils.config import MkdConfig
from make_it_dense.utils.vdb_utils import vdb_to_torch


class KITTIVDBSequence(Dataset):
    def __init__(self, config: MkdConfig, sequence: int):
        self.config = config
        self.use_cache = self.config.cache.use_cache
        self.cache_dir = os.path.join(self.config.cache.cache_dir, f"{sequence:02}/vdb_grids")
        self.cache = get_cache(directory=self.cache_dir, size_limit=self.config.cache.size_limit)
        self.voxel_sizes_cm = self.config.fusion.voxel_sizes
        self.voxel_size = min(self.config.fusion.voxel_sizes) / 100.0
        self.voxel_trunc = self.config.fusion.voxel_trunc
        self.sdf_trunc = self.voxel_trunc * self.voxel_size
        self.min_weight = self.config.fusion.min_weight
        self.sequence = KITTIOdometrySequence(config, sequence)

    def __len__(self):
        return len(self.sequence)

    @memoize()
    def __getitem__(self, idx: int):
        return self._get_leaf_node_pairs(self._get_vdb_grids(idx))

    def _get_leaf_node_pairs(self, vdb_grids: Dict):
        # base_grid is the biggest resolution grid, typically 40cm
        base_grid = vdb_grids[f"gt_tsdf_{max(self.voxel_sizes_cm)}"]
        outputs = []
        for origin_ijk, _ in LeafNodeGrid(base_grid):
            origin_xyz = base_grid.transform.indexToWorld(origin_ijk.tolist())
            out = {key: vdb_to_torch(vdb_grids[key], origin_xyz) for key in vdb_grids.keys()}
            if None in out.values():
                continue
            outputs.append(out)
        return default_collate(outputs)

    def _get_vdb_grids(self, idx) -> Dict:
        """This is the main method to of this data module.

        It's responsible for providing the (x, y) description of the data, this is, the
        input VDB volume (x) and the GT VDB volume that will be used for supervising the
        network.
        """
        # Dictionary which has all the information in the world
        vdb_grids = {}

        # Finest grid, 16 beams input data
        tsdf_volume = VDBVolume(self.voxel_size, self.sdf_trunc)
        tsdf_volume.integrate(self.sequence.get_low_res_scan(idx), self.sequence.poses[idx])
        voxel_size_cm = int(100 * self.voxel_size)
        vdb_grids[f"in_tsdf_{voxel_size_cm}"] = normalize_grid(tsdf_volume.tsdf)

        # Generate the GT TSDF accumlated scans
        for voxel_size_cm in self.voxel_sizes_cm:
            voxel_size = voxel_size_cm / 100.0
            sdf_trunc = self.voxel_trunc * voxel_size
            gt_volume = VDBVolume(voxel_size, sdf_trunc)
            min_idx = max(idx - self.config.fusion.acc_scans, 0)
            max_idx = min(idx + self.config.fusion.acc_scans, len(self))
            for nidx in range(min_idx, max_idx):
                scan = self.sequence.get_high_res_scan(nidx)
                pose = self.sequence.poses[nidx]
                gt_volume.integrate(scan, pose)
            # Store the results in the output dictionary
            vdb_grids[f"gt_tsdf_{voxel_size_cm}"] = normalize_grid(gt_volume.prune(self.min_weight))
        return vdb_grids
