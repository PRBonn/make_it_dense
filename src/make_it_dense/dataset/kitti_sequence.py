import glob
import os

import numpy as np
import pandas as pd

from make_it_dense.utils.cache import get_cache, memoize
from make_it_dense.utils.config import MkdConfig


class KITTIOdometrySequence:
    def __init__(self, config: MkdConfig, sequence: int):
        # Config
        self.config = config
        self.sequence = f"{sequence:02}"

        # Cache
        self.use_cache = self.config.cache.use_cache
        self.cache_dir = os.path.join(self.config.cache.cache_dir, self.sequence)
        self.cache = get_cache(directory=self.cache_dir, size_limit=self.config.cache.size_limit)

        # Read stuff
        self.kitti_root_dir = os.path.realpath(self.config.data.kitti_root_dir)
        self.kitti_sequence_dir = os.path.join(self.kitti_root_dir, "sequences", self.sequence)
        self.velodyne_dir = os.path.join(self.kitti_sequence_dir, "velodyne/")
        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
        self.calibration = self.read_calib_file(os.path.join(self.kitti_sequence_dir, "calib.txt"))
        self.poses = self.load_poses(
            os.path.join(self.kitti_root_dir, f"poses/{self.sequence}.txt")
        )

    def __len__(self):
        return len(self.scan_files)

    @memoize()
    def read_point_cloud(self, idx: int):
        scan_file = self.scan_files[idx]
        points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :3]
        return self.preprocess(points)

    @memoize()
    def get_high_res_scan(self, idx: int):
        points = self.read_point_cloud(idx)
        points = self.transform_points(points, self.poses[idx])
        return points

    @memoize()
    def get_low_res_scan(self, idx: int):
        points = self.read_point_cloud(idx)
        points = self.downsample_scan(points)
        points = self.transform_points(points, self.poses[idx])
        return points

    @staticmethod
    def downsample_scan(points):
        # TODO: Implement
        #  vertex_map = np.moveaxis(vertex_map, 0, -1) if vertex_map.shape[0] == 3 else vertex_map
        #  vertex_map = vertex_map[::scale_down_factor]

        #  H = vertex_map.shape[0]
        #  W = vertex_map.shape[1]
        #  points = vertex_map.reshape(H * W, 3)
        return points

    def load_poses(self, poses_file):
        def _lidar_pose_gt(poses_gt):
            _tr = self.calibration["Tr"].reshape(3, 4)
            tr = np.eye(4, dtype=np.float64)
            tr[:3, :4] = _tr
            left = np.einsum("...ij,...jk->...ik", np.linalg.inv(tr), poses_gt)
            right = np.einsum("...ij,...jk->...ik", left, tr)
            return right

        poses = pd.read_csv(poses_file, sep=" ", header=None).values
        n = poses.shape[0]
        poses = np.concatenate(
            (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)), axis=1
        )
        poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
        return _lidar_pose_gt(poses)

    @staticmethod
    def preprocess(points, z_th=-2.9, min_range=2.75):
        z = points[:, 2]
        points = points[z > z_th]
        points = points[np.linalg.norm(points, axis=1) >= min_range]
        return points

    @staticmethod
    def read_calib_file(file_path: str) -> dict:
        calib_dict = {}
        with open(file_path, "r") as calib_file:
            for line in calib_file.readlines():
                tokens = line.split(" ")
                if tokens[0] == "calib_time:":
                    continue
                # Only read with float data
                if len(tokens) > 0:
                    values = [float(token) for token in tokens[1:]]
                    values = np.array(values, dtype=np.float32)

                    # The format in KITTI's file is <key>: <f1> <f2> <f3> ...\n -> Remove the ':'
                    key = tokens[0][:-1]
                    calib_dict[key] = values
        return calib_dict

    @staticmethod
    def transform_points(points, matrix, translate=True):
        """
        Implementation borrowed trom the trimesh library
        """
        points = np.asanyarray(points, dtype=np.float64)
        # no points no cry
        if len(points) == 0:
            return points.copy()

        matrix = np.asanyarray(matrix, dtype=np.float64)
        if len(points.shape) != 2 or (points.shape[1] + 1 != matrix.shape[1]):
            raise ValueError(
                "matrix shape ({}) doesn't match points ({})".format(matrix.shape, points.shape)
            )

        # check to see if we've been passed an identity matrix
        identity = np.abs(matrix - np.eye(matrix.shape[0])).max()
        if identity < 1e-8:
            return np.ascontiguousarray(points.copy())

        dimension = points.shape[1]
        column = np.zeros(len(points)) + int(bool(translate))
        stacked = np.column_stack((points, column))
        transformed = np.dot(matrix, stacked.T).T[:, :dimension]
        transformed = np.ascontiguousarray(transformed)
        return transformed
