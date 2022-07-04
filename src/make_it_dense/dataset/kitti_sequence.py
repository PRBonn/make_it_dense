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

    def downsample_scan(self, points):
        """Adapted from RangeNet"""
        W = self.config.data.lidar_width
        H = self.config.data.lidar_height
        fov_up = np.deg2rad(self.config.data.up_fov)
        fov_down = np.deg2rad(self.config.data.down_fov)
        fov = abs(fov_down) + abs(fov_up)  # get field of view total in radians

        # get depth of all points
        depth = np.linalg.norm(points, axis=1)

        # get scan components
        scan_x = points[:, 0]
        scan_y = points[:, 1]
        scan_z = points[:, 2]

        # get angles of all points
        yaw = -np.arctan2(scan_y, scan_x)
        pitch = np.arcsin(scan_z / depth)

        # get projections in image coords
        proj_x = 0.5 * (yaw / np.pi + 1.0)  # in [0.0, 1.0]
        proj_y = 1.0 - (pitch + abs(fov_down)) / fov  # in [0.0, 1.0]

        # scale to image size using angular resolution
        proj_x *= W  # in [0.0, W]
        proj_y *= H  # in [0.0, H]

        # round and clamp for use as index
        proj_x = np.floor(proj_x)
        proj_x = np.minimum(W - 1, proj_x)
        proj_x = np.maximum(0, proj_x).astype(np.int32)

        proj_y = np.floor(proj_y)
        proj_y = np.minimum(H - 1, proj_y)
        proj_y = np.maximum(0, proj_y).astype(np.int32)

        # order in decreasing depth
        order = np.argsort(depth)[::-1]
        depth = depth[order]
        proj_y = proj_y[order]
        proj_x = proj_x[order]

        scan_x = scan_x[order]
        scan_y = scan_y[order]
        scan_z = scan_z[order]

        vertex_map = np.full((H, W, 3), -1, dtype=np.float32)
        vertex_map[proj_y, proj_x] = np.array([scan_x, scan_y, scan_z]).T
        vertex_map = vertex_map[:: self.config.data.scale_down_factor]
        points_ds = vertex_map.reshape(int(H * W / self.config.data.scale_down_factor), 3)
        return points_ds

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
