#!/usr/bin/env python
from functools import reduce
import os
from pathlib import Path
import sys
import time

import numpy as np
import open3d as o3d
import torch
from tqdm import trange
import typer
from vdbfusion import VDBVolume

from make_it_dense.dataset import KITTIOdometrySequence
from make_it_dense.evaluation import run_completion_pipeline
from make_it_dense.models import CompletionNet
from make_it_dense.utils.config import MkdConfig, load_config, write_config


class ReFusionPipeline:
    def __init__(
        self,
        dataset,
        config: MkdConfig.refusion,
        checkpoint: Path,
        map_name: str,
        jump: int = 0,
        n_scans: int = -1,
    ):
        self._dataset = dataset
        self._config: MkdConfig.refusion = config
        self._n_scans = len(dataset) if n_scans == -1 else n_scans
        self._jump = jump
        self._map_name = f"{map_name}_{self._n_scans}_scans"
        self._global_map = VDBVolume(
            self._config.voxel_size,
            self._config.vox_trunc * self._config.voxel_size,
            self._config.space_carving,
        )
        self._res = {}
        # Refusion memebers
        self._cuda = self._config.cuda
        model_cfg = MkdConfig.from_dict(torch.load(checkpoint)["hyper_parameters"])
        self._model = CompletionNet.load_from_checkpoint(checkpoint, config=model_cfg)
        self._model.eval()
        self._model = self._model.cuda() if self._cuda else self._model

    def run(self):
        self._run_tsdf_pipeline()
        self._write_ply()
        self._write_cfg()
        self._write_vdb()
        self._print_tim()
        self._print_metrics()

    def visualize(self):
        o3d.visualization.draw_geometries([self._res["mesh"]])

    def __len__(self):
        return len(self._dataset)

    def _run_tsdf_pipeline(self):
        times = []
        for idx in trange(self._jump, self._jump + self._n_scans, unit=" frames"):
            scan = self._dataset.get_low_res_scan(idx)
            pose = self._dataset.poses[idx]
            tic = time.perf_counter_ns()
            self._global_map.integrate(scan, pose, weight=self._config.eta)
            self._global_map.integrate(grid=self.make_it_dense(scan), weight=1 - self._config.eta)
            toc = time.perf_counter_ns()
            times.append(toc - tic)
        self._res = {"mesh": self._get_o3d_mesh(self._global_map, self._config), "times": times}

    def make_it_dense(self, scan):
        return run_completion_pipeline(
            scan,
            self._config.voxel_size,
            self._config.vox_trunc,
            self._model,
            self._cuda,
        )[0]

    def _write_vdb(self):
        os.makedirs(self._config.out_dir, exist_ok=True)
        filename = os.path.join(self._config.out_dir, self._map_name) + ".vdb"
        self._global_map.extract_vdb_grids(filename)

    def _write_ply(self):
        os.makedirs(self._config.out_dir, exist_ok=True)
        filename = os.path.join(self._config.out_dir, self._map_name) + ".ply"
        o3d.io.write_triangle_mesh(filename, self._res["mesh"])

    def _write_cfg(self):
        os.makedirs(self._config.out_dir, exist_ok=True)
        filename = os.path.join(self._config.out_dir, self._map_name) + ".yml"
        write_config(dict(self._config), filename)

    def _print_tim(self):
        total_time_ns = reduce(lambda a, b: a + b, self._res["times"])
        total_time = total_time_ns * 1e-9
        total_scans = self._n_scans - self._jump
        self.fps = float(total_scans / total_time)

    @staticmethod
    def _get_o3d_mesh(tsdf_volume, cfg):
        vertices, triangles = tsdf_volume.extract_triangle_mesh(cfg.fill_holes, cfg.min_weight)
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(vertices),
            o3d.utility.Vector3iVector(triangles),
        )
        mesh.compute_vertex_normals()
        mesh.compute_triangle_normals()
        return mesh

    def _print_metrics(self):
        # If PYOPENVDB_SUPPORT has not been enabled then we can't report any metrics
        if not self._global_map.pyopenvdb_support_enabled:
            print("No metrics available, please compile with PYOPENVDB_SUPPORT")
            return

        # Compute the dimensions of the volume mapped
        grid = self._global_map.tsdf
        bbox = grid.evalActiveVoxelBoundingBox()
        dim = np.abs(np.asarray(bbox[1]) - np.asarray(bbox[0]))
        volume_extent = np.ceil(self._config.voxel_size * dim).astype(np.int32)
        volume_extent = f"{volume_extent[0]} x {volume_extent[1]} x {volume_extent[2]}"

        # Compute memory footprint
        total_voxels = int(np.prod(dim))
        float_size = 4
        # Always 2 grids
        mem_footprint = 2 * grid.memUsage() / (1024 * 1024)
        dense_equivalent = 2 * (float_size * total_voxels) / (1024 * 1024 * 1024)  # GB

        # compute size of .vdb file
        filename = os.path.join(self._config.out_dir, self._map_name) + ".vdb"
        file_size = float(os.stat(filename).st_size) / (1024 * 1024)

        # print metrics
        trunc_voxels = self._config.vox_trunc

        filename = os.path.join(self._config.out_dir, self._map_name) + ".txt"
        with open(filename, "w") as f:
            stdout = sys.stdout
            sys.stdout = f  # Change the standard output to the file we created.
            print(f"--------------------------------------------------")
            print(f"Results for dataset {self._map_name}:")
            print(f"--------------------------------------------------")
            print(f"voxel size       = {self._config.voxel_size} [m]")
            print(f"truncation       = {trunc_voxels} [voxels]")
            print(f"space carving    = {self._config.space_carving}")
            print(f"Avg FPS          = {self.fps:.2f} [Hz]")
            print(f"--------------------------------------------------")
            print(f"volume extent    = {volume_extent} [m x m x m]")
            print(f"memory footprint = {mem_footprint:.2f} [MB]")
            print(f"dense equivalent = {dense_equivalent:.2f} [GB]")
            print(f"size on disk     = {file_size:.2f} [MB]")
            print(f"--------------------------------------------------")
            print(f"number of scans  = {len(self)}")
            print(f"points per scan  = {len(self._dataset.get_low_res_scan(0))}")
            print(f"--------------------------------------------------")
            sys.stdout = stdout
        # Print it
        os.system(f"cat {filename}")


def main(
    checkpoint: Path = typer.Option(Path("models/make_it_dense.ckpt"), exists=True),
    config_file: Path = typer.Option(Path("config/kitti.yaml"), exists=True),
    sequence: int = typer.Option(-1, "--sequence", "-s", show_default=False),
    n_scans: int = typer.Option(-1, "--n-scans", "-n", show_default=False),
    jump: int = typer.Option(0, "--jump", "-j", show_default=False),
    visualize: bool = typer.Option(False, "--visualize"),
):
    map_name = f"kitti_{str(sequence).zfill(2)}"
    config: MkdConfig = load_config(config_file)
    dataset = KITTIOdometrySequence(config, sequence)
    pipeline = ReFusionPipeline(dataset, config.refusion, checkpoint, map_name, jump, n_scans)
    pipeline.run()
    pipeline.visualize() if visualize else None


if __name__ == "__main__":
    typer.run(main)
