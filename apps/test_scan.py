#!/usr/bin/env python
import os
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import typer

from vdb_to_numpy import vdb_to_triangle_mesh

from make_it_dense.evaluation import run_completion_pipeline
from make_it_dense.models import CompletionNet
from make_it_dense.utils import MkdConfig


def read_point_cloud(filename):
    ext = os.path.splitext(filename)[-1]
    if ext == ".bin":
        scan = np.fromfile(filename, dtype=np.float32).reshape((-1, 4))[:, :3]
    else:
        scan = o3d.io.read_point_cloud(filename).points
    return np.asarray(scan, dtype=np.float64)


def test(
    pointcloud: Path,
    checkpoint: Path = typer.Option(Path("models/make_it_dense.ckpt"), exists=True),
    cuda: bool = typer.Option(False, "--cuda"),
):
    config = MkdConfig.from_dict(torch.load(checkpoint)["hyper_parameters"])
    model = CompletionNet.load_from_checkpoint(checkpoint_path=str(checkpoint), config=config)
    model.eval()

    # Run make-it-dense pipeline
    out_grid, in_grid = run_completion_pipeline(
        scan=read_point_cloud(pointcloud),
        voxel_size=min(model.voxel_sizes) / 100.0,
        voxel_trunc=model.voxel_trunc,
        model=model,
        cuda=cuda,
    )

    in_mesh = vdb_to_triangle_mesh(in_grid.tsdf)
    out_mesh = vdb_to_triangle_mesh(out_grid)
    scan_name = pointcloud.name.split(".")[0]
    o3d.io.write_triangle_mesh(f"results/{scan_name}_in.ply", in_mesh)
    o3d.io.write_triangle_mesh(f"results/{scan_name}_out.ply", out_mesh)
    in_mesh.translate([0, +50, 0])
    o3d.visualization.draw_geometries([in_mesh, out_mesh])


if __name__ == "__main__":
    typer.run(test)
