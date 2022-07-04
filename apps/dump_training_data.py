#!/usr/bin/env python
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import typer

from make_it_dense.dataset import KITTIVDBSequence
from make_it_dense.utils import load_config, write_to_vdb


def store_network_input(
    config_file: Path = typer.Option(Path("config/kitti.yaml"), "--config", "-c", exists=True),
    sequence: int = typer.Option(-1, "--sequence", "-s", show_default=False),
):
    """Small script to store all the data used during training to the disk in vdb format."""
    tsdf_path = f"data/models/seq/{sequence:02}/tsdf"
    gt_path = f"data/models/seq/{sequence:02}/gt"
    os.makedirs(tsdf_path, exist_ok=True)
    os.makedirs(gt_path, exist_ok=True)

    # Run
    config = load_config(config_file)
    torch.multiprocessing.set_sharing_strategy(config.settings.sharing_strategy)
    dataloader = DataLoader(
        KITTIVDBSequence(config, sequence),
        num_workers=config.settings.num_workers,
        shuffle=False,
        batch_size=1,  # Batch size == 1 means we only take the first batch (the whole scan)
        batch_sampler=None,
    )

    for idx, batch in enumerate(tqdm(dataloader, desc="Storing VDBs", unit=" scans")):
        key = f"in_tsdf_{min(config.fusion.voxel_sizes)}"
        write_to_vdb(
            leaf_nodes_t=batch[key]["nodes"].squeeze(0).squeeze(1),
            coords_ijk_t=batch[key]["origin"].squeeze(0),
            voxel_size=min(config.fusion.voxel_sizes) / 100.0,
            voxel_trunc=config.fusion.voxel_trunc,
            name=str(idx).zfill(6),
            path=tsdf_path,
        )

        for voxel_size_cm in config.fusion.voxel_sizes:
            key = f"gt_tsdf_{voxel_size_cm}"
            write_to_vdb(
                leaf_nodes_t=batch[key]["nodes"].squeeze(0).squeeze(1),
                coords_ijk_t=batch[key]["origin"].squeeze(0),
                voxel_size=voxel_size_cm / 100.0,
                voxel_trunc=config.fusion.voxel_trunc,
                name=str(idx).zfill(6),
                path=os.path.join(gt_path, key),
            )


if __name__ == "__main__":
    typer.run(store_network_input)
