#!/usr/bin/env python
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import typer

from make_it_dense.dataset import KITTIVDBMultiSequence
from make_it_dense.utils import load_config


def precache(
    config_file: Path = typer.Option(Path("config/kitti.yaml"), "--config", "-c", exists=True),
    sequence: int = typer.Option(-1, "--sequence", "-s", show_default=False),
):
    """This module shall be used to pre-cache all the data before starting a full training loop."""
    config = load_config(config_file)
    torch.multiprocessing.set_sharing_strategy(config.settings.sharing_strategy)
    sequences = [sequence] if sequence else config.data.train_sequences + config.data.eval_sequences
    dataloader = DataLoader(
        KITTIVDBMultiSequence(config, sequences),
        num_workers=config.settings.num_workers,
        shuffle=False,
        batch_size=1,
        batch_sampler=None,
    )
    for _ in tqdm(dataloader, desc="Caching data", unit=" models"):
        pass


if __name__ == "__main__":
    typer.run(precache)
