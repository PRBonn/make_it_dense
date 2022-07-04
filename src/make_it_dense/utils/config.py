from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

from easydict import EasyDict
import yaml


# Adapted from PyLiDAR-SLAM
@dataclass
class KITTIConfig:
    kitti_root_dir: str = ""
    lidar_height: int = 64
    lidar_width: int = 1024
    up_fov: int = 3
    down_fov: int = -24
    train_sequences: list = field(
        default_factory=lambda: ["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10"]
    )
    eval_sequences: list = field(default_factory=lambda: [f"{i:02}" for i in range(22)])


@dataclass
class MkdConfig:
    data: KITTIConfig

    @dataclass
    class settings:
        gpu: bool = True
        num_workers: int = 0
        sharing_strategy: Optional[str] = ""

    @dataclass
    class fusion:
        voxel_sizes: list = field(default_factory=lambda: [40, 20, 10])
        voxel_trunc: int = 3
        acc_scans: int = 25
        scale_down_factor: int = 4
        min_weight: float = 5.0

    @dataclass
    class model:
        occ_th: list = field(default_factory=lambda: [0.99, 0.99, 0.99])
        f_maps: list = field(default_factory=lambda: [4, 16, 32, 64])
        layers_down: list = field(default_factory=lambda: [1, 2, 3, 4])
        layers_up: list = field(default_factory=lambda: [3, 2, 1])

    @dataclass
    class cache:
        use_cache: bool = True
        cache_dir: str = "data/kitti-odometry/cache"
        size_limit: int = 100

    @dataclass
    class loss:
        mask_plane_loss: bool = True
        mask_l1_loss: bool = True
        use_log_transform: bool = True
        mask_occ_weight: float = 0.9

    @dataclass
    class optimization:
        lr: float = 10.0e-03
        weight_decay: float = 1.0e-3
        div_factor: float = 10000

    @dataclass
    class training:
        n_epochs: int = 329
        train_batch_size: int = 4
        train_shuffle: bool = True
        val_batch_size: int = 2
        val_shuffle: bool = False

    @dataclass
    class logging:
        save_dir: str = "logs/"
        name: str = "kitti"
        weights_summary: str = "full"
        log_every_n_steps: int = 1
        log_graph: bool = True
        lr_monitor_step: str = "step"

    @dataclass
    class checkpoints:
        dirpath: str = "checkpoints/"
        monitor: str = "train/train_loss"
        save_top_k: int = -1
        mode: str = "min"

    @dataclass
    class refusion:
        # VDBFUSION
        voxel_size: float = 0.1
        vox_trunc: int = 3
        space_carving: bool = False
        out_dir: str = "results/"

        # Reconstruction
        fill_holes: bool = False
        min_weight: float = 5.0

        # Make it Dense
        cuda: bool = True
        eta: float = 0.7

    @staticmethod
    def from_dict(config: Dict) -> MkdConfig:
        return EasyDict(config)


def load_config(path: Path) -> MkdConfig:
    try:
        return EasyDict(yaml.safe_load(open(path)))
    except FileNotFoundError as err:
        raise FileNotFoundError("{} file doesn't exist".format(path)) from err


def write_config(config: MkdConfig, filename: str):
    with open(filename, "w") as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
