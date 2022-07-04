from typing import Tuple

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from make_it_dense.dataset.kitti_vdb_sequence import KITTIVDBSequence
from make_it_dense.utils.config import MkdConfig
from make_it_dense.utils.data import collate_models


class KITTIVDBMultiSequence(Dataset):
    """A thin wrapper around KITTIVDBSequence, but containing multiple sequences."""

    def __init__(self, config: MkdConfig, sequences: list):
        super().__init__()
        if sequences is None or len(sequences) == 0:
            return None
        self.config = config
        self.sequences = {
            sequence_id: KITTIVDBSequence(self.config, sequence_id) for sequence_id in sequences
        }
        # N scans is the sum of all the scans of all sequence in this sub-datasets
        self.n_scans = sum(map(len, self.sequences.values()))

        # Create a map between global indices and per sequence index, taken from MOS - Benedikt
        self.idx_mapper = []
        for sequence_id, seq in self.sequences.items():
            for scan_idx in range(len(seq)):
                self.idx_mapper.append((sequence_id, scan_idx))

        # If this doesn't work something else is quite wrong
        assert len(self.idx_mapper) == self.n_scans

    def __len__(self):
        return self.n_scans

    def __getitem__(self, idx: int):
        sequence_id, scan_idx = self.idx_mapper[idx]
        return self.sequences[sequence_id][scan_idx]


class KITTIVDBDataModule(pl.LightningDataModule):
    def __init__(self, config: MkdConfig, overfit_sequence: str = None):
        super().__init__()
        self.dataset = None
        self.train_set = None
        self.val_set = None
        self.config = config
        # Override training and validation sequences if specified by the user
        if overfit_sequence:
            print(f"[WARNING] User specified to override sequence {overfit_sequence}")
            self.config.data.train_sequences = [overfit_sequence]
            self.config.data.eval_sequences = [overfit_sequence]
        self.train_set = KITTIVDBMultiSequence(self.config, self.config.data.train_sequences)
        self.val_set = KITTIVDBMultiSequence(self.config, self.config.data.eval_sequences)

    def collate_fn(self, batch: Tuple[torch.Tensor]) -> Tuple[torch.Tensor]:
        return collate_models(batch, batch_size=self.config.training.effective_batch_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.config.training.train_batch_size,
            collate_fn=self.collate_fn,
            num_workers=self.config.settings.num_workers,
            shuffle=self.config.training.train_shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.config.training.val_batch_size,
            num_workers=self.config.settings.num_workers,
            collate_fn=self.collate_fn,
            shuffle=self.config.training.val_shuffle,
        )
