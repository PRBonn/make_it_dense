import numpy as np
from torch import nn
import torch.nn.functional as F

from make_it_dense.utils.config import MkdConfig


class SDFLoss(nn.Module):
    def __init__(self, config: MkdConfig):
        super().__init__()
        self.config = config.loss
        self.voxel_sizes = config.fusion.voxel_sizes
        self.sdf_trunc = np.float32(1.0)
        self.l1_loss = nn.L1Loss()

    @staticmethod
    def log_transform(sdf):
        return sdf.sign() * (sdf.abs() + 1.0).log()

    def forward(self, output, mask_occupied, targets):
        losses = {}
        for i, voxel_size_cm in enumerate(self.voxel_sizes):
            pred = output[f"out_tsdf_{voxel_size_cm}"]
            trgt = targets[f"gt_tsdf_{voxel_size_cm}"]["nodes"]

            # Apply masking for the loss function
            mask_observed = trgt.abs() < self.config.mask_occ_weight * self.sdf_trunc
            planes = trgt == self.sdf_trunc
            # Search for truncated planes along the target volume on X, Y, Z directions
            if self.config.mask_plane_loss:
                mask_planes = (
                    planes.all(-1, keepdim=True)
                    | planes.all(-2, keepdim=True)
                    | planes.all(-3, keepdim=True)
                )
                mask = mask_observed | mask_planes
            else:
                mask = mask_observed
            mask &= mask_occupied[i - 1] if (i != 0 and self.config.mask_l1_loss) else True

            if self.config.use_log_transform:
                pred = self.log_transform(pred)
                trgt = self.log_transform(trgt)

            losses[f"{voxel_size_cm}"] = F.l1_loss(pred[mask], trgt[mask])
        return losses
