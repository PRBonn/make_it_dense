# Inspired by https://github.com/magicleap/Atlas
import torch
import torch.nn as nn
import torch.nn.functional as F

from make_it_dense.models.blocks3d import FeatureExtractor, Unet3D
from make_it_dense.utils.config import MkdConfig


class AtlasNet(nn.Module):
    def __init__(self, config: MkdConfig):
        super().__init__()
        self.config = config
        self.voxel_sizes = self.config.fusion.voxel_sizes
        self.occ_th = self.config.model.occ_th
        self.f_maps = self.config.model.f_maps
        self.layers_down = self.config.model.layers_down
        self.layers_up = self.config.model.layers_up

        # Network
        self.feature_extractor = FeatureExtractor(channels=self.f_maps[0])
        self.unet = Unet3D(
            channels=self.f_maps,
            layers_down=self.layers_down,
            layers_up=self.layers_up,
        )
        self.decoders = nn.ModuleList(
            [nn.Conv3d(c, 1, 1, bias=False) for c in self.f_maps[:-1]][::-1]
        )

    def forward(self, xs):
        feats = self.feature_extractor(xs)
        out = self.unet(feats)

        output = {}
        mask_occupied = []
        for i, (decoder, x) in enumerate(zip(self.decoders, out)):
            # regress the TSDF
            tsdf = torch.tanh(decoder(x)) * 1.05

            # use previous scale to sparsify current scale
            if i > 0:
                tsdf_prev = output[f"out_tsdf_{self.voxel_sizes[i - 1]}"]
                tsdf_prev = F.interpolate(tsdf_prev, scale_factor=2)
                mask_truncated = tsdf_prev.abs() >= self.occ_th[i - 1]
                tsdf[mask_truncated] = tsdf_prev[mask_truncated].sign()
                mask_occupied.append(~mask_truncated)
            output[f"out_tsdf_{ self.voxel_sizes[i]}"] = tsdf
        return output, mask_occupied
