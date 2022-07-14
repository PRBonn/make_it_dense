import pytorch_lightning as pl
import torch

from make_it_dense.loss import SDFLoss
from make_it_dense.models.atlas import AtlasNet
from make_it_dense.utils.config import MkdConfig


class CompletionNet(pl.LightningModule):
    def __init__(self, config: MkdConfig):
        super().__init__()
        self.config = config
        self.model = AtlasNet(self.config)
        self.loss = SDFLoss(self.config)
        self.lr = self.config.optimization.lr
        self.voxel_sizes = self.config.fusion.voxel_sizes
        self.voxel_trunc = self.config.fusion.voxel_trunc
        self.save_hyperparameters(config)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.config.optimization.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "train/train_loss",
        }

    def step(self, batch, batch_idx, mode: str):
        inputs, targets = batch
        outputs, masks = self(inputs["nodes"])
        pred_tsdf_t = outputs["out_tsdf_10"]

        # Compute Loss function
        losses = self.loss(outputs, masks, targets)
        loss = sum(losses.values())

        self.log(mode + "/train_loss", loss)
        for key in losses.keys():
            self.log(mode + "/losses/" + key, losses[key])

        # Log some metrics
        self.log(mode + "/metrics/max_sdf", pred_tsdf_t.max())
        self.log(mode + "/metrics/min_sdf", pred_tsdf_t.min())
        self.log(mode + "/metrics/mean_sdf", pred_tsdf_t.mean())

        return loss

    def training_step(self, train_batch, batch_idx):
        return self.step(train_batch, batch_idx, mode="train")

    def validation_step(self, val_batch, batch_idx):
        self.step(val_batch, batch_idx, mode="val")

    @torch.no_grad()
    def predict_step(self, input_tsdf_t):
        return self(input_tsdf_t)[0]
