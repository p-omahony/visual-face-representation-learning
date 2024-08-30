import time

import boto3
import lightning as L
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models import ResNet18_Weights, resnet18

s3_client = boto3.client("s3")


class FaceLearner(L.LightningModule):
    def __init__(self, contrastive_loss_margin):
        super(FaceLearner, self).__init__()
        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = nn.Identity()

        self.contrastive_loss_margin = contrastive_loss_margin

        self.training_step_losses = []
        self.validation_step_losses = []
        self.validation_step_labels = []

        self.start_time = 0.0

    def forward(self, ims):
        out = self.resnet(ims)

        return out

    def criterion(self, emb1, emb2, label):
        distance = (emb1 - emb2).pow(2).sum(1)
        pos = (1 - label) * distance
        neg = (label) * torch.relu(self.contrastive_loss_margin - distance)
        return torch.mean(pos + neg)

    def training_step(self, batch, batch_idx):
        anchor, comp, label = batch
        anchor_emb = self.forward(anchor)
        comp_emb = self.forward(comp)

        loss = self.criterion(anchor_emb, comp_emb, label)

        self.training_step_losses.append(loss)

        return loss

    def on_train_epoch_start(self):
        self.start_time = time.time()

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_losses).mean()
        duration = time.time() - self.start_time
        print(
            f"Epoch {self.current_epoch} - train_loss={avg_loss} - duration={duration/60:.2f} m"
        )

        self.log("train_loss", avg_loss, prog_bar=True, logger=True)

        self.training_step_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        lr_reducer = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=5, verbose=True
        )
        scheduler = {
            "scheduler": lr_reducer,
            "monitor": "train_loss",  # The metric to monitor
            "interval": "epoch",  # How often to update the learning rate
            "frequency": 1,  # How often to check the metric
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


class FaceLearnerTriplet(L.LightningModule):
    def __init__(self, hyperparameters):
        super(FaceLearnerTriplet, self).__init__()
        self.hyperparameters = hyperparameters
        self.save_hyperparameters(self.hyperparameters.to_dict())

        if self.hyperparameters.model.pretrained:
            self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.resnet = resnet18(weights=None)
        self.resnet.fc = nn.Identity()
        
        if self.hyperparameters.model.projector.active:
            embed_dim = self.hyperparameters.model.projector.embed_dim
            self.projector = nn.Sequential(
                nn.BatchNorm1d(512),
                nn.Linear(512, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(embed_dim, embed_dim),
            )

        self.start_time = 0.0

        self.training_step_losses = []
        self.validation_step_losses = []
        self.validation_step_losses_hol = []
        self.validation_step_labels = []

    def forward(self, im):
        out = self.resnet(im)
        if self.hyperparameters.model.projector.active:
            out = self.projector(out)

        return out

    def criterion(self, anchor_emb, positive_emb, negative_emb):
        distance_positive = (anchor_emb - positive_emb).pow(2).sum(1)
        distance_negative = (anchor_emb - negative_emb).pow(2).sum(1)
        margin = self.hyperparameters.training.loss.margin
        losses = torch.relu(distance_positive - distance_negative + margin)
        return losses.mean()

    def training_step(self, batch, batch_idx):
        anchor_im, positive_im, negative_im = batch

        anchor_emb = self.forward(anchor_im)
        positive_emb = self.forward(positive_im)
        negative_emb = self.forward(negative_im)

        loss = self.criterion(anchor_emb, positive_emb, negative_emb)

        self.training_step_losses.append(loss)

        return loss

    def validation_step(self, batch, batch_idx):
        anchor_im, positive_im, negative_im = batch

        anchor_emb = self.forward(anchor_im)
        positive_emb = self.forward(positive_im)
        negative_emb = self.forward(negative_im)

        loss = self.criterion(anchor_emb, positive_emb, negative_emb)

        self.validation_step_losses.append(loss)

    def on_train_epoch_start(self):
        self.start_time = time.time()

    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.training_step_losses).mean()

        duration = time.time() - self.start_time
        print(
            f"Epoch {self.current_epoch} - train_loss={avg_loss} - duration={duration/60:.2f} m"
        )

        self.log("train_loss", avg_loss, prog_bar=True, logger=True)

        self.training_step_losses.clear()

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_losses).mean()

        print(f"Val Epoch {self.current_epoch} - val_loss={avg_loss}")

        self.log("val_loss", avg_loss, prog_bar=True, logger=True)

        self.validation_step_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hyperparameters.training.learning_rate.start_value,
        )
        if self.hyperparameters.training.learning_rate.reduce:
            scheduler = {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=self.hyperparameters.training.learning_rate.reduce_factor,
                    patience=self.hyperparameters.training.learning_rate.reduce_patience,
                    verbose=True,
                ),
                "monitor": "train_loss",  # The metric to monitor
                "interval": "epoch",  # How often to update the learning rate
                "frequency": 1,  # How often to check the metric
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler}
        return optimizer
