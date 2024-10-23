import lightning as L
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torchmetrics import Accuracy, MaxMetric, MeanMetric
from torchmetrics.classification import Accuracy


class TimmClassifier(L.LightningModule):
    def __init__(
        self,
        base_model: str,
        num_classes: int,
        pretrained: bool = False,
        lr: float = 1e-3,
        optimizer: str = "Adam",
        weight_decay: float = 1e-5,
        scheduler_patience: int = 3,
        scheduler_factor: float = 0.1,
        drop_path_rate: float = 0.0,
        head_init_scale: float = 1.0,
        horizontal_flip: bool = False,
        random_crop: bool = False,
        random_rotation: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the model
        self.model = timm.create_model(
            base_model,
            pretrained=pretrained,
            num_classes=num_classes,
            drop_path_rate=drop_path_rate,
            head_init_scale=head_init_scale,
            **kwargs,
        )

        # Define loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        self.val_acc_best = MaxMetric()
        self.test_acc_best = MaxMetric()

    def forward(self, x):
        return self.model(x)

    def model_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log(
            "train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log(
            "val/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True
        )

    def test_step(self, batch, batch_idx):
        loss, preds, targets = self.model_step(batch)
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log(
            "test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        self.test_acc_best(self.test_acc.compute())
        self.log(
            "test/acc_best", self.test_acc_best.compute(), on_epoch=True, prog_bar=True
        )

    def configure_optimizers(self):
        optimizer_class = getattr(optim, self.hparams.optimizer)
        optimizer = optimizer_class(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.hparams.scheduler_factor,
            patience=self.hparams.scheduler_patience,
            min_lr=1e-6,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }
