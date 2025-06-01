import abc

import torchmetrics.classification as metrics
from lightning import LightningModule
from torch import Tensor, nn, optim
from torchmetrics import MetricCollection


class AudioClassificationModel(abc.ABC, LightningModule):
    def __init__(self, num_classes: int, initial_lr: float, lr_gamma=0.9, idx_to_class: dict[int, str] = None):
        super().__init__()

        self.save_hyperparameters()

        self.num_classes = num_classes
        self.initial_lr = initial_lr
        self.lr_gamma = lr_gamma
        self.idx_to_class = idx_to_class if idx_to_class else {}

        self.loss = nn.CrossEntropyLoss()
        self.metrics = MetricCollection(
            {
                "val/accuracy": metrics.MulticlassAccuracy(num_classes),
                "val/f1_score": metrics.MulticlassF1Score(num_classes),
                "val/precision": metrics.MulticlassPrecision(num_classes),
                "val/recall": metrics.MulticlassRecall(num_classes),
            }
        )

    def forward(self, x: Tensor) -> Tensor:
        embeddings = self.get_embeddings(x)
        logits = self.classify(embeddings)
        return logits

    def decode(self, logits: Tensor) -> list[dict[str | int, float]]:
        probabilities = nn.functional.softmax(logits, dim=1)
        result = [{self.idx_to_class.get(i, i): p.item() for i, p in enumerate(prob)} for prob in probabilities]
        return result

    @abc.abstractmethod
    def get_embeddings(self, x: Tensor) -> Tensor:
        pass

    @abc.abstractmethod
    def classify(self, embeddings: Tensor) -> Tensor:
        pass

    def training_step(self, batch: tuple[Tensor, Tensor]) -> Tensor:
        data, target = batch

        output = self.forward(data)
        output = output.squeeze()

        loss = self.loss(output, target)

        optimizers = self.optimizers()
        optimizer = optimizers[0] if isinstance(optimizers, list) else optimizers
        lr = optimizer.optimizer.param_groups[0]["lr"]

        self.log_dict({"train/loss": loss, "train/lr": lr}, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor]) -> None:
        data, target = batch

        output = self.forward(data)
        output = output.squeeze()

        loss = self.loss(output, target)

        self.log("val/loss", loss)
        self.log_dict(self.metrics(output, target))

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.initial_lr)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.lr_gamma)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
