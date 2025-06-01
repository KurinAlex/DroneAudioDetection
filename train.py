import os

from lightning import Trainer, LightningModule, LightningDataModule
from lightning.pytorch import callbacks, loggers
from torch import nn

import augmentations
import callbacks as custom_callbacks
import utils
from datamodules import DronesAudioDataModule, DronesAudio
from models import *


def train_model(
        model: LightningModule,
        data_module: LightningDataModule,
        epochs: int,
        patience: int,
        logs_dir: str) -> tuple[str, str]:
    timer_callback = custom_callbacks.TimerCallback()
    early_stop_callback = callbacks.EarlyStopping(monitor="val/accuracy", mode="max", patience=patience)
    checkpoint_callback = callbacks.ModelCheckpoint(
        monitor="val/accuracy", mode="max", filename="best", save_last=True
    )

    csv_logger = loggers.CSVLogger(logs_dir, name=model.__class__.__name__)
    tensorboard_logger = loggers.TensorBoardLogger(logs_dir, name=csv_logger.name, version=csv_logger.version)

    trainer = Trainer(
        max_epochs=epochs,
        logger=[csv_logger, tensorboard_logger],
        callbacks=[checkpoint_callback, early_stop_callback, timer_callback],
    )

    trainer.fit(model, datamodule=data_module)

    log_dir_path = csv_logger.log_dir
    best_model_path = checkpoint_callback.best_model_path
    return log_dir_path, best_model_path


def train_models(
        model_types: list[type[AudioClassificationModel]],
        data_root_path: str,
        batch_size: int = 16,
        epochs: int = 30,
        patience: int = 5,
        logs_dir: str = "runs"):
    dataset = DronesAudio(data_root_path)
    data_module = DronesAudioDataModule(
        data_root_path, batch_size=batch_size,
        transform=nn.Sequential(augmentations.RandomNoise(), augmentations.RandomVolume())
    )

    os.makedirs(logs_dir, exist_ok=True)
    run_index = len(os.listdir(logs_dir))
    run_path = os.path.join(logs_dir, f"run_{run_index}")

    metrics_dirs = []
    models_names = []
    for model_type in model_types:
        model = model_type(num_classes=len(dataset.classes), idx_to_class=dataset.idx_to_class)
        log_dir_path, best_model_path = train_model(model, data_module, epochs, patience, run_path)

        metrics_dirs.append(log_dir_path)
        models_names.append(model_type.__name__)

        best_model = model_type.load_from_checkpoint(best_model_path)
        utils.plot_tsne(best_model, dataset, best_model.idx_to_class, log_dir_path)

    summary_path = os.path.join(run_path, "summary")
    os.makedirs(summary_path, exist_ok=True)
    metrics_files = [os.path.join(metrics_dir, "metrics.csv") for metrics_dir in metrics_dirs]
    utils.plot_metrics(metrics_files, models_names, summary_path)


if __name__ == "__main__":
    train_models([M5, VGG, CRNN, Wav2Vec], "./data/Drone_Audio")
