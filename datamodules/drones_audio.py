from collections import Counter
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torchaudio
from lightning import LightningDataModule
from torch.utils import data
from torchvision.datasets.folder import DatasetFolder


class DronesAudio(DatasetFolder):
    def __init__(self, root: str | Path, transform: Callable = None):
        super().__init__(
            root,
            loader=lambda path: torchaudio.load(path)[0],
            extensions=(".wav",),
            transform=transform,
        )

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}


class DronesAudioDataModule(LightningDataModule):
    def __init__(self, root: str | Path, batch_size: int, transform: Callable = None, train_size: float = 0.8):
        super().__init__()

        self.root = root
        self.batch_size = batch_size
        self.transform = transform
        self.train_size = train_size

        self.train_dataset: data.Dataset | None = None
        self.val_dataset: data.Dataset | None = None
        self.train_sampler: data.Sampler | None = None
        self.val_sampler: data.Sampler | None = None

    def setup(self, stage=None):
        dataset = DronesAudio(self.root, self.transform)

        test_size = 1.0 - self.train_size
        train_indices, val_indices = data.random_split(range(len(dataset)), [self.train_size, test_size])

        train_targets = np.array(dataset.targets)[train_indices]
        val_targets = np.array(dataset.targets)[val_indices]

        train_class_counter = Counter(train_targets)
        val_class_counter = Counter(val_targets)

        self.train_dataset = data.Subset(dataset, train_indices)
        self.val_dataset = data.Subset(dataset, val_indices)

        self.train_sampler = data.WeightedRandomSampler(
            [1.0 / train_class_counter[dataset.targets[i]] for i in train_indices], len(train_indices)
        )
        self.val_sampler = data.WeightedRandomSampler(
            [1.0 / val_class_counter[dataset.targets[i]] for i in val_indices], len(val_indices)
        )

    @classmethod
    def collate_fn(cls, batch):
        tensors, targets = zip(*batch)
        tensors = torch.stack(tensors)
        targets = torch.tensor(targets)
        return tensors, targets

    def train_dataloader(self):
        return data.DataLoader(
            self.train_dataset,
            sampler=self.train_sampler,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
        )

    def val_dataloader(self):
        return data.DataLoader(
            self.val_dataset,
            sampler=self.val_sampler,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
        )
