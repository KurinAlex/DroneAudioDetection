import json
import os
import sys
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import MaxNLocator
from sklearn.manifold import TSNE
from torch import nn
from tqdm import tqdm


def plot_metrics(csv_files: Sequence[str], models_names: Sequence[str], destination_path: str, ndigits=2):
    logs = {}
    metrics = set[str]()
    for file_path, model_name in zip(csv_files, models_names):
        df = pd.read_csv(file_path)
        logs[model_name] = df
        metrics.update(df.columns)

    metrics.remove("epoch")
    metrics.remove("step")

    metrics_summary = {models_name: {"epochs": len(logs[models_name]["epoch"].unique())} for models_name in
                       models_names}

    markers = ["P", "X", "s", "o", "*"]
    for i, metric in enumerate(metrics):
        fig, ax = plt.subplots()
        for j, (model_name, df) in enumerate(logs.items()):
            if metric not in df.columns:
                continue

            data = df[["epoch", metric]].dropna()
            ax.plot(data["epoch"], data[metric], label=model_name, marker=markers[j % len(markers)])

            if metric in ["train/lr", "train/loss", "val/loss"]:
                continue

            if metric == "train/time":
                metrics_summary[model_name][metric] = round(data[metric].sum() / 60, ndigits)
            else:
                metrics_summary[model_name][metric] = round(data[metric].max() * 100, ndigits)

        ax.set_title(metric)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend()

        metric_file_name = metric.replace("/", "_")
        path = os.path.join(destination_path, f"{metric_file_name}.png")
        fig.savefig(path)

    metrics_file_path = os.path.join(destination_path, "metrics.json")
    with open(metrics_file_path, "w") as metrics_file:
        json.dump(metrics_summary, metrics_file, indent=4)


def plot_tsne(model: nn.Module, dataset: Sequence[tuple[torch.Tensor, int]], idx_to_class: dict[int, str],
              destination_path: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()

    embeddings = []
    targets = []
    with torch.inference_mode():
        for sample, target in tqdm(dataset, desc="Computing embeddings", file=sys.stdout):
            waveform = sample.unsqueeze(0).to(device)
            logits = model.get_embeddings(waveform)
            logits = logits[0].cpu().numpy()

            embeddings.append(logits)
            targets.append(target)

    tsne = TSNE()
    embeddings_2d = tsne.fit_transform(np.array(embeddings))

    targets = np.array(targets)

    fig, ax = plt.subplots()
    for class_id in np.unique(targets):
        idx = targets == class_id
        label = idx_to_class.get(class_id, f"Class {class_id}")
        ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label, alpha=0.6)

    ax.legend()
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.grid(True)

    fig.suptitle("t-SNE")
    fig.savefig(os.path.join(destination_path, "t-sne.png"))
