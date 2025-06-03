from torch import nn

from models.base import AudioClassificationModel


class M5(AudioClassificationModel):

    def __init__(self, num_classes: int, input_channels: int = 1, idx_to_class: dict[int, str] = None):
        super().__init__(num_classes, initial_lr=1e-3, idx_to_class=idx_to_class)

        n_channel = 128

        self.features = nn.Sequential(
            nn.Conv1d(input_channels, n_channel, kernel_size=80, stride=16),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(n_channel, n_channel, kernel_size=3),
            nn.BatchNorm1d(n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3),
            nn.BatchNorm1d(2 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.Conv1d(2 * n_channel, 4 * n_channel, kernel_size=3),
            nn.BatchNorm1d(4 * n_channel),
            nn.ReLU(),
            nn.MaxPool1d(4),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(4 * n_channel, 1000),
            nn.Dropout1d(0.3),
            nn.BatchNorm1d(1000),
        )
        self.classifier = nn.Linear(1000, num_classes)

    def get_embeddings(self, x):
        logits = self.features(x)
        return logits

    def classify(self, embeddings):
        y = self.classifier(embeddings)
        return y
