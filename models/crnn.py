from torch import nn
from torchaudio.transforms import Spectrogram

from models.base import AudioClassificationModel


class CRNN(AudioClassificationModel):

    def __init__(self, num_classes: int, input_channels: int = 1, idx_to_class: dict[int, str] = None):
        super().__init__(num_classes, initial_lr=1e-3, idx_to_class=idx_to_class)

        self.cnn = nn.Sequential(
            Spectrogram(),
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), stride=(2, 1)),
        )

        self.rnn = nn.LSTM(
            input_size=6400,
            hidden_size=256,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )

        self.classifier = nn.Linear(10240, num_classes)

    def get_embeddings(self, x):
        # apply CNN
        x = self.cnn(x)

        # reshape for RNN
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(b, w, c * h)

        # apply RNN
        x, _ = self.rnn(x)
        x = x.flatten(1)
        return x

    def classify(self, embeddings):
        y = self.classifier(embeddings)
        return y
