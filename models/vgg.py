import torchvision.models as models
from torch import nn
from torchaudio.transforms import Spectrogram

from models.base import AudioClassificationModel


class VGG(AudioClassificationModel):
    def __init__(self, num_classes: int, num_channels: int = 1, idx_to_class: dict[int, str] = None):
        super().__init__(num_classes, initial_lr=1e-4, idx_to_class=idx_to_class)

        model = models.vgg16(weights="DEFAULT")

        features = model.features
        first_conv = features[0]
        features[0] = nn.Conv2d(
            num_channels,
            first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            padding=first_conv.padding,
        )

        classifier = model.classifier
        last_linear = classifier[-1]
        classifier[-1] = nn.Linear(last_linear.in_features, num_classes)

        self.spectrogram = Spectrogram()
        self.features = features
        self.average_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = classifier

    def get_embeddings(self, x):
        spec = self.spectrogram(x)
        embeddings = self.features(spec)
        embeddings = self.average_pool(embeddings)
        embeddings = embeddings.flatten(1)
        return embeddings

    def classify(self, embeddings):
        y = self.classifier(embeddings)
        return y
