from torch import nn
from torchaudio import models

from models.base import AudioClassificationModel


class Wav2Vec(AudioClassificationModel):
    def __init__(self, num_classes: int, idx_to_class: dict[int, str] = None):
        super().__init__(num_classes, initial_lr=1e-5, idx_to_class=idx_to_class)

        self.wav2vec = models.wav2vec2_base(aux_num_out=num_classes)
        self.classifier = self.wav2vec.aux
        self.wav2vec.aux = None

    def get_embeddings(self, x):
        x = x.squeeze(1)
        embeddings = self.wav2vec(x)[0].mean(axis=1)
        embeddings = nn.functional.normalize(embeddings)
        return embeddings

    def classify(self, embeddings):
        y = self.classifier(embeddings)
        return y
