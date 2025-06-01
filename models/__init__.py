from models.base import AudioClassificationModel
from models.crnn import CRNN
from models.m5 import M5
from models.vgg import VGG
from models.wav2vec import Wav2Vec

__all__ = ["AudioClassificationModel", "M5", "VGG", "CRNN", "Wav2Vec"]
