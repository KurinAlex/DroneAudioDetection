# Drone Audio Detection

This repository contains the code for my bachelor's thesis:  
**"Technologies of UAV Identification by Audio Characteristics in Challenging Acoustic Conditions."**

## Project Overview

The goal of this project is to detect and classify the type of UAV (if any) present in an input audio stream. This is framed as a machine learning classification task. Various deep learning architectures were implemented and evaluated using standard training and evaluation metrics.

Each model is trained to classify 1-second audio clips into one of the following classes:

- `background`: No UAV detected;
- `fpv`: FPV drone detected;
- `shahed`: [Shahed-136](https://en.wikipedia.org/wiki/HESA_Shahed_136) drone detected.

In addition to model training and evaluation, the project includes a desktop application that continuously listens through a microphone and uses a trained model to detect UAV sounds in real-time.

---

## Data Sources

- [ESC-50](https://github.com/karolpiczak/ESC-50): used for samples of the `background` class.
- [DroneAudioDataset](https://github.com/saraalemadi/DroneAudioDataset): used for samples of the `fpv` class.
- Public news sources (e.g. Telegram, YouTube):used for `shahed` samples. Original audio clips can be found in the [`data/Drone_Audio_shahed`](data/Drone_Audio_shahed) directory.

---

## Models Implemented

- **M5**: based on the architecture from [this paper](https://arxiv.org/pdf/1610.00087)
- **VGG-16**: from the original [VGG paper](https://arxiv.org/pdf/1409.1556)
- **Wav2Vec 2.0**: based on [this paper](https://arxiv.org/pdf/2006.11477)
- **Custom CRNN**: a recurrent convolutional neural network implemented [in this file](models/crnn.py)

---

## Frameworks & Libraries Used

- [PyTorch](https://pytorch.org/) and [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) — for deep learning model development and training.
- [scikit-learn](https://scikit-learn.org/) — for metric calculations.
- [Pydub](https://pydub.com/), [MoviePy](https://zulko.github.io/moviepy/), and [PyAudio](https://people.csail.mit.edu/hubert/pyaudio/) — for audio/video processing and recording.
- [pandas](https://pandas.pydata.org/) and [Matplotlib](https://matplotlib.org/) — for data handling and visualization.
- [tkinter](https://docs.python.org/3/library/tkinter.html) — for building the desktop app interface.

---

## Model Comparison Results

| Model         | Accuracy | Recall | Precision | F1     | Training Time (min)  |
|---------------|----------|--------|-----------|--------|----------------------|
| M5            | 94.94%   | 94.94% | 95.07%    | 94.94% | **7.52**             |
| VGG-16        | **95.66%** | **95.66%** | **96.06%** | **95.33%** | 67.2      |
| Custom CRNN   | 95.58%   | 95.58% | 95.50%    | 95.05% | 24.09                |
| Wav2Vec 2.0   | 89.93%   | 89.93% | 90.91%    | 89.11% | 113.05               |

---

## Running the Project

### Training

To train models, run the [train.py](train.py) script.  
This will create a `/runs` directory with all training and evaluation metrics saved in both CSV and [TensorBoard](https://www.tensorflow.org/tensorboard) formats.  
Additionally, you will find metric plots and [T-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) visualizations for model embeddings.

### Desktop Application

To run the desktop app, execute [app.py](app.py).  
The app uses the microphone to classify live audio using trained models.

**Note:**  
The app expects a `/checkpoints` directory containing trained models.  
Each model should be saved with a filename matching the class name of the model (e.g., `M5.ckpt` for the M5 model).  
These files can be obtained from training results in `/runs` directory and should be in a format compatible with PyTorch.

---
