import torch
from torchaudio import functional


class RandomNoise(torch.nn.Module):
    def __init__(self, snr_min=5, snr_max=20):
        super().__init__()

        self.snr_min = snr_min
        self.snr_range = snr_max - snr_min

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        noise = torch.rand_like(waveform)
        snr = torch.rand(waveform.shape[:-1]) * self.snr_range + self.snr_min
        waveform = functional.add_noise(waveform, noise, snr)
        return waveform
