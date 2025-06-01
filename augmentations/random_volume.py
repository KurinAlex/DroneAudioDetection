import torch


class RandomVolume(torch.nn.Module):
    def __init__(self, vol_min=0.7, vol_max=1.2):
        super().__init__()

        self.vol_min = vol_min
        self.vol_range = vol_max - vol_min

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        gain = torch.rand(1).item() * self.vol_range + self.vol_min
        waveform = waveform * gain
        waveform = torch.clamp(waveform, -1, 1)
        return waveform
