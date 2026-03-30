import numpy as np
import torch

def resolve_device(device: str = "auto") -> torch.device:
    """Resolve device string to a torch device with CUDA fallback."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


class TorchSpectrogramExtractor:
    """Extract log-spectrogram features using PyTorch."""
    def __init__(
        self,
        device: str = "auto",
        n_fft: int = 512,
        hop_length: int = 160,
        win_length: int = 400,
    ):
        self.device = resolve_device(device)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.feature_dim = (n_fft // 2) + 1
        self.window = torch.hann_window(self.win_length, device=self.device)

    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract per-frame log-spectrogram features."""
        if sr != 16000:
            raise ValueError("TorchSpectrogramExtractor expects 16 kHz audio.")

        waveform = torch.as_tensor(np.asarray(audio, dtype=np.float32), device=self.device)
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=0)
        if waveform.numel() == 0:
            return np.zeros((1, self.feature_dim), dtype=np.float32)
        if torch.max(torch.abs(waveform)) > 0:
            waveform = waveform / torch.max(torch.abs(waveform))

        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        power = stft.abs().pow(2.0)
        log_spec = torch.log1p(power).transpose(0, 1)
        return log_spec.detach().cpu().numpy().astype(np.float32)


# Backwards-compatible alias for the rest of the codebase.
YAMNetExtractor = TorchSpectrogramExtractor
