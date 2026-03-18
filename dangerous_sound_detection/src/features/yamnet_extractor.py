import importlib
import numpy as np

class YAMNetExtractor:
    """YAMNet feature extractor."""
    def __init__(self, model_url: str, model=None):
        self.model_url = model_url
        self.model = model or self._load_model()

    def _load_model(self):
        try:
            hub = importlib.import_module("tensorflow_hub")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "tensorflow-hub is not installed. Install the project requirements to use YAMNet."
            ) from exc
        return hub.load(self.model_url)

    def extract(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Extract embeddings from audio."""
        if sr != 16000:
            raise ValueError("YAMNet expects 16 kHz audio.")
        audio = np.asarray(audio, dtype=np.float32)
        _, embeddings, _ = self.model(audio)
        if hasattr(embeddings, "numpy"):
            return embeddings.numpy()
        return np.asarray(embeddings, dtype=np.float32)
