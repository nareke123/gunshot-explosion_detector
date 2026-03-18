import numpy as np
from src.features.yamnet_extractor import YAMNetExtractor

class DummyTensor:
    def __init__(self, array: np.ndarray):
        self._array = array

    def numpy(self) -> np.ndarray:
        return self._array

class DummyModel:
    def __call__(self, audio: np.ndarray):
        del audio
        embeddings = np.ones((2, 1024), dtype=np.float32)
        return None, DummyTensor(embeddings), None

def test_yamnet_shape():
    extractor = YAMNetExtractor("unused", model=DummyModel())
    y = np.random.randn(16000).astype(np.float32)
    emb = extractor.extract(y, 16000)
    assert emb.shape[1] == 1024
