import numpy as np
from src.features.yamnet_extractor import TorchSpectrogramExtractor

def test_yamnet_shape():
    extractor = TorchSpectrogramExtractor(device='cpu')
    y = np.random.randn(16000).astype(np.float32)
    emb = extractor.extract(y, 16000)
    assert emb.shape[1] == extractor.feature_dim
