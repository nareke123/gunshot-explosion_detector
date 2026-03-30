from src.utils.io import resolve_split_path


def test_resolve_split_path_appends_split_before_suffix():
    assert resolve_split_path("data/processed/features.npy", "val") == "data\\processed\\features_val.npy"


def test_resolve_split_path_without_split_returns_original():
    assert resolve_split_path("data/processed/features.npy", None) == "data/processed/features.npy"
