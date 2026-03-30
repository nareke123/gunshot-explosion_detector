import csv
from pathlib import Path
import uuid

from src.data.import_urbansound8k import import_urbansound8k


def test_import_urbansound8k_copies_gunshot_and_normal():
    tmp_dir = Path(".tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    test_root = tmp_dir / f"urbansound_import_{uuid.uuid4().hex}"
    archive_dir = test_root / "archive"
    audio_root = archive_dir / "UrbanSound8K" / "UrbanSound8K" / "audio"
    fold1 = audio_root / "fold1"
    fold1.mkdir(parents=True, exist_ok=True)

    metadata_path = archive_dir / "UrbanSound8K.csv"
    gunshot_file = fold1 / "1000-6-0-0.wav"
    normal_file = fold1 / "1001-3-0-0.wav"
    gunshot_file.write_bytes(b"gunshot")
    normal_file.write_bytes(b"normal")

    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slice_file_name", "fold", "class"])
        writer.writeheader()
        writer.writerow({"slice_file_name": gunshot_file.name, "fold": "1", "class": "gun_shot"})
        writer.writerow({"slice_file_name": normal_file.name, "fold": "1", "class": "dog_bark"})

    raw_dir = test_root / "raw"
    counts = import_urbansound8k(str(archive_dir), str(raw_dir))

    assert counts["gunshot"] == 1
    assert counts["normal"] == 1
    assert counts["missing"] == 0
    assert counts["linked"] + counts["copied"] == 2
    assert counts["skipped_existing"] == 0
    assert (raw_dir / "gunshot" / gunshot_file.name).exists()
    assert (raw_dir / "normal" / normal_file.name).exists()
