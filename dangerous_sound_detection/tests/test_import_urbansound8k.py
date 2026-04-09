import csv
from pathlib import Path
import uuid

from src.data.import_urbansound8k import import_urbansound8k


def test_import_urbansound8k_maps_classes_into_event_taxonomy():
    tmp_dir = Path(".tmp")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    test_root = tmp_dir / f"urbansound_import_{uuid.uuid4().hex}"
    archive_dir = test_root / "archive"
    audio_root = archive_dir / "UrbanSound8K" / "UrbanSound8K" / "audio"
    fold1 = audio_root / "fold1"
    fold1.mkdir(parents=True, exist_ok=True)

    metadata_path = archive_dir / "UrbanSound8K.csv"
    impulse_file = fold1 / "1000-6-0-0.wav"
    background_file = fold1 / "1001-0-0-0.wav"
    child_file = fold1 / "1002-2-0-0.wav"
    music_file = fold1 / "1003-9-0-0.wav"
    drill_file = fold1 / "1004-4-0-0.wav"
    engine_file = fold1 / "1005-5-0-0.wav"
    skipped_file = fold1 / "1006-3-0-0.wav"
    impulse_file.write_bytes(b"impulse")
    background_file.write_bytes(b"background")
    child_file.write_bytes(b"child")
    music_file.write_bytes(b"music")
    drill_file.write_bytes(b"drill")
    engine_file.write_bytes(b"engine")
    skipped_file.write_bytes(b"skip")

    with metadata_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["slice_file_name", "fold", "class"])
        writer.writeheader()
        writer.writerow({"slice_file_name": impulse_file.name, "fold": "1", "class": "gun_shot"})
        writer.writerow({"slice_file_name": background_file.name, "fold": "1", "class": "air_conditioner"})
        writer.writerow({"slice_file_name": child_file.name, "fold": "1", "class": "children_playing"})
        writer.writerow({"slice_file_name": music_file.name, "fold": "1", "class": "street_music"})
        writer.writerow({"slice_file_name": drill_file.name, "fold": "1", "class": "drilling"})
        writer.writerow({"slice_file_name": engine_file.name, "fold": "1", "class": "engine_idling"})
        writer.writerow({"slice_file_name": skipped_file.name, "fold": "1", "class": "dog_bark"})

    raw_dir = test_root / "raw"
    counts = import_urbansound8k(str(archive_dir), str(raw_dir))

    assert counts["danger_noise"] == 3
    assert counts["normal"] == 0
    assert counts["safety_noise"] == 1
    assert counts["discipline_violation"] == 2
    assert counts["missing"] == 0
    assert counts["skipped_unknown_class"] == 1
    assert counts["linked"] + counts["copied"] == 6
    assert counts["skipped_existing"] == 0
    assert (raw_dir / "danger_noise" / impulse_file.name).exists()
    assert (raw_dir / "danger_noise" / drill_file.name).exists()
    assert (raw_dir / "danger_noise" / engine_file.name).exists()
    assert (raw_dir / "safety_noise" / background_file.name).exists()
    assert (raw_dir / "discipline_violation" / child_file.name).exists()
    assert (raw_dir / "discipline_violation" / music_file.name).exists()
    assert not (raw_dir / "discipline_violation" / skipped_file.name).exists()
