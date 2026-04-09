import typer
from pathlib import Path
import json
from src.config import load_config
from src.data.import_urbansound8k import import_urbansound8k
from src.data.prepare_dataset import prepare_dataset
from src.data.make_splits import make_splits
from src.data.dataset_summary import dataset_summary
from src.features.feature_pipeline import extract_all_features
from src.features.yamnet_extractor import YAMNetExtractor
from src.training.train_classifier import train_classifier
from src.training.evaluate import evaluate_file_level, evaluate_model
from src.training.torch_model import load_torch_classifier
from src.utils.io import load_features, load_labels, resolve_split_path
from src.inference.predict_audio import predict_audio
from src.inference.predict_video import predict_video
from src.inference.predict_stream import predict_stream
import joblib

app = typer.Typer()


def _split_paths(config: dict, split: str) -> tuple[str, str]:
    return (
        resolve_split_path(config['features_path'], split),
        resolve_split_path(config['labels_path'], split),
    )

@app.command(name="prepare-dataset")
def prepare_dataset_cmd(config: str = typer.Option("configs/default.yaml", "--config")):
    config = load_config(config)
    prepare_dataset(config['data_raw_dir'], config['data_processed_dir'], config.get('label_groups'))

@app.command(name="import-urbansound8k")
def import_urbansound8k_cmd(
    archive: str = typer.Option("data/archive", "--archive"),
    raw_dir: str = typer.Option("data/raw", "--raw-dir"),
):
    counts = import_urbansound8k(archive, raw_dir)
    print(counts)

@app.command(name="make-splits")
def make_splits_cmd(config: str = typer.Option("configs/default.yaml", "--config")):
    config = load_config(config)
    make_splits(
        config['data_processed_dir'],
        config['data_splits_dir'],
        config['train_split'],
        config['val_split'],
        config['test_split'],
        config.get('model_class_names', config['class_names']),
    )

@app.command(name="dataset-summary")
def dataset_summary_cmd(config: str = typer.Option("configs/default.yaml", "--config")):
    config = load_config(config)
    dataset_summary(config['data_splits_dir'])

@app.command(name="extract-features")
def extract_features_cmd(
    config: str = typer.Option("configs/default.yaml", "--config"),
    split: str = typer.Option("train", "--split"),
):
    config = load_config(config)
    extractor = YAMNetExtractor(
        device=config.get('device', 'auto'),
        n_fft=config.get('n_fft', 512),
        hop_length=config.get('stft_hop_length', 160),
        win_length=config.get('win_length', 400),
    )
    features_path, labels_path = _split_paths(config, split)
    extract_config = dict(config)
    extract_config['features_path'] = features_path
    extract_config['labels_path'] = labels_path
    extract_all_features(Path(config['data_splits_dir']) / f'{split}.csv', extractor, extract_config)

@app.command(name="train")
def train_cmd(
    config: str = typer.Option("configs/default.yaml", "--config"),
    split: str = typer.Option("train", "--split"),
):
    config = load_config(config)
    features_path, labels_path = _split_paths(config, split)
    features = load_features(features_path)
    labels = load_labels(labels_path)
    train_classifier(features, labels, config)

@app.command(name="evaluate")
def evaluate_cmd(
    config: str = typer.Option("configs/default.yaml", "--config"),
    split: str = typer.Option("val", "--split"),
):
    config = load_config(config)
    clf, device = load_torch_classifier(config['model_save_path'], config.get('device', 'auto'))
    le = joblib.load(config['label_encoder_path'])
    features_path, labels_path = _split_paths(config, split)
    features = load_features(features_path)
    labels = load_labels(labels_path)
    metrics = {
        'window_level': evaluate_model(clf, features, labels, le, device),
        'clip_level': evaluate_file_level(
            clf,
            Path(config['data_splits_dir']) / f'{split}.csv',
            config,
            le,
            device,
        ),
    }
    print(json.dumps(metrics, indent=2))


@app.command(name="run-pipeline")
def run_pipeline_cmd(config: str = typer.Option("configs/default.yaml", "--config")):
    config = load_config(config)
    prepare_dataset(config['data_raw_dir'], config['data_processed_dir'], config.get('label_groups'))
    make_splits(
        config['data_processed_dir'],
        config['data_splits_dir'],
        config['train_split'],
        config['val_split'],
        config['test_split'],
        config.get('model_class_names', config['class_names']),
    )

    extractor = YAMNetExtractor(
        device=config.get('device', 'auto'),
        n_fft=config.get('n_fft', 512),
        hop_length=config.get('stft_hop_length', 160),
        win_length=config.get('win_length', 400),
    )
    for split in ('train', 'val'):
        features_path, labels_path = _split_paths(config, split)
        extract_config = dict(config)
        extract_config['features_path'] = features_path
        extract_config['labels_path'] = labels_path
        extract_all_features(Path(config['data_splits_dir']) / f'{split}.csv', extractor, extract_config)

    train_features_path, train_labels_path = _split_paths(config, 'train')
    train_classifier(
        load_features(train_features_path),
        load_labels(train_labels_path),
        config,
    )

    clf, device = load_torch_classifier(config['model_save_path'], config.get('device', 'auto'))
    le = joblib.load(config['label_encoder_path'])
    val_features_path, val_labels_path = _split_paths(config, 'val')
    metrics = evaluate_model(
        clf,
        load_features(val_features_path),
        load_labels(val_labels_path),
        le,
        device,
    )
    file_metrics = evaluate_file_level(
        clf,
        Path(config['data_splits_dir']) / 'val.csv',
        config,
        le,
        device,
    )
    print(json.dumps({'window_level': metrics, 'clip_level': file_metrics}, indent=2))

@app.command(name="predict-audio")
def predict_audio_cmd(
    input: str = typer.Option(..., "--input"),
    output: str = typer.Option(..., "--output"),
    config: str = typer.Option("configs/default.yaml", "--config"),
    confidence_threshold: float | None = typer.Option(None, "--confidence-threshold"),
    include_windows: bool = typer.Option(False, "--include-windows"),
):
    config = load_config(config)
    result = predict_audio(
        input,
        config,
        confidence_threshold=confidence_threshold,
        include_windows=include_windows,
    )
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

@app.command(name="predict-video")
def predict_video_cmd(
    input: str = typer.Option(..., "--input"),
    output: str = typer.Option(..., "--output"),
    config: str = typer.Option("configs/default.yaml", "--config"),
    confidence_threshold: float | None = typer.Option(None, "--confidence-threshold"),
    include_windows: bool = typer.Option(False, "--include-windows"),
):
    config = load_config(config)
    result = predict_video(
        input,
        config,
        confidence_threshold=confidence_threshold,
        include_windows=include_windows,
    )
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

@app.command(name="predict-stream")
def predict_stream_cmd(
    rtsp: str = typer.Option(..., "--rtsp"),
    config: str = typer.Option("configs/default.yaml", "--config"),
):
    config = load_config(config)
    predict_stream(rtsp, config)

if __name__ == "__main__":
    app()
