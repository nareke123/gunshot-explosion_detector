import typer
from pathlib import Path
import json
from src.config import load_config
from src.data.prepare_dataset import prepare_dataset
from src.data.make_splits import make_splits
from src.data.dataset_summary import dataset_summary
from src.features.feature_pipeline import extract_all_features
from src.features.yamnet_extractor import YAMNetExtractor
from src.training.train_classifier import train_classifier
from src.training.evaluate import evaluate_model
from src.utils.io import load_features, load_labels
from src.inference.predict_audio import predict_audio
from src.inference.predict_video import predict_video
from src.inference.predict_stream import predict_stream
import joblib

app = typer.Typer()

@app.command(name="prepare-dataset")
def prepare_dataset_cmd(config: str = typer.Option("configs/default.yaml", "--config")):
    config = load_config(config)
    prepare_dataset(config['data_raw_dir'], config['data_processed_dir'])

@app.command(name="make-splits")
def make_splits_cmd(config: str = typer.Option("configs/default.yaml", "--config")):
    config = load_config(config)
    make_splits(config['data_processed_dir'], config['data_splits_dir'], config['train_split'], config['val_split'], config['test_split'])

@app.command(name="dataset-summary")
def dataset_summary_cmd(config: str = typer.Option("configs/default.yaml", "--config")):
    config = load_config(config)
    dataset_summary(config['data_splits_dir'])

@app.command(name="extract-features")
def extract_features_cmd(config: str = typer.Option("configs/default.yaml", "--config")):
    config = load_config(config)
    extractor = YAMNetExtractor(config['yamnet_model_url'])
    extract_all_features(Path(config['data_splits_dir']) / 'train.csv', extractor, config)

@app.command(name="train")
def train_cmd(config: str = typer.Option("configs/default.yaml", "--config")):
    config = load_config(config)
    features = load_features(config['features_path'])
    labels = load_labels(config['labels_path'])
    train_classifier(features, labels, config)

@app.command(name="evaluate")
def evaluate_cmd(config: str = typer.Option("configs/default.yaml", "--config")):
    config = load_config(config)
    clf = joblib.load(config['model_save_path'])
    le = joblib.load(config['label_encoder_path'])
    features = load_features(config['features_path'])
    labels = load_labels(config['labels_path'])
    metrics = evaluate_model(clf, features, labels, le)
    print(metrics)

@app.command(name="predict-audio")
def predict_audio_cmd(
    input: str = typer.Option(..., "--input"),
    output: str = typer.Option(..., "--output"),
    config: str = typer.Option("configs/default.yaml", "--config"),
):
    config = load_config(config)
    result = predict_audio(input, config)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)

@app.command(name="predict-video")
def predict_video_cmd(
    input: str = typer.Option(..., "--input"),
    output: str = typer.Option(..., "--output"),
    config: str = typer.Option("configs/default.yaml", "--config"),
):
    config = load_config(config)
    result = predict_video(input, config)
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
