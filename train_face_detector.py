from __future__ import annotations

import argparse
import errno
import shutil
from pathlib import Path
from typing import Iterable

import torch
import yaml
from ultralytics import YOLO

REPO_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_YAML = REPO_DIR / "data" / "data.yaml"
DEFAULT_MODEL = REPO_DIR / "models" / "yolo26l.pt"
GENERATED_DIR = REPO_DIR / "data" / "_generated"
DEFAULT_PROJECT = REPO_DIR / "runs" / "face_training"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a face detector on the local WIDER Face dataset.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--smoke-train", type=int, default=64)
    parser.add_argument("--smoke-val", type=int, default=32)
    parser.add_argument("--copy-best-to", type=Path, default=None)
    return parser.parse_args()


def load_dataset_config(data_yaml: Path) -> dict:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")
    with data_yaml.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def resolve_config_path(data_yaml: Path, config_value: str) -> Path:
    candidate = Path(config_value)
    if candidate.is_absolute():
        return candidate
    return (data_yaml.parent / candidate).resolve()


def resolve_dataset_dirs(data_yaml: Path) -> tuple[dict, Path, Path]:
    config = load_dataset_config(data_yaml)
    base_path = resolve_config_path(data_yaml, config.get("path", ".")) if config.get("path") else data_yaml.parent.resolve()
    train_path = (base_path / config["train"]).resolve()
    val_path = (base_path / config["val"]).resolve()
    return config, train_path, val_path


def list_images(images_dir: Path) -> list[Path]:
    images = [path for path in sorted(images_dir.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]
    if not images:
        raise FileNotFoundError(f"No images found in {images_dir}")
    return images


def write_list_file(path: Path, image_paths: Iterable[Path]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(image_path.resolve()) for image_path in image_paths]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def copy_file_without_metadata(source_path: Path, destination_path: Path) -> None:
    try:
        shutil.copyfile(source_path, destination_path)
        return
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EACCES}:
            raise
        destination_path.unlink(missing_ok=True)
        with source_path.open("rb") as src, destination_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)


def build_runtime_full_yaml(full_data_yaml: Path) -> Path:
    config, train_images_dir, val_images_dir = resolve_dataset_dirs(full_data_yaml)
    runtime_yaml = GENERATED_DIR / "full_data.yaml"
    runtime_config = {
        "train": str(train_images_dir),
        "val": str(val_images_dir),
        "nc": int(config["nc"]),
        "names": config["names"],
    }
    runtime_yaml.parent.mkdir(parents=True, exist_ok=True)
    with runtime_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(runtime_config, f, sort_keys=False, allow_unicode=False)
    return runtime_yaml


def build_smoke_yaml(full_data_yaml: Path, train_limit: int, val_limit: int) -> tuple[Path, dict]:
    config, train_images_dir, val_images_dir = resolve_dataset_dirs(full_data_yaml)
    smoke_train_images = list_images(train_images_dir)[:train_limit]
    smoke_val_images = list_images(val_images_dir)[:val_limit]
    if not smoke_train_images or not smoke_val_images:
        raise RuntimeError("Smoke dataset is empty.")

    smoke_train_txt = write_list_file(GENERATED_DIR / "smoke_train.txt", smoke_train_images)
    smoke_val_txt = write_list_file(GENERATED_DIR / "smoke_val.txt", smoke_val_images)
    smoke_yaml = GENERATED_DIR / "smoke_data.yaml"
    smoke_config = {
        "train": str(smoke_train_txt),
        "val": str(smoke_val_txt),
        "nc": int(config["nc"]),
        "names": config["names"],
    }
    with smoke_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(smoke_config, f, sort_keys=False, allow_unicode=False)
    return smoke_yaml, {
        "train_count": len(smoke_train_images),
        "val_count": len(smoke_val_images),
        "train_list": smoke_train_txt,
        "val_list": smoke_val_txt,
    }


def train(args: argparse.Namespace) -> Path:
    data_yaml = args.data.resolve()
    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Base model not found: {model_path}")

    if args.mode == "smoke":
        run_data_yaml, smoke_meta = build_smoke_yaml(data_yaml, args.smoke_train, args.smoke_val)
        epochs = args.epochs or 1
        name = args.name or "smoke"
        print(f"Smoke dataset: train={smoke_meta['train_count']} val={smoke_meta['val_count']}")
        print(f"Smoke train list: {smoke_meta['train_list']}")
        print(f"Smoke val list: {smoke_meta['val_list']}")
    else:
        run_data_yaml = build_runtime_full_yaml(data_yaml)
        epochs = args.epochs or 50
        name = args.name or "full"

    print(f"Model: {model_path}")
    print(f"Data: {run_data_yaml}")
    print(f"Device: {args.device}")
    print(f"Epochs: {epochs}")
    print(f"Batch: {args.batch}")
    print(f"Image size: {args.imgsz}")

    model = YOLO(str(model_path))
    results = model.train(
        data=str(run_data_yaml),
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        device=args.device,
        project=str(args.project),
        name=name,
        exist_ok=True,
        pretrained=True,
        patience=args.patience,
        verbose=True,
    )
    best_path = Path(results.save_dir) / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(f"Training completed but best weights were not found: {best_path}")

    print(f"Training output: {results.save_dir}")
    print(f"Best weights: {best_path}")

    if args.copy_best_to is not None:
        target_path = args.copy_best_to.resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        copy_file_without_metadata(best_path, target_path)
        print(f"Copied best weights to: {target_path}")

    return best_path


def main() -> None:
    args = parse_args()
    best_path = train(args)
    print(f"Done: {best_path}")


if __name__ == "__main__":
    main()
