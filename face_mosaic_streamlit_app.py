from __future__ import annotations

import copy
import errno
import hashlib
import json
import math
import os
import pickle
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import streamlit as st
import torch
from ultralytics import YOLO

APP_DIR = Path(__file__).resolve().parent
WORK_DIR = APP_DIR / ".work"
UPLOAD_DIR = WORK_DIR / "uploads"
CACHE_DIR = WORK_DIR / "cache"
RENDER_DIR = WORK_DIR / "rendered"
MODELS_DIR = APP_DIR / "models"
RESULTS_DIR = APP_DIR / "Results"

VIDEO_EXTS = {"mp4", "mov", "avi", "mkv", "m4v", "webm"}
MODEL_EXTS = {".pt", ".onnx", ".engine", ".torchscript", ".tflite"}
CACHE_FORMAT_VERSION = "face-only-v2"
RENDER_FORMAT_VERSION = "render-v2"
DEFAULT_VIDEO_FPS = 30.0
FACE_CLASS_KEYWORDS = (
    "face",
    "faces",
    "human face",
    "person face",
    "facial",
    "head",
    "heads",
    "顔",
    "頭",
)


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0

    def clipped(self, width: int, height: int) -> "Box":
        x1 = max(0.0, min(float(width - 1), self.x1))
        y1 = max(0.0, min(float(height - 1), self.y1))
        x2 = max(0.0, min(float(width - 1), self.x2))
        y2 = max(0.0, min(float(height - 1), self.y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return Box(x1=x1, y1=y1, x2=x2, y2=y2, conf=self.conf)

    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        return max(0.0, self.width() * self.height())

    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    def diag(self) -> float:
        return math.hypot(self.width(), self.height())

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2, self.conf]

    @staticmethod
    def from_list(values: Sequence[float]) -> "Box":
        conf = float(values[4]) if len(values) >= 5 else 1.0
        return Box(
            x1=float(values[0]),
            y1=float(values[1]),
            x2=float(values[2]),
            y2=float(values[3]),
            conf=conf,
        )


@st.cache_resource(show_spinner=False)
def load_model_cached(model_path: str) -> YOLO:
    return YOLO(model_path)


def ensure_dirs() -> None:
    for directory in [WORK_DIR, UPLOAD_DIR, CACHE_DIR, RENDER_DIR, MODELS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def file_sha1(data: bytes) -> str:
    return hashlib.sha1(data).hexdigest()


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in stem).strip("_") or "video"


def normalize_model_names(raw_names) -> Dict[int, str]:
    if isinstance(raw_names, dict):
        return {int(class_id): str(name) for class_id, name in raw_names.items()}
    if isinstance(raw_names, (list, tuple)):
        return {class_id: str(name) for class_id, name in enumerate(raw_names)}
    return {}


def find_face_class_ids(model) -> List[int]:
    names = normalize_model_names(getattr(model, "names", None))
    if not names:
        return [0]

    face_ids: List[int] = []
    for class_id, name in names.items():
        normalized = name.strip().lower().replace("_", " ").replace("-", " ")
        if any(keyword in normalized for keyword in FACE_CLASS_KEYWORDS):
            face_ids.append(class_id)

    if face_ids:
        return face_ids
    if len(names) == 1:
        return list(names.keys())
    return []


def describe_face_classes(model, face_class_ids: Sequence[int]) -> str:
    names = normalize_model_names(getattr(model, "names", None))
    if not names:
        return "class 0"
    labels = [names.get(class_id, str(class_id)) for class_id in face_class_ids]
    return ", ".join(labels)


def copy_file_without_metadata(source_path: Path, destination_path: Path) -> Path:
    try:
        shutil.copyfile(source_path, destination_path)
        return destination_path
    except OSError as exc:
        if exc.errno not in {errno.EPERM, errno.EACCES}:
            raise
        destination_path.unlink(missing_ok=True)
        with source_path.open("rb") as src, destination_path.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        return destination_path


def make_non_conflicting_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def save_uploaded_video(uploaded_file) -> Path:
    data = uploaded_file.getvalue()
    digest = file_sha1(data)
    ext = Path(uploaded_file.name).suffix.lower() or ".mp4"
    filename = f"{safe_stem(uploaded_file.name)}_{digest[:12]}{ext}"
    output_path = UPLOAD_DIR / filename
    if not output_path.exists():
        output_path.write_bytes(data)
    return output_path


def detect_preferred_model_file() -> Optional[Path]:
    if not MODELS_DIR.exists():
        return None
    candidates = sorted(
        [p for p in MODELS_DIR.iterdir() if p.is_file() and p.suffix.lower() in MODEL_EXTS],
        key=lambda p: p.name.lower(),
    )
    for preferred_name in [
        "yolo26l_face_full.engine",
        "yolo26l_face_full.pt",
        "yolo26l_face_smoke_gpu.engine",
        "yolo26l_face_smoke_gpu.pt",
        "yolo26l_face_smoke.engine",
        "yolo26l_face_smoke.pt",
        "yolo11_face.engine",
        "yolo11_face.pt",
        "yolo11n_face.pt",
        "yolov11n_face.pt",
        "yolo11_face.onnx",
        "yolo26l.engine",
        "yolo26l.pt",
    ]:
        exact = MODELS_DIR / preferred_name
        if exact.exists():
            return exact
    return candidates[0] if candidates else None


def parse_positive_float(value: object) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.0
    return parsed if math.isfinite(parsed) and parsed > 0 else 0.0


def parse_fractional_fps(value: object) -> float:
    if value is None:
        return 0.0
    text = str(value).strip()
    if not text or text in {"0/0", "N/A"}:
        return 0.0
    if "/" in text:
        num_text, den_text = text.split("/", 1)
        numerator = parse_positive_float(num_text)
        denominator = parse_positive_float(den_text)
        if denominator <= 0:
            return 0.0
        return numerator / denominator
    return parse_positive_float(text)


def probe_video_stream_info(video_path: Path) -> Dict[str, float]:
    ffprobe_path = shutil.which("ffprobe")
    if not ffprobe_path:
        return {}

    cmd = [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate,r_frame_rate,nb_frames,width,height,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(video_path),
    ]
    try:
        completed = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        payload = json.loads(completed.stdout or "{}")
    except Exception:
        return {}

    streams = payload.get("streams") or []
    if not streams:
        return {}

    stream = streams[0]
    container = payload.get("format") or {}
    fps = parse_fractional_fps(stream.get("avg_frame_rate")) or parse_fractional_fps(stream.get("r_frame_rate"))
    duration = parse_positive_float(stream.get("duration")) or parse_positive_float(container.get("duration"))
    frame_count = int(round(parse_positive_float(stream.get("nb_frames"))))
    if frame_count <= 0 and duration > 0 and fps > 0:
        frame_count = int(round(duration * fps))

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": int(parse_positive_float(stream.get("width"))),
        "height": int(parse_positive_float(stream.get("height"))),
        "duration": duration,
    }


def get_video_info(video_path: Path) -> Dict[str, float]:
    probed = probe_video_stream_info(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")
    cap_fps = parse_positive_float(cap.get(cv2.CAP_PROP_FPS))
    cap_frame_count = int(round(parse_positive_float(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
    cap_width = int(round(parse_positive_float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))))
    cap_height = int(round(parse_positive_float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    cap.release()

    fps = parse_positive_float(probed.get("fps")) or cap_fps
    frame_count = cap_frame_count or int(probed.get("frame_count", 0))
    width = cap_width or int(probed.get("width", 0))
    height = cap_height or int(probed.get("height", 0))

    duration = parse_positive_float(probed.get("duration"))
    if duration <= 0 and frame_count > 0 and fps > 0:
        duration = frame_count / fps
    if fps <= 0 and frame_count > 0 and duration > 0:
        fps = frame_count / duration
    if fps <= 0:
        fps = DEFAULT_VIDEO_FPS

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration,
    }


def choose_runtime_profile() -> Dict[str, object]:
    profile: Dict[str, object] = {
        "device": "cpu",
        "device_label": "CPU",
        "half": False,
        "batch_size": 1,
        "notes": [],
    }

    try:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
    except Exception:
        pass

    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
    except Exception:
        pass

    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    if torch.cuda.is_available():
        device_index = 0
        gpu_name = torch.cuda.get_device_name(device_index)
        total_mem_gb = 0.0
        backend_name = "ROCm" if getattr(torch.version, "hip", None) else "CUDA"
        try:
            props = torch.cuda.get_device_properties(device_index)
            total_mem_gb = props.total_memory / (1024**3)
        except Exception:
            pass

        if total_mem_gb >= 20:
            batch_size = 16
        elif total_mem_gb >= 10:
            batch_size = 8
        elif total_mem_gb >= 6:
            batch_size = 4
        else:
            batch_size = 2

        profile.update(
            {
                "device": "cuda:0",
                "device_label": f"{backend_name} / {gpu_name}",
                "half": True,
                "batch_size": batch_size,
            }
        )
        profile["notes"] = [
            f"{backend_name} を使用します。",
            "FP16 を有効化して推論を高速化します。",
            f"検出バッチサイズを {batch_size} に設定します。",
        ]
        return profile

    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()  # type: ignore[attr-defined]
    )
    if mps_available:
        profile.update(
            {
                "device": "mps",
                "device_label": "Apple Metal (MPS)",
                "half": False,
                "batch_size": 2,
            }
        )
        profile["notes"] = [
            "Apple Silicon の MPS を使用します。",
            "MPS では安定性優先で FP32 を使います。",
            "検出バッチサイズを 2 に設定します。",
        ]
        return profile

    cpu_count = os.cpu_count() or 4
    batch_size = 2 if cpu_count >= 8 else 1
    profile.update({"batch_size": batch_size})
    profile["notes"] = [
        "GPU が見つからないため CPU 実行です。",
        f"CPU コア数に応じて検出バッチサイズを {batch_size} に設定します。",
    ]
    return profile


def iou(a: Box, b: Box) -> float:
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    denom = a.area() + b.area() - inter
    if denom <= 0:
        return 0.0
    return inter / denom


def box_distance_score(a: Box, b: Box) -> float:
    ax, ay = a.center()
    bx, by = b.center()
    center_dist = math.hypot(ax - bx, ay - by)
    ref = max(a.diag(), b.diag(), 1.0)
    overlap = iou(a, b)
    return (center_dist / ref) - overlap


def match_boxes(prev_boxes: Sequence[Box], next_boxes: Sequence[Box]) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    if not prev_boxes or not next_boxes:
        return [], list(range(len(prev_boxes))), list(range(len(next_boxes)))

    candidates: List[Tuple[float, int, int]] = []
    for i, prev_box in enumerate(prev_boxes):
        for j, next_box in enumerate(next_boxes):
            cdist = math.hypot(prev_box.center()[0] - next_box.center()[0], prev_box.center()[1] - next_box.center()[1])
            threshold = max(prev_box.diag(), next_box.diag(), 40.0) * 2.5
            if cdist <= threshold or iou(prev_box, next_box) >= 0.02:
                candidates.append((box_distance_score(prev_box, next_box), i, j))

    candidates.sort(key=lambda x: x[0])
    used_prev = set()
    used_next = set()
    matches: List[Tuple[int, int]] = []
    for _score, i, j in candidates:
        if i in used_prev or j in used_next:
            continue
        used_prev.add(i)
        used_next.add(j)
        matches.append((i, j))

    unmatched_prev = [i for i in range(len(prev_boxes)) if i not in used_prev]
    unmatched_next = [j for j in range(len(next_boxes)) if j not in used_next]
    return matches, unmatched_prev, unmatched_next


def lerp_box(a: Box, b: Box, alpha: float, width: int, height: int) -> Box:
    return Box(
        x1=a.x1 + (b.x1 - a.x1) * alpha,
        y1=a.y1 + (b.y1 - a.y1) * alpha,
        x2=a.x2 + (b.x2 - a.x2) * alpha,
        y2=a.y2 + (b.y2 - a.y2) * alpha,
        conf=a.conf + (b.conf - a.conf) * alpha,
    ).clipped(width, height)


def interpolate_keyframes(
    total_frames: int,
    keyframe_indices: Sequence[int],
    keyframe_boxes: Dict[int, List[Box]],
    width: int,
    height: int,
) -> List[List[List[float]]]:
    all_boxes: List[List[List[float]]] = [[] for _ in range(total_frames)]

    for frame_idx in keyframe_indices:
        all_boxes[frame_idx] = [box.clipped(width, height).to_list() for box in keyframe_boxes.get(frame_idx, [])]

    for prev_idx, next_idx in zip(keyframe_indices[:-1], keyframe_indices[1:]):
        gap = next_idx - prev_idx
        if gap <= 1:
            continue
        prev_boxes = keyframe_boxes.get(prev_idx, [])
        next_boxes = keyframe_boxes.get(next_idx, [])
        matches, unmatched_prev, unmatched_next = match_boxes(prev_boxes, next_boxes)
        midpoint = gap / 2.0

        for offset in range(1, gap):
            alpha = offset / gap
            current_boxes: List[Box] = []

            for i, j in matches:
                current_boxes.append(lerp_box(prev_boxes[i], next_boxes[j], alpha, width, height))

            if offset <= midpoint:
                current_boxes.extend(copy.deepcopy(prev_boxes[i]).clipped(width, height) for i in unmatched_prev)
            if offset > midpoint:
                current_boxes.extend(copy.deepcopy(next_boxes[j]).clipped(width, height) for j in unmatched_next)

            all_boxes[prev_idx + offset] = [box.to_list() for box in current_boxes]

    return all_boxes


def extract_boxes_from_result(
    result,
    width: int,
    height: int,
    allowed_class_ids: Optional[Sequence[int]] = None,
) -> List[Box]:
    if result.boxes is None or len(result.boxes) == 0:
        return []
    xyxy = result.boxes.xyxy.detach().cpu().numpy()
    confs = result.boxes.conf.detach().cpu().numpy() if result.boxes.conf is not None else np.ones((len(xyxy),), dtype=np.float32)
    classes = (
        result.boxes.cls.detach().cpu().numpy().astype(np.int32)
        if result.boxes.cls is not None
        else np.zeros((len(xyxy),), dtype=np.int32)
    )
    allowed_class_id_set = set(allowed_class_ids) if allowed_class_ids is not None else None
    boxes = [
        Box(float(x1), float(y1), float(x2), float(y2), float(conf)).clipped(width, height)
        for (x1, y1, x2, y2), conf, class_id in zip(xyxy, confs, classes)
        if allowed_class_id_set is None or int(class_id) in allowed_class_id_set
    ]
    return sorted(boxes, key=lambda b: b.conf, reverse=True)


def make_cache_filename(video_path: Path, model_path: Path, detect_every_n: int, conf: float, imgsz: int) -> Path:
    sig = (
        f"{CACHE_FORMAT_VERSION}|{video_path.resolve()}|{video_path.stat().st_mtime_ns}|"
        f"{model_path.resolve()}|{model_path.stat().st_mtime_ns}|{detect_every_n}|{conf:.4f}|{imgsz}"
    )
    digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{safe_stem(video_path.name)}_{digest}.pkl"


def detect_faces_on_keyframes(
    video_path: Path,
    model_path: Path,
    detect_every_n: int,
    conf: float,
    imgsz: int,
    profile: Dict[str, object],
    progress_text_slot,
    progress_bar,
) -> Tuple[Path, Dict[str, object]]:
    info = get_video_info(video_path)
    total_frames = int(info["frame_count"])
    width = int(info["width"])
    height = int(info["height"])
    fps = float(info["fps"])

    if total_frames <= 0 or width <= 0 or height <= 0:
        raise RuntimeError("動画のフレーム情報を取得できませんでした。")

    model = load_model_cached(str(model_path))
    face_class_ids = find_face_class_ids(model)
    if not face_class_ids:
        available_names = normalize_model_names(getattr(model, "names", None))
        available_labels = ", ".join(available_names.values()) if available_names else "不明"
        raise RuntimeError(
            "このモデルには顔クラスが見つかりませんでした。"
            f" 顔検出用の重みを使ってください。検出クラス: {available_labels}"
        )
    face_class_label = describe_face_classes(model, face_class_ids)

    cache_path = make_cache_filename(video_path, model_path, detect_every_n, conf, imgsz)
    if cache_path.exists():
        with cache_path.open("rb") as f:
            cached = pickle.load(f)
        return cache_path, cached

    keyframe_indices = list(range(0, total_frames, max(1, detect_every_n)))
    if keyframe_indices[-1] != total_frames - 1:
        keyframe_indices.append(total_frames - 1)

    keyframe_set = set(keyframe_indices)
    keyframe_boxes: Dict[int, List[Box]] = {}
    batch_frames: List[np.ndarray] = []
    batch_frame_indices: List[int] = []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    processed_frames = 0
    start_time = time.time()

    def flush_batch() -> None:
        nonlocal batch_frames, batch_frame_indices, processed_frames
        if not batch_frames:
            return
        results = model.predict(
            source=batch_frames,
            conf=conf,
            imgsz=imgsz,
            device=str(profile["device"]),
            half=bool(profile["half"]),
            classes=face_class_ids,
            verbose=False,
            agnostic_nms=True,
            max_det=200,
        )
        for frame_idx, result in zip(batch_frame_indices, results):
            keyframe_boxes[frame_idx] = extract_boxes_from_result(
                result,
                width=width,
                height=height,
                allowed_class_ids=face_class_ids,
            )
        processed_frames += len(batch_frame_indices)
        progress = min(processed_frames / max(1, len(keyframe_indices)), 1.0)
        elapsed = max(time.time() - start_time, 1e-6)
        fps_detect = processed_frames / elapsed
        progress_text_slot.text(
            f"顔検出中: {processed_frames}/{len(keyframe_indices)} キーフレーム | 約 {fps_detect:.2f} キーフレーム/秒"
        )
        progress_bar.progress(progress)
        batch_frames = []
        batch_frame_indices = []

    batch_size = int(profile.get("batch_size", 1))
    frame_idx = 0
    ok, frame = cap.read()
    while ok:
        if frame_idx in keyframe_set:
            batch_frames.append(frame)
            batch_frame_indices.append(frame_idx)
            if len(batch_frames) >= batch_size:
                flush_batch()
        frame_idx += 1
        ok, frame = cap.read()

    flush_batch()
    cap.release()

    progress_text_slot.text("顔位置を補間しています…")
    all_frame_boxes = interpolate_keyframes(
        total_frames=total_frames,
        keyframe_indices=keyframe_indices,
        keyframe_boxes=keyframe_boxes,
        width=width,
        height=height,
    )

    payload = {
        "video_path": str(video_path),
        "model_path": str(model_path),
        "video_info": info,
        "detect_every_n": detect_every_n,
        "conf": conf,
        "imgsz": imgsz,
        "profile": profile,
        "face_class_ids": list(face_class_ids),
        "face_class_label": face_class_label,
        "keyframe_indices": keyframe_indices,
        "frame_boxes": all_frame_boxes,
        "created_at": time.time(),
    }

    with cache_path.open("wb") as f:
        pickle.dump(payload, f)

    progress_text_slot.text("顔検出キャッシュを保存しました。")
    progress_bar.progress(1.0)
    return cache_path, payload


def apply_mosaic_to_region(frame: np.ndarray, box: Box, mosaic_block_size: int) -> None:
    h, w = frame.shape[:2]
    clipped = box.clipped(w, h)
    x1 = int(round(clipped.x1))
    y1 = int(round(clipped.y1))
    x2 = int(round(clipped.x2))
    y2 = int(round(clipped.y2))

    if x2 <= x1 or y2 <= y1:
        return

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return

    cell = max(1, int(mosaic_block_size))
    down_w = max(1, int(round((x2 - x1) / cell)))
    down_h = max(1, int(round((y2 - y1) / cell)))
    small = cv2.resize(roi, (down_w, down_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = mosaic


def try_mux_audio(original_video: Path, silent_video: Path, final_video: Path) -> Path:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        copy_file_without_metadata(silent_video, final_video)
        return final_video

    cmd = [
        ffmpeg_path,
        "-y",
        "-i",
        str(silent_video),
        "-i",
        str(original_video),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-shortest",
        str(final_video),
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return final_video
    except Exception:
        copy_file_without_metadata(silent_video, final_video)
        return final_video


def make_rendered_video_filename(video_path: Path, detection_cache_path: Path, mosaic_block_size: int) -> Path:
    sig = (
        f"{RENDER_FORMAT_VERSION}|{video_path.resolve()}|{detection_cache_path.resolve()}|"
        f"{detection_cache_path.stat().st_mtime_ns}|{mosaic_block_size}"
    )
    digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:16]
    return RENDER_DIR / f"{safe_stem(video_path.name)}_mosaic_{digest}.mp4"


def render_mosaic_video(
    video_path: Path,
    detection_cache_path: Path,
    mosaic_block_size: int,
    progress_text_slot,
    progress_bar,
) -> Path:
    with detection_cache_path.open("rb") as f:
        payload = pickle.load(f)

    frame_boxes: List[List[List[float]]] = payload["frame_boxes"]
    info = get_video_info(video_path)
    frame_count = len(frame_boxes) or int(info["frame_count"])
    width = int(info["width"])
    height = int(info["height"])
    fps = max(parse_positive_float(info["fps"]), 1.0)

    rendered_video_path = make_rendered_video_filename(video_path, detection_cache_path, mosaic_block_size)
    if rendered_video_path.exists():
        return rendered_video_path

    temp_silent_path = RENDER_DIR / f"{rendered_video_path.stem}_silent.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"動画を開けませんでした: {video_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(temp_silent_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("出力動画を書き込めませんでした。OpenCV の codec 設定を確認してください。")

    start_time = time.time()
    frame_idx = 0
    ok, frame = cap.read()
    while ok:
        boxes = frame_boxes[frame_idx] if frame_idx < len(frame_boxes) else []
        for raw_box in boxes:
            apply_mosaic_to_region(frame, Box.from_list(raw_box), mosaic_block_size)
        writer.write(frame)
        frame_idx += 1
        if frame_idx % 10 == 0 or frame_idx == frame_count:
            progress = min(frame_idx / max(1, frame_count), 1.0)
            elapsed = max(time.time() - start_time, 1e-6)
            render_fps = frame_idx / elapsed
            progress_text_slot.text(
                f"モザイク動画を生成中: {frame_idx}/{frame_count} フレーム | 約 {render_fps:.2f} フレーム/秒"
            )
            progress_bar.progress(progress)
        ok, frame = cap.read()

    writer.release()
    cap.release()

    final_video = try_mux_audio(video_path, temp_silent_path, rendered_video_path)
    if temp_silent_path.exists():
        temp_silent_path.unlink(missing_ok=True)
    progress_text_slot.text("モザイク動画を生成しました。")
    progress_bar.progress(1.0)
    return final_video


def save_rendered_video_to_results(rendered_video_path: Path, output_name: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = rendered_video_path.suffix.lower() or ".mp4"
    output_name = output_name.strip()
    if not output_name:
        output_name = rendered_video_path.stem
    output_name = safe_stem(output_name)
    requested_path = RESULTS_DIR / f"{output_name}{suffix}"
    final_path = make_non_conflicting_path(requested_path)
    copy_file_without_metadata(rendered_video_path, final_path)
    return final_path


def human_readable_time(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours:
        return f"{hours}時間 {mins}分 {secs}秒"
    if mins:
        return f"{mins}分 {secs}秒"
    return f"{secs}秒"


def render_model_help(model_path_value: str) -> None:
    st.warning("顔検出モデルがまだ見つかっていません。")
    st.markdown(
        f"""
**あなたにお願いしたい作業**

1. 顔専用に学習された YOLO26 互換モデルを 1 つ用意してください。  
   例: `yolo26l_face_full.pt`, `yolo26l_face_smoke.pt`, `yolo26l_face_smoke_gpu.pt`
2. そのファイルを **`{MODELS_DIR}`** に置いてください。
3. 置いたファイル名が違う場合は、下の「モデルパス」でそのパスを入力してください。
4. 一般物体検出用の `yolo26l.pt` ではなく、**顔を検出できる重み** を置いてください。
5. モデルを置いたら、もう一度この画面を再読み込みしてください。

**補足**
- `.pt` のほか、環境によっては `.onnx` や `.engine` でも動く場合があります。
- NVIDIA GPU を使う場合は、まず `.pt` 版で動作確認してから高速化を検討するのがおすすめです。
- このアプリは、モデルが無い状態では顔検出を開始しません。
"""
    )
    st.code(
        f"models/\n  └─ {Path(model_path_value).name if model_path_value else 'yolo26l_face_full.pt'}",
        language="text",
    )


def main() -> None:
    ensure_dirs()
    st.set_page_config(page_title="YOLO26 顔モザイク", layout="wide")
    st.title("YOLO26 顔モザイク動画アプリ")
    st.caption("アップロード → 顔検出 → モザイク動画生成 → 保存 → Results フォルダ")

    runtime_profile = choose_runtime_profile()
    with st.expander("実行環境と高速化設定", expanded=True):
        st.write(f"**使用デバイス**: {runtime_profile['device_label']}")
        for note in runtime_profile["notes"]:
            st.write(f"- {note}")

    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "detection_cache_path" not in st.session_state:
        st.session_state.detection_cache_path = None
    if "rendered_video_path" not in st.session_state:
        st.session_state.rendered_video_path = None
    if "last_mosaic_block_size" not in st.session_state:
        st.session_state.last_mosaic_block_size = None
    if st.session_state.get("cache_format_version") != CACHE_FORMAT_VERSION:
        st.session_state.detection_cache_path = None
        st.session_state.rendered_video_path = None
        st.session_state.last_mosaic_block_size = None
        st.session_state.cache_format_version = CACHE_FORMAT_VERSION

    st.header("1. 動画アップロード")
    uploaded_file = st.file_uploader("動画ファイルを選択してください", type=sorted(VIDEO_EXTS))
    if uploaded_file is not None:
        try:
            video_path = save_uploaded_video(uploaded_file)
            previous_video_path = st.session_state.video_path
            st.session_state.video_path = str(video_path)
            if previous_video_path != str(video_path):
                st.session_state.detection_cache_path = None
                st.session_state.rendered_video_path = None
                st.session_state.last_mosaic_block_size = None
            st.success(f"アップロード完了: {video_path.name}")
            info = get_video_info(video_path)
            st.write(
                f"解像度: {int(info['width'])}x{int(info['height'])} / "
                f"FPS: {info['fps']:.2f} / "
                f"フレーム数: {int(info['frame_count'])} / "
                f"長さ: {human_readable_time(float(info['duration']))}"
            )
            st.caption("アプリ内での動画プレビュー表示は省略しています。")
        except Exception as e:
            st.error(f"アップロードした動画の処理に失敗しました: {e}")
            st.stop()

    if not st.session_state.video_path:
        st.info("先に動画をアップロードしてください。")
        st.stop()

    video_path = Path(st.session_state.video_path)

    st.header("2. 顔検出処理")
    default_model = detect_preferred_model_file()
    default_model_path = str(default_model) if default_model else str(MODELS_DIR / "yolo26l_face_full.pt")
    model_path_str = st.text_input("モデルパス", value=default_model_path)
    model_path = Path(model_path_str)

    col1, col2, col3 = st.columns(3)
    with col1:
        detect_every_n = st.slider("何フレームごとに顔検出するか", min_value=1, max_value=30, value=5, step=1)
    with col2:
        conf = st.slider("顔検出の信頼度しきい値", min_value=0.05, max_value=0.95, value=0.30, step=0.05)
    with col3:
        imgsz = st.select_slider("推論画像サイズ", options=[320, 416, 512, 640, 768, 960], value=640)

    if not model_path.exists():
        render_model_help(model_path_str)
        st.stop()

    try:
        preview_model = load_model_cached(str(model_path))
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました: {e}")
        st.stop()

    preview_face_class_ids = find_face_class_ids(preview_model)
    preview_model_names = normalize_model_names(getattr(preview_model, "names", None))
    if not preview_face_class_ids:
        available_labels = ", ".join(preview_model_names.values()) if preview_model_names else "不明"
        st.error(
            "このモデルには顔クラスが見つかりません。"
            f" 顔検出用の重みを指定してください。検出クラス: {available_labels}"
        )
        st.stop()

    st.caption(f"顔として使う検出クラス: {describe_face_classes(preview_model, preview_face_class_ids)}")

    detection_progress_text = st.empty()
    detection_progress_bar = st.progress(0.0)
    if st.button("顔検出を実行", type="primary", use_container_width=True):
        try:
            cache_path, payload = detect_faces_on_keyframes(
                video_path=video_path,
                model_path=model_path,
                detect_every_n=detect_every_n,
                conf=conf,
                imgsz=imgsz,
                profile=runtime_profile,
                progress_text_slot=detection_progress_text,
                progress_bar=detection_progress_bar,
            )
            st.session_state.detection_cache_path = str(cache_path)
            st.session_state.rendered_video_path = None
            st.session_state.last_mosaic_block_size = None
            total_detected = sum(len(frame_boxes) for frame_boxes in payload["frame_boxes"])
            st.success("顔検出が完了しました。")
            st.write(f"検出キャッシュ: {cache_path.name}")
            st.write(f"全フレームに展開した顔ボックス総数: {total_detected}")
        except Exception as e:
            st.error(f"顔検出に失敗しました: {e}")
            st.stop()

    if not st.session_state.detection_cache_path:
        st.info("顔検出を実行すると、次のモザイク処理へ進めます。")
        st.stop()

    detection_cache_path = Path(st.session_state.detection_cache_path)

    st.header("3. モザイク動画生成")
    mosaic_block_size = st.slider("モザイクの粗さ（大きいほど荒くなる）", min_value=4, max_value=80, value=18, step=2)
    mosaic_progress_text = st.empty()
    mosaic_progress_bar = st.progress(0.0)
    if st.button("モザイク動画を生成", use_container_width=True):
        try:
            rendered_video_path = render_mosaic_video(
                video_path=video_path,
                detection_cache_path=detection_cache_path,
                mosaic_block_size=mosaic_block_size,
                progress_text_slot=mosaic_progress_text,
                progress_bar=mosaic_progress_bar,
            )
            st.session_state.rendered_video_path = str(rendered_video_path)
            st.session_state.last_mosaic_block_size = mosaic_block_size
            st.success("モザイク動画の生成が完了しました。")
            st.write(f"生成ファイル: {rendered_video_path.name}")
        except Exception as e:
            st.error(f"モザイク処理に失敗しました: {e}")
            st.stop()

    st.header("4. モザイク大きさ調整")
    st.write("モザイクの粗さを変更したら、もう一度「モザイク動画を生成」を押してください。顔検出は再実行されません。")
    rendered_video_path_str = st.session_state.rendered_video_path
    if rendered_video_path_str:
        rendered_video_path = Path(rendered_video_path_str)
        if rendered_video_path.exists():
            st.write(f"現在のモザイクサイズ: {st.session_state.last_mosaic_block_size}")
            st.write(f"直近の生成ファイル: {rendered_video_path.name}")
        else:
            st.warning("生成済みモザイク動画が見つかりません。再度モザイク動画を生成してください。")
    else:
        st.info("モザイク動画を生成すると、保存できる状態になります。")

    st.header("5. 保存")
    default_output_name = f"{safe_stem(video_path.name)}_mosaic"
    output_name = st.text_input("保存ファイル名（拡張子不要）", value=default_output_name)
    if st.button("Results フォルダへ保存", use_container_width=True):
        if not st.session_state.rendered_video_path:
            st.error("先にモザイク動画を生成してください。")
            st.stop()
        rendered_video_path = Path(st.session_state.rendered_video_path)
        try:
            final_path = save_rendered_video_to_results(rendered_video_path, output_name)
            st.success("保存が完了しました。")
            st.write(f"保存先: {final_path}")
        except Exception as e:
            st.error(f"保存に失敗しました: {e}")
            st.stop()

    st.header("6. Results フォルダ")
    results_files = sorted(RESULTS_DIR.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if results_files:
        for file_path in results_files[:20]:
            st.write(f"- {file_path.name}")
    else:
        st.write("まだ保存されたファイルはありません。")

    with st.expander("補足と注意点", expanded=False):
        st.markdown(
            """
- 顔検出は **キーフレームのみ** YOLO26 で推論し、その間の顔位置は線形補間しています。
- そのため、`何フレームごとに顔検出するか` を大きくすると高速になりますが、急な動きに弱くなります。
- モザイクサイズの調整時は、顔検出キャッシュを再利用するため、通常は再検出より軽くなります。
- OpenCV のみで出力した場合は音声が消えますが、`ffmpeg` が見つかれば元動画の音声を再合成します。
- 顔モデルそのものの精度は、あなたが用意する重みに依存します。
            """
        )


if __name__ == "__main__":
    main()
