from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Optional


@dataclass
class Recommendation:
    profile: str
    os_name: str
    machine: str
    gpu_name: Optional[str]
    driver_version: Optional[str]
    cuda_version: Optional[str]
    reason: str
    notes: list[str]


def run_command(cmd: list[str]) -> Optional[str]:
    try:
        return subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=8,
        )
    except Exception:
        return None


def parse_nvidia_smi() -> tuple[Optional[str], Optional[str], Optional[str]]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None, None, None

    query = run_command([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"])
    gpu_name = None
    driver_version = None
    if query:
        first_line = query.strip().splitlines()[0] if query.strip() else ""
        if first_line:
            parts = [p.strip() for p in first_line.split(",", 1)]
            gpu_name = parts[0] if parts else None
            driver_version = parts[1] if len(parts) > 1 else None

    full = run_command([exe])
    cuda_version = None
    if full:
        match = re.search(r"CUDA Version:\s*([0-9]+(?:\.[0-9]+)?)", full)
        if match:
            cuda_version = match.group(1)

    return gpu_name, driver_version, cuda_version


def recommend_profile() -> Recommendation:
    os_name = platform.system()
    machine = platform.machine().lower()
    notes: list[str] = []

    if os_name == "Darwin":
        if machine in {"arm64", "aarch64"}:
            notes.append("Apple Silicon を検出しました。uv extra は cpu を使い、実行時に PyTorch の MPS が使えればアプリが自動で MPS を選びます。")
            return Recommendation(
                profile="cpu",
                os_name=os_name,
                machine=machine,
                gpu_name="Apple Silicon GPU",
                driver_version=None,
                cuda_version=None,
                reason="macOS では CUDA ビルドを使わないため、cpu extra を推奨します。",
                notes=notes,
            )
        notes.append("Intel Mac を検出しました。MPS は通常 Apple Silicon 向けなので、cpu extra を推奨します。")
        return Recommendation(
            profile="cpu",
            os_name=os_name,
            machine=machine,
            gpu_name=None,
            driver_version=None,
            cuda_version=None,
            reason="macOS では CUDA extra は通常使わないため、cpu extra を推奨します。",
            notes=notes,
        )

    gpu_name, driver_version, cuda_version = parse_nvidia_smi()
    if gpu_name:
        notes.append(f"NVIDIA GPU を検出しました: {gpu_name}")
        if driver_version:
            notes.append(f"Driver version: {driver_version}")
        if cuda_version:
            notes.append(f"nvidia-smi が報告した CUDA Version: {cuda_version}")
            try:
                major_minor = cuda_version.split(".")
                major = int(major_minor[0])
                minor = int(major_minor[1]) if len(major_minor) > 1 else 0
            except ValueError:
                major = 0
                minor = 0

            if major >= 13:
                profile = "cu130"
            elif major == 12 and minor >= 8:
                profile = "cu128"
            elif major == 12 and minor >= 6:
                profile = "cu126"
            else:
                profile = "cpu"
                notes.append("CUDA バージョンが 12.6 未満に見えるため、まずは cpu extra を推奨します。")

            return Recommendation(
                profile=profile,
                os_name=os_name,
                machine=machine,
                gpu_name=gpu_name,
                driver_version=driver_version,
                cuda_version=cuda_version,
                reason="NVIDIA GPU と CUDA バージョンから uv extra を推定しました。最終判断はあなたのドライバと PyTorch 互換性を優先してください。",
                notes=notes,
            )

        notes.append("nvidia-smi の CUDA Version を読み取れなかったため、cu128 を仮の推奨値にします。")
        return Recommendation(
            profile="cu128",
            os_name=os_name,
            machine=machine,
            gpu_name=gpu_name,
            driver_version=driver_version,
            cuda_version=None,
            reason="NVIDIA GPU は見つかりましたが CUDA Version が取れなかったため、一般的な候補として cu128 を仮提案します。",
            notes=notes,
        )

    notes.append("nvidia-smi が見つからなかったため、NVIDIA CUDA 環境は検出できませんでした。")
    return Recommendation(
        profile="cpu",
        os_name=os_name,
        machine=machine,
        gpu_name=None,
        driver_version=None,
        cuda_version=None,
        reason="GPU が確定できないため、まずは cpu extra を推奨します。",
        notes=notes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="uv 用の PyTorch extra 推奨値を表示します。")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--only-profile", action="store_true")
    parser.add_argument("--python-version", default="3.11")
    args = parser.parse_args()

    rec = recommend_profile()

    if args.only_profile:
        print(rec.profile)
        return 0

    if args.as_json:
        print(json.dumps(asdict(rec), ensure_ascii=False, indent=2))
        return 0

    print("推奨 uv extra:", rec.profile)
    print("OS:", rec.os_name)
    print("Machine:", rec.machine)
    if rec.gpu_name:
        print("GPU:", rec.gpu_name)
    if rec.driver_version:
        print("Driver:", rec.driver_version)
    if rec.cuda_version:
        print("CUDA Version:", rec.cuda_version)
    print("理由:", rec.reason)
    if rec.notes:
        print("メモ:")
        for note in rec.notes:
            print(f"- {note}")
    print()
    print("次に実行するコマンド例:")
    print(f"uv python install {args.python_version}")
    print(f"uv sync --extra {rec.profile}")
    print(f"uv run --extra {rec.profile} -- streamlit run face_mosaic_streamlit_app.py")
    print()
    print("注意: これは自動推定です。NVIDIA 環境では nvidia-smi の内容と PyTorch の互換性を優先してください。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
