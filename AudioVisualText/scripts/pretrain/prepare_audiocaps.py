#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
import sys
import socket
import shutil
import tempfile
from tqdm import tqdm
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare an AudioCaps subset for MokA audio pretraining.")
    parser.add_argument("--dataset", default="d0rj/audiocaps", help="Hugging Face dataset id.")
    parser.add_argument("--train_n", type=int, default=500, help="Number of train samples.")
    parser.add_argument("--val_n", type=int, default=100, help="Number of validation samples.")
    parser.add_argument("--out_dir", default="prepared_datasets/AudioCaps", help="Output folder (relative to AudioVisualText root).")
    parser.add_argument("--retry", type=int, default=2, help="Retry count per sample when download fails.")
    parser.add_argument("--hf_endpoint", default="", help="Optional HF endpoint, e.g. https://hf-mirror.com")
    parser.add_argument("--hf_timeout", type=int, default=60, help="HF hub timeout seconds.")
    parser.add_argument("--cookies_from_browser", default="", help="yt-dlp cookies-from-browser value, e.g. chrome")
    parser.add_argument("--cookies", default="", help="Path to a cookies.txt file for yt-dlp")
    parser.add_argument("--skip_download", action="store_true", help="Only export train/val json, do not download wav.")
    return parser.parse_args()


def ensure_deps() -> None:
    try:
        import datasets  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Missing dependency: datasets. Install with `pip install datasets`." ) from exc


def configure_hf_env(hf_endpoint: str, hf_timeout: int) -> None:
    # These env vars are consumed by huggingface_hub when loading datasets.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", str(hf_timeout))
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", str(hf_timeout))
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint


def warn_if_hf_dns_loopback() -> None:
    try:
        resolved = socket.gethostbyname("huggingface.co")
    except OSError:
        return

    if resolved.startswith("127."):
        print(
            "[warn] huggingface.co resolves to loopback address "
            f"{resolved}. HF access may fail in this environment."
        )
        print("[warn] You can set --hf_endpoint https://hf-mirror.com to bypass this.")


def normalize_row(row: Dict) -> Dict:
    return {
        "audiocap_id": str(row["audiocap_id"]),
        "youtube_id": row["youtube_id"],
        "start_time": float(row["start_time"]),
        "caption": row["caption"],
    }


def dump_json(path: str, rows: List[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)


def load_subset(dataset_id: str, train_n: int, val_n: int) -> Tuple[List[Dict], List[Dict]]:
    from datasets import load_dataset

    try:
        print(f"[prepare] loading dataset {dataset_id} ...")
        ds = load_dataset(dataset_id)
    except Exception as exc:
        raise RuntimeError(
            "Failed to load dataset from Hugging Face. "
            "Check network/DNS or use --hf_endpoint https://hf-mirror.com. "
            f"Original error: {exc}"
        ) from exc
    if "train" not in ds:
        raise RuntimeError(f"Dataset {dataset_id} has no split: train")

    val_split_name = "validation" if "validation" in ds else ("val" if "val" in ds else None)
    if val_split_name is None:
        raise RuntimeError(f"Dataset {dataset_id} has no split: validation/val")

    train_subset = ds["train"].select(range(min(train_n, len(ds["train"])) ))
    val_subset = ds[val_split_name].select(range(min(val_n, len(ds[val_split_name])) ))

    train_rows = [normalize_row(x) for x in train_subset]
    val_rows = [normalize_row(x) for x in val_subset]
    return train_rows, val_rows


def download_clip(
    youtube_id: str,
    start_time: float,
    out_template: str,
    cookies_from_browser: str,
    cookies_file: str,
) -> int:
    end_time = start_time + 10.0
    url = f"https://www.youtube.com/watch?v={youtube_id}"
    common = [
        "yt-dlp",
        "--no-playlist",
        "--force-overwrites",
        "--retries",
        "5",
        "--fragment-retries",
        "5",
        "--socket-timeout",
        "30",
        "--force-ipv4",
        "--remote-components",
        "ejs:github",
    ]

    if shutil.which("node"):
        common += ["--js-runtimes", "node"]

    if cookies_from_browser:
        common += ["--cookies-from-browser", cookies_from_browser]
    elif cookies_file:
        common += ["--cookies", cookies_file]

    if cookies_from_browser or cookies_file:
        clients = ["web", "mweb", "ios"]
    else:
        clients = ["android"]

    # Robust path: download full audio first, then trim locally with ffmpeg.
    # This avoids remote section cutting, which is fragile on unstable networks.
    tmp_dir = tempfile.mkdtemp(prefix="audiocaps_")
    try:
        full_template = os.path.join(tmp_dir, "%(id)s.%(ext)s")
        proc_full = None
        for client in clients:
            full_cmd = common + [
                "--extractor-args",
                f"youtube:player_client={client}",
                "--http-chunk-size",
                "10M",
                "-f",
                "bestaudio/best",
                "-o",
                full_template,
                url,
            ]
            proc_full = subprocess.run(full_cmd, check=False)
            if proc_full.returncode == 0:
                break
        if proc_full is None or proc_full.returncode != 0:
            return 1

        candidates = [
            os.path.join(tmp_dir, name)
            for name in os.listdir(tmp_dir)
            if os.path.isfile(os.path.join(tmp_dir, name))
        ]
        if not candidates:
            return 1

        source_audio = candidates[0]
        wav_path = out_template.replace("%(ext)s", "wav")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_time),
            "-t",
            "10",
            "-i",
            source_audio,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            wav_path,
        ]
        proc_trim = subprocess.run(ffmpeg_cmd, check=False)
        return proc_trim.returncode
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def download_audio(
    rows: List[Dict],
    data_dir: str,
    retry: int,
    cookies_from_browser: str,
    cookies_file: str,
) -> Tuple[int, int, int, List[str]]:
    seen = set()
    ok = 0
    fail = 0
    skip = 0
    missing = []

    print(f"[prepare] start downloading {len(rows)} audio clips ...")
    for row in tqdm(rows, desc="Downloading audio"):
        audiocap_id = row["audiocap_id"]
        if audiocap_id in seen:
            continue
        seen.add(audiocap_id)

        wav_path = os.path.join(data_dir, f"{audiocap_id}.wav")
        if os.path.exists(wav_path):
            skip += 1
            continue

        output_template = os.path.join(data_dir, f"{audiocap_id}.%(ext)s")
        success = False
        for _ in range(retry + 1):
            code = download_clip(
                row["youtube_id"],
                row["start_time"],
                output_template,
                cookies_from_browser,
                cookies_file,
            )
            if code == 0 and os.path.exists(wav_path):
                success = True
                break

        if success:
            ok += 1
            tqdm.write(f"[ok] {audiocap_id}")
        else:
            fail += 1
            missing.append(audiocap_id)
            tqdm.write(f"[fail] {audiocap_id}")

    return ok, fail, skip, missing


def main() -> int:
    args = parse_args()
    configure_hf_env(args.hf_endpoint, args.hf_timeout)
    warn_if_hf_dns_loopback()
    ensure_deps()

    out_dir = os.path.abspath(args.out_dir)
    data_dir = os.path.join(out_dir, "data")
    os.makedirs(data_dir, exist_ok=True)

    train_rows, val_rows = load_subset(args.dataset, args.train_n, args.val_n)

    train_json = os.path.join(out_dir, "train.json")
    val_json = os.path.join(out_dir, "val.json")
    dump_json(train_json, train_rows)
    dump_json(val_json, val_rows)

    print(f"[prepare] dataset={args.dataset}")
    print(f"[prepare] train={len(train_rows)}, val={len(val_rows)}")
    print(f"[prepare] json written: {train_json}, {val_json}")

    if args.skip_download:
        print("[prepare] skip_download=True, stop after json export")
        return 0

    if args.cookies_from_browser:
        print(f"[prepare] yt-dlp auth via browser cookies: {args.cookies_from_browser}")
    elif args.cookies:
        print(f"[prepare] yt-dlp auth via cookies file: {args.cookies}")
    else:
        print("[prepare] yt-dlp auth: none (may fail with YouTube bot checks)")

    all_rows = train_rows + val_rows
    ok, fail, skip, missing = download_audio(
        all_rows,
        data_dir,
        args.retry,
        args.cookies_from_browser,
        args.cookies,
    )

    missing_path = os.path.join(out_dir, "missing_ids.txt")
    with open(missing_path, "w", encoding="utf-8") as f:
        for item in missing:
            f.write(item + "\n")

    print(f"[prepare] done: ok={ok}, fail={fail}, skip_existing={skip}, total_unique={len(set([r['audiocap_id'] for r in all_rows]))}")
    print(f"[prepare] wav dir: {data_dir}")
    print(f"[prepare] missing list: {missing_path}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
