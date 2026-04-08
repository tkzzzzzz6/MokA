#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Dict

from huggingface_hub import hf_hub_download
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a selective subset of Video-LLaVA dataset for pretraining/smoke tests."
    )
    parser.add_argument("--repo_id", default="LanguageBind/Video-LLaVA", help="HF dataset repo id")
    parser.add_argument("--out_dir", default="prepared_datasets/video-llava", help="Local output root")
    parser.add_argument("--image_json", default="train_json/llava_image_.json", help="Image annotation json path in repo")
    parser.add_argument("--video_json", default="train_json/valid_valley_.json", help="Video annotation json path in repo")
    parser.add_argument("--image_n", type=int, default=0, help="How many image samples to download (0 means skip)")
    parser.add_argument("--video_n", type=int, default=0, help="How many video samples to download (0 means skip)")
    parser.add_argument("--token", default=None, help="HF token if needed")
    parser.add_argument(
        "--write_default_json_names",
        action="store_true",
        help="Also write subset json to default filenames used by pretrain_dataset.py",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def download_repo_file(repo_id: str, filename: str, out_dir: Path, token: str = None) -> Path:
    local_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,
        token=token,
    )
    return Path(local_path)


def load_json(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: List[Dict]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def unique_keep_order(paths: Iterable[str]) -> List[str]:
    seen = set()
    out = []
    for p in paths:
        if not p or p in seen:
            continue
        seen.add(p)
        out.append(p)
    return out


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[step] download annotation files from {args.repo_id}")
    image_json_local = download_repo_file(args.repo_id, args.image_json, out_dir, args.token)
    video_json_local = download_repo_file(args.repo_id, args.video_json, out_dir, args.token)

    image_samples = load_json(image_json_local)
    video_samples = load_json(video_json_local)

    image_subset = image_samples[: args.image_n] if args.image_n > 0 else []
    video_subset = video_samples[: args.video_n] if args.video_n > 0 else []

    subset_dir = out_dir / "train_json"
    subset_image_json = subset_dir / "llava_image_subset.json"
    subset_video_json = subset_dir / "valid_valley_subset.json"
    dump_json(subset_image_json, image_subset)
    dump_json(subset_video_json, video_subset)

    if args.write_default_json_names:
        dump_json(subset_dir / "llava_image_.json", image_subset)
        dump_json(subset_dir / "valid_valley_.json", video_subset)

    image_files = unique_keep_order([item.get("image", "") for item in image_subset])
    video_files = unique_keep_order([item.get("video", "") for item in video_subset])

    print(
        f"[step] subset ready: image_samples={len(image_subset)}, video_samples={len(video_subset)}, "
        f"image_files={len(image_files)}, video_files={len(video_files)}"
    )

    if not image_files and not video_files:
        print("[done] no media requested (image_n/video_n both 0).")
        return 0

    all_media = [("image", p) for p in image_files] + [("video", p) for p in video_files]

    print("[step] downloading referenced media files")
    failed = []
    for kind, rel_path in tqdm(all_media, desc="media"):
        try:
            download_repo_file(args.repo_id, rel_path, out_dir, args.token)
        except Exception as exc:
            failed.append((kind, rel_path, str(exc)))

    failed_path = out_dir / "train_json" / "download_failed.txt"
    if failed:
        ensure_parent(failed_path)
        with failed_path.open("w", encoding="utf-8") as f:
            for kind, rel_path, err in failed:
                f.write(f"{kind}\t{rel_path}\t{err}\n")
        print(f"[warn] failed files: {len(failed)}; see {failed_path}")

    print("[done] subset download finished")
    print(f"[out] subset image json: {subset_image_json}")
    print(f"[out] subset video json: {subset_video_json}")
    if args.write_default_json_names:
        print("[out] default train json names were overwritten with subset")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
