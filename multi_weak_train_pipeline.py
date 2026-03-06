import argparse
import csv
import os
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from feature_utils import compute_clip_features
from split_video import split_video_to_clips


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


@dataclass
class MatchPair:
    match_id: str
    full_video: str
    highlight_video: str


def clip_sort_key(filename: str) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", filename)
    if match:
        return int(match.group(1))
    return 10**9


def list_video_files(folder: str) -> List[str]:
    files = [f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTENSIONS)]
    files.sort(key=clip_sort_key)
    return files


def compute_feature_matrix(folder: str, files: List[str]) -> Tuple[List[str], np.ndarray]:
    valid_files: List[str] = []
    features: List[np.ndarray] = []
    for idx, filename in enumerate(files, start=1):
        path = os.path.join(folder, filename)
        print(f"[MultiTrain] Feature ({idx}/{len(files)}): {path}")
        try:
            features.append(compute_clip_features(path))
            valid_files.append(filename)
        except Exception as exc:
            print(f"[MultiTrain] Skipped {path}: {exc}")
    if not features:
        return [], np.zeros((0, 7), dtype=np.float32)
    return valid_files, np.vstack(features).astype(np.float32)


def match_highlights_to_full(
    full_features: np.ndarray,
    highlight_features: np.ndarray,
    min_similarity: float,
) -> List[int]:
    scaler = StandardScaler()
    full_scaled = scaler.fit_transform(full_features)
    highlight_scaled = scaler.transform(highlight_features)
    sim = cosine_similarity(highlight_scaled, full_scaled)

    positive_indices: List[int] = []
    for i in range(sim.shape[0]):
        best_j = int(np.argmax(sim[i]))
        best_sim = float(sim[i, best_j])
        if best_sim >= min_similarity:
            positive_indices.append(best_j)
    return positive_indices


def sample_negative_indices(
    total: int,
    positives: List[int],
    exclusion_radius: int,
    negative_multiplier: int,
    rng: random.Random,
) -> List[int]:
    positive_set = set(positives)
    blocked = set()
    for idx in positive_set:
        for k in range(max(0, idx - exclusion_radius), min(total, idx + exclusion_radius + 1)):
            blocked.add(k)

    candidates = [i for i in range(total) if i not in positive_set and i not in blocked]
    take = min(len(candidates), max(1, len(positive_set)) * max(1, negative_multiplier))
    if take == 0:
        return []
    return rng.sample(candidates, take)


def load_pairs_csv(path: str) -> List[MatchPair]:
    if not os.path.isfile(path):
        raise SystemExit(f"Pairs CSV not found: {path}")

    pairs: List[MatchPair] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise SystemExit("Pairs CSV has no header.")

        fields = {k.strip().lower(): k for k in reader.fieldnames}
        full_col = fields.get("full_video")
        highlight_col = fields.get("highlight_video")
        id_col = fields.get("match_id")

        if not full_col or not highlight_col:
            raise SystemExit(
                "Pairs CSV must include columns: full_video,highlight_video "
                "(optional: match_id)."
            )

        for idx, row in enumerate(reader, start=1):
            full_video = str(row.get(full_col, "")).strip()
            highlight_video = str(row.get(highlight_col, "")).strip()
            match_id = str(row.get(id_col, "")).strip() if id_col else ""
            if not full_video or not highlight_video:
                continue
            if not match_id:
                match_id = f"match_{idx:03d}"
            pairs.append(
                MatchPair(
                    match_id=match_id,
                    full_video=full_video,
                    highlight_video=highlight_video,
                )
            )

    if not pairs:
        raise SystemExit("No valid rows found in pairs CSV.")
    return pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one highlight model from multiple full/highlight match pairs."
    )
    parser.add_argument(
        "--pairs-csv",
        default="match_pairs.csv",
        help=(
            "CSV with columns full_video,highlight_video and optional match_id "
            "(default: match_pairs.csv)"
        ),
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=5.0,
        help="Clip duration in seconds (default: 5)",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.75,
        help="Minimum cosine similarity for auto positives (default: 0.75)",
    )
    parser.add_argument(
        "--negative-multiplier",
        type=int,
        default=3,
        help="Negatives sampled per positive (default: 3)",
    )
    parser.add_argument(
        "--exclusion-radius",
        type=int,
        default=1,
        help="Do not sample negatives this many clip IDs around positives (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--tmp-dir",
        default="tmp_multi_train",
        help="Temporary workspace for split clips (default: tmp_multi_train)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    pairs = load_pairs_csv(args.pairs_csv)

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.makedirs("features", exist_ok=True)

    all_features: List[np.ndarray] = []
    all_labels: List[int] = []
    all_filenames: List[str] = []
    labels_rows: List[Tuple[str, int, str, str]] = []

    used_matches = 0
    for pair in pairs:
        full_clip_dir = os.path.join(args.tmp_dir, f"{pair.match_id}_clips")
        highlight_clip_dir = os.path.join(args.tmp_dir, f"{pair.match_id}_highlights")

        print(
            f"[MultiTrain] Processing {pair.match_id} | full={pair.full_video} | "
            f"highlight={pair.highlight_video}"
        )

        if not os.path.isfile(pair.full_video):
            print(f"[MultiTrain] Missing full video: {pair.full_video} (skipped)")
            continue
        if not os.path.isfile(pair.highlight_video):
            print(f"[MultiTrain] Missing highlight video: {pair.highlight_video} (skipped)")
            continue

        split_video_to_clips(pair.full_video, full_clip_dir, args.clip_duration)
        split_video_to_clips(pair.highlight_video, highlight_clip_dir, args.clip_duration)

        full_files = list_video_files(full_clip_dir)
        highlight_files = list_video_files(highlight_clip_dir)
        if not full_files or not highlight_files:
            print(f"[MultiTrain] No clips for {pair.match_id} (skipped)")
            continue

        full_valid, full_feat = compute_feature_matrix(full_clip_dir, full_files)
        high_valid, high_feat = compute_feature_matrix(highlight_clip_dir, highlight_files)
        if len(full_valid) == 0 or len(high_valid) == 0:
            print(f"[MultiTrain] No valid features for {pair.match_id} (skipped)")
            continue

        positive_indices = match_highlights_to_full(
            full_features=full_feat,
            highlight_features=high_feat,
            min_similarity=args.min_similarity,
        )
        positive_indices = sorted(set(positive_indices))
        if not positive_indices:
            print(f"[MultiTrain] No positive matches for {pair.match_id} (skipped)")
            continue

        negative_indices = sample_negative_indices(
            total=len(full_valid),
            positives=positive_indices,
            exclusion_radius=args.exclusion_radius,
            negative_multiplier=args.negative_multiplier,
            rng=rng,
        )
        negative_set = set(negative_indices)
        positive_set = set(positive_indices)

        for idx, clip_name in enumerate(full_valid):
            if idx in positive_set:
                label = 1
                source = "auto_match"
            elif idx in negative_set:
                label = 0
                source = "auto_negative"
            else:
                continue

            all_features.append(full_feat[idx])
            all_labels.append(label)
            key = f"{pair.match_id}/{clip_name}"
            all_filenames.append(key)
            labels_rows.append((key, label, source, pair.match_id))

        used_matches += 1
        print(
            f"[MultiTrain] {pair.match_id} labeled: "
            f"positives={len(positive_set)} negatives={len(negative_set)}"
        )

    if not all_features:
        raise SystemExit("No training samples generated from provided pairs.")

    X = np.vstack(all_features).astype(np.float32)
    y = np.array(all_labels, dtype=np.int32)
    names = np.array(all_filenames)

    np.save("features/features.npy", X)
    np.save("features/labels.npy", y)
    np.save("features/filenames.npy", names)

    with open("labels.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label", "source", "match_id"])
        writer.writerows(labels_rows)

    print(
        f"[MultiTrain] Dataset ready: samples={len(X)} "
        f"positives={(y == 1).sum()} negatives={(y == 0).sum()} "
        f"matches_used={used_matches}"
    )

    cmd = [sys.executable, "train_model.py"]
    print(f"[MultiTrain] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        raise SystemExit("[MultiTrain] Training failed at train_model.py")
    print("[MultiTrain] Completed: highlight_model.pkl updated")


if __name__ == "__main__":
    main()
