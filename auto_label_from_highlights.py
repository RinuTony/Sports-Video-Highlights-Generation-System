import argparse
import csv
import os
import random
import re
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from feature_utils import compute_clip_features


VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def clip_sort_key(filename: str) -> int:
    match = re.search(r"(\d+)(?=\.[^.]+$)", filename)
    if match:
        return int(match.group(1))
    return 10**9


def list_video_files(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    files = [f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTENSIONS)]
    files.sort(key=clip_sort_key)
    return files


def extract_features_map(folder: str, files: List[str]) -> Dict[str, np.ndarray]:
    feature_map: Dict[str, np.ndarray] = {}
    for idx, filename in enumerate(files, start=1):
        filepath = os.path.join(folder, filename)
        print(f"[AutoLabel] Features ({idx}/{len(files)}): {filepath}")
        try:
            feature_map[filename] = compute_clip_features(filepath)
        except Exception as exc:
            print(f"[AutoLabel] Skipped {filepath}: {exc}")
    return feature_map


def auto_match_by_similarity(
    full_files: List[str],
    full_features: np.ndarray,
    highlight_files: List[str],
    highlight_features: np.ndarray,
    min_similarity: float,
) -> Tuple[List[int], List[Tuple[str, str, float]]]:
    scaler = StandardScaler()
    full_scaled = scaler.fit_transform(full_features)
    highlight_scaled = scaler.transform(highlight_features)

    sim_matrix = cosine_similarity(highlight_scaled, full_scaled)
    positives: List[int] = []
    details: List[Tuple[str, str, float]] = []

    for i, h_name in enumerate(highlight_files):
        best_j = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_j])
        if best_sim >= min_similarity:
            positives.append(best_j)
            details.append((h_name, full_files[best_j], best_sim))

    return positives, details


def write_labels_csv(
    output_csv: str,
    full_files: List[str],
    positive_indices: List[int],
    negative_count_multiplier: int,
    exclusion_radius: int,
    seed: int,
) -> Tuple[int, int]:
    rng = random.Random(seed)
    positive_set = set(positive_indices)

    blocked_negatives = set()
    for idx in positive_set:
        for k in range(
            max(0, idx - exclusion_radius), min(len(full_files), idx + exclusion_radius + 1)
        ):
            blocked_negatives.add(k)

    negative_candidates = [
        idx for idx in range(len(full_files)) if idx not in positive_set and idx not in blocked_negatives
    ]

    target_negatives = min(
        len(negative_candidates),
        max(len(positive_set), 1) * max(1, negative_count_multiplier),
    )
    negative_indices = set(rng.sample(negative_candidates, target_negatives))

    with open(output_csv, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["filename", "label", "source"])
        for idx, filename in enumerate(full_files):
            if idx in positive_set:
                writer.writerow([filename, 1, "auto_match"])
            elif idx in negative_indices:
                writer.writerow([filename, 0, "auto_negative"])

    return len(positive_set), len(negative_indices)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate weak labels automatically by matching highlight clips against "
            "full-match clips using feature similarity."
        )
    )
    parser.add_argument(
        "--clips-dir",
        default="clips",
        help="Folder containing full-match clips (default: clips)",
    )
    parser.add_argument(
        "--highlights-dir",
        default="highlights",
        help="Folder containing highlight clips (default: highlights)",
    )
    parser.add_argument(
        "--output-csv",
        default="labels.csv",
        help="Output labels CSV path (default: labels.csv)",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.75,
        help="Minimum cosine similarity for accepting a positive match (default: 0.75)",
    )
    parser.add_argument(
        "--negative-multiplier",
        type=int,
        default=3,
        help="How many negatives to sample per positive (default: 3)",
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
        help="Random seed for negative sampling (default: 42)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    full_files = list_video_files(args.clips_dir)
    highlight_files = list_video_files(args.highlights_dir)

    if not full_files:
        raise SystemExit(
            f"No full-match clips found in '{args.clips_dir}'. Run split_video.py first."
        )
    if not highlight_files:
        raise SystemExit(
            f"No highlight clips found in '{args.highlights_dir}'. "
            "Split your highlight video to clips first."
        )

    print(
        f"[AutoLabel] Stage started: full_clips={len(full_files)}, "
        f"highlight_clips={len(highlight_files)}"
    )

    full_map = extract_features_map(args.clips_dir, full_files)
    highlight_map = extract_features_map(args.highlights_dir, highlight_files)

    full_files = [f for f in full_files if f in full_map]
    highlight_files = [f for f in highlight_files if f in highlight_map]

    if not full_files or not highlight_files:
        raise SystemExit("Unable to compute enough features for auto-labeling.")

    full_features = np.vstack([full_map[f] for f in full_files])
    highlight_features = np.vstack([highlight_map[f] for f in highlight_files])

    positive_indices, match_details = auto_match_by_similarity(
        full_files=full_files,
        full_features=full_features,
        highlight_files=highlight_files,
        highlight_features=highlight_features,
        min_similarity=args.min_similarity,
    )

    if not positive_indices:
        raise SystemExit(
            "No positive matches found. Try lowering --min-similarity (example: 0.65)."
        )

    positives, negatives = write_labels_csv(
        output_csv=args.output_csv,
        full_files=full_files,
        positive_indices=positive_indices,
        negative_count_multiplier=args.negative_multiplier,
        exclusion_radius=args.exclusion_radius,
        seed=args.seed,
    )

    print("[AutoLabel] Stage finished")
    print(f"[AutoLabel] Output: {args.output_csv}")
    print(f"[AutoLabel] Positives: {positives}")
    print(f"[AutoLabel] Negatives: {negatives}")
    print("[AutoLabel] Top matches:")
    for h_name, c_name, sim in sorted(match_details, key=lambda x: x[2], reverse=True)[:10]:
        print(f"  - {h_name} -> {c_name} (similarity={sim:.3f})")


if __name__ == "__main__":
    main()
