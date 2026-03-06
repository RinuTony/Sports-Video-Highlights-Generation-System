import argparse
import subprocess
import sys


def run_step(command: list[str]) -> None:
    print(f"[Pipeline] Running: {' '.join(command)}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(f"[Pipeline] Failed: {' '.join(command)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Weakly supervised training pipeline using a full match video and a "
            "pre-created highlight video."
        )
    )
    parser.add_argument(
        "--full-video",
        default="videos/input.mp4",
        help="Path to full match video (default: videos/input.mp4)",
    )
    parser.add_argument(
        "--highlight-video",
        required=True,
        help="Path to pre-created highlight video for the same match",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=5.0,
        help="Clip duration in seconds for both full/highlight videos (default: 5)",
    )
    parser.add_argument(
        "--min-similarity",
        type=float,
        default=0.75,
        help="Minimum cosine similarity for auto positive match (default: 0.75)",
    )
    parser.add_argument(
        "--negative-multiplier",
        type=int,
        default=3,
        help="Auto negatives sampled per positive (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable

    # 1) Split full match to clips
    run_step(
        [
            py,
            "split_video.py",
            "--input",
            args.full_video,
            "--output",
            "clips",
            "--clip-duration",
            str(args.clip_duration),
        ]
    )

    # 2) Split existing highlight video to highlight clips
    run_step(
        [
            py,
            "split_video.py",
            "--input",
            args.highlight_video,
            "--output",
            "highlights",
            "--clip-duration",
            str(args.clip_duration),
        ]
    )

    # 3) Auto-generate weak labels
    run_step(
        [
            py,
            "auto_label_from_highlights.py",
            "--clips-dir",
            "clips",
            "--highlights-dir",
            "highlights",
            "--output-csv",
            "labels.csv",
            "--min-similarity",
            str(args.min_similarity),
            "--negative-multiplier",
            str(args.negative_multiplier),
        ]
    )

    # 4) Build features with generated labels
    run_step([py, "extract_features.py"])

    # 5) Train highlight model
    run_step([py, "train_model.py"])

    print("[Pipeline] Completed: generated labels.csv and highlight_model.pkl")


if __name__ == "__main__":
    main()
