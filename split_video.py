import argparse
import os
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def split_video_to_clips(video_path: str, output_folder: str, clip_duration: float) -> int:
    os.makedirs(output_folder, exist_ok=True)

    for existing in os.listdir(output_folder):
        existing_path = os.path.join(output_folder, existing)
        if os.path.isfile(existing_path) and existing.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        ):
            os.remove(existing_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Missing or unreadable video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise SystemExit(f"Invalid FPS for video: {video_path}")

    frames_per_clip = max(1, int(fps * clip_duration))
    count = 0
    clip_id = 0

    print(
        f"[Split] Stage started: source={video_path}, output={output_folder}, "
        f"clip_length={clip_duration}s"
    )
    while True:
        ret, _frame = cap.read()
        if not ret:
            break

        count += 1

        if count == frames_per_clip:
            clip_start = (clip_id * frames_per_clip) / fps
            clip_end = clip_start + clip_duration
            ffmpeg_extract_subclip(
                video_path,
                clip_start,
                clip_end,
                targetname=os.path.join(output_folder, f"clip_{clip_id}.mp4"),
            )
            count = 0
            clip_id += 1

    if count > 0:
        clip_start = (clip_id * frames_per_clip) / fps
        ffmpeg_extract_subclip(
            video_path,
            clip_start,
            float(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps),
            targetname=os.path.join(output_folder, f"clip_{clip_id}.mp4"),
        )
        clip_id += 1

    cap.release()
    print(f"[Split] Stage finished: created {clip_id} clip(s) in {output_folder}")
    return clip_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split a video into fixed-duration clips."
    )
    parser.add_argument(
        "--input",
        default="videos/input.mp4",
        help="Input video path (default: videos/input.mp4)",
    )
    parser.add_argument(
        "--output",
        default="clips",
        help="Output folder for generated clips (default: clips)",
    )
    parser.add_argument(
        "--clip-duration",
        type=float,
        default=5.0,
        help="Clip duration in seconds (default: 5)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_video_to_clips(args.input, args.output, args.clip_duration)
