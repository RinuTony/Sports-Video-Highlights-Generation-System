import cv2
import numpy as np
from moviepy.editor import VideoFileClip


def compute_clip_features(clip_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open clip: {clip_path}")

    diff_means = []
    frame_brightness = []
    prev_gray = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_brightness.append(float(np.mean(gray)))

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            diff_means.append(float(np.mean(diff)))

        prev_gray = gray
        frame_count += 1

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()

    if frame_count == 0:
        raise RuntimeError(f"Clip has no frames: {clip_path}")

    motion_mean = float(np.mean(diff_means)) if diff_means else 0.0
    motion_std = float(np.std(diff_means)) if diff_means else 0.0
    scene_change_ratio = (
        float(np.mean(np.array(diff_means) > 30.0)) if diff_means else 0.0
    )
    brightness_mean = float(np.mean(frame_brightness)) if frame_brightness else 0.0
    duration = float(frame_count / fps) if fps > 0 else 0.0

    audio_rms = 0.0
    audio_peak = 0.0
    clip = VideoFileClip(clip_path)
    try:
        if clip.audio is not None:
            chunks = []
            for chunk in clip.audio.iter_chunks(
                chunksize=50000,
                fps=22050,
                quantize=False,
                nbytes=2,
            ):
                if chunk is None:
                    continue
                arr = np.asarray(chunk, dtype=np.float32)
                if arr.size > 0:
                    chunks.append(arr)

            if chunks:
                samples = np.concatenate(chunks, axis=0)
                audio_rms = float(np.sqrt(np.mean(samples ** 2)))
                audio_peak = float(np.max(np.abs(samples)))
    except Exception:
        # Keep processing robust even if a clip has problematic audio metadata/codec.
        audio_rms = 0.0
        audio_peak = 0.0
    finally:
        clip.close()

    return np.array(
        [
            motion_mean,
            motion_std,
            scene_change_ratio,
            brightness_mean,
            audio_rms,
            audio_peak,
            duration,
        ],
        dtype=np.float32,
    )


FEATURE_NAMES = [
    "motion_mean",
    "motion_std",
    "scene_change_ratio",
    "brightness_mean",
    "audio_rms",
    "audio_peak",
    "duration",
]
