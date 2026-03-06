import os
import joblib
import shutil
import re

from feature_utils import compute_clip_features

payload = joblib.load("highlight_model.pkl")
model_type = payload["model_type"]
model = payload["model"]

clips_folder = "clips"
highlight_folder = "highlights"

os.makedirs(highlight_folder, exist_ok=True)
for existing in os.listdir(highlight_folder):
    existing_path = os.path.join(highlight_folder, existing)
    if os.path.isfile(existing_path):
        os.remove(existing_path)

def clip_sort_key(filename: str):
    match = re.search(r"(\d+)(?=\.[^.]+$)", filename)
    if match:
        return int(match.group(1))
    return float("inf")


if not os.path.isdir(clips_folder):
    raise SystemExit("Missing 'clips' folder. Run split_video.py first.")

video_files = [
    f
    for f in os.listdir(clips_folder)
    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
]
video_files.sort(key=clip_sort_key)

if not video_files:
    raise SystemExit("No clips found in 'clips' folder.")

print(f"[Detect] Stage started: clips={len(video_files)} | model_type={model_type}")
highlight_count = 0
skipped_count = 0

for idx, file in enumerate(video_files, start=1):
    path = os.path.join(clips_folder, file)
    print(f"[Detect] ({idx}/{len(video_files)}) {file}")

    try:
        feature_vector = compute_clip_features(path).reshape(1, -1)
    except Exception as exc:
        skipped_count += 1
        print(f"[Detect] Skipped {file}: {exc}")
        continue

    is_highlight = False

    if model_type == "classifier":
        prediction = model.predict(feature_vector)[0]
        is_highlight = prediction == 1
    elif model_type == "isolation_forest":
        # IsolationForest outputs -1 for anomaly and 1 for inlier.
        prediction = model.predict(feature_vector)[0]
        is_highlight = prediction == -1
    else:
        raise RuntimeError(f"Unknown model type in highlight_model.pkl: {model_type}")

    if is_highlight:
        shutil.copy(path, os.path.join(highlight_folder, file))
        highlight_count += 1
        print(f"[Detect] Highlight selected: {file}")

print("[Detect] Stage finished")
print(f"[Detect] Highlights selected: {highlight_count}")
print(f"[Detect] Clips skipped: {skipped_count}")
