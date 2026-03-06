import os
import numpy as np
import re
import csv

from feature_utils import FEATURE_NAMES, compute_clip_features

clips_folder = "clips"
labels_csv_path = "labels.csv"
features = []
labels = []
filenames = []


def infer_label_from_name(filename: str) -> int:
    name = filename.lower()
    positive_keywords = ("goal", "wicket", "six", "four", "score", "highlight")
    negative_keywords = ("normal", "boring", "nonhighlight", "nohighlight")

    if any(k in name for k in positive_keywords):
        return 1
    if any(k in name for k in negative_keywords):
        return 0
    return -1


def parse_label(value: str) -> int:
    text = str(value).strip().lower()
    if text in {"1", "highlight", "positive", "pos", "yes", "true"}:
        return 1
    if text in {"0", "normal", "negative", "neg", "no", "false"}:
        return 0
    return -1


def load_labels_from_csv(path: str) -> dict:
    if not os.path.isfile(path):
        return {}

    labels_map = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            return {}

        field_map = {f.strip().lower(): f for f in reader.fieldnames}
        filename_col = None
        label_col = None

        for candidate in ("filename", "file", "clip", "path"):
            if candidate in field_map:
                filename_col = field_map[candidate]
                break

        for candidate in ("label", "class", "target", "y"):
            if candidate in field_map:
                label_col = field_map[candidate]
                break

        if not filename_col or not label_col:
            print("labels.csv found but missing required columns (filename,label).")
            return {}

        for row in reader:
            raw_name = str(row.get(filename_col, "")).strip()
            raw_label = str(row.get(label_col, "")).strip()
            if not raw_name:
                continue
            label = parse_label(raw_label)
            if label == -1:
                continue

            key = os.path.basename(raw_name).lower()
            labels_map[key] = label

    return labels_map


if not os.path.isdir(clips_folder):
    raise SystemExit("Missing 'clips' folder. Run split_video.py first.")


def clip_sort_key(filename: str):
    match = re.search(r"(\d+)(?=\.[^.]+$)", filename)
    if match:
        return int(match.group(1))
    return float("inf")

video_files = [
    f
    for f in os.listdir(clips_folder)
    if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
]
video_files.sort(key=clip_sort_key)

if not video_files:
    raise SystemExit("No clips found in 'clips' folder.")

csv_labels = load_labels_from_csv(labels_csv_path)
print(f"[Features] Stage started: {len(video_files)} clip(s) found")
if csv_labels:
    print(f"[Features] Labels source: {labels_csv_path} ({len(csv_labels)} rows)")
else:
    print("[Features] Labels source: filename keyword fallback (labels.csv missing/unusable)")

csv_labeled_count = 0
fallback_labeled_count = 0
skipped_count = 0

for idx, filename in enumerate(video_files, start=1):
    filepath = os.path.join(clips_folder, filename)
    print(f"[Features] ({idx}/{len(video_files)}) {filename}")

    try:
        feature_vector = compute_clip_features(filepath)
        features.append(feature_vector)

        csv_label = csv_labels.get(filename.lower(), -1)
        if csv_label != -1:
            label = csv_label
            csv_labeled_count += 1
        else:
            label = infer_label_from_name(filename)
            if label != -1:
                fallback_labeled_count += 1

        labels.append(label)
        filenames.append(filename)
    except Exception as exc:
        skipped_count += 1
        print(f"[Features] Skipped {filename}: {exc}")

if not features:
    raise SystemExit("No valid clips were processed.")

X = np.vstack(features).astype(np.float32)
y = np.array(labels, dtype=np.int32)
files = np.array(filenames)

os.makedirs("features", exist_ok=True)
np.save("features/features.npy", X)
np.save("features/labels.npy", y)
np.save("features/filenames.npy", files)

print("[Features] Stage finished")
print(f"[Features] Saved samples: {len(X)}")
print(f"[Features] Skipped clips: {skipped_count}")
print(f"[Features] Feature columns: {FEATURE_NAMES}")
print(f"[Features] Labeled samples: {(y != -1).sum()} / {len(y)}")
print(f"[Features] Labeled from CSV: {csv_labeled_count}")
print(f"[Features] Labeled from filename fallback: {fallback_labeled_count}")
