from moviepy.editor import VideoFileClip, concatenate_videoclips
import os
import re

clips = []


def clip_sort_key(filename: str):
    match = re.search(r"(\d+)(?=\.[^.]+$)", filename)
    if match:
        return int(match.group(1))
    return float("inf")

if not os.path.isdir("highlights"):
    raise SystemExit("Missing 'highlights' folder. Run detect_highlights.py first.")

for file in sorted(os.listdir("highlights"), key=clip_sort_key):
    if not file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue
    clip = VideoFileClip(os.path.join("highlights", file))
    clips.append(clip)

if not clips:
    raise SystemExit("No highlight clips found. Nothing to merge.")

print(f"[Create] Stage started: merging {len(clips)} highlight clip(s)")
final = concatenate_videoclips(clips)

final.write_videofile(
    "final_highlights.mp4",
    codec="libx264",
    audio_codec="aac",
)
final.close()
for c in clips:
    c.close()

print("[Create] Stage finished: final_highlights.mp4 created")
