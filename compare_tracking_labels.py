import os
from glob import glob
from collections import defaultdict
from natsort import natsorted
import numpy as np

experiments = {
    "Original": "tracking_label_results/original",
    "Adaptive_Burst": "tracking_label_results/adaptive_burst",
    "Original_Retinexformer": "tracking_label_results/original_enhanced",
    "Adaptive_Burst_Retinexformer": "tracking_label_results/adaptive_enhanced",
}

PERSON_ID = 0

def analyze_tracking(label_dir):
    txt_files = natsorted(glob(os.path.join(label_dir, "*.txt")))

    # track_id -> list of frame indices
    track_frames = defaultdict(list)

    tracked_person_frames = 0
    total_person_detections = 0
    person_confidences = []

    for frame_idx, txt_file in enumerate(txt_files):
        has_person = False

        with open(txt_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()

            if len(parts) < 6:
                continue

            class_id = int(float(parts[0]))

            if class_id != PERSON_ID:
                continue

            has_person = True
            total_person_detections += 1

            # YOLO track txt format may be:
            # class x y w h track_id conf
            # or class x y w h conf
            track_id = None
            conf = None

            if len(parts) >= 7:
                conf = float(parts[5])
                track_id = int(float(parts[6]))
            else:
                # Fallback if no track id exists
                track_id = -1
                conf = float(parts[5])

            track_frames[track_id].append(frame_idx)
            person_confidences.append(conf)

        if has_person:
            tracked_person_frames += 1

    track_lengths = [len(frames) for frames in track_frames.values()]

    # Longest continuous track segment
    longest_continuous = 0
    for frames in track_frames.values():
        frames = sorted(frames)
        current = 1

        for i in range(1, len(frames)):
            if frames[i] == frames[i - 1] + 1:
                current += 1
            else:
                longest_continuous = max(longest_continuous, current)
                current = 1

        if frames:
            longest_continuous = max(longest_continuous, current)

    return {
        "label_files": len(txt_files),
        "tracked_person_frames": tracked_person_frames,
        "total_person_detections": total_person_detections,
        "unique_person_track_ids": len(track_frames),
        "avg_track_length": np.mean(track_lengths) if track_lengths else 0,
        "max_track_length": np.max(track_lengths) if track_lengths else 0,
        "longest_continuous_track": longest_continuous,
        "avg_person_conf": np.mean(person_confidences) if person_confidences else 0,
        "max_person_conf": np.max(person_confidences) if person_confidences else 0,
    }

print("\nYOLOv8s + ByteTrack Tracking Comparison\n")
print(
    f"{'Experiment':35s} | {'Files':>6s} | {'Tracked Frames':>14s} | "
    f"{'Person Det':>10s} | {'Track IDs':>9s} | {'Avg Track':>9s} | "
    f"{'Max Track':>9s} | {'Longest Cont.':>13s} | {'Avg Conf':>8s}"
)
print("-" * 135)

for name, label_dir in experiments.items():
    if not os.path.exists(label_dir):
        print(f"{name:35s} | folder not found: {label_dir}")
        continue

    r = analyze_tracking(label_dir)

    print(
        f"{name:35s} | "
        f"{r['label_files']:6d} | "
        f"{r['tracked_person_frames']:14d} | "
        f"{r['total_person_detections']:10d} | "
        f"{r['unique_person_track_ids']:9d} | "
        f"{r['avg_track_length']:9.2f} | "
        f"{r['max_track_length']:9.0f} | "
        f"{r['longest_continuous_track']:13d} | "
        f"{r['avg_person_conf']:8.3f}"
    )