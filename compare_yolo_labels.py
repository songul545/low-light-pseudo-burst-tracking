import os
from glob import glob
import numpy as np

# file paths for YOLO label folders
experiments = {
    "Original": "yolo_label_results/original",
    "Adaptive_Burst": "yolo_label_results/adaptive_burst",
    "Original_Retinexformer": "yolo_label_results/original_enhanced",
    "Adaptive_Burst_Retinexformer": "yolo_label_results/adaptive_enhanced",
}
PERSON_ID = 0
VEHICLE_IDS = {2, 3, 5, 7}  # car, motorcycle, bus, truck

def analyze_labels(label_dir):
    txt_files = glob(os.path.join(label_dir, "*.txt"))

    person_detected_frames = 0
    total_person_detections = 0
    person_confidences = []

    vehicle_detected_frames = 0
    total_vehicle_detections = 0
    vehicle_confidences = []

    total_detections = 0

    for txt_file in txt_files:
        has_person = False
        has_vehicle = False

        with open(txt_file, "r") as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            class_id = int(float(parts[0]))
            conf = float(parts[5])
            total_detections += 1

            if class_id == PERSON_ID:
                has_person = True
                total_person_detections += 1
                person_confidences.append(conf)

            if class_id in VEHICLE_IDS:
                has_vehicle = True
                total_vehicle_detections += 1
                vehicle_confidences.append(conf)

        if has_person:
            person_detected_frames += 1

        if has_vehicle:
            vehicle_detected_frames += 1

    return {
        "label_files": len(txt_files),
        "person_detected_frames": person_detected_frames,
        "total_person_detections": total_person_detections,
        "avg_person_conf": np.mean(person_confidences) if person_confidences else 0,
        "max_person_conf": np.max(person_confidences) if person_confidences else 0,
        "vehicle_detected_frames": vehicle_detected_frames,
        "total_vehicle_detections": total_vehicle_detections,
        "avg_vehicle_conf": np.mean(vehicle_confidences) if vehicle_confidences else 0,
        "max_vehicle_conf": np.max(vehicle_confidences) if vehicle_confidences else 0,
        "total_detections": total_detections,
    }

print("\nYOLO Detection Comparison\n")
print(
    f"{'Experiment':35s} | {'Files':>6s} | {'Person Frames':>13s} | "
    f"{'Person Det':>10s} | {'Avg P Conf':>10s} | {'Max P Conf':>10s} | "
    f"{'Vehicle Frames':>14s} | {'Vehicle Det':>11s}"
)
print("-" * 130)

for name, label_dir in experiments.items():
    if not os.path.exists(label_dir):
        print(f"{name:35s} | labels folder not found: {label_dir}")
        continue

    r = analyze_labels(label_dir)

    print(
        f"{name:35s} | "
        f"{r['label_files']:6d} | "
        f"{r['person_detected_frames']:13d} | "
        f"{r['total_person_detections']:10d} | "
        f"{r['avg_person_conf']:10.3f} | "
        f"{r['max_person_conf']:10.3f} | "
        f"{r['vehicle_detected_frames']:14d} | "
        f"{r['total_vehicle_detections']:11d}"
    )