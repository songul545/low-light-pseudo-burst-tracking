# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from glob import glob
from natsort import natsorted

folders = {
    "Original": "video_frames_yakin/original",
    "Adaptive_Burst": "video_frames_yakin/adaptive_burst",
    "Original_Retinexformer": "video_frames_yakin/original_enhanced",
    "Adaptive_Burst_Retinexformer": "video_frames_yakin/adaptive_enhanced",
}

def load_gray(path):
    img = cv2.imread(path)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray.astype(np.float32)

def brightness(gray):
    return float(np.mean(gray))

def contrast(gray):
    return float(np.std(gray))

def laplacian_variance(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    return float(np.var(lap))

def temporal_difference(gray_frames):
    diffs = []
    for i in range(1, len(gray_frames)):
        diff = np.mean(np.abs(gray_frames[i] - gray_frames[i - 1]))
        diffs.append(diff)
    return float(np.mean(diffs)) if diffs else 0.0

def analyze_folder(folder_path):
    paths = natsorted(glob(os.path.join(folder_path, "*.png")))

    if len(paths) == 0:
        return None

    grays = [load_gray(p) for p in paths]

    b_values = [brightness(g) for g in grays]
    c_values = [contrast(g) for g in grays]
    l_values = [laplacian_variance(g) for g in grays]

    return {
        "frames": len(paths),
        "mean_brightness": np.mean(b_values),
        "mean_contrast": np.mean(c_values),
        "mean_laplacian_var": np.mean(l_values),
        "temporal_diff": temporal_difference(grays),
    }

print("\nImage Quality and Temporal Stability Metrics\n")
print(
    f"{'Input Type':35s} | {'Frames':>6s} | {'Brightness':>10s} | "
    f"{'Contrast':>10s} | {'Laplacian Var':>14s} | {'Temporal Diff':>14s}"
)
print("-" * 105)

for name, folder in folders.items():
    if not os.path.exists(folder):
        print(f"{name:35s} | folder not found: {folder}")
        continue

    result = analyze_folder(folder)

    if result is None:
        print(f"{name:35s} | no images found")
        continue

    print(
        f"{name:35s} | "
        f"{result['frames']:6d} | "
        f"{result['mean_brightness']:10.2f} | "
        f"{result['mean_contrast']:10.2f} | "
        f"{result['mean_laplacian_var']:14.2f} | "
        f"{result['temporal_diff']:14.2f}"
    )

# Extra comparison: pixel-level difference between original and adaptive burst
orig_paths = natsorted(glob(os.path.join(folders["Original"], "*.png")))
burst_paths = natsorted(glob(os.path.join(folders["Adaptive_Burst"], "*.png")))

if len(orig_paths) > 0 and len(orig_paths) == len(burst_paths):
    mad_values = []

    for op, bp in zip(orig_paths, burst_paths):
        o = load_gray(op)
        b = load_gray(bp)
        mad_values.append(np.mean(np.abs(o - b)))

    print("\nOriginal vs Adaptive Burst")
    print(f"Mean Absolute Difference: {np.mean(mad_values):.2f}")
else:
    print("\nOriginal vs Adaptive Burst comparison could not be calculated.")