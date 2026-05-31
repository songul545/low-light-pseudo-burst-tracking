import os
import cv2
import numpy as np
from glob import glob
from natsort import natsorted

input_dir = "video_frames_yakin/original"
output_dir = "video_frames_yakin/adaptive_burst"

os.makedirs(output_dir, exist_ok=True)

frame_paths = natsorted(glob(os.path.join(input_dir, "*.png")))

if len(frame_paths) == 0:
    raise RuntimeError("No frames found.")

window_radius = 1
eps = 1e-6

def brightness_score(gray):
    mean_val = np.mean(gray) / 255.0
    return np.clip(mean_val, 0.05, 1.0)

def noise_score(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    noise_val = np.var(lap)
    score = 1.0 / (1.0 + noise_val / 500.0)
    return np.clip(score, 0.05, 1.0)

def similarity_score(center_gray, gray):
    diff = np.mean(np.abs(center_gray.astype(np.float32) - gray.astype(np.float32))) / 255.0
    score = np.exp(-diff * 6.0)
    return np.clip(score, 0.05, 1.0)

for i in range(len(frame_paths)):
    center_img = cv2.imread(frame_paths[i])
    center_gray = cv2.cvtColor(center_img, cv2.COLOR_BGR2GRAY)

    selected_imgs = []
    raw_weights = []

    for offset in range(-window_radius, window_radius + 1):
        idx = i + offset
        idx = max(0, min(idx, len(frame_paths) - 1))

        img = cv2.imread(frame_paths[idx])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        b = brightness_score(gray)
        n = noise_score(gray)
        s = similarity_score(center_gray, gray)

        center_boost = 3.0 if offset == 0 else 1.0

        weight = b * n * s * center_boost

        selected_imgs.append(img.astype(np.float32))
        raw_weights.append(weight)

    raw_weights = np.array(raw_weights, dtype=np.float32)
    norm_weights = raw_weights / (np.sum(raw_weights) + eps)

    fused = np.zeros_like(selected_imgs[0], dtype=np.float32)

    for img, w in zip(selected_imgs, norm_weights):
        fused += img * w

    fused = np.clip(fused, 0, 255).astype(np.uint8)

    filename = os.path.basename(frame_paths[i])
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, fused)

print(f"Adaptive pseudo-burst fused frames saved to: {output_dir}")
print(f"Total frames: {len(frame_paths)}")