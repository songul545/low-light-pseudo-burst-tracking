import os
import cv2
import shutil
import subprocess
import numpy as np
from glob import glob
from natsort import natsorted


def extract_frames(video_path, output_dir, frame_step=1):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Video could not be opened: {video_path}")

    frame_id = 0
    saved_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_step == 0:
            out_path = os.path.join(output_dir, f"frame_{saved_id:05d}.png")
            cv2.imwrite(out_path, frame)
            saved_id += 1

        frame_id += 1

    cap.release()

    print("[1] Frame extraction completed.")
    print(f"    Total frames read: {frame_id}")
    print(f"    Saved frames: {saved_id}")
    print(f"    Output folder: {output_dir}")

    return saved_id


def brightness_score(gray):
    mean_val = np.mean(gray) / 255.0
    return np.clip(mean_val, 0.05, 1.0)


def noise_score(gray):
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    noise_val = np.var(lap)
    score = 1.0 / (1.0 + noise_val / 500.0)
    return np.clip(score, 0.05, 1.0)


def similarity_score(center_gray, gray):
    diff = np.mean(
        np.abs(center_gray.astype(np.float32) - gray.astype(np.float32))
    ) / 255.0

    score = np.exp(-diff * 6.0)
    return np.clip(score, 0.05, 1.0)


def adaptive_pseudo_burst_fusion(input_dir, output_dir):
    """
    Center-preserving adaptive pseudo-burst fusion.

    For each target frame F_t, this function uses:
    F(t-1), F(t), F(t+1)

    Neighboring frames are weighted according to:
    - brightness score
    - noise score
    - similarity score
    - center-frame boost
    """

    os.makedirs(output_dir, exist_ok=True)

    frame_paths = natsorted(glob(os.path.join(input_dir, "*.png")))

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {input_dir}")

    window_radius = 1
    eps = 1e-6

    for i in range(len(frame_paths)):
        center_img = cv2.imread(frame_paths[i])

        if center_img is None:
            raise RuntimeError(f"Could not read center frame: {frame_paths[i]}")

        center_gray = cv2.cvtColor(center_img, cv2.COLOR_BGR2GRAY)

        selected_imgs = []
        raw_weights = []

        for offset in range(-window_radius, window_radius + 1):
            idx = i + offset
            idx = max(0, min(idx, len(frame_paths) - 1))

            img = cv2.imread(frame_paths[idx])

            if img is None:
                raise RuntimeError(f"Could not read frame: {frame_paths[idx]}")

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

    print("[2] Adaptive pseudo-burst fusion completed.")
    print(f"    Input folder: {input_dir}")
    print(f"    Output folder: {output_dir}")
    print(f"    Total frames: {len(frame_paths)}")

    return len(frame_paths)


def run_retinexformer(input_dir, output_dir, opt_path, weights_path, use_cpu=True):
    os.makedirs(output_dir, exist_ok=True)

    cpu_flag = "--cpu" if use_cpu else ""

    command = (
        f'python Enhancement/test_my_images.py '
        f'--input "{input_dir}" '
        f'--output "{output_dir}" '
        f'--opt "{opt_path}" '
        f'--weights "{weights_path}" '
        f'{cpu_flag}'
    )

    print("[3] Running Retinexformer enhancement...")
    print(command)

    subprocess.run(command, shell=True, check=True)

    print("[3] Retinexformer enhancement completed.")
    print(f"    Output folder: {output_dir}")


def frames_to_video(input_dir, output_video, fps=30):
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    frame_paths = natsorted(glob(os.path.join(input_dir, "*.png")))

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {input_dir}")

    first_frame = cv2.imread(frame_paths[0])

    if first_frame is None:
        raise RuntimeError(f"Could not read first frame: {frame_paths[0]}")

    h, w = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    for path in frame_paths:
        frame = cv2.imread(path)
        writer.write(frame)

    writer.release()

    print("[4] Frames converted to video.")
    print(f"    Output video: {output_video}")
    print(f"    Frames: {len(frame_paths)}")

    return output_video


def find_latest_yolo_track_dir():
    """
    Finds the latest YOLO tracking output directory.
    Ultralytics may save results to the default user runs directory.
    """

    possible_roots = [
        os.path.join(os.getcwd(), "runs", "track"),
        os.path.join(os.getcwd(), "runs", "detect"),
        os.path.join(os.path.expanduser("~"), "runs", "track"),
        os.path.join(os.path.expanduser("~"), "runs", "detect"),
    ]

    candidates = []

    for root in possible_roots:
        if not os.path.exists(root):
            continue

        for name in os.listdir(root):
            path = os.path.join(root, name)
            if os.path.isdir(path):
                candidates.append(path)

    if len(candidates) == 0:
        return None

    latest_dir = max(candidates, key=os.path.getmtime)
    return latest_dir


def run_yolo_tracking(video_path, yolo_model, output_project, output_name, conf=0.10):
    output_project = os.path.abspath(output_project)
    os.makedirs(output_project, exist_ok=True)

    command = (
        f'yolo track '
        f'model="{yolo_model}" '
        f'source="{video_path}" '
        f'save=True '
        f'save_txt=True '
        f'save_conf=True '
        f'tracker=bytetrack.yaml '
        f'conf={conf}'
    )

    print("[5] Running YOLOv8s + ByteTrack tracking...")
    print(command)

    subprocess.run(command, shell=True, check=True)

    latest_dir = find_latest_yolo_track_dir()

    if latest_dir is None:
        print("[5] Tracking completed, but YOLO output folder could not be found.")
        return None

    final_result_dir = os.path.join(output_project, output_name)

    if os.path.exists(final_result_dir):
        shutil.rmtree(final_result_dir)

    shutil.copytree(latest_dir, final_result_dir)

    print("[5] Tracking completed.")
    print(f"    YOLO original output folder: {latest_dir}")
    print(f"    Copied final results to: {final_result_dir}")

    return final_result_dir