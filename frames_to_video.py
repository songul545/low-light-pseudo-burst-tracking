# -*- coding: utf-8 -*-

import cv2
import os
from glob import glob
from natsort import natsorted

def frames_to_video(input_dir, output_path, fps=30):
    frame_paths = natsorted(glob(os.path.join(input_dir, "*.png")))

    if len(frame_paths) == 0:
        raise RuntimeError(f"No frames found in {input_dir}")

    first = cv2.imread(frame_paths[0])
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    for p in frame_paths:
        frame = cv2.imread(p)
        out.write(frame)

    out.release()
    print(f"Saved video: {output_path}")
    print(f"Frames: {len(frame_paths)}")

os.makedirs("tracking_videos", exist_ok=True)

frames_to_video("video_frames_yakin/original", "tracking_videos/original.mp4", fps=30)
frames_to_video("video_frames_yakin/adaptive_burst", "tracking_videos/adaptive_burst.mp4", fps=30)
frames_to_video("video_frames_yakin/original_enhanced", "tracking_videos/original_enhanced.mp4", fps=30)
frames_to_video("video_frames_yakin/adaptive_enhanced", "tracking_videos/adaptive_enhanced.mp4", fps=30)