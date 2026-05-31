# Low-Light Pseudo-Burst Object Tracking

This repository contains an experimental end-to-end inference pipeline for illumination enhancement and object tracking in low-light videos.

The system takes a low-light input video and produces a YOLOv8s + ByteTrack tracking output video. The pipeline includes frame extraction, center-preserving adaptive pseudo-burst fusion, Retinexformer-based illumination enhancement, video reconstruction, and object tracking.

This project was developed as part of a graduation thesis study on low-light video object tracking.

---

## Pipeline Overview

The proposed pipeline follows these stages:

```text
Input low-light video
        ↓
Frame extraction
        ↓
Center-preserving adaptive pseudo-burst fusion
        ↓
Retinexformer illumination enhancement
        ↓
Video reconstruction
        ↓
YOLOv8s + ByteTrack tracking
        ↓
Output tracking video
```

The main purpose of the pseudo-burst module is to test whether consecutive RGB video frames can be used as a lightweight burst-like temporal representation before enhancement and tracking. Instead of directly averaging neighboring frames, the proposed fusion module preserves the center frame and weights adjacent frames according to brightness, noise, and similarity.

---

## Main Contribution

The main contribution of this repository is not a newly trained monolithic neural network. Instead, it provides a modular experimental inference pipeline for analyzing low-light object tracking.

The project includes:

- A center-preserving adaptive pseudo-burst fusion module
- Retinexformer integration for low-light enhancement
- YOLOv8s + ByteTrack integration for object tracking
- Detection, tracking, and image-quality evaluation scripts
- An end-to-end pipeline script that produces a final tracking video from an input low-light video

---

## Project Structure

```text
.
├── main_pipeline.py
├── pipeline_modules.py
├── adaptive_pseudo_burst_fusion.py
├── extract_frames.py
├── frames_to_video.py
├── compare_yolo_labels.py
├── compare_tracking_labels.py
├── evaluate_image_metrics.py
├── Enhancement/
├── Options/
├── basicsr/
├── README.md
└── .gitignore
```

### Main Files

- `main_pipeline.py`  
  Runs the complete end-to-end inference pipeline.

- `pipeline_modules.py`  
  Contains reusable functions for frame extraction, adaptive pseudo-burst fusion, Retinexformer execution, video reconstruction, and YOLO tracking.

- `adaptive_pseudo_burst_fusion.py`  
  Standalone version of the adaptive pseudo-burst fusion module used during development.

- `extract_frames.py`  
  Extracts frames from the input video.

- `frames_to_video.py`  
  Converts processed frames back into video format.

- `compare_yolo_labels.py`  
  Computes YOLO detection metrics from saved label files.

- `compare_tracking_labels.py`  
  Computes tracking metrics from YOLOv8s + ByteTrack label outputs.

- `evaluate_image_metrics.py`  
  Computes brightness, contrast, Laplacian variance, temporal difference, and pixel-level difference metrics.

---

## Adaptive Pseudo-Burst Fusion

For each target frame `F_t`, the pipeline uses a short temporal window:

```text
F(t-1), F(t), F(t+1)
```

The fused frame is computed using adaptive weights:

```text
w_i = brightness_i × noise_i × similarity_i × center_boost_i
```

The weights are normalized before fusion:

```text
w'_i = w_i / sum(w_i)
```

Then the pseudo-burst frame is obtained as:

```text
F'_t = sum(w'_i × F_i)
```

The center frame receives a higher weight in order to reduce motion blur and preserve the target frame structure.

---

## Requirements

The project was tested with Python 3.13 on Windows.

Install the required packages:

```bash
pip install opencv-python numpy natsort ultralytics torch torchvision torchaudio
pip install einops timm scipy scikit-image tqdm matplotlib
```

Depending on the Retinexformer environment, additional dependencies from the original Retinexformer implementation may be required.

---

## Pretrained Weights

Pretrained model weights are not included in this repository due to file size limitations.

Place the Retinexformer SMID checkpoint under:

```text
pretrained_weights/SMID.pth
```

YOLOv8s weights can be downloaded automatically by Ultralytics, or they can be placed manually as:

```text
yolov8s.pt
```

---

## Usage

Run the end-to-end pipeline:

```bash
set PYTHONPATH=%cd%
python main_pipeline.py --video videos/input_video.mp4 --output final_pipeline_output --retinex_weights pretrained_weights/SMID.pth --retinex_opt Options/RetinexFormer_SMID.yml --yolo_model yolov8s.pt --conf 0.10 --cpu
```

The `--cpu` flag runs Retinexformer on CPU. CPU inference is slow, so GPU acceleration is recommended for practical use.

---

## Output

The main output is the final YOLOv8s + ByteTrack tracking video.

The pipeline also saves intermediate outputs for analysis and reproducibility:

```text
final_pipeline_output/
├── 01_original_frames/
├── 02_adaptive_burst_frames/
├── 03_retinexformer_enhanced_frames/
├── 04_enhanced_video/
└── 05_tracking_results/
```

The final tracking result is saved under:

```text
final_pipeline_output/05_tracking_results/
```

Depending on the local Ultralytics configuration, YOLO may also create temporary output folders under the default `runs/track/` directory.

---

## Evaluation Scripts

### YOLO Detection Comparison

```bash
python compare_yolo_labels.py
```

This script calculates:

- person detected frame count
- total person detections
- average person confidence
- maximum person confidence
- vehicle detection count

### Tracking Comparison

```bash
python compare_tracking_labels.py
```

This script calculates:

- tracked person frames
- total person detections
- unique track IDs
- average track length
- maximum track length
- longest continuous track
- average confidence

### Image Quality and Temporal Stability Metrics

```bash
python evaluate_image_metrics.py
```

This script calculates:

- mean brightness
- mean contrast
- Laplacian variance
- temporal difference
- mean absolute difference between original and adaptive burst frames

---

## Experimental Notes

The pipeline was evaluated on a 303-frame low-light video segment recorded from a fixed surveillance-like viewpoint. The input video was captured using a Samsung Galaxy A16 smartphone in MP4 format with H.264 encoding, 720 × 1280 resolution, and 30 FPS.

The study compares four configurations:

```text
Original
Adaptive Burst
Original + Retinexformer
Adaptive Burst + Retinexformer
```

The experiments showed that adaptive pseudo-burst fusion improved temporal stability, but this improvement did not automatically translate into better YOLO detection or tracking performance. Retinexformer increased visual brightness, but in the tested video segment it reduced YOLO person detection and tracking stability.

This result suggests that low-light video object tracking should be evaluated using downstream detection and tracking metrics, not only visual enhancement quality.

---

## Limitations

The current implementation is intended for experimental offline analysis.

Main limitations:

- Retinexformer inference is slow on CPU.
- The pseudo-burst module does not use explicit motion compensation.
- YOLOv8s was not fine-tuned on enhanced low-light frames.
- The pipeline was evaluated on a limited low-light video segment.
- Real-time deployment would require GPU acceleration and optimization.

Future improvements may include:

- optical-flow-based motion compensation
- detection-aware enhancement
- YOLO fine-tuning on low-light/enhanced data
- ROI-based enhancement
- real-time GPU optimization

---

## Acknowledgment

This project uses Retinexformer as the low-light enhancement backbone. The original Retinexformer implementation and pretrained checkpoints belong to their respective authors.

This repository adds an adaptive pseudo-burst video preprocessing module, YOLOv8s + ByteTrack tracking integration, end-to-end pipeline execution, and evaluation scripts for low-light video object tracking.

---

## License

This repository includes components from the original Retinexformer implementation, which is distributed under the MIT License. Please refer to the included license file and the original Retinexformer project for license details.

The additional code developed for this thesis includes the adaptive pseudo-burst fusion module, YOLOv8s + ByteTrack integration, end-to-end pipeline, and evaluation scripts.
