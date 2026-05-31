import os
import argparse

from pipeline_modules import (
    extract_frames,
    adaptive_pseudo_burst_fusion,
    run_retinexformer,
    frames_to_video,
    run_yolo_tracking,
)


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pseudo-burst low-light object tracking pipeline"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Input low-light video path"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="final_pipeline_output",
        help="Output folder"
    )

    parser.add_argument(
        "--retinex_opt",
        type=str,
        default="Options/RetinexFormer_SMID.yml",
        help="Retinexformer option file"
    )

    parser.add_argument(
        "--retinex_weights",
        type=str,
        default="pretrained_weights/SMID.pth",
        help="Retinexformer pretrained weights"
    )

    parser.add_argument(
        "--yolo_model",
        type=str,
        default="yolov8s.pt",
        help="YOLO model path"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.10,
        help="YOLO confidence threshold"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Output video FPS"
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run Retinexformer on CPU"
    )

    args = parser.parse_args()

    output_root = args.output

    original_frames = os.path.join(output_root, "01_original_frames")
    adaptive_burst_frames = os.path.join(output_root, "02_adaptive_burst_frames")
    enhanced_frames = os.path.join(output_root, "03_retinexformer_enhanced_frames")

    enhanced_video = os.path.join(
        output_root,
        "04_enhanced_video",
        "adaptive_burst_enhanced_video.mp4"
    )

    tracking_project = os.path.join(output_root, "05_tracking_results")
    tracking_name = "adaptive_burst_retinexformer_yolov8s_bytetrack"

    print("\n===================================================")
    print("ADAPTIVE PSEUDO-BURST LOW-LIGHT OBJECT TRACKING PIPELINE")
    print("===================================================\n")

    print(f"Input video: {args.video}")
    print(f"Output folder: {output_root}")
    print(f"Retinexformer weights: {args.retinex_weights}")
    print(f"YOLO model: {args.yolo_model}")
    print(f"Confidence threshold: {args.conf}\n")

    # 1. Video -> original frames
    extract_frames(
        video_path=args.video,
        output_dir=original_frames,
        frame_step=1
    )

    # 2. Original frames -> adaptive pseudo-burst frames
    adaptive_pseudo_burst_fusion(
        input_dir=original_frames,
        output_dir=adaptive_burst_frames
    )

    # 3. Adaptive pseudo-burst frames -> Retinexformer enhanced frames
    run_retinexformer(
        input_dir=adaptive_burst_frames,
        output_dir=enhanced_frames,
        opt_path=args.retinex_opt,
        weights_path=args.retinex_weights,
        use_cpu=args.cpu
    )

    # 4. Enhanced frames -> enhanced video
    frames_to_video(
        input_dir=enhanced_frames,
        output_video=enhanced_video,
        fps=args.fps
    )

    # 5. Enhanced video -> YOLOv8s + ByteTrack tracking output
    tracking_result_dir = run_yolo_tracking(
        video_path=enhanced_video,
        yolo_model=args.yolo_model,
        output_project=tracking_project,
        output_name=tracking_name,
        conf=args.conf
    )

    print("\n===================================================")
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("Final tracking result folder:")
    print(tracking_result_dir)
    print("===================================================\n")


if __name__ == "__main__":
    main()