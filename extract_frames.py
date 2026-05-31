import cv2
import os

video_path = "videos/otopark_yakin.mp4"
output_dir = "video_frames_yakin/original"

os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Video could not be opened: {video_path}")

frame_id = 0
saved_id = 0

# Save one frame every 1 frames
frame_step = 1

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

print(f"Total frames read: {frame_id}")
print(f"Saved frames: {saved_id}")
print(f"Output folder: {output_dir}")