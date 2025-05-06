import cv2
import os
import numpy as np


def pad_to_square_with_extra_width(image, extra_ratio=1):
    """Pads image horizontally to make width = extra_ratio × height, if needed."""
    height, width, _ = image.shape
    target_width = int(height * extra_ratio)
    if target_width <= width:
        return image  # No padding needed
    pad_total = target_width - width
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    padded = cv2.copyMakeBorder(
        image, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded


def extract_frames(video_path, num_frames=6, extra_width_ratio=1.1):
    # Derive output directory from video file name
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.path.dirname(
        video_path), f"{base_name}_frames")
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")

    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    print(f"Total frames in video: {total_frames}")
    print(f"Extracting frames at: {frame_indices}")

    count = 0
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if i in frame_indices:
            padded_frame = pad_to_square_with_extra_width(
                frame, extra_width_ratio)
            frame_filename = os.path.join(output_dir, f"frame_{count + 1}.jpg")
            cv2.imwrite(frame_filename, padded_frame)
            print(f"Saved: {frame_filename}")
            count += 1

    cap.release()
    print(f"Frames saved in: {output_dir}")


# Example usage
video_path = 'banana_success.mp4'
extract_frames(video_path)
