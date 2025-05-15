import cv2
import whisper
import os
from moviepy import VideoFileClip
import numpy as np

# --- Step 1: Transcribe Video using Whisper ---
def transcribe_video(video_path, model_name="base"):
    audio_path = "temp_audio.wav"
    try:
        print("Extracting audio...")
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()

        print("Transcribing...")
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path)
        return result["text"]

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

# --- Step 2: Convert text into batches of N words ---
def get_word_batches(text, batch_size=5):
    words = text.strip().split()
    return [' '.join(words[i:i + batch_size]) for i in range(0, len(words), batch_size)]

# --- Step 3: Overlay text onto video and save ---
def overlay_text_on_video(video_path, output_path, word_batches, batch_duration=1):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 255, 0)
    thickness = 2
    line_type = cv2.LINE_AA

    current_batch_index = 0
    frames_per_batch = int(fps * batch_duration)

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Change text every `frames_per_batch` frames
        if frame_idx // frames_per_batch < len(word_batches):
            current_batch_index = frame_idx // frames_per_batch
        else:
            current_batch_index = len(word_batches) - 1

        text = word_batches[current_batch_index]

        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = (frame.shape[0] + text_size[1]) // 2

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness, line_type)
        out.write(frame)

    cap.release()
    out.release()
    print(f"Video saved to {output_path}")

# --- Step 4: Run Everything ---
if __name__ == "__main__":
    video_file = "test_video.mp4"  # <--- Your video here
    output_file = "output_video.mp4"
    batch_size = 4  # Number of words per batch
    batch_display_time = 1  # In seconds

    if not os.path.exists(video_file):
        print(f"Video file not found: {video_file}")
    else:
        # Transcribe and split text
        full_text = transcribe_video(video_file)
        word_batches = get_word_batches(full_text, batch_size=batch_size)

        # Overlay onto video
        overlay_text_on_video(video_file, output_file, word_batches, batch_duration=batch_display_time)
