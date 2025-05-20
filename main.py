import cv2
import whisper
import os
from moviepy import VideoFileClip
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import shutil


# --- Step 1: Transcribe Video using Whisper ---
def transcribe_video_with_timestamps(video_path, model_name="base"):
    audio_path = "temp_audio.wav"
    try:
        print("Extracting audio...")
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()

        print("Transcribing...")
        model = whisper.load_model(model_name)
        result = model.transcribe(audio_path, word_timestamps=True)
        
        words = []
        for segment in result["segments"]:
            words.extend(segment["words"])
        
        return words  # Each word has 'start', 'end', and 'word'

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


# --- Step 2: Convert text into batches of N words ---
def group_words_into_batches(words, batch_size=4):
    batches = []
    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        text = ' '.join([w['word'].strip() for w in batch])
        start_time = batch[0]['start']
        end_time = batch[-1]['end']
        batches.append({"text": text, "start": start_time, "end": end_time})
    return batches


# --- Step 3: Overlay text onto video and save silent video ---
def overlay_precise_text_on_video(video_path, silent_output_path, batches, font_path=None):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(silent_output_path, fourcc, fps, (width, height))

    base_font_size = 40  # Can be adjusted

    current_batch_idx = 0
    current_batch = batches[current_batch_idx] if batches else None

    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_idx / fps

        while current_batch and current_time > current_batch["end"]:
            current_batch_idx += 1
            if current_batch_idx >= len(batches):
                current_batch = None
                break
            current_batch = batches[current_batch_idx]

        if current_batch and current_batch["start"] <= current_time <= current_batch["end"]:
            text = current_batch["text"]
            duration = current_batch["end"] - current_batch["start"]
            elapsed = current_time - current_batch["start"]

            # --- Scaling Up Animation ---
            if elapsed < 0.2:
                scale = 0.8 + 0.2 * (elapsed / 0.2)
            else:
                scale = 1.0

            font_size = int(base_font_size * scale)

            # --- Use PIL to draw custom fonts with border ---
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)

            if font_path:
                font = ImageFont.truetype(font_path, font_size)
            else:
                font = ImageFont.truetype("arial.ttf", font_size)  # fallback

            bbox = font.getbbox(text)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            text_x = (width - text_width) // 2
            text_y = int(height * 0.85)

            # Draw border by drawing text multiple times around original position
            for dx in [-2, 0, 2]:
                for dy in [-2, 0, 2]:
                    if dx != 0 or dy != 0:
                        draw.text((text_x + dx, text_y + dy), text, font=font, fill=(0, 0, 0))

            # Draw main text
            draw.text((text_x, text_y), text, font=font, fill=(255, 255, 255))

            # Convert back to OpenCV
            frame = np.array(pil_img)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Silent video saved to {silent_output_path}")



# --- Step 4: Add original audio to silent video ---
def add_audio_to_video(original_video_path, silent_video_path, final_output_path):
    print("Merging audio with silent video...")
    original_clip = VideoFileClip(original_video_path)
    silent_clip = VideoFileClip(silent_video_path)

    # Use with_audio() instead of set_audio()
    final_clip = silent_clip.with_audio(original_clip.audio)
    final_clip.write_videofile(final_output_path, codec='libx264', audio_codec='aac')
    print(f"Final video with audio saved to {final_output_path}")
    original_clip.close()
    silent_clip.close()
    final_clip.close()

# --- Step 5: Run Everything ---
# if __name__ == "__main__":
#     video_file = "test_video.mp4"
#     silent_video_file = "silent_output.mp4"
#     final_output_file = "output_video_with_audio.mp4"
#     batch_size = 2

#     if not os.path.exists(video_file):
#         print(f"Video file not found: {video_file}")
#     else:
#         # Step 1: Transcribe with timestamps
#         words = transcribe_video_with_timestamps(video_file)
        
#         # Step 2: Group words into timed batches
#         batches = group_words_into_batches(words, batch_size=batch_size)
        
#         # Step 3: Overlay precise text on video
#         overlay_precise_text_on_video(video_file, silent_video_file, batches, font_path='')
        
#         # Step 4: Add original audio back
#         add_audio_to_video(video_file, silent_video_file, final_output_file)


