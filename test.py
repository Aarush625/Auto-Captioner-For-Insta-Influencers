import cv2
import whisper

# Load whisper model and transcribe with segments
model = whisper.load_model("base")
result = model.transcribe("temp_audio.wav")
segments = result["segments"]

cap = cv2.VideoCapture("test_video.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output_cv2.mp4", fourcc, fps, (width, height))

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = frame_idx / fps

    # Find segment text for this time
    current_text = ""
    for seg in segments:
        if seg['start'] <= current_time <= seg['end']:
            current_text = seg['text']
            break

    if current_text:
        # Draw text on frame (bottom center)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        thickness = 2
        text_size = cv2.getTextSize(current_text, font, font_scale, thickness)[0]
        text_x = (width - text_size[0]) // 2
        text_y = height - 30

        # Add a black rectangle background for readability
        cv2.rectangle(frame, (text_x-5, text_y - text_size[1] - 5), 
                      (text_x + text_size[0] + 5, text_y + 5), (0,0,0), -1)

        # Put white text on top
        cv2.putText(frame, current_text, (text_x, text_y), font, font_scale, (255,255,255), thickness)

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()
