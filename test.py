import cv2

# Load the video
video_path = 'test_video.mp4'
cap = cv2.VideoCapture(video_path)

# Get the video frame properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object to save the output
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Define the font and text properties
font = cv2.FONT_HERSHEY_SIMPLEX
text = "Hello, OpenCV!"
font_scale = 1
font_color = (0, 255, 0)  # Green color in BGR
thickness = 2
line_type = cv2.LINE_AA

while True:
    # Read each frame from the video
    ret, frame = cap.read()
    
    if not ret:
        break  # End of video
    
    # Get the text size to center it
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = (frame.shape[1] - text_size[0]) // 2
    text_y = (frame.shape[0] + text_size[1]) // 2

    # Put the text on the frame
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, thickness, line_type)

    # Write the frame to the output video
    out.write(frame)

    # Display the frame with the text
    cv2.imshow('Frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
