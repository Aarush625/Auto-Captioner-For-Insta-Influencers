import whisper
from moviepy import VideoFileClip
import os

# --- Your function to read file in batches ---
def read_file_in_batches(file_path, batch_size=4):
    try:
        with open(file_path, 'r', encoding='utf-8') as file: # Added encoding for wider compatibility
            # Read the entire content of the file
            text = file.read()
            
            # Split the content into words
            words = text.split()
            
            if not words:
                print(f"The file '{file_path}' is empty or contains only whitespace.")
                return

            print(f"\n--- Reading '{file_path}' in batches of {batch_size} words ---")
            # Loop over words in batches of 'batch_size'
            for i in range(0, len(words), batch_size):
                # Slice the list to get a batch of words
                batch = words[i:i + batch_size]
                print(' '.join(batch))  # Print the batch of words as a line
            print("--- End of batched reading ---")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found for batched reading.")
    except Exception as e:
        print(f"An error occurred while reading file in batches: {e}")

# --- Main transcription function (from previous example) ---
def transcribe_video(video_path, model_name="base"):

    audio_path = "temp_audio.wav"  # Define audio_path here to ensure it's in scope for finally
    try:
        # 1. Extract Audio from Video
        print(f"Extracting audio from {video_path}...")
        video_clip = VideoFileClip(video_path)
        video_clip.audio.write_audiofile(audio_path)
        video_clip.close()
        print(f"Audio extracted successfully to {audio_path}")

        # 2. Transcribe Audio using Whisper
        print(f"Loading Whisper model: {model_name}...")
        model = whisper.load_model(model_name)
        print("Model loaded. Starting transcription...")

        result = model.transcribe(audio_path)
        transcribed_text = result["text"]
        print("Transcription complete.")

        return transcribed_text

    except Exception as e:
        print(f"An error occurred during transcription: {e}")
        return None
    finally:
        # Clean up the temporary audio file
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                print(f"Temporary audio file {audio_path} deleted.")
            except Exception as e:
                print(f"Error deleting temporary audio file {audio_path}: {e}")


if __name__ == "__main__":
    # --- Configuration ---
    video_file_path = "test_video.mp4"  # <--- CHANGE THIS TO YOUR VIDEO FILE PATH
    selected_model = "base"
    batch_read_size = 5 # You can change the batch size for reading the transcription
    # --- End Configuration ---

    if not os.path.exists(video_file_path):
        print(f"Error: Video file not found at {video_file_path}")
        print("Please update the 'video_file_path' variable in the script.")
    else:
        transcription = transcribe_video(video_file_path, selected_model)

        if transcription:
            print("\n--- Full Transcription ---")
            print(transcription)

            # Define the output text file name based on the video file name
            output_txt_file = os.path.splitext(video_file_path)[0] + "_transcription.txt"
            
            try:
                with open(output_txt_file, "w", encoding="utf-8") as f:
                    f.write(transcription)
                print(f"\nTranscription saved to: {output_txt_file}")

                # Now, read the saved transcription file in batches
                read_file_in_batches(output_txt_file, batch_size=batch_read_size)

            except Exception as e:
                print(f"Error saving transcription or reading in batches: {e}")
        else:
            print("Transcription failed. Cannot proceed to read in batches.")