import os
from flask import Flask, render_template, request, send_file, redirect, url_for, after_this_request
from werkzeug.utils import secure_filename
import shutil

# Import your captioning functions here
from main import (
    transcribe_video_with_timestamps,
    group_words_into_batches,
    overlay_precise_text_on_video,
    add_audio_to_video
)

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi', 'mkv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return "No file part", 400

    file = request.files['video']
    if file.filename == '':
        return "No selected file", 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)

        # 🔹 Get batch size from form input
        batch_size = request.form.get('batch_size', type=int)
        if not batch_size or batch_size < 1:
            return "Invalid batch size", 400

        # Paths for intermediate and output files
        silent_path = os.path.join(app.config['OUTPUT_FOLDER'], f"silent_{filename}")
        final_path = os.path.join(app.config['OUTPUT_FOLDER'], f"captioned_{filename}")

        # Run your pipeline
        words = transcribe_video_with_timestamps(input_path)
        batches = group_words_into_batches(words, batch_size=batch_size)
        overlay_precise_text_on_video(input_path, silent_path, batches)
        add_audio_to_video(input_path, silent_path, final_path)

        # Serve final video to user
        return redirect(url_for('download_file', filename=f"captioned_{filename}"))

    return "Invalid file format", 400

@app.route('/download/<filename>')
def download_file(filename):
    final_path = os.path.join(app.config['OUTPUT_FOLDER'], filename)
    original_filename = filename.replace("captioned_", "")
    original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
    silent_path = os.path.join(app.config['OUTPUT_FOLDER'], f"silent_{original_filename}")

    if os.path.exists(final_path):
        @after_this_request
        def remove_files(response):
            try:
                shutil.rmtree("./uploads")
                shutil.rmtree("./outputs")
                print("✅ Deleted all related files.")
            except Exception as e:
                print(f"❌ Error deleting files: {e}")
            return response

        return send_file(final_path, as_attachment=True)

    return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)
