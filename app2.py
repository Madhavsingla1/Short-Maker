from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import yt_dlp
import whisper
import librosa
import numpy as np
import re
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import AudioFileClip
import hashlib  # For generating unique file names

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'shorts'
app.config['VIDEO_CACHE'] = 'cached_videos'

# Ensure the output and cache folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEO_CACHE'], exist_ok=True)

model = whisper.load_model("base")  # or "small", "medium", "large"

def generate_video_filename(url):
    """Generate a unique filename based on the video URL."""
    url_hash = hashlib.md5(url.encode()).hexdigest()
    return f"{url_hash}.mp4"

def download_youtube_video(url, output_dir='cached_videos'):
    video_filename = generate_video_filename(url)
    video_path = os.path.join(output_dir, video_filename)

    # Download only if the video doesn't already exist
    if not os.path.exists(video_path):
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
            'outtmpl': video_path,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Downloaded new video: {video_path}")
    else:
        print(f"Using cached video: {video_path}")

    return video_path

def transcribe_video(video_path):
    audio = whisper.load_audio(video_path)
    audio = whisper.pad_or_trim(audio)
    result = model.transcribe(audio)
    return result["text"]

def find_reaction_segments(audio_path, transcript, threshold=0.7):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        energy = librosa.feature.rms(y=y)[0]
        normalized_energy = energy / np.max(energy)
        reaction_times = np.where(normalized_energy > threshold)[0]

        times = librosa.frames_to_time(reaction_times, sr=sr, hop_length=512)
        interest_segments = []
        for match in re.finditer(r"(important|exciting|now|key point|watch|listen)", transcript, re.IGNORECASE):
            start_time = match.start() // sr
            end_time = min(start_time + 60, len(y) // sr)
            interest_segments.append((start_time, end_time))

        combined_segments = set()
        combined_segments.update(interest_segments)
        combined_segments.update([(times[i], times[i+1]) for i in range(len(times) - 1) if times[i+1] - times[i] > 1])
        final_segments = [(start, end) for start, end in combined_segments if 15 <= (end - start) <= 60]

        return final_segments
    except Exception as e:
        print(f"Error processing audio and transcript: {e}")
        return []

def create_shorts(video_path, output_dir='shorts'):
    transcript = transcribe_video(video_path)
    reaction_segments = find_reaction_segments(video_path, transcript)

    if not reaction_segments:
        print("No suitable reaction segments found.")
        return []

    clip = VideoFileClip(video_path)
    shorts = []
    for i, (start_time, end_time) in enumerate(reaction_segments):
        short_clip = clip.subclip(start_time, end_time)
        short_clip = short_clip.resize(height=1080).crop(x_center=short_clip.w // 2, y_center=short_clip.h // 2, width=608, height=1080)

        if not short_clip.audio:
            print(f"Warning: Skipping segment {i+1} due to empty audio.")
            continue

        output_path = os.path.join(output_dir, f'short_{i+1}.mp4')
        short_clip.write_videofile(output_path, codec='libx264')
        shorts.append(f'short_{i+1}.mp4')

    clip.close()
    return shorts

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        video_path = download_youtube_video(url)
        audio_path = 'audio.wav'
        os.system(f'ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y')

        shorts = create_shorts(video_path, app.config['UPLOAD_FOLDER'])
        return render_template('index.html', shorts=shorts)

    return render_template('index.html', shorts=None)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
