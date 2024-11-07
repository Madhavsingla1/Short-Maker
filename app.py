from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import yt_dlp
import whisper
import librosa
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'shorts'

# Ensure the output folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def download_youtube_video(url, output_path='downloaded_video.mp4'):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def transcribe_video(video_path):
    model = whisper.load_model("base")
    result = model.transcribe(video_path)
    return result['segments']

def find_reaction_segments(audio_path, threshold=0.7):
    y, sr = librosa.load(audio_path, sr=None)
    energy = librosa.feature.rms(y=y)[0]
    normalized_energy = energy / np.max(energy)
    reaction_times = np.where(normalized_energy > threshold)[0]
    times = librosa.frames_to_time(reaction_times, sr=sr, hop_length=512)
    
    segments = []
    start = times[0]
    for i in range(1, len(times)):
        if times[i] - times[i-1] > 1:
            segments.append((start, times[i-1]))
            start = times[i]
    segments.append((start, times[-1]))
    return [(start, end) for start, end in segments if end - start <= 60]

def create_shorts(video_path, reaction_segments, output_dir='shorts'):
    clip = VideoFileClip(video_path)
    shorts = []
    for i, (start_time, end_time) in enumerate(reaction_segments):
        short_clip = clip.subclip(start_time, end_time)
        output_path = os.path.join(output_dir, f'short_{i+1}.mp4')
        short_clip.write_videofile(output_path, codec='libx264')
        shorts.append(f'short_{i+1}.mp4')
        if i >= 2:  # Limit to 3 shorts
            break
    clip.close()
    return shorts

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        video_path = download_youtube_video(url)
        audio_path = 'audio.wav'
        os.system(f'ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y')
        
        reaction_segments = find_reaction_segments(audio_path)
        shorts = create_shorts(video_path, reaction_segments, app.config['UPLOAD_FOLDER'])
        
        return render_template('index.html', shorts=shorts)
    return render_template('index.html', shorts=None)

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
