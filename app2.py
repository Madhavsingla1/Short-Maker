from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import yt_dlp
import whisper
import librosa
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import AudioFileClip, VideoFileClip
import re
import heapq


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'shorts'

# Ensure the output folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

model = whisper.load_model("base")  # or "small", "medium", "large"

def transcribe_video(video_path):
    audio = whisper.load_audio(video_path)
    audio = whisper.pad_or_trim(audio)
    result = model.transcribe(audio)
    return result["text"]

def download_youtube_video(url, output_path='downloaded_video.mp4'):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path


def find_all_possible_segments(audio_path, transcript, energy_threshold=0.6, min_duration=10, max_duration=60):
    # Step 1: Load audio and calculate energy
    y, sr = librosa.load(audio_path, sr=None)
    energy = librosa.feature.rms(y=y)[0]
    normalized_energy = energy / np.max(energy)
    
    # Step 2: Find all energy-based segments
    reaction_times = np.where(normalized_energy > energy_threshold)[0]
    times = librosa.frames_to_time(reaction_times, sr=sr, hop_length=512)
    
    segments = []
    start = times[0]
    for i in range(1, len(times)):
        if times[i] - times[i-1] > 1:
            segments.append((start, times[i-1]))
            start = times[i]
    segments.append((start, times[-1]))
    
    # Debug: Print all identified segments
    print("All possible segments based on energy threshold:", segments)
    
    # Step 3: Filter by duration constraints (min_duration, max_duration)
    valid_segments = [(start, end) for start, end in segments if min_duration <= (end - start) <= max_duration]
    
    # Debug: Print valid segments after filtering by duration
    print("Valid segments after duration filtering:", valid_segments)

    # Step 4: Analyze transcript to find keywords indicating interest (e.g., "important", "exciting")
    interest_segments = []
    for match in re.finditer(r"(important|exciting|key point|watch|listen)", transcript, re.IGNORECASE):
        start_time = match.start() // sr
        end_time = min(start_time + 60, len(y) // sr)
        interest_segments.append((start_time, end_time))
    
    # Debug: Print interest-based segments
    print("Interest segments based on transcript keywords:", interest_segments)
    
    # Step 5: Combine energy-based and interest-based segments
    combined_segments = valid_segments + interest_segments

    return combined_segments

def score_segment(segment, video_path, transcript):
    start, end = segment
    # Open the video clip and extract the relevant subclip
    clip = VideoFileClip(video_path).subclip(start, end)
    
    # Example scoring: You could use segment length and transcript keyword count as part of the score
    length_score = min(clip.duration / 60, 1)  # Normalize length to 1 for 1 minute clips
    keyword_score = len(re.findall(r"(important|exciting|key point|watch|listen)", transcript, re.IGNORECASE))  # Simple keyword count
    
    # Combine these scores (you can use a weighted sum or more complex model)
    score = length_score + keyword_score
    
    # Debug: Print the score for each segment
    print(f"Scoring segment {start}-{end} with score {score}")
    
    return score

def create_best_shorts(video_path, audio_path, transcript, num_best=10):
    # Step 1: Find all possible segments
    segments = find_all_possible_segments(audio_path, transcript)
    
    # Step 2: Score each segment
    scored_segments = []
    for segment in segments:
        score = score_segment(segment, video_path, transcript)
        scored_segments.append((segment, score))
    
    # Debug: Print all scored segments
    print("Scored segments:", scored_segments)
    
    # Step 3: Sort segments by score, and pick the top N best
    scored_segments.sort(key=lambda x: x[1], reverse=True)
    top_segments = scored_segments[:num_best]
    
    # Debug: Print top N segments
    print("Top segments based on score:", top_segments)
    
    # Step 4: Create the shorts based on the top segments
    clip = VideoFileClip(video_path)
    shorts = []
    for i, (segment, score) in enumerate(top_segments):
        start_time, end_time = segment
        short_clip = clip.subclip(start_time, end_time)
        short_clip = short_clip.resize(height=1080).crop(x_center=short_clip.w // 2, y_center=short_clip.h // 2, width=608, height=1080)
        output_path = f'shorts/short_{i+1}.mp4'
        short_clip.write_videofile(output_path, codec='libx264')
        shorts.append(f'short_{i+1}.mp4')
    
    clip.close()
    return shorts

# Example usage in Flask route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        url = request.form['url']
        video_path = download_youtube_video(url)
        audio_path = 'audio.wav'
        os.system(f'ffmpeg -i {video_path} -q:a 0 -map a {audio_path} -y')
        
        # Transcribe video to get the transcript
        transcript = transcribe_video(video_path)
        
        shorts = create_best_shorts(video_path, audio_path, transcript)
        
        if shorts:
            return render_template('index.html', shorts=shorts)
        else:
            return render_template('index.html', message="No suitable shorts found.")
    return render_template('index.html', shorts=None)
@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
