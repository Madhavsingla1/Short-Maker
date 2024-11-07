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
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import yt_dlp
import whisper
import librosa
import numpy as np
import re
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.editor import AudioFileClip, VideoFileClip
import whisper




app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'shorts'



# Ensure the output folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

model = whisper.load_model("base")  # or "small", "medium", "large"

def transcribe_video(video_path):
    # Load the audio file
    audio = whisper.load_audio(video_path)
    audio = whisper.pad_or_trim(audio)

    # Transcribe the audio
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



def find_reaction_segments(audio_path, transcript, threshold=0.7):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) == 0:
            print("Error: Audio file is empty or not loaded correctly.")
            return []

        energy = librosa.feature.rms(y=y)[0]
        normalized_energy = energy / np.max(energy)

        # Find segments above the threshold for energy
        reaction_times = np.where(normalized_energy > threshold)[0]
        if len(reaction_times) == 0:
            print("No reactions detected above the threshold.")
            return []

        times = librosa.frames_to_time(reaction_times, sr=sr, hop_length=512)

        # Analyze transcript for keywords indicating viewer interest
        interest_segments = []
        for match in re.finditer(r"(important|exciting|now|key point|watch|listen)", transcript, re.IGNORECASE):
            start_time = match.start() // sr
            end_time = min(start_time + 60, len(y) // sr)  # Cap at 60 seconds
            interest_segments.append((start_time, end_time))

        # Combine high-energy and transcript-based segments
        combined_segments = set()
        combined_segments.update(interest_segments)
        combined_segments.update([(times[i], times[i+1]) for i in range(len(times) - 1) if times[i+1] - times[i] > 1])

        # Filter segments between 15 and 60 seconds
        final_segments = [(start, end) for start, end in combined_segments if 15 <= (end - start) <= 60]

        if not final_segments:
            print("No valid segments found based on audio and transcript analysis.")
            return []

        return final_segments
    except Exception as e:
        print(f"Error processing audio and transcript: {e}")
        return []



def create_shorts(video_path, output_dir='shorts'):
    try:
        # Extract transcript
        transcript = transcribe_video(video_path)
        print("Transcript: ", transcript)
        # Extract reaction segments based on both audio and transcript
        reaction_segments = find_reaction_segments(video_path, transcript)

        if not reaction_segments:
            print("No suitable reaction segments found.")
            return []

        clip = VideoFileClip(video_path)
        shorts = []
        for i, (start_time, end_time) in enumerate(reaction_segments):
            segment_length = end_time - start_time
            if segment_length < 15:
                print(f"Segment {i+1} is too short: {segment_length} seconds. Skipping.")
                continue
            elif segment_length > 60:
                end_time = start_time + 60  # Cap the length to 60 seconds

            short_clip = clip.subclip(start_time, end_time)

            # Resize to 9:16 aspect ratio for YouTube shorts
            short_clip = short_clip.resize(height=1080).crop(x_center=short_clip.w // 2, y_center=short_clip.h // 2, width=608, height=1080)

            if not short_clip.audio or AudioFileClip(video_path).duration == 0:
                print(f"Warning: Skipping segment {i+1} due to empty audio.")
                continue

            output_path = os.path.join(output_dir, f'short_{i+1}.mp4')
            short_clip.write_videofile(output_path, codec='libx264')
            shorts.append(f'short_{i+1}.mp4')

            if i >= 2:  # Limit to 3 shorts
                break

        clip.close()
        return shorts
    except Exception as e:
        print(f"Error while creating shorts: {e}")
        return []
        # return []


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
