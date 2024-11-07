# YouTube Shorts Maker

## Overview

YouTube Shorts Maker is a web application designed to create short videos from full-length YouTube videos. This application identifies the most engaging moments based on audio analysis and transcriptions and extracts them into short video clips formatted for YouTube Shorts.

## Features

- Extracts key moments from a YouTube video based on reactions and viewer interest.
- Transcribes video audio using OpenAI's Whisper model.
- Clips videos to create engaging short-format content (up to 60 seconds).
- Outputs the videos in an MP4 format with the correct aspect ratio (9:16) for YouTube Shorts.

## Tech Stack

- **Backend**: Python, Flask
- **Audio Processing**: Whisper by OpenAI, moviepy, and yt-dlp
- **Frontend**: HTML, CSS
- **Dependencies**: ffmpeg for video processing

## Installation and Setup

### Prerequisites

- Python 3.x installed
- `ffmpeg` installed and added to the system path
- Ensure you have `pip` installed

### Step-by-step Setup

1. **Clone the repository**:
   ```
   bash
   git clone https://github.com/yourusername/youtube-shorts-maker.git
   cd youtube-shorts-maker
   Install the required dependencies:
   ```
2. **Install the required dependencies**:

   ```
   pip install -r requirements.txt
   ```

3. **Install ffmpeg**:

   - On Ubuntu/Debian:
     ```
     sudo apt update
     sudo apt install ffmpeg
     ```
   - On macOS:

     ```
     brew install ffmpeg
     ```

   - On Windows: Download ffmpeg from ffmpeg.org and add it to the system path.
     Run the Flask app:

4. **Run the Flask app**:

   ```
   python app.py
   ```

5. **Access the application**:
   Open a web browser and navigate to http://127.0.0.1:5000.:

# YouTube Shorts Maker

## How It Works

### Main Workflow

1. **Download Video**: The URL input is processed using `yt-dlp` to download the video.
2. **Transcription**: The audio from the video is transcribed using the Whisper model to generate a text transcript.
3. **Reaction Analysis**: Key segments are identified based on the transcript and audio cues to determine engaging moments.
4. **Shorts Creation**: The identified segments are clipped using `moviepy` and formatted for YouTube Shorts (9:16 aspect ratio).
5. **Output**: The created short videos are saved in an `output` folder as `.mp4` files.

## Code Structure

```bash
youtube-shorts-maker/
│
├── app.py               # Main Flask app file
├── static/              # Folder for static files (CSS, JS)
├── templates/           # HTML templates for the web interface
├── utils.py             # Utility functions for audio processing
├── requirements.txt     # List of Python dependencies
└── README.md            # This file
```

# YouTube Shorts Maker

## Key Functions

- `transcribe_video(video_path)`: Loads and transcribes the video audio.
- `find_reaction_segments(audio_path, transcript)`: Analyzes audio and transcript to find the most engaging segments.
- `create_shorts(video_path, segments, output_folder)`: Creates and saves the short videos.

## Common Issues

- **AttributeError: module 'whisper' has no attribute 'load_model'**: Ensure that you have installed the correct `openai-whisper` library.
- **IndexError during video processing**: Check that the audio file is correctly processed and not empty.
- **File path errors**: Ensure paths are correct and have the necessary read/write permissions.

## Future Enhancements

- Add advanced analytics for better segment detection.
- Implement a user-friendly drag-and-drop interface.
- Provide options for custom video duration and additional editing tools.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Contributors

- **Madhav Singla**
