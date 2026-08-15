# How to Use

Quick start guide for making simple videos for music, incorporating hard coded subtitles, a visualizer, and different elements to convert mp3 to mp4 ready for youtube

## PART 0: PREREQUISITES

Python 3.9+ must already be installed and available on your PATH.

### Linux/macOS:
```python3 --version
```

### Windows (Command Prompt):
```python --version
```

Confirm the printed version is 3.9 or higher. If not, download it from
https://www.python.org/downloads/ (Windows users: check "Add Python to PATH"
during install).

ffmpeg must also be installed and on your PATH:

### Linux (Fedora):
```sudo dnf install ffmpeg
```

Linux (Ubuntu):
```sudo apt install ffmpeg
```

### Windows:
```choco install ffmpeg   (or download from https://ffmpeg.org/download.html)
```

Check it worked:
```ffmpeg -version
```

## PART 1: SETUP

### 1. Create and activate the virtual environment

Open a terminal inside this project folder, then run the command for your OS.

#### a) Linux / macOS (bash/zsh):
```python3 -m venv venv && source venv/bin/activate
```

#### b) Windows (Command Prompt):
```python -m venv venv && venv\Scripts\activate.bat
```

You should see (venv) appear at the start of your terminal prompt.

### 2. GPU support (optional — speeds up transcription only)

This project uses faster-whisper, which runs on CTranslate2.
GPU acceleration is NVIDIA-only.

#### VIDIA GPUs:
Check you have a working NVIDIA driver:
```nvidia-smi
```
Then install the two GPU support packages:
```pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

#### AMD GPUs:
Not supported. CTranslate2 (the engine behind faster-whisper) has no ROCm/AMD
backend. Transcription will automatically run on CPU — it still works, just
slower. Video encoding will also fall back to CPU (libx264) since NVENC is
NVIDIA-only. No install step needed — just skip this section.

#### No GPU / CPU only:
Skip this step entirely. The script auto-detects and falls back to CPU.

### 3. Install dependencies
```pip install -r requirements.txt
```

### 4. (Optional) Verify GPU is detected
```python -c "import ctranslate2; print('CUDA devices found:', ctranslate2.get_cuda_device_count())"
```

A number greater than 0 means GPU acceleration is available. 0 means it'll run
on CPU — the script handles this automatically either way, no action required.

### 5. Add fonts (OPTIONAL)

Use the folder named "fonts" next to the script, and drop .ttf/.otf files into
it. Several fonts (filtered by active/markers) from Google Fonts have been pre-selected:
  Righteous, Bangers, Permanent Marker, Kalam, Protest, Riot, Lemon, Margarine

To get other fonts:

  a) Go to fonts.google.com, search the name or use the filters, click "Download family"
  b) Extract the downloaded .zip
  c) Copy the .ttf file(s) into the "fonts" folder next to this script
     (skip files like OFL.txt / README.txt — only .ttf/.otf files are needed)
     
### 6. Background types

The background can now be an image (jpg, png, webp, bmp, tiff, gif, svg) or
a short video (mp4, mkv, mov, webm, avi, m4v). If you choose a video, it
will simply loop for the entire length of the song — e.g. a 20-second
driving clip will repeat seamlessly behind a 4-minute song.

The center visualizer circle is always a static image — even if your
background is a video, pick a still image for the center (or just press
Enter to let it grab a clear frame from your background video automatically).

## PART 2: RUNNING

Re-activating later (skip if you just finished Part 1):
Once set up, you don't need to recreate the venv — just activate it each
session from inside the project folder:

### Linux/macOS:
```source venv/bin/activate
```

### Windows (Command Prompt):
```venv\Scripts\activate.bat
```

To exit the venv when done: type 'deactivate', or just close the terminal.

## PART 3: USING

a) Open a terminal inside the project folder and activate your venv (see above)
b) Type: ```python gen_music_video.py "your video.mp4"```
   (or drag the video file into the terminal after typing the command + a space)
c) Choose your output resolution (1080p or 720p)
d) Pick a subtitle font from a random offer — press:

     1 = use it
     2 = try another random one
     3 = pick a specific font from the fonts/ folder by name
     
e) Pick a background — an image or a looping video (file dialog opens in
   the video's own folder)
   
f) Choose whether the center circle should use a different image (optional)

g) Wait while it extracts audio and transcribes the lyrics

h) EDIT PAUSE — open the generated .srt file (same folder, same name as your
   video) in any text editor, fix any wrong lyrics, save it, then in the
   terminal press:
   
     1 = refresh (re-reads the file — use this after saving your edits)
     2 = continue (locks in your corrections and moves on)
     3 = cancel (press 3 again to confirm and abort)
     
i) Wait while the visualizer renders and the final video is encoded

OUTPUT:

- video name_music_video.mp4 — your finished, YouTube-ready video
- video name.srt — kept in the folder afterward, in case you want to reuse or further edit it later

That's it. Thanks for using my script!
