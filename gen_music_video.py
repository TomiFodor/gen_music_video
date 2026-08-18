#!/usr/bin/env python3
# =============================================================================
# gen_music_video — circular audio visualizer + burned-in, hand-corrected lyrics
# Usage: gen_music_video "song.mp4" [--model small|medium|large-v3]
# Full setup & usage instructions: see README.md
# =============================================================================

import os
import io
import sys
import shutil
import random
import subprocess
from pathlib import Path

import numpy as np
import librosa
import matplotlib
matplotlib.use('Agg')  # non-interactive backend, renders straight to file
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from tqdm import tqdm
from faster_whisper import WhisperModel
from fontTools.ttLib import TTFont

# =============================================================================
# --- SETTINGS ---
# =============================================================================

# --- Output / encode ---
FPS           = 30
WIDTH         = 1920             # overridden by resolution choice at runtime
HEIGHT        = 1080             # overridden by resolution choice at runtime
CRF           = 18               # quality (lower = better)
AUDIO_BITRATE = "192k"

# --- Transcription ---
DEFAULT_MODEL = "medium"

# --- Visualizer bar response ---
NUM_BARS       = 280
SMOOTHING      = 0.12            # lower = snappier, higher = laggier
BAR_SCALE_LOW  = 1.0             # multiplier for lowest-frequency bar (bass) — no bass clipping
BAR_SCALE_HIGH = 1.45            # multiplier for highest-frequency bar (treble)
N_FFT          = 2048
DPI            = 100

# --- Radii (normalized plot units, Y axis = -1 to 1) ---
COVER_R       = 0.28
INNER_R       = 0.30
OUTER_R       = 0.80

# --- Scale ring radii (cyberpunk gauge look) ---
SCALE_RINGS   = [0.84, 0.89, 0.94]

# --- Tick settings per ring: [num_ticks, length, alpha, linewidth] ---
TICK_CONFIGS  = [
    [360, 0.012, 0.6, 0.6],
    [72,  0.025, 0.8, 1.0],
    [36,  0.040, 1.0, 1.4],
]

OVERLAY_COLOR = (0, 0, 10, 190)  # dark layer over background so neon pops

GLOW_ALPHA    = 0.18
GLOW_WIDTH    = 8.0
BAR_WIDTH     = 2.0

# --- Neon accent rings: (radius, color_hex, linewidth, alpha) ---
NEON_RINGS = [
    (INNER_R,        '#00ffff', 1.5, 0.7),
    (OUTER_R + 0.01, '#ff00ff', 1.0, 0.5),
    (SCALE_RINGS[0], '#00ffff', 0.6, 0.3),
    (SCALE_RINGS[1], '#ff00ff', 0.6, 0.25),
    (SCALE_RINGS[2], '#00ffff', 0.8, 0.35),
]

# --- Subtitle style (burned in via ffmpeg/libass, drives a generated .ass file) ---
FONTS_DIR     = Path(__file__).resolve().parent / "fonts"  # drop .ttf/.otf files here
SUB_FONTSIZE  = 68
SUB_BOLD      = 0                 # display fonts are already bold by design
SUB_OUTLINE   = 6
SUB_SHADOW    = 2
SUB_MARGIN_V  = 70
FADE_MS         = 200             # subtitle fade in/out duration (ms)
MIN_SUB_SECONDS = 0.7               # shortest a subtitle line will stay on screen, even a single word
FALLBACK_FONT = "Liberation Sans" # used if fonts/ is empty or missing

# --- Supported file types ---
SUPPORTED_VIDEO   = ['.mp4', '.mkv', '.mov', '.webm', '.avi', '.m4v']  # main input + background video
IMAGE_EXTENSIONS  = ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.tif', '.gif', '.svg']

# =============================================================================
# --- USAGE / HELP ---
# =============================================================================

def print_usage():
    print("""
=============================================================
  gen_music_video — Usage Instructions
=============================================================

RUN:
  gen_music_video "song.mp4"
  gen_music_video "song.mp4" --model medium   (small | medium | large-v3)

STEPS WHEN IT RUNS:
  1. Choose output resolution, then pick a subtitle font.
  2. Pick a background — an image (jpg/png/svg/etc) OR a short video that
     will loop for the whole song. Optionally pick a separate center image.
  3. Audio is extracted and transcribed to a .srt file.
  4. EDIT PAUSE — open the .srt, fix any wrong lyrics, save, then:
       1 = refresh (re-read the file after saving)
       2 = continue
       3 = cancel (press 3 again to confirm)
  5. Visualizer frames render.
  6. Final video is encoded with your corrected, faded-in/out subtitles burned in.

FONTS:
  Drop .ttf / .otf files into a "fonts" folder next to this script.
  A random one is offered each run — you can cycle or pick a specific one.
  If the folder is empty, Liberation Sans Bold is used as a fallback.

BACKGROUND TYPES:
  Images: .jpg .jpeg .png .webp .bmp .tiff .gif .svg
  Videos (looped): .mp4 .mkv .mov .webm .avi .m4v
  The center visualizer circle is ALWAYS a static image (SVG included) —
  it does not support video, even if the background does.

OUTPUT:
  <video name>_music_video.mp4 — saved next to the input video.
  <video name>.srt is kept (not deleted) so you can re-use or re-edit it.
=============================================================
""")

# =============================================================================
# --- FONT SELECTION ---
# =============================================================================

def get_font_family_name(font_path):
    """Read a font file's family name directly from its internal name table."""
    try:
        font = TTFont(str(font_path), fontNumber=0, lazy=True)
        for name_id in (16, 1):  # 16 = Typographic Family, 1 = Font Family
            name = font['name'].getDebugName(name_id)
            if name:
                font.close()
                return name
        font.close()
    except Exception:
        pass
    return Path(font_path).stem

def scan_fonts(fonts_dir):
    fonts, seen = [], set()
    if not fonts_dir.is_dir():
        return fonts
    for f in sorted(fonts_dir.iterdir()):
        if f.suffix.lower() not in ('.ttf', '.otf'):
            continue
        family = get_font_family_name(f)
        if family in seen:
            continue
        seen.add(family)
        url = f"https://fonts.google.com/specimen/{family.replace(' ', '+')}"
        fonts.append({"family": family, "path": f, "url": url})
    return fonts

def choose_font(fonts_dir):
    fonts = scan_fonts(fonts_dir)
    if not fonts:
        print(f"\n  No fonts found in: {fonts_dir}")
        print(f"  Using fallback font: {FALLBACK_FONT}")
        return None, FALLBACK_FONT

    pool = fonts.copy()
    random.shuffle(pool)
    idx = 0

    while True:
        current = pool[idx]
        print("\n" + "="*60)
        print(f"  Font: {current['family']}")
        print(f"  Preview: {current['url']}")
        print("="*60)
        print("    1 = use this font")
        print("    2 = try another random font")
        print("    3 = choose a specific font from the fonts folder")

        choice = input("\n  > ").strip()

        if choice in ('1', ''):
            print(f"  Using font: {current['family']}")
            return current['path'], current['family']

        elif choice == '2':
            idx += 1
            if idx >= len(pool):
                random.shuffle(pool)
                idx = 0

        elif choice == '3':
            fonts = scan_fonts(fonts_dir)
            print("\n  Available fonts:")
            for i, f in enumerate(fonts, 1):
                print(f"    [{i}] {f['family']}")
            pick = input("  Enter number: ").strip()
            if pick.isdigit() and 1 <= int(pick) <= len(fonts):
                chosen = fonts[int(pick) - 1]
                print(f"  Using font: {chosen['family']}")
                return chosen['path'], chosen['family']
            print("  [!] Invalid selection.")

        else:
            print("  [!] Enter 1, 2, or 3.")

# =============================================================================
# --- AUDIO EXTRACTION ---
# =============================================================================

def extract_audio(video_path, wav_path):
    print("  Extracting audio track...", end='', flush=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vn", "-ac", "2", "-ar", "44100", "-f", "wav", str(wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    print(" done.")

# =============================================================================
# --- TRANSCRIPTION (faster-whisper) ---
# =============================================================================

from functools import lru_cache

@lru_cache(maxsize=2)  # small cache: keeps last GPU model + last CPU fallback model in memory
def _load_whisper_model(model_size, device, compute_type):
    """
    Loads (and caches) a WhisperModel instance. If this process transcribes
    more than one file with the same model_size/device/compute_type — e.g. a
    future batch mode that loops inside one Python process — the model is
    reused instead of being reloaded from disk into memory every time.
    """
    return WhisperModel(model_size, device=device, compute_type=compute_type)

def transcribe(audio_path, model_size):
    print(f"  Loading '{model_size}' model and analyzing audio...")
    def run_transcribe(device, compute_type):
        model = _load_whisper_model(model_size, device, compute_type)
        return model.transcribe(
            str(audio_path),
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=300,  # shorter gaps still count as "silence" between lyric lines
                threshold=0.35,               # more lenient — avoids treating quiet singing as non-speech
            ),
            beam_size=8,                      # wider search, better for noisy/musical audio
            condition_on_previous_text=False,
            temperature=0.0,
            no_speech_threshold=0.6,
            compression_ratio_threshold=2.4,
            initial_prompt="These are song lyrics, sung over music.",
        )

    try:
        segments, info = run_transcribe("cuda", "float16")
    except Exception as e:
        print(f"  (GPU path failed: {e})")
        print("  Falling back to CPU. This will be slower.")
        segments, info = run_transcribe("cpu", "int8")

    print(f"  Detected language: {info.language} ({info.language_probability:.2f})")
    segments_list = []
    with tqdm(total=round(info.duration, 2), unit="sec", desc="  Transcribing") as pbar:
        last_end = 0.0
        for segment in segments:
            segments_list.append(segment)
            pbar.update(segment.end - last_end)
            last_end = segment.end
    return info.language, segments_list

def format_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = round((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(srt_path, segments):
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            start = seg.start
            end = seg.end

            # If this line is shorter than our minimum, stretch its end time —
            # but never past the next line's start, so we don't create overlap.
            if end - start < MIN_SUB_SECONDS:
                next_start = segments[i].start if i < len(segments) else None
                stretched_end = start + MIN_SUB_SECONDS
                end = min(stretched_end, next_start) if next_start else stretched_end

            f.write(f"{i}\n")
            f.write(f"{format_time(start)} --> {format_time(end)}\n")
            f.write(f"{seg.text.strip()}\n\n")

# =============================================================================
# --- EDIT PAUSE ---
# =============================================================================

def wait_for_edits(srt_path):
    def count_entries():
        try:
            return Path(srt_path).read_text(encoding="utf-8").strip().count("-->")
        except Exception:
            return 0

    def print_menu():
        print("\n  Then choose:")
        print("    1 = refresh (re-read the file after saving)")
        print("    2 = done editing, continue")
        print("    3 = cancel (press 3 again to confirm)")

    print("\n" + "="*60)
    print("  EDIT YOUR SUBTITLES")
    print("="*60)
    print(f"  File: {srt_path}")
    print("  Open it, correct any wrong lyrics, and SAVE.")
    print(f"  Current subtitle entries: {count_entries()}")
    print_menu()

    while True:
        choice = input("\n  > ").strip()
        if choice == '1':
            print(f"  Refreshed. Subtitle entries now: {count_entries()}")
            print_menu()
        elif choice == '2':
            print("  Continuing with your edited subtitles...")
            return True
        elif choice == '3':
            if input("  Are you sure? Press 3 again to cancel: ").strip() == '3':
                return False
            print("  Cancellation aborted — resuming.")
        else:
            print("  [!] Enter 1 (refresh), 2 (continue), or 3 (cancel).")

# =============================================================================
# --- SRT -> ASS CONVERSION ---
# =============================================================================

def parse_srt(srt_path):
    segments = []
    blocks = Path(srt_path).read_text(encoding="utf-8").strip().split("\n\n")
    for block in blocks:
        lines = block.strip().splitlines()
        if len(lines) < 2:
            continue
        start_str, end_str = [t.strip() for t in lines[1].split("-->")]
        text = "\n".join(lines[2:]).strip()
        segments.append((srt_time_to_seconds(start_str), srt_time_to_seconds(end_str), text))
    return segments

def srt_time_to_seconds(t):
    h, m, s_ms = t.split(":")
    s, ms = s_ms.split(",")
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

def seconds_to_ass_time(seconds):
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"

def build_ass(segments, ass_path, font_family, width, height):
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_family},{SUB_FONTSIZE},&H00FFFFFF,&H000000FF,&H00000000,&H00000000,{SUB_BOLD},0,0,0,100,100,0,0,1,{SUB_OUTLINE},{SUB_SHADOW},2,10,10,{SUB_MARGIN_V},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [header]
    for start, end, text in segments:
        text_ass = text.replace("\n", "\\N")
        fade = f"{{\\fad({FADE_MS},{FADE_MS})}}"
        lines.append(f"Dialogue: 0,{seconds_to_ass_time(start)},{seconds_to_ass_time(end)},"
                     f"Default,,0,0,0,,{fade}{text_ass}\n")
    Path(ass_path).write_text("".join(lines), encoding="utf-8")

# =============================================================================
# --- IMAGE LOADING (handles raster formats + SVG) ---
# =============================================================================

def rasterize_svg(svg_path, target_size):
    """Render an SVG to a PIL RGBA image at the given (width, height) via resvg_py."""
    import resvg_py
    svg_string = Path(svg_path).read_text(encoding="utf-8")
    width, height = target_size if target_size else (WIDTH, HEIGHT)
    try:
        png_bytes = resvg_py.svg_to_bytes(svg_string=svg_string, width=width, height=height)
    except TypeError:
        # Older/newer API without width/height kwargs — render native, then resize.
        png_bytes = resvg_py.svg_to_bytes(svg_string=svg_string)
    img = Image.open(io.BytesIO(bytes(png_bytes))).convert("RGBA")
    if img.size != (width, height):
        img = img.resize((width, height), Image.LANCZOS)
    return img

def load_image_any(source, target_size=None):
    """
    Load an image from a path (any supported format, including SVG) or accept
    an already-opened PIL Image directly. Returns an RGBA image, resized if
    target_size is given.
    """
    if isinstance(source, Image.Image):
        img = source.convert("RGBA")
    else:
        path = Path(source)
        if path.suffix.lower() == '.svg':
            img = rasterize_svg(path, target_size)
        else:
            img = Image.open(path).convert("RGBA")

    if target_size and img.size != target_size:
        img = img.resize(target_size, Image.LANCZOS)
    return img

# =============================================================================
# --- BACKGROUND SOURCE SELECTION ---
# Background may be a static image (any supported type) or a video that
# loops for the full song duration. Center circle is always a static image.
# =============================================================================

def is_video_file(path):
    return Path(path).suffix.lower() in SUPPORTED_VIDEO

def select_background_source(video_path):
    """Prompt for a background image OR video path typed into the terminal. Returns a path, or None."""
    for candidate in find_background_by_keyword(video_path):
        print(f"  Found a likely background file: {os.path.basename(candidate)}")
        use_match = input("  Use this as the background? (Y/n): ").strip().lower()
        if use_match in ('', 'y'):
            return candidate
        # 'n' (or anything else) — move on and ask about the next candidate, if any.

    print("  Enter the path to a background image or video (loops if a video).")
    print("  Tip: drag the file into the terminal, or paste a copied path.")
    typed = input("  Background path (ENTER to skip): ").strip().strip("'\"")
    if not typed:
        return None

    path = os.path.expanduser(typed)
    if not os.path.isfile(path):
        print(f"  [!] File not found: {path} — skipping background.")
        return None
    return path

def select_visualizer_image(video_path, background_path, background_is_video, fallback_image):
    """Optional separate STATIC image for the center circle. Never a video."""
    default_result = fallback_image if background_is_video else background_path

    print("\n  Center visualizer circle image:")
    if background_is_video:
        print("  (Background is a video — center circle needs a still image.)")

    # Auto-detect an image sharing the video's exact base name (e.g. "song.png"
    # next to "song.mp4") — this is the common case, since the background video
    # (if used) has to be named differently anyway to avoid colliding with it.
    match = find_matching_file(video_path, suffix="", extensions=IMAGE_EXTENSIONS)

    # Fall back to suffixed names like "song_center.png" or "song_visualizer.png"
    # in case the plain same-name match isn't what's there.
    if not match:
        for suffix in ("_center", "_visualizer"):
            match = find_matching_file(video_path, suffix=suffix, extensions=IMAGE_EXTENSIONS)
            if match:
                break

    if match:
        print(f"  Found a matching center image: {os.path.basename(match)}")
        use_match = input("  Use this for the center circle? (Y/n): ").strip().lower()
        if use_match in ('', 'y'):
            return match

    use_alt = input("  Use a different image than the background? (y/N): ").strip().lower()
    if use_alt != 'y':
        return default_result

    print("  Enter the path to the image you want in the center circle.")
    typed = input("  Center image path (ENTER for default): ").strip().strip("'\"")
    if typed:
        typed = os.path.expanduser(typed)
        if os.path.isfile(typed):
            return typed
        print(f"  [!] File not found: {typed} — using default instead.")

    return default_result

def find_matching_file(video_path, suffix="", extensions=None):
    """
    Find a file next to the input sharing its base name, optionally with an
    added suffix (e.g. suffix="_center" looks for "song_center.png" next to
    "song.mp4"). Searches `extensions` if given, otherwise images + videos.
    """
    base = os.path.splitext(str(video_path))[0] + suffix
    exts = extensions if extensions is not None else (IMAGE_EXTENSIONS + SUPPORTED_VIDEO)
    for ext in exts:
        for candidate in (base + ext, base + ext.upper()):
            if os.path.isfile(candidate):
                return candidate
    return None

def find_background_by_keyword(video_path):
    """
    Scan the video's folder for every image/video file with 'background'
    anywhere in its filename (case-insensitive) — catches naming styles like
    "background.jpg", "song-background.mp4", "My_Background_v2.png", etc.,
    without requiring it to match the input video's exact base name.
    Returns a list of matching paths (possibly empty), not just one file —
    the caller is responsible for asking the user to confirm each one.
    """
    folder = Path(video_path).parent
    exts = set(e.lower() for e in IMAGE_EXTENSIONS + SUPPORTED_VIDEO)
    matches = [
        f for f in sorted(folder.iterdir())
        if f.is_file() and f.suffix.lower() in exts and "background" in f.name.lower()
    ]
    return [str(f) for f in matches]


# =============================================================================
# --- BACKGROUND PREPARATION ---
# Returns a dict: {'type': 'image', 'image': <RGB PIL>}
#             or  {'type': 'video', 'frames': [sorted frame paths]}
# =============================================================================

def prepare_background(background_path):
    """Static image background: load, stretch to WxH, apply dark overlay."""
    if background_path and os.path.isfile(background_path):
        print(f"  Background image: {os.path.basename(background_path)}")
        bg = load_image_any(background_path, target_size=(WIDTH, HEIGHT))
    else:
        print("  No background provided — generating dark gradient.")
        bg_array = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
        for row in range(HEIGHT):
            val = int(20 * (1 - row / HEIGHT))
            bg_array[row, :] = [0, val, val * 2, 255]
        bg = Image.fromarray(bg_array, 'RGBA')

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), OVERLAY_COLOR)
    rgb = Image.alpha_composite(bg, overlay).convert("RGB")
    return {"type": "image", "image": rgb}

def prepare_background_video(video_path, frames_dir):
    """
    Extracts a looping background video into WxH frames at FPS, then darkens
    each one. Frame 0 is saved separately BEFORE darkening, for use as a
    fallback center-circle image (keeps it bright/clear, not dimmed).
    Returns ({'type': 'video', 'frames': [...]}, first_frame_raw_path).
    """
    print(f"  Background video: {os.path.basename(video_path)} (will loop)")
    os.makedirs(frames_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"scale={WIDTH}:{HEIGHT}",
        "-r", str(FPS),
        os.path.join(frames_dir, "bg_%06d.png"),
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    frame_paths = sorted(Path(frames_dir).glob("bg_*.png"))
    if not frame_paths:
        raise RuntimeError("No frames extracted from background video.")

    # Keep an unmodified copy of the first frame for the center-circle fallback.
    first_frame_raw_path = Path(frames_dir) / "_first_frame_raw.png"
    shutil.copy(frame_paths[0], first_frame_raw_path)

    # Darken every extracted frame in place so it matches the image path's look.
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), OVERLAY_COLOR)
    for fp in tqdm(frame_paths, desc="  Darkening loop frames", unit="frame"):
        frame = Image.open(fp).convert("RGBA")
        frame = Image.alpha_composite(frame, overlay).convert("RGB")
        frame.save(fp)

    return {"type": "video", "frames": [str(p) for p in frame_paths]}, str(first_frame_raw_path)

def peek_first_frame(video_path, bg_frames_dir):
    """Grab just frame 0 of the background video — cheap preview for the
    center-circle picker, before we commit to full extraction later."""
    os.makedirs(bg_frames_dir, exist_ok=True)
    preview_path = Path(bg_frames_dir) / "_first_frame_raw.png"
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vframes", "1", "-vf", f"scale={WIDTH}:{HEIGHT}",
        str(preview_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(preview_path)

def get_background_frame(background, frame_idx):
    """Fetch the correct background for this output frame (loops if video)."""
    if background["type"] == "image":
        return background["image"].copy().convert("RGBA")
    frames = background["frames"]
    path = frames[frame_idx % len(frames)]   # modulo = seamless loop
    return Image.open(path).convert("RGBA")

# =============================================================================
# --- CENTER CIRCLE PREP (always a static image) ---
# =============================================================================

def prepare_cover_circle(source):
    """Crop a source image into a circle with a neon ring for the center."""
    diameter = int(HEIGHT * COVER_R * 2)

    if source and (isinstance(source, Image.Image) or os.path.isfile(source)):
        # Load at native resolution first (no target_size) so we can crop a
        # centered square using the image's actual aspect ratio, instead of
        # force-stretching a rectangle into a square.
        img = load_image_any(source)
        w, h = img.size
        side = min(w, h)  # shorter dimension becomes the square's size
        left = (w - side) // 2
        top = (h - side) // 2
        img = img.crop((left, top, left + side, top + side))
        img = img.resize((diameter, diameter), Image.LANCZOS)
    else:
        img = Image.new("RGBA", (diameter, diameter), (10, 10, 20, 255))

    mask = Image.new("L", (diameter, diameter), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, diameter - 1, diameter - 1), fill=255)
    img.putalpha(mask)

    bw = 3
    ImageDraw.Draw(img).ellipse((bw, bw, diameter - bw - 1, diameter - bw - 1),
                                outline=(0, 255, 255, 220), width=bw)

    paste_x = WIDTH // 2 - diameter // 2
    paste_y = HEIGHT // 2 - diameter // 2
    return img, paste_x, paste_y

# =============================================================================
# --- AUDIO ANALYSIS ---
# =============================================================================

def analyze_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    total_frames = int(duration * FPS)
    print(f"  Duration: {duration:.1f}s | SR: {sr}Hz | Frames: {total_frames}")

    hop_length = int(sr / FPS)
    stft = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=hop_length))
    stft_db = librosa.amplitude_to_db(stft, ref=np.max)

    s_min, s_max = stft_db.min(), stft_db.max()
    stft_norm = (stft_db - s_min) / (s_max - s_min + 1e-9)

    num_freq_bins = stft_norm.shape[0]
    num_time_frames = stft_norm.shape[1]

    log_bins = np.logspace(0, np.log10(num_freq_bins - 1), NUM_BARS + 1).astype(int)
    log_bins = np.clip(log_bins, 0, num_freq_bins - 1)

    return stft_norm, log_bins, total_frames, num_time_frames

# =============================================================================
# --- FRAME RENDERING ---
# Each frame = background (image or looped video frame) + visualizer + circle
# =============================================================================

def render_frames(stft_norm, log_bins, total_frames, num_time_frames,
                  background, cover_circle, cover_x, cover_y, frames_dir):

    os.makedirs(frames_dir, exist_ok=True)

    cmap = plt.get_cmap('hsv')
    angles = np.linspace(0, 2 * np.pi, NUM_BARS, endpoint=False)
    bar_colors = [cmap(i / NUM_BARS) for i in range(NUM_BARS)]
    bar_scales = np.linspace(BAR_SCALE_LOW, BAR_SCALE_HIGH, NUM_BARS)  # bass unboosted, treble boosted

    prev_bars = np.zeros(NUM_BARS)

    fig = plt.figure(figsize=(WIDTH / DPI, HEIGHT / DPI))
    fig.patch.set_alpha(0.0)
    ax = fig.add_axes([0, 0, 1, 1])
    aspect = WIDTH / HEIGHT

    for frame_idx in tqdm(range(total_frames), desc="  Rendering", unit="frame"):
        ax.cla()
        ax.patch.set_alpha(0.0)
        ax.axis('off')
        ax.set_xlim(-aspect, aspect)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')

        stft_idx = min(frame_idx, num_time_frames - 1)
        frame_spectrum = stft_norm[:, stft_idx]
        bars = np.zeros(NUM_BARS)
        for i in range(NUM_BARS):
            b0, b1 = log_bins[i], log_bins[i + 1]
            bars[i] = np.mean(frame_spectrum[b0:b1]) if b1 > b0 else frame_spectrum[b0]

        bars = SMOOTHING * prev_bars + (1.0 - SMOOTHING) * bars
        bars = np.clip(bars * bar_scales, 0, 1)
        prev_bars = bars.copy()

        for ring_idx, ring_r in enumerate(SCALE_RINGS):
            num_ticks, tick_len, tick_alpha, tick_lw = TICK_CONFIGS[ring_idx]
            for ta in np.linspace(0, 2 * np.pi, num_ticks, endpoint=False):
                ax.plot([ring_r * np.cos(ta), (ring_r + tick_len) * np.cos(ta)],
                        [ring_r * np.sin(ta), (ring_r + tick_len) * np.sin(ta)],
                        color='white', alpha=tick_alpha, linewidth=tick_lw,
                        solid_capstyle='butt')

        for ring_r, ring_color, ring_lw, ring_alpha in NEON_RINGS:
            ax.add_patch(plt.Circle((0, 0), ring_r, color=ring_color, fill=False,
                                    linewidth=ring_lw, alpha=ring_alpha))

        bar_range = OUTER_R - INNER_R
        for i in range(NUM_BARS):
            angle, height = angles[i], bars[i]
            r1 = INNER_R + height * bar_range
            x0, y0 = INNER_R * np.cos(angle), INNER_R * np.sin(angle)
            x1, y1 = r1 * np.cos(angle), r1 * np.sin(angle)
            base = bar_colors[i]

            ax.plot([x0, x1], [y0, y1],
                    color=(base[0], base[1], base[2], GLOW_ALPHA),
                    linewidth=GLOW_WIDTH, solid_capstyle='round')
            core_alpha = 0.5 + height * 0.5
            ax.plot([x0, x1], [y0, y1],
                    color=(base[0], base[1], base[2], core_alpha),
                    linewidth=BAR_WIDTH, solid_capstyle='round')

        vis_path = os.path.join(frames_dir, "_vis_temp.png")
        fig.savefig(vis_path, dpi=DPI, transparent=True, facecolor='none')

        frame = get_background_frame(background, frame_idx)
        frame = Image.alpha_composite(frame, Image.open(vis_path).convert("RGBA"))
        if cover_circle is not None:
            frame.paste(cover_circle, (cover_x, cover_y), cover_circle)

        frame.convert("RGB").save(os.path.join(frames_dir, f"frame_{frame_idx:06d}.png"))

    plt.close(fig)
    temp = os.path.join(frames_dir, "_vis_temp.png")
    if os.path.exists(temp):
        os.remove(temp)

# =============================================================================
# --- FINAL ENCODE: frames + audio + burned subtitles ---
# =============================================================================

def _escape_sub_path(path):
    return str(path).replace('\\', '\\\\').replace(':', '\\:').replace("'", "\\'")

def encode_final_video(frames_dir, audio_source, ass_path, output_path, fonts_dir):
    print(f"\n  Encoding: {os.path.basename(output_path)}")

    sub_filter = f"subtitles=filename='{_escape_sub_path(ass_path)}'"
    if fonts_dir:
        sub_filter += f":fontsdir='{_escape_sub_path(fonts_dir)}'"

    check = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
    if "h264_nvenc" in check.stdout:
        print("  Using NVIDIA nvenc GPU encoder.")
        video_flags = ["-c:v", "h264_nvenc", "-preset", "p4", "-cq", str(CRF), "-b:v", "0"]
    else:
        print("  nvenc not found — using libx264 CPU encoder.")
        video_flags = ["-c:v", "libx264", "-preset", "slow", "-crf", str(CRF)]

    cmd = [
        "ffmpeg",
        "-framerate", str(FPS),
        "-i", os.path.join(frames_dir, "frame_%06d.png"),
        "-i", str(audio_source),
        "-map", "0:v:0",      # video = our rendered frames (input 0), not input 1
        "-map", "1:a:0",      # audio = input 1's audio track only
        "-vf", sub_filter,
        *video_flags,
        "-c:a", "aac", "-b:a", AUDIO_BITRATE,
        "-pix_fmt", "yuv420p",
        "-shortest", "-y",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  [!] FFmpeg error:\n{result.stderr}")
        return False

    print(f"  Saved: {output_path}")
    return True

# =============================================================================
# --- RESOLUTION CHOICE ---
# =============================================================================

def get_resolution_choice():
    print("\n  Select output resolution:")
    print("    [1] 1080p — 1920x1080 (slower render)")
    print("    [2]  720p — 1280x720  (faster render)")
    while True:
        choice = input("\n  Enter 1 or 2 (default = 1): ").strip()
        if choice in ('', '1'):
            print("  Resolution set to 1080p.")
            return 1920, 1080
        if choice == '2':
            print("  Resolution set to 720p.")
            return 1280, 720
        print("  [!] Invalid choice. Enter 1 or 2.")

# =============================================================================
# --- MAIN ---
# =============================================================================

def main():
    global WIDTH, HEIGHT

    args = sys.argv[1:]
    if not args or args[0] in ('-h', '--help', 'help'):
        print_usage()
        sys.exit(0)

    video_path = Path(args[0]).expanduser()
    if not video_path.is_file():
        print(f"  [!] File not found: {video_path}")
        sys.exit(1)
    if video_path.suffix.lower() not in SUPPORTED_VIDEO:
        print(f"  [!] Unsupported video type '{video_path.suffix}'.")
        sys.exit(1)

    model_size = DEFAULT_MODEL
    if "--model" in args:
        model_size = args[args.index("--model") + 1]

    folder        = video_path.parent
    stem          = video_path.stem
    srt_path      = folder / f"{stem}.srt"
    ass_path      = folder / f"_{stem}.ass"
    wav_path      = folder / f"_audio_{stem}.wav"
    frames_dir    = folder / f"_frames_{stem}"
    bg_frames_dir = folder / f"_bgframes_{stem}"
    output_path   = folder / f"{stem}_music_video.mp4"

    print("\n" + "="*60)
    print("  gen_music_video")
    print("="*60)
    print(f"  Video  : {video_path.name}")
    print(f"  Model  : {model_size}")
    print(f"  Output : {output_path.name}")
    print("="*60)

    try:
        print("\n[Stage 1/6] Resolution + font selection")
        WIDTH, HEIGHT = get_resolution_choice()
        font_path, font_family = choose_font(FONTS_DIR)

        print("\n[Stage 2/6] Background (img/vid) + visualizer image selection")
        background_source = select_background_source(video_path)
        background_is_video = background_source and is_video_file(background_source)

        # Just grab a raw first-frame preview now (cheap) for the visualizer
        # picker — the full extraction + darkening pass happens later in
        # Stage 5, once every other choice is locked in.
        first_frame_raw = None
        if background_is_video:
            first_frame_raw = peek_first_frame(background_source, str(bg_frames_dir))

        visualizer_source = select_visualizer_image(
            video_path, background_source, background_is_video, first_frame_raw
        )

        print("\n[Stage 3/6] Extracting audio and transcribing lyrics...")
        extract_audio(video_path, wav_path)
        language, segments = transcribe(wav_path, model_size)
        write_srt(srt_path, segments)
        print(f"  Subtitles ({language}) saved to: {srt_path}")

        print("\n[Stage 4/6] Manual subtitle correction")
        if not wait_for_edits(srt_path):
            print("\n  Cancelled by user.")
            return

        print("\n[Stage 5/6] Rendering visualizer frames...")
        if background_is_video:
            background, _ = prepare_background_video(background_source, str(bg_frames_dir))
        else:
            background = prepare_background(background_source)
        cover_circle, cover_x, cover_y = prepare_cover_circle(visualizer_source)

        stft_norm, log_bins, total_frames, num_time_frames = analyze_audio(wav_path)
        render_frames(stft_norm, log_bins, total_frames, num_time_frames,
                      background, cover_circle, cover_x, cover_y, str(frames_dir))

        print("\n[Stage 6/6] Encoding final video with burned subtitles...")
        corrected_segments = parse_srt(srt_path)
        build_ass(corrected_segments, ass_path, font_family, WIDTH, HEIGHT)
        fonts_dir_for_ffmpeg = FONTS_DIR if font_path else None
        ok = encode_final_video(str(frames_dir), wav_path, ass_path, output_path,
                                fonts_dir_for_ffmpeg)

        if ok:
            size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\n  Output file : {output_path}")
            print(f"  File size   : {size_mb:.1f} MB")
            print(f"  Subtitles   : {srt_path}  (kept for reference/re-editing)")

    except KeyboardInterrupt:
        print("\n\n  Interrupted.")
    except Exception as e:
        print(f"\n  [!] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # .srt is kept intentionally — everything else is temp working data.
        for temp_dir in (frames_dir, bg_frames_dir):
            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
        if wav_path.exists():
            wav_path.unlink()
        if ass_path.exists():
            ass_path.unlink()

    print(f"\n{'='*60}\n  Done.\n{'='*60}\n")

if __name__ == "__main__":
    main()
