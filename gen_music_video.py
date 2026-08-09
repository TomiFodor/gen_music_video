#!/usr/bin/env python3
# =============================================================================
# gen_music_video — circular audio visualizer + burned-in, hand-corrected lyrics
# Usage: gen_music_video "song.mp4" [--model small|medium|large-v3]
# Full setup & usage instructions: see README.md
# =============================================================================

import os
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
SMOOTHING      = 0.1             # lower = snappier, higher = laggier
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
SUB_FONTSIZE  = 26
SUB_BOLD      = 0                 # display fonts are already bold by design
SUB_OUTLINE   = 3
SUB_SHADOW    = 1
SUB_MARGIN_V  = 70
FADE_MS       = 200               # subtitle fade in/out duration (ms)
FALLBACK_FONT = "Liberation Sans" # used if fonts/ is empty or missing

SUPPORTED_VIDEO  = ['.mp4', '.mkv', '.mov', '.webm', '.avi', '.m4v']
COVER_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.webp']

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
  2. Pick a background image, and optionally a separate center-circle image.
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

OUTPUT:
  <video name>_music_video.mp4 — saved next to the input video.
  <video name>.srt is kept (not deleted) so you can re-use or re-edit it.

SUPPORTED VIDEO:  .mp4 .mkv .mov .webm .avi .m4v
SUPPORTED IMAGES: .jpg .jpeg .png .webp
=============================================================
""")

# =============================================================================
# --- FONT SELECTION ---
# Scans fonts/ for .ttf/.otf, reads each font's real family name (not just the
# filename), and lets the user cycle through a shuffled queue or pick by name.
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
    return Path(font_path).stem  # fallback: use filename

def scan_fonts(fonts_dir):
    """Return a deduplicated list of {family, path, url} dicts from fonts_dir."""
    fonts, seen = [], set()
    if not fonts_dir.is_dir():
        return fonts
    for f in sorted(fonts_dir.iterdir()):
        if f.suffix.lower() not in ('.ttf', '.otf'):
            continue
        family = get_font_family_name(f)
        if family in seen:
            continue  # skip extra weights of a family already listed
        seen.add(family)
        url = f"https://fonts.google.com/specimen/{family.replace(' ', '+')}"
        fonts.append({"family": family, "path": f, "url": url})
    return fonts

def choose_font(fonts_dir):
    """Interactive font picker. Returns (font_path, family_name)."""
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
        print("    3 = choose a specific font from the fonts/ folder")

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
            fonts = scan_fonts(fonts_dir)  # rescan in case a file was just added
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

def transcribe(audio_path, model_size):
    """Run Whisper on the audio and return (language, [segments])."""
    print(f"  Loading '{model_size}' model and analyzing audio...")
    try:
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    except Exception:
        print("  (No CUDA — using CPU. This will be slower.)")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        str(audio_path),
        vad_filter=False,                   # avoids skipping lyrics over loud beats
        condition_on_previous_text=False,   # stops one bad guess snowballing
        temperature=0.0,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )
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
    """SRT timestamp: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = round((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def write_srt(srt_path, segments):
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(f"{i}\n")
            f.write(f"{format_time(seg.start)} --> {format_time(seg.end)}\n")
            f.write(f"{seg.text.strip()}\n\n")

# =============================================================================
# --- EDIT PAUSE ---
# Holds the program so the user can hand-correct the .srt lyrics.
# =============================================================================

def wait_for_edits(srt_path):
    def count_entries():
        try:
            return Path(srt_path).read_text(encoding="utf-8").strip().count("-->")
        except Exception:
            return 0

    print("\n" + "="*60)
    print("  EDIT YOUR SUBTITLES")
    print("="*60)
    print(f"  File: {srt_path}")
    print("  Open it, correct any wrong lyrics, and SAVE.")
    print(f"  Current subtitle entries: {count_entries()}")
    print("\n  Then choose:")
    print("    1 = refresh (re-read the file after saving)")
    print("    2 = done editing, continue")
    print("    3 = cancel (press 3 again to confirm)")

    while True:
        choice = input("\n  > ").strip()
        if choice == '1':
            print(f"  Refreshed. Subtitle entries now: {count_entries()}")
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
# The edited .srt is parsed back and rebuilt as an .ass file so we can attach
# a fade in/out tag and the chosen font/style to every line before burning.
# =============================================================================

def parse_srt(srt_path):
    """Read a (possibly hand-edited) .srt back into (start, end, text) tuples."""
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
    """Write an .ass subtitle file with the chosen style and per-line fades."""
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
# --- IMAGE SELECTION ---
# Background image, plus an optional separate image for the center circle.
# =============================================================================

def select_cover_image(default_dir, video_path):
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title="Select background image",
            initialdir=default_dir,
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp"), ("All files", "*.*")],
        )
        root.destroy()
        if path:
            return path
    except Exception:
        pass

    match = find_cover(video_path)
    if match:
        print(f"  Using matching cover: {os.path.basename(match)}")
        return match

    typed = input("  Image path (ENTER to skip): ").strip().strip("'\"")
    return os.path.expanduser(typed) if typed else None

def select_visualizer_image(default_dir, background_path):
    """Optional separate image for the center circle. Defaults to background."""
    print("\n  Center visualizer circle image:")
    use_alt = input("  Use a different image than the background? (y/N): ").strip().lower()
    if use_alt != 'y':
        return background_path

    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title="Select center visualizer image",
            initialdir=default_dir,
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp"), ("All files", "*.*")],
        )
        root.destroy()
        return path if path else background_path
    except Exception:
        typed = input("  Path to center image (ENTER to reuse background): ").strip().strip("'\"")
        if not typed:
            return background_path
        typed = os.path.expanduser(typed)
        return typed if os.path.isfile(typed) else background_path

def find_cover(video_path):
    """Find an image next to the video sharing its base name."""
    base = os.path.splitext(str(video_path))[0]
    for ext in COVER_EXTENSIONS:
        for candidate in (base + ext, base + ext.upper()):
            if os.path.isfile(candidate):
                return candidate
    return None

# =============================================================================
# --- BACKGROUND + CENTER CIRCLE PREP ---
# =============================================================================

def prepare_background(cover_path):
    if cover_path and os.path.isfile(cover_path):
        print(f"  Background image: {os.path.basename(cover_path)}")
        bg = Image.open(cover_path).convert("RGBA").resize((WIDTH, HEIGHT), Image.LANCZOS)
    else:
        print("  No cover art — generating dark gradient background.")
        bg_array = np.zeros((HEIGHT, WIDTH, 4), dtype=np.uint8)
        for row in range(HEIGHT):
            val = int(20 * (1 - row / HEIGHT))
            bg_array[row, :] = [0, val, val * 2, 255]
        bg = Image.fromarray(bg_array, 'RGBA')

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), OVERLAY_COLOR)
    return Image.alpha_composite(bg, overlay).convert("RGB")

def prepare_cover_circle(cover_path):
    diameter = int(HEIGHT * COVER_R * 2)

    if cover_path and os.path.isfile(cover_path):
        img = Image.open(cover_path).convert("RGBA").resize((diameter, diameter), Image.LANCZOS)
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

    # Log-spaced bin edges: bar index 0 = bass, bar index NUM_BARS-1 = treble
    log_bins = np.logspace(0, np.log10(num_freq_bins - 1), NUM_BARS + 1).astype(int)
    log_bins = np.clip(log_bins, 0, num_freq_bins - 1)

    return stft_norm, log_bins, total_frames, num_time_frames

# =============================================================================
# --- FRAME RENDERING ---
# Each frame = background + visualizer (transparent) + center cover circle.
# =============================================================================

def render_frames(stft_norm, log_bins, total_frames, num_time_frames,
                  background_img, cover_circle, cover_x, cover_y, frames_dir):

    os.makedirs(frames_dir, exist_ok=True)

    cmap = plt.get_cmap('hsv')
    angles = np.linspace(0, 2 * np.pi, NUM_BARS, endpoint=False)
    bar_colors = [cmap(i / NUM_BARS) for i in range(NUM_BARS)]

    # Per-bar scale multiplier: bass (index 0) barely boosted, treble boosted more.
    # This is what stops the bass ring from slamming into a permanent max/clip.
    bar_scales = np.linspace(BAR_SCALE_LOW, BAR_SCALE_HIGH, NUM_BARS)

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

        # Scale rings + ticks
        for ring_idx, ring_r in enumerate(SCALE_RINGS):
            num_ticks, tick_len, tick_alpha, tick_lw = TICK_CONFIGS[ring_idx]
            for ta in np.linspace(0, 2 * np.pi, num_ticks, endpoint=False):
                ax.plot([ring_r * np.cos(ta), (ring_r + tick_len) * np.cos(ta)],
                        [ring_r * np.sin(ta), (ring_r + tick_len) * np.sin(ta)],
                        color='white', alpha=tick_alpha, linewidth=tick_lw,
                        solid_capstyle='butt')

        # Neon accent rings
        for ring_r, ring_color, ring_lw, ring_alpha in NEON_RINGS:
            ax.add_patch(plt.Circle((0, 0), ring_r, color=ring_color, fill=False,
                                    linewidth=ring_lw, alpha=ring_alpha))

        # Bars: glow pass then sharp core pass
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

        frame = background_img.copy().convert("RGBA")
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

    folder      = video_path.parent
    stem        = video_path.stem
    srt_path    = folder / f"{stem}.srt"
    ass_path    = folder / f"_{stem}.ass"
    wav_path    = folder / f"_audio_{stem}.wav"
    frames_dir  = folder / f"_frames_{stem}"
    output_path = folder / f"{stem}_music_video.mp4"

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

        print("\n[Stage 2/6] Cover + visualizer image selection")
        background_path = select_cover_image(str(folder), video_path)
        visualizer_path = select_visualizer_image(str(folder), background_path)

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
        background_img = prepare_background(background_path)
        cover_circle, cover_x, cover_y = prepare_cover_circle(visualizer_path)
        stft_norm, log_bins, total_frames, num_time_frames = analyze_audio(wav_path)
        render_frames(stft_norm, log_bins, total_frames, num_time_frames,
                      background_img, cover_circle, cover_x, cover_y, str(frames_dir))

        print("\n[Stage 6/6] Encoding final video with burned subtitles...")
        corrected_segments = parse_srt(srt_path)
        build_ass(corrected_segments, ass_path, font_family, WIDTH, HEIGHT)
        fonts_dir_for_ffmpeg = FONTS_DIR if font_path else None
        ok = encode_final_video(str(frames_dir), video_path, ass_path, output_path,
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
        if frames_dir.exists():
            shutil.rmtree(frames_dir, ignore_errors=True)
        if wav_path.exists():
            wav_path.unlink()
        if ass_path.exists():
            ass_path.unlink()

    print(f"\n{'='*60}\n  Done.\n{'='*60}\n")

if __name__ == "__main__":
    main()
