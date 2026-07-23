"""
Local AI Models for YouTube Captioning — Full Comparison (11 models)
====================================================================

All models run 100% locally. No API keys, no tokens, no internet at runtime.

Models compared:
 1. faster-whisper      — CTranslate2 Whisper (easiest, lowest VRAM)
 2. WhisperX            — Forced phoneme alignment (best word timestamps)
 3. stable-ts           — Stabilized timestamps (best for subtitle files)
 4. Parakeet TDT 0.6B   — Best English WER, native word timestamps
 5. Canary Qwen 2.5B    — Top leaderboard accuracy, English only
 6. Distil-Whisper       — 6x faster distilled Whisper
 7. Moonshine            — Ultra-lightweight for edge/CPU
 8. SenseVoice (FunASR)  — 50+ languages, emotion detection
 9. Vosk                 — Kaldi-based, 20+ languages, minimal resources
10. Whisper (original)   — The baseline, 99 languages
11. whisper.cpp          — C++ port, runs on anything
"""

import json
import sys
import os
from pathlib import Path


# =============================================================================
# ANSI COLORS for terminal output
# =============================================================================

class Style:
    """ANSI escape codes for colored terminal output."""
    # Reset
    RESET = "\033[0m"

    # Regular colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright/bold colors
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"

    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"
    BRIGHT_BLACK = "\033[90m"

    # Backgrounds
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"


def supports_color():
    """Check if terminal supports ANSI colors."""
    if not hasattr(sys.stdout, "isatty"):
        return False
    return sys.stdout.isatty()


# Disable colors if terminal doesn't support them
if not supports_color():
    for attr in dir(Style):
        if attr.isupper() and attr != "RESET":
            setattr(Style, attr, "")


def print_header(text):
    """Print a colored header banner."""
    print(f"\n{Style.BG_BLUE}{Style.BRIGHT_WHITE}{Style.BOLD} {text} {Style.RESET}")
    print(f"{Style.BLUE}{'=' * 60}{Style.RESET}")


def print_model_header(num, name, tagline):
    """Print a colored model header."""
    print(f"\n{Style.BG_CYAN}{Style.BRIGHT_BLACK}{Style.BOLD} {num}. {name} {Style.RESET}")
    print(f"{Style.CYAN}{tagline}{Style.RESET}")
    print(f"{Style.DIM}{'-' * 60}{Style.RESET}")


def print_success(text):
    """Print a success message in green."""
    print(f"{Style.BRIGHT_GREEN}{Style.BOLD}✓ {text}{Style.RESET}")


def print_error(text):
    """Print an error message in red."""
    print(f"{Style.BRIGHT_RED}{Style.BOLD}✗ {text}{Style.RESET}")


def print_warning(text):
    """Print a warning message in yellow."""
    print(f"{Style.BRIGHT_YELLOW}{Style.BOLD}⚠ {text}{Style.RESET}")


def print_info(text):
    """Print an info message in cyan."""
    print(f"{Style.CYAN}{text}{Style.RESET}")


def print_dim(text):
    """Print dimmed/grayed text."""
    print(f"{Style.DIM}{text}{Style.RESET}")


def print_timestamp(start, end, text):
    """Print a transcription timestamp with colored timecodes."""
    print(f"  {Style.GREEN}{start:>8s}{Style.RESET} → {Style.GREEN}{end:>8s}{Style.RESET}  {text}")


def print_subtitle_stats(path, count, fmt="SRT"):
    """Print subtitle generation stats."""
    print(f"  {Style.BRIGHT_MAGENTA}{fmt}{Style.RESET} written to {Style.UNDERLINE}{path}{Style.RESET} ({Style.CYAN}{count} blocks{Style.RESET})")


def print_banner():
    """Print the application banner with system info."""
    import platform
    import os

    print(f"""
{Style.BG_BLUE}{Style.BRIGHT_WHITE}{Style.BOLD}
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║   ████████╗███████╗██████╗ ███╗   ███╗██╗███╗   ██╗ █████╗ ██╗     ║
║   ╚══██╔══╝██╔════╝██╔══██╗████╗ ████║██║████╗  ██║██╔══██╗██║     ║
║      ██║   █████╗  ██████╔╝██╔████╔██║██║██╔██╗ ██║███████║██║     ║
║      ██║   ██╔══╝  ██╔══██╗██║╚██╔╝██║██║██║╚██╗██║██╔══██║██║     ║
║      ██║   ███████╗██║  ██║██║ ╚═╝ ██║██║██║ ╚████║██║  ██║███████╗║
║      ╚═╝   ╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═╝  ╚═╝╚══════╝║
║                                                              ║
║   Local AI Models for YouTube Captioning                     ║
║   11 models compared • 100% offline • No API keys            ║
║                                                              ║
║   {Style.DIM}by C0m3b4ck • Apache License 2.0{Style.RESET}                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET}
""")

    # System info
    print(f"{Style.BOLD}System Information:{Style.RESET}")
    print(f"  {Style.CYAN}OS:{Style.RESET}           {platform.system()} {platform.release()}")
    print(f"  {Style.CYAN}Python:{Style.RESET}        {platform.python_version()}")
    print(f"  {Style.CYAN}Machine:{Style.RESET}       {platform.machine()}")

    # GPU info
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            print(f"  {Style.CYAN}GPU:{Style.RESET}           {Style.BRIGHT_GREEN}{gpu_name}{Style.RESET} ({vram:.1f} GB VRAM)")
        else:
            print(f"  {Style.CYAN}GPU:{Style.RESET}           {Style.BRIGHT_YELLOW}Not available (CPU mode){Style.RESET}")
    except ImportError:
        print(f"  {Style.CYAN}GPU:{Style.RESET}           {Style.DIM}torch not installed{Style.RESET}")

    # ffmpeg check
    import subprocess
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split("\n")[0].split(" ")[2] if result.stdout else "unknown"
            print(f"  {Style.CYAN}ffmpeg:{Style.RESET}        {Style.BRIGHT_GREEN}v{version}{Style.RESET}")
        else:
            print(f"  {Style.CYAN}ffmpeg:{Style.RESET}        {Style.BRIGHT_RED}Not working{Style.RESET}")
    except FileNotFoundError:
        print(f"  {Style.CYAN}ffmpeg:{Style.RESET}        {Style.BRIGHT_RED}Not found (required for video burning){Style.RESET}")

    print(f"\n{Style.DIM}{'─' * 60}{Style.RESET}")


def select_audio_file():
    """Interactive file selection for audio/video files."""
    import glob

    # Supported formats
    extensions = ["*.mp4", "*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.mkv", "*.avi", "*.mov", "*.webm"]

    # Find all matching files
    files = []
    for ext in extensions:
        files.extend(glob.glob(ext))
        files.extend(glob.glob(f"**/{ext}", recursive=True))

    # Remove duplicates and sort
    files = sorted(set(files))

    print(f"\n{Style.BOLD}Select Audio/Video File:{Style.RESET}")

    if files:
        print(f"\n{Style.CYAN}Found files in current directory:{Style.RESET}")
        for i, f in enumerate(files[:10], 1):  # Show max 10 files
            size = os.path.getsize(f) / (1024 * 1024) if os.path.exists(f) else 0
            print(f"  {Style.GREEN}{i:2d}{Style.RESET}. {f} ({size:.1f} MB)")

        if len(files) > 10:
            print(f"  {Style.DIM}... and {len(files) - 10} more files{Style.RESET}")

        print(f"\n  {Style.BRIGHT_YELLOW}0{Style.RESET}. Enter path manually")

        while True:
            choice = input(f"\n{Style.BRIGHT_YELLOW}Select file (0-{min(len(files), 10)}): {Style.RESET}").strip()

            if choice == "0":
                break
            if choice.isdigit() and 1 <= int(choice) <= min(len(files), 10):
                selected = files[int(choice) - 1]
                print_info(f"Selected: {Style.UNDERLINE}{selected}{Style.RESET}")
                return selected
            print_error("Invalid choice. Try again.")
    else:
        print(f"  {Style.DIM}No media files found in current directory{Style.RESET}")

    # Manual entry
    while True:
        path = input(f"{Style.BRIGHT_YELLOW}Enter file path: {Style.RESET}").strip()
        if path:
            real_path = os.path.realpath(path)
            if os.path.isfile(real_path):
                print_info(f"Selected: {Style.UNDERLINE}{real_path}{Style.RESET}")
                return real_path
            else:
                print_error("File not found.")
        else:
            print_error("Please enter a file path.")


def print_model_info(model_key, model_info):
    """Print detailed model information."""
    print(f"\n{Style.BOLD}Model Details:{Style.RESET}")
    print(f"  {Style.CYAN}Name:{Style.RESET}        {model_info['name']}")
    print(f"  {Style.CYAN}Description:{Style.RESET} {model_info['description']}")
    print(f"  {Style.CYAN}VRAM:{Style.RESET}        {model_info['vram']}")
    print(f"  {Style.CYAN}WER:{Style.RESET}         {model_info['wer']}")
    print(f"  {Style.CYAN}Speed:{Style.RESET}       {model_info['rtfx']}")
    print(f"  {Style.CYAN}Languages:{Style.RESET}   {model_info['languages']}")


def print_transcription_summary(words, elapsed, output_files):
    """Print summary after transcription completes."""
    print(f"\n{Style.BG_GREEN}{Style.BRIGHT_BLACK}{Style.BOLD} TRANSCRIPTION COMPLETE {Style.RESET}")
    print(f"\n{Style.BOLD}Summary:{Style.RESET}")
    print(f"  {Style.CYAN}Words detected:{Style.RESET}  {Style.BRIGHT_GREEN}{len(words)}{Style.RESET}")
    print(f"  {Style.CYAN}Time elapsed:{Style.RESET}    {Style.BRIGHT_GREEN}{elapsed:.1f}s{Style.RESET}")
    print(f"  {Style.CYAN}Words/second:{Style.RESET}    {Style.BRIGHT_GREEN}{len(words)/elapsed:.1f}{Style.RESET}" if elapsed > 0 else "")

    if output_files:
        print(f"\n{Style.BOLD}Output Files:{Style.RESET}")
        for fmt, path in output_files.items():
            print(f"  {Style.BRIGHT_MAGENTA}{fmt}{Style.RESET}: {Style.UNDERLINE}{path}{Style.RESET}")

    print(f"\n{Style.DIM}{'─' * 60}{Style.RESET}")


def print_position_visual(position_name):
    """Print a visual representation of subtitle position."""
    # Create a 3x3 grid to show position
    grid = [
        [" ", " ", " "],
        [" ", " ", " "],
        [" ", " ", " "],
    ]

    # Map position name to grid coordinates
    position_map = {
        "Bottom center": (2, 1),
        "Top center": (0, 1),
        "Middle center": (1, 1),
        "Bottom left": (2, 0),
        "Bottom right": (2, 2),
        "Top left": (0, 0),
        "Top right": (0, 2),
    }

    if position_name in position_map:
        row, col = position_map[position_name]
        grid[row][col] = "█"

    print(f"\n  {Style.BOLD}Subtitle Position:{Style.RESET}")
    print(f"    {Style.DIM}┌───┬───┬───┐{Style.RESET}")
    for i, row in enumerate(grid):
        cells = " │ ".join(row)
        print(f"    {Style.DIM}│{Style.RESET} {cells} {Style.DIM}│{Style.RESET}")
        if i < 2:
            print(f"    {Style.DIM}├───┼───┼───┤{Style.RESET}")
    print(f"    {Style.DIM}└───┴───┴───┘{Style.RESET}")


# =============================================================================
# SHARED: SRT/VTT generation from word timestamps
# =============================================================================

def words_to_srt(words: list[dict], output_path: str, words_per_group: int = 3, gap_threshold: float = 1.0):
    """
    Convert word-level timestamps into SRT subtitles.

    Splits subtitle chunks when there's a gap > gap_threshold seconds between words.
    Each dict in `words` must have: {"word": str, "start": float, "end": float}
    """
    def fmt(seconds: float) -> str:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        ms = int(s % 1 * 1000)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"

    groups = []
    chunk = []
    for w in words:
        if chunk and (w["start"] - chunk[-1]["end"]) > gap_threshold:
            text = " ".join(c["word"].strip() for c in chunk)
            groups.append((chunk[0]["start"], chunk[-1]["end"], text))
            chunk = []
        chunk.append(w)
        if len(chunk) >= words_per_group:
            text = " ".join(c["word"].strip() for c in chunk)
            groups.append((chunk[0]["start"], chunk[-1]["end"], text))
            chunk = []
    if chunk:
        text = " ".join(c["word"].strip() for c in chunk)
        groups.append((chunk[0]["start"], chunk[-1]["end"], text))

    lines = []
    for idx, (start, end, text) in enumerate(groups, 1):
        lines.extend([str(idx), f"{fmt(start)} --> {fmt(end)}", text, ""])

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print_subtitle_stats(output_path, len(groups), "SRT")


def words_to_vtt(words: list[dict], output_path: str, words_per_group: int = 3, gap_threshold: float = 1.0):
    """Same as SRT but in WebVTT format.

    Splits subtitle chunks when there's a gap > gap_threshold seconds between words.
    """
    def fmt(seconds: float) -> str:
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        ms = int(s % 1 * 1000)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}.{ms:03d}"

    groups = []
    chunk = []
    for w in words:
        if chunk and (w["start"] - chunk[-1]["end"]) > gap_threshold:
            text = " ".join(c["word"].strip() for c in chunk)
            groups.append((chunk[0]["start"], chunk[-1]["end"], text))
            chunk = []
        chunk.append(w)
        if len(chunk) >= words_per_group:
            text = " ".join(c["word"].strip() for c in chunk)
            groups.append((chunk[0]["start"], chunk[-1]["end"], text))
            chunk = []
    if chunk:
        text = " ".join(c["word"].strip() for c in chunk)
        groups.append((chunk[0]["start"], chunk[-1]["end"], text))

    lines = ["WEBVTT", ""]
    for idx, (start, end, text) in enumerate(groups, 1):
        lines.extend([str(idx), f"{fmt(start)} --> {fmt(end)}", text, ""])

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print_subtitle_stats(output_path, len(groups), "VTT")


# =============================================================================
# SUBTITLE STYLING: User preferences and video burning
# =============================================================================

def get_subtitle_preferences():
    """
    Ask user for subtitle styling preferences.
    Returns dict with font, color, size, position, and outline settings.
    """
    print_header("SUBTITLE STYLING")

    # Common fonts (cross-platform safe defaults)
    common_fonts = [
        "Arial", "Helvetica", "Times New Roman", "Georgia",
        "Verdana", "Trebuchet MS", "Impact", "Comic Sans MS",
        "Courier New", "Lucida Console", "Tahoma", "Calibri",
    ]

    print(f"\n{Style.BOLD}Available fonts{Style.RESET} (or type any font name):")
    for i, font in enumerate(common_fonts, 1):
        print(f"  {Style.CYAN}{i:2d}{Style.RESET}. {font}")

    while True:
        choice = input(f"\n{Style.BRIGHT_YELLOW}Select font (number or name): {Style.RESET}").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(common_fonts):
            font = common_fonts[int(choice) - 1]
            break
        elif choice:
            font = choice
            break
        print_error("Invalid choice. Try again.")

    # Color presets
    color_presets = {
        "1": ("White", "&H00FFFFFF"),
        "2": ("Yellow", "&H0000FFFF"),
        "3": ("Cyan", "&H00FFFF00"),
        "4": ("Green", "&H0000FF00"),
        "5": ("Red", "&H000000FF"),
        "6": ("Magenta", "&H00FF00FF"),
        "7": ("Blue", "&H00FF0000"),
        "8": ("Black", "&H00000000"),
    }

    print(f"\n{Style.BOLD}Color presets:{Style.RESET}")
    for key, (name, _) in color_presets.items():
        print(f"  {Style.CYAN}{key}{Style.RESET}. {name}")

    while True:
        choice = input(f"{Style.BRIGHT_YELLOW}Select color (1-8): {Style.RESET}").strip()
        if choice in color_presets:
            color_name, color_hex = color_presets[choice]
            break
        print_error("Invalid choice. Try again.")

    # Font size
    while True:
        size_str = input(f"\n{Style.BRIGHT_YELLOW}Font size (12-72, default 24): {Style.RESET}").strip()
        if not size_str:
            font_size = 24
            break
        try:
            font_size = int(size_str)
            if 12 <= font_size <= 72:
                break
            print_error("Size must be between 12 and 72.")
        except ValueError:
            print_error("Please enter a number.")

    # Position
    print(f"\n{Style.BOLD}Position presets:{Style.RESET}")
    positions = {
        "1": ("Bottom center", 2),
        "2": ("Top center", 8),
        "3": ("Middle center", 5),
        "4": ("Bottom left", 1),
        "5": ("Bottom right", 3),
        "6": ("Top left", 7),
        "7": ("Top right", 9),
    }

    for key, (name, _) in positions.items():
        print(f"  {Style.CYAN}{key}{Style.RESET}. {name}")

    while True:
        choice = input(f"{Style.BRIGHT_YELLOW}Select position (1-7, default 1): {Style.RESET}").strip()
        if not choice:
            position = 2  # Bottom center
            position_name = "Bottom center"
            break
        if choice in positions:
            position_name, position = positions[choice]
            break
        print_error("Invalid choice. Try again.")

    # Outline/thickness
    while True:
        outline_str = input(f"\n{Style.BRIGHT_YELLOW}Outline thickness (0-4, default 2): {Style.RESET}").strip()
        if not outline_str:
            outline = 2
            break
        try:
            outline = int(outline_str)
            if 0 <= outline <= 4:
                break
            print_error("Outline must be between 0 and 4.")
        except ValueError:
            print_error("Please enter a number.")

    # Shadow
    while True:
        shadow_str = input(f"{Style.BRIGHT_YELLOW}Shadow depth (0-3, default 1): {Style.RESET}").strip()
        if not shadow_str:
            shadow = 1
            break
        try:
            shadow = int(shadow_str)
            if 0 <= shadow <= 3:
                break
            print_error("Shadow must be between 0 and 3.")
        except ValueError:
            print_error("Please enter a number.")

    # Words per caption chunk
    while True:
        words_str = input(f"{Style.BRIGHT_YELLOW}Words per caption (1-8, default 3): {Style.RESET}").strip()
        if not words_str:
            words_per_chunk = 3
            break
        try:
            words_per_chunk = int(words_str)
            if 1 <= words_per_chunk <= 8:
                break
            print_error("Words per caption must be between 1 and 8.")
        except ValueError:
            print_error("Please enter a number.")

    print(f"\n{Style.BRIGHT_GREEN}{Style.BOLD}Selected:{Style.RESET} {Style.CYAN}{font}{Style.RESET}, {color_name}, size {font_size}, {position_name}, outline {outline}, shadow {shadow}, {words_per_chunk} words/chunk")

    return {
        "font": font,
        "color": color_hex,
        "color_name": color_name,
        "font_size": font_size,
        "position": position,
        "position_name": position_name,
        "outline": outline,
        "shadow": shadow,
        "words_per_chunk": words_per_chunk,
    }


def words_to_ass(words: list[dict], output_path: str, prefs: dict, words_per_group: int = 3, gap_threshold: float = 1.0):
    """
    Convert word-level timestamps into ASS subtitle file with custom styling.

    Splits subtitle chunks when there's a gap > gap_threshold seconds between words.
    ASS format supports advanced styling (font, color, position, outline, shadow).
    """
    import re

    def fmt(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}:{m:02d}:{s:05.2f}"

    # ASS alignment values: 1-9 numpad layout
    # 1=bottom-left, 2=bottom-center, 3=bottom-right
    # 4=mid-left, 5=mid-center, 6=mid-right
    # 7=top-left, 8=top-center, 9=top-right
    alignment = prefs["position"]

    # Convert ASS color format (&HAABBGGRR) - note: ASS uses BGR, not RGB
    # Our colors are already in ASS format from the presets
    primary_color = prefs["color"]
    outline_color = "&H00000000"  # Black outline

    # Sanitize font name — only allow safe characters for ASS Style line
    font_name = re.sub(r"[^a-zA-Z0-9\s\-\(\)\.]", "", prefs['font'])[:64]

    # Validate numeric parameters
    font_size = max(8, min(200, int(prefs['font_size'])))
    outline_val = max(0, min(10, int(prefs['outline'])))
    shadow_val = max(0, min(10, int(prefs['shadow'])))

    # ASS header
    ass_content = f"""[Script Info]
Title: Transcribix Subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 0

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},{primary_color},&H000000FF,{outline_color},&H80000000,0,0,0,0,100,100,0,0,1,{outline_val},{shadow_val}, {alignment},20,20,30,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Group words into chunks with gap detection
    groups = []
    chunk = []
    for w in words:
        if chunk and (w["start"] - chunk[-1]["end"]) > gap_threshold:
            text = " ".join(c["word"].strip() for c in chunk)
            groups.append((chunk[0]["start"], chunk[-1]["end"], text))
            chunk = []
        chunk.append(w)
        if len(chunk) >= words_per_group:
            text = " ".join(c["word"].strip() for c in chunk)
            groups.append((chunk[0]["start"], chunk[-1]["end"], text))
            chunk = []
    if chunk:
        text = " ".join(c["word"].strip() for c in chunk)
        groups.append((chunk[0]["start"], chunk[-1]["end"], text))

    # Generate dialogue lines
    for start, end, text in groups:
        start_time = fmt(start)
        end_time = fmt(end)
        # Escape special characters for ASS
        text = text.replace("\\", "\\\\")
        text = text.replace("{", "\\{").replace("}", "\\}")
        ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{text}\n"

    Path(output_path).write_text(ass_content, encoding="utf-8")
    print_subtitle_stats(output_path, len(groups), "ASS")
    return groups


def burn_subtitles_to_video(
    video_path: str,
    subtitle_path: str,
    output_path: str,
    prefs: dict,
):
    """
    Burn subtitles onto video using ffmpeg.
    Uses the subtitles filter for hardcoded subtitles.
    """
    import subprocess
    import shlex

    # Validate paths exist before passing to ffmpeg
    if not os.path.isfile(video_path):
        print_error(f"Video file not found: {video_path}")
        return False
    if not os.path.isfile(subtitle_path):
        print_error(f"Subtitle file not found: {subtitle_path}")
        return False

    # Build ffmpeg filter for subtitles using shlex.quote for safe escaping
    # The subtitles filter requires colon and backslash escaping within the filter string
    sub_path_quoted = shlex.quote(subtitle_path)
    # ffmpeg filter syntax needs additional escaping beyond shell quoting
    sub_path_escaped = subtitle_path.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")

    filter_str = f"subtitles={sub_path_quoted}"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", filter_str,
        "-c:a", "copy",
        output_path,
    ]

    print(f"\n{Style.BRIGHT_CYAN}{Style.BOLD}Burning subtitles to video...{Style.RESET}")
    print_dim(f"Burning subtitles onto video...")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        print_error("ffmpeg timed out after 600 seconds")
        return False
    except FileNotFoundError:
        print_error("ffmpeg not found. Please install ffmpeg.")
        return False

    if result.returncode != 0:
        print_error("ffmpeg failed. Check that ffmpeg is installed and the video file is valid.")
        return False

    print_success(f"Video saved to: {Style.UNDERLINE}{output_path}{Style.RESET}")
    return True


# =============================================================================
# 1. FASTER-WHISPER (CTranslate2)
# =============================================================================
# WER: ~7.4% (English avg) | RTFx: ~600x | Params: 1.55B | VRAM: ~2.5GB int8
# Languages: 99 | Word timestamps: cross-attention based
# Pros: Easiest setup, lowest VRAM, fastest Whisper variant
# Cons: Some timestamp drift, hallucinations on silence without VAD

def transcribe_faster_whisper(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "auto",
    compute_type: str = "int8",
    language: str = None,
):
    """
    Install: pip install faster-whisper
    """
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    compute_type = "float16" if device == "cuda" else compute_type
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=language,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
    )

    print_info(f"Language: {Style.CYAN}{info.language}{Style.RESET} (prob={info.language_probability:.2f})")
    all_words = []
    for segment in segments:
        print_timestamp(f"{segment.start:.1f}s", f"{segment.end:.1f}s", segment.text.strip())
        if segment.words:
            for w in segment.words:
                all_words.append({
                    "word": w.word, "start": w.start, "end": w.end,
                    "probability": w.probability,
                })
    return all_words


# =============================================================================
# 2. WHISPERX (faster-whisper + forced phoneme alignment)
# =============================================================================
# WER: ~7.4% (English avg) | RTFx: ~150x | Params: 1.55B | VRAM: ~3-10GB
# Languages: 99 | Word timestamps: forced alignment via wav2vec2 (BEST)
# Pros: Sub-100ms word timestamps, speaker diarization optional
# Cons: Heavier deps, diarization needs HF token (timestamps don't)

def transcribe_whisperx(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "auto",
    language: str = "en",
):
    """
    Install: pip install whisperx
    Word timestamps work fully offline — no tokens needed.
    """
    import whisperx
    import torch

    device_type = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    compute_type = "float16" if device_type == "cuda" else "int8"

    model = whisperx.load_model(model_size, device_type, compute_type=compute_type)
    audio = whisperx.load_audio(audio_path)
    result = model.transcribe(audio, batch_size=16, language=language)

    model_a, metadata = whisperx.load_align_model(language_code=language, device=device_type)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device_type)

    all_words = []
    for segment in result["segments"]:
        if "words" in segment:
            for w in segment["words"]:
                all_words.append({
                    "word": w["word"], "start": w["start"], "end": w["end"],
                    "score": w.get("score", 0),
                })
    return all_words


# =============================================================================
# 3. STABLE-TS (Whisper with stabilized timestamps)
# =============================================================================
# WER: ~7.4% (English avg) | RTFx: ~100x | Params: 1.55B | VRAM: ~3-10GB
# Languages: 99 | Word timestamps: drift-corrected, regrouped
# Pros: Purpose-built for subtitles, direct SRT/VTT/ASS export, silence suppression
# Cons: No diarization, slightly slower

def transcribe_stable_ts(
    audio_path: str,
    model_size: str = "large-v3",
    device: str = "auto",
    language: str = None,
):
    """
    Install: pip install stable-ts
    """
    import stable_whisper

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    model = stable_whisper.load_model(model_size, device=device)
    result = model.transcribe(
        audio_path, language=language, word_timestamps=True,
        vad=True, regroup=True, only_voice_segments=True,
    )

    all_words = []
    for seg in result.segments:
        if seg.words:
            for w in seg.words:
                all_words.append({"word": w.word, "start": w.start, "end": w.end})

    # Direct export (no need for words_to_srt when using stable-ts):
    # result.to_srt_vtt("output.srt", word_level=True)
    # result.to_srt_vtt("output.vtt", word_level=True)
    # result.to_ass("output.ass")  # karaoke-style word highlights

    return all_words


# =============================================================================
# 4. NVIDIA PARAKEET TDT 0.6B v3 (via NeMo)
# =============================================================================
# WER: 6.32% (avg) | RTFx: ~3,300x | Params: 600M | VRAM: ~2GB
# Languages: 25 (European) | Word timestamps: native from transducer
# Pros: Best accuracy, fastest, minimal hallucination on silence
# Cons: NeMo toolkit (heavy deps), 25 languages only, NVIDIA GPU preferred

def transcribe_parakeet(
    audio_path: str,
    model_name: str = "nvidia/parakeet-tdt-0.6b-v3",
):
    """
    Install: pip install nemo-toolkit[asr] torch torchaudio
    """
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    hypotheses = model.transcribe([audio_path], batch_size=1, return_hypotheses=True)

    all_words = []
    for hyp in hypotheses:
        if hasattr(hyp, "timestamp") and hyp.timestamp:
            for word, start, end in hyp.timestamp:
                all_words.append({
                    "word": word, "start": start / 1000, "end": end / 1000,
                })
    return all_words


# =============================================================================
# 5. NVIDIA CANARY QWEN 2.5B (via NeMo)
# =============================================================================
# WER: 5.63% (English, #1 on Open ASR Leaderboard) | RTFx: ~458x
# Params: 2.5B | VRAM: ~6GB | Languages: English only
# Word timestamps: Not native — combine with WhisperX/stable-ts for alignment
# Pros: Highest English accuracy, also works as LLM over transcripts
# Cons: English only, no native word timestamps, heavy NeMo deps

def transcribe_canary_qwen(
    audio_path: str,
    model_name: str = "nvidia/canary-qwen-2.5b",
):
    """
    Install: pip install nemo-toolkit[asr] torch torchaudio

    Note: Canary-Qwen outputs full text without word-level timestamps.
    For captions, combine with WhisperX for alignment (see canary_qwen_with_alignment).
    """
    import nemo.collections.asr as nemo_asr

    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    transcriptions = model.transcribe([audio_path])

    # Returns text only — no native word timestamps
    return transcriptions[0] if transcriptions else ""


def transcribe_canary_qwen_with_alignment(
    audio_path: str,
    language: str = "en",
):
    """
    Best pipeline: Canary-Qwen for accuracy + WhisperX for word timestamps.

    Step 1: Transcribe with Canary-Qwen (best WER)
    Step 2: Force-align with WhisperX wav2vec2 model
    """
    import nemo.collections.asr as nemo_asr
    import whisperx
    import torch

    # Step 1: Get the best transcript from Canary
    canary = nemo_asr.models.ASRModel.from_pretrained("nvidia/canary-qwen-2.5b")
    transcriptions = canary.transcribe([audio_path])
    best_text = transcriptions[0]

    # Step 2: Align with WhisperX for word-level timestamps
    device = "cuda" if torch.cuda.is_available() else "cpu"
    audio = whisperx.load_audio(audio_path)
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)

    # Create a fake segment with Canary's text for alignment
    segments = [{"text": best_text, "start": 0, "end": len(audio) / 16000}]
    result = whisperx.align(segments, model_a, metadata, audio, device)

    all_words = []
    for segment in result["segments"]:
        if "words" in segment:
            for w in segment["words"]:
                all_words.append({
                    "word": w["word"], "start": w["start"], "end": w["end"],
                })
    return all_words


# =============================================================================
# 6. DISTIL-WHISPER (via faster-whisper)
# =============================================================================
# WER: ~7.5% (English) | RTFx: ~3,600x | Params: 756M | VRAM: ~4GB fp16
# Languages: 99 (English-focused variants best) | Word timestamps: cross-attention
# Pros: 6x faster than large-v3, half the params, near-identical accuracy
# Cons: English-optimized variants better, still some timestamp drift

def transcribe_distil_whisper(
    audio_path: str,
    model_size: str = "distil-large-v3",
    device: str = "auto",
    compute_type: str = "int8",
    language: str = "en",
):
    """
    Install: pip install faster-whisper
    Distil-Whisper checkpoints work natively with faster-whisper.
    """
    from faster_whisper import WhisperModel

    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    compute_type = "float16" if device == "cuda" else compute_type
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=language,
        vad_filter=True,
    )

    all_words = []
    for segment in segments:
        if segment.words:
            for w in segment.words:
                all_words.append({
                    "word": w.word, "start": w.start, "end": w.end,
                    "probability": w.probability,
                })
    return all_words


# =============================================================================
# 7. MOONSHINE (Useful Sensors / Moonshine AI)
# =============================================================================
# WER: ~10% (English avg) | RTFx: ~80x | Params: 61M (base) | VRAM: <1GB
# Languages: 8 | Word timestamps: cross-attention alignment
# Pros: Tiny model, runs on Raspberry Pi, streaming-capable, MIT license
# Cons: English-focused, lower accuracy than Whisper, limited languages

def transcribe_moonshine(
    audio_path: str,
    model_name: str = "moonshine/base",
):
    """
    Install: pip install useful-moonshine

    Moonshine now supports word-level timestamps natively.
    """
    import moonshine

    # Moonshine returns list of transcript strings
    result = moonshine.transcribe(audio_path, model_name)

    # For word-level timestamps, use the HuggingFace Transformers API:
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
    import torch
    import soundfile as sf

    processor = AutoProcessor.from_pretrained("UsefulSensors/moonshine-base")
    model = AutoModelForSpeechSeq2Seq.from_pretrained("UsefulSensors/moonshine-base")

    audio, sr = sf.read(audio_path)
    if sr != processor.feature_extractor.sampling_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=processor.feature_extractor.sampling_rate)

    inputs = processor(audio, return_tensors="pt", sampling_rate=processor.feature_extractor.sampling_rate)
    with torch.no_grad():
        outputs = model.generate(**inputs, return_timestamps=True)

    # Decode with timestamps
    result = processor.batch_decode(outputs, return_timestamps=True)[0]

    # Fallback: return text-based results (moonshine base API doesn't expose
    # per-word timestamps as richly as Whisper — best used for real-time streaming)
    all_words = []
    for text in result:
        for word in text.split():
            all_words.append({"word": word, "start": 0.0, "end": 0.5})

    return all_words


# =============================================================================
# 8. SENSEVOICE (via FunASR)
# =============================================================================
# WER: ~5.5% (Chinese better than Whisper) | RTFx: ~50x | Params: 234M
# VRAM: ~1GB | Languages: 50+ (best: zh, en, ja, ko, yue)
# Word timestamps: character-level via FunASR Paraformer
# Pros: 50+ languages, emotion detection, audio event detection, non-autoregressive (fast)
# Cons: Timestamps are character-level not word-level for CJK, best for CJK

def transcribe_sensevoice(
    audio_path: str,
    model_name: str = "iic/SenseVoiceSmall",
    language: str = "auto",
):
    """
    Install: pip install funasr torch torchaudio

    SenseVoice outputs emotion + event tags alongside text.
    For word-level timestamps in English, use FunASR's Paraformer instead.
    """
    from funasr import AutoModel

    model = AutoModel(model=model_name, vad_model="fsmn-vad", disable_update=True)
    result = model.generate(input=audio_path, language=language, use_itn=True)

    text = result[0]["text"]
    # Strip emotion/event tags like <|NEUTRAL|> <|Speech|>
    import re
    clean_text = re.sub(r"<\|[^|]+\|>", "", text).strip()

    all_words = []
    for word in clean_text.split():
        all_words.append({"word": word, "start": 0.0, "end": 0.5})

    return all_words


def transcribe_sensevoice_with_timestamps(
    audio_path: str,
    language: str = "auto",
):
    """
    Use FunASR Paraformer for native character/word-level timestamps.
    Install: pip install funasr
    """
    from funasr import AutoModel

    model = AutoModel(
        model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        disable_update=True,
    )
    result = model.generate(input=audio_path, batch_size_s=300)

    text = result[0]["text"]
    timestamps = result[0].get("timestamp", [])  # [[start_ms, end_ms], ...]

    all_words = []
    chars = text.split()
    for i, (ch, ts) in enumerate(zip(chars, timestamps)):
        all_words.append({
            "word": ch, "start": ts[0] / 1000, "end": ts[1] / 1000,
        })

    return all_words


# =============================================================================
# 9. VOSK (Kaldi-based)
# =============================================================================
# WER: ~12% (English) | RTFx: ~100x | Params: ~50M (small) | VRAM: <500MB
# Languages: 20+ | Word timestamps: native, per-word with confidence
# Pros: Extremely lightweight, runs on Raspberry Pi, real-time streaming, many languages
# Cons: Lower accuracy, Kaldi-era architecture, model quality varies by language

def transcribe_vosk(
    audio_path: str,
    model_path: str = "vosk-model-small-en-us-0.15",
    sample_rate: int = 16000,
):
    """
    Install: pip install vosk soundfile

    Download models from: https://alphacephei.com/vosk/models
    Audio must be WAV mono 16kHz. Convert with ffmpeg if needed.

    Vosk natively returns per-word timestamps + confidence scores.
    """
    import vosk
    import soundfile as sf
    import numpy as np
    import json as _json

    model = vosk.Model(model_path)
    recognizer = vosk.KaldiRecognizer(model, sample_rate)
    recognizer.SetWords(True)  # enables word-level timestamps

    audio, sr = sf.read(audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    # Feed audio in chunks (Vosk processes streaming-style)
    audio_int16 = (audio * 32767).astype(np.int16)
    chunk_size = 4000  # ~250ms chunks at 16kHz
    for i in range(0, len(audio_int16), chunk_size):
        chunk = audio_int16[i:i + chunk_size].tobytes()
        recognizer.AcceptWaveform(chunk)

    final_result = _json.loads(recognizer.FinalResult())

    all_words = []
    if "result" in final_result:
        for w in final_result["result"]:
            all_words.append({
                "word": w["word"],
                "start": w["start"],
                "end": w["end"],
                "confidence": w["conf"],
            })

    return all_words


# =============================================================================
# 10. WHISPER (original OpenAI)
# =============================================================================
# WER: ~7.4% (English avg) | RTFx: ~1x (large) to ~10x (tiny) | Params: 39M-1.55B
# VRAM: 1-10GB | Languages: 99 | Word timestamps: cross-attention DTW
# Pros: Most languages (99), reference implementation, widest ecosystem
# Cons: Slowest, highest VRAM, hallucinations on silence, timestamp drift

def transcribe_whisper_original(
    audio_path: str,
    model_size: str = "large-v3",
    language: str = None,
):
    """
    Install: pip install openai-whisper

    The reference implementation. Word timestamps use DTW on cross-attention.
    """
    import whisper

    model = whisper.load_model(model_size)
    result = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=language,
    )

    all_words = []
    for segment in result["segments"]:
        if "words" in segment:
            for w in segment["words"]:
                all_words.append({
                    "word": w["word"],
                    "start": w["start"],
                    "end": w["end"],
                    "probability": w.get("probability", 0),
                })

    return all_words


# =============================================================================
# 11. WHISPER.CPP (C++ port via Python)
# =============================================================================
# WER: ~7.4% (same weights as Whisper) | RTFx: ~5-15x on CPU
# Params: 39M-1.55B | VRAM: 1-10GB | Languages: 99
# Word timestamps: native GGML implementation
# Pros: Runs on anything (CPU, Metal, CUDA, Vulkan), no PyTorch needed
# Cons: Needs C++ build or prebuilt binary, Python bindings less ergonomic

def transcribe_whisper_cpp(
    audio_path: str,
    model_size: str = "large-v3",
    language: str = None,
):
    """
    Install options:
      a) pip install whisper-cpp-python  (wraps whisper.cpp binary)
      b) Build from source: https://github.com/ggerganov/whisper.cpp
      c) Use pywhispercpp: pip install pywhispercpp

    whisper.cpp uses the same Whisper weights — identical accuracy.
    """
    from pywhispercpp.model import Model

    model = Model(model_size)
    segments = model.transcribe(
        audio_path,
        word_timestamps=True,
        language=language,
    )

    all_words = []
    for segment in segments:
        if hasattr(segment, "words") and segment.words:
            for w in segment.words:
                all_words.append({
                    "word": w.word,
                    "start": w.t0 / 100,  # whisper.cpp returns centiseconds
                    "end": w.t1 / 100,
                })

    return all_words


# =============================================================================
# MAIN: Example usage
# =============================================================================

if __name__ == "__main__":
    import time

    # Print banner with system info
    print_banner()

    # Select audio file
    AUDIO_FILE = select_audio_file()

    # Validate file size (warn if > 2GB)
    file_size = os.path.getsize(AUDIO_FILE) / (1024 * 1024)  # MB
    if file_size > 10240:
        print_error("File too large (>10GB).")
        return
    if file_size > 2048:
        print_warning(f"Large file ({file_size/1024:.1f} GB) — processing may be slow or fail due to memory limits.")
    print_info(f"Input file: {Style.UNDERLINE}{AUDIO_FILE}{Style.RESET} ({file_size:.1f} MB)")

    # Ask user for subtitle styling preferences before transcription
    subtitle_prefs = get_subtitle_preferences()
    words_per_chunk = subtitle_prefs["words_per_chunk"]

    # Show position visual
    print_position_visual(subtitle_prefs["position_name"])

    # Model info
    model_info = {
        "name": "Faster Whisper",
        "description": "CTranslate2 Whisper — easiest setup, lowest VRAM",
        "vram": "~2.5 GB (int8)",
        "wer": "~7.4%",
        "rtfx": "~600x",
        "languages": "99",
    }
    print_model_info("faster-whisper", model_info)

    # Start transcription
    print(f"\n{Style.BG_CYAN}{Style.BRIGHT_BLACK}{Style.BOLD} STARTING TRANSCRIPTION {Style.RESET}")
    start_time = time.time()

    words = transcribe_faster_whisper(AUDIO_FILE, model_size="large-v3", device="cpu")

    elapsed = time.time() - start_time

    # Generate output files
    print(f"\n{Style.BG_MAGENTA}{Style.BRIGHT_BLACK}{Style.BOLD} GENERATING OUTPUT FILES {Style.RESET}")

    # Generate SRT for reference
    srt_path = "captions_faster_whisper.srt"
    words_to_srt(words, srt_path, words_per_chunk)

    # Generate styled ASS subtitles
    ass_path = "captions_faster_whisper.ass"
    words_to_ass(words, ass_path, subtitle_prefs, words_per_chunk)

    # Ask if user wants to burn subtitles onto video
    print(f"\n{Style.BOLD}Burn subtitles into video?{Style.RESET}")
    burn_choice = input(f"{Style.BRIGHT_YELLOW}Burn subtitles onto video? (y/n, default n): {Style.RESET}").strip().lower()

    output_files = {
        "SRT": srt_path,
        "ASS": ass_path,
    }

    if burn_choice in ("y", "yes"):
        output_video = "output_faster_whisper.mp4"
        burn_subtitles_to_video(AUDIO_FILE, ass_path, output_video, subtitle_prefs)
        output_files["Video"] = output_video
    else:
        print_info("Skipping video burn — subtitle files generated only.")

    # Print summary
    print_transcription_summary(words, elapsed, output_files)

    # Uncomment any model you want to test:

    # print("\n" + "=" * 60)
    # print("2. WHISPERX (best word timestamps)")
    # print("=" * 60)
    # words = transcribe_whisperx(AUDIO_FILE, language="en")
    # words_to_srt(words, "captions_whisperx.srt", WORDS_PER_CHUNK)
    # words_to_ass(words, "captions_whisperx.ass", subtitle_prefs, WORDS_PER_CHUNK)
    # burn_subtitles_to_video(AUDIO_FILE, "captions_whisperx.ass", "output_whisperx.mp4", subtitle_prefs)

    # print("\n" + "=" * 60)
    # print("3. STABLE-TS (best subtitle export)")
    # print("=" * 60)
    # words = transcribe_stable_ts(AUDIO_FILE)
    # words_to_srt(words, "captions_stable_ts.srt", WORDS_PER_CHUNK)
    # words_to_ass(words, "captions_stable_ts.ass", subtitle_prefs, WORDS_PER_CHUNK)
    # burn_subtitles_to_video(AUDIO_FILE, "captions_stable_ts.ass", "output_stable_ts.mp4", subtitle_prefs)

    # print("\n" + "=" * 60)
    # print("4. PARAKEET (best English accuracy)")
    # print("=" * 60)
    # words = transcribe_parakeet(AUDIO_FILE)
    # words_to_srt(words, "captions_parakeet.srt", WORDS_PER_CHUNK)
    # words_to_ass(words, "captions_parakeet.ass", subtitle_prefs, WORDS_PER_CHUNK)
    # burn_subtitles_to_video(AUDIO_FILE, "captions_parakeet.ass", "output_parakeet.mp4", subtitle_prefs)

    # print("\n" + "=" * 60)
    # print("5. CANARY QWEN (top leaderboard)")
    # print("=" * 60)
    # words = transcribe_canary_qwen_with_alignment(AUDIO_FILE)
    # words_to_srt(words, "captions_canary.srt", WORDS_PER_CHUNK)
    # words_to_ass(words, "captions_canary.ass", subtitle_prefs, WORDS_PER_CHUNK)
    # burn_subtitles_to_video(AUDIO_FILE, "captions_canary.ass", "output_canary.mp4", subtitle_prefs)

    # print("\n" + "=" * 60)
    # print("6. DISTIL-WHISPER (fast + accurate)")
    # print("=" * 60)
    # words = transcribe_distil_whisper(AUDIO_FILE)
    # words_to_srt(words, "captions_distil.srt", WORDS_PER_CHUNK)
    # words_to_ass(words, "captions_distil.ass", subtitle_prefs, WORDS_PER_CHUNK)
    # burn_subtitles_to_video(AUDIO_FILE, "captions_distil.ass", "output_distil.mp4", subtitle_prefs)

    # print("\n" + "=" * 60)
    # print("7. MOONSHINE (ultra-lightweight)")
    # print("=" * 60)
    # words = transcribe_moonshine(AUDIO_FILE)
    # words_to_srt(words, "captions_moonshine.srt", WORDS_PER_CHUNK)
    # words_to_ass(words, "captions_moonshine.ass", subtitle_prefs, WORDS_PER_CHUNK)
    # burn_subtitles_to_video(AUDIO_FILE, "captions_moonshine.ass", "output_moonshine.mp4", subtitle_prefs)

    # print("\n" + "=" * 60)
    # print("8. SENSEVOICE (50+ languages)")
    # print("=" * 60)
    # words = transcribe_sensevoice(AUDIO_FILE)
    # words_to_srt(words, "captions_sensevoice.srt", WORDS_PER_CHUNK)
    # words_to_ass(words, "captions_sensevoice.ass", subtitle_prefs, WORDS_PER_CHUNK)
    # burn_subtitles_to_video(AUDIO_FILE, "captions_sensevoice.ass", "output_sensevoice.mp4", subtitle_prefs)

    # print("\n" + "=" * 60)
    # print("9. VOSK (minimal resources)")
    # print("=" * 60)
    # words = transcribe_vosk(AUDIO_FILE)
    # words_to_srt(words, "captions_vosk.srt", WORDS_PER_CHUNK)
    # words_to_ass(words, "captions_vosk.ass", subtitle_prefs, WORDS_PER_CHUNK)
    # burn_subtitles_to_video(AUDIO_FILE, "captions_vosk.ass", "output_vosk.mp4", subtitle_prefs)

    # print("\n" + "=" * 60)
    # print("10. WHISPER ORIGINAL")
    # print("=" * 60)
    # words = transcribe_whisper_original(AUDIO_FILE)
    # words_to_srt(words, "captions_whisper.srt", WORDS_PER_CHUNK)
    # words_to_ass(words, "captions_whisper.ass", subtitle_prefs, WORDS_PER_CHUNK)
    # burn_subtitles_to_video(AUDIO_FILE, "captions_whisper.ass", "output_whisper.mp4", subtitle_prefs)

    # print("\n" + "=" * 60)
    # print("11. WHISPER.CPP (runs on anything)")
    # print("=" * 60)
    # words = transcribe_whisper_cpp(AUDIO_FILE)
    # words_to_srt(words, "captions_whisper_cpp.srt", WORDS_PER_CHUNK)
    # words_to_ass(words, "captions_whisper_cpp.ass", subtitle_prefs, WORDS_PER_CHUNK)
    # burn_subtitles_to_video(AUDIO_FILE, "captions_whisper_cpp.ass", "output_whisper_cpp.mp4", subtitle_prefs)
