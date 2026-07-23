import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr

# Import from the main script
from captioning_comparison import (
    transcribe_faster_whisper,
    transcribe_whisperx,
    transcribe_stable_ts,
    transcribe_parakeet,
    transcribe_canary_qwen,
    transcribe_canary_qwen_with_alignment,
    transcribe_distil_whisper,
    transcribe_moonshine,
    transcribe_sensevoice,
    transcribe_vosk,
    transcribe_whisper_original,
    transcribe_whisper_cpp,
    words_to_srt,
    words_to_vtt,
    words_to_ass,
    burn_subtitles_to_video,
    Style,
)


# ========================= MODEL REGISTRY =========================

MODEL_REGISTRY = {
    "faster-whisper": {
        "name": "Faster Whisper",
        "description": "CTranslate2 Whisper — easiest setup, lowest VRAM",
        "vram": "~2.5 GB (int8)",
        "wer": "~7.4%",
        "rtfx": "~600x",
        "languages": "99",
        "function": transcribe_faster_whisper,
        "params": {"model_size": "large-v3", "device": "auto"},
    },
    "whisperx": {
        "name": "WhisperX",
        "description": "Forced phoneme alignment — best word timestamps",
        "vram": "~3-10 GB",
        "wer": "~7.4%",
        "rtfx": "~150x",
        "languages": "99",
        "function": transcribe_whisperx,
        "params": {"model_size": "large-v3", "device": "auto", "language": "en"},
    },
    "stable-ts": {
        "name": "Stable-TS",
        "description": "Stabilized timestamps — best for subtitle files",
        "vram": "~3-10 GB",
        "wer": "~7.4%",
        "rtfx": "~100x",
        "languages": "99",
        "function": transcribe_stable_ts,
        "params": {"model_size": "large-v3", "device": "auto"},
    },
    "parakeet": {
        "name": "Parakeet TDT 0.6B",
        "description": "Best English WER, native word timestamps",
        "vram": "~2 GB",
        "wer": "6.32%",
        "rtfx": "~3,300x",
        "languages": "25",
        "function": transcribe_parakeet,
        "params": {"model_name": "nvidia/parakeet-tdt-0.6b-v3"},
    },
    "canary": {
        "name": "Canary Qwen 2.5B",
        "description": "Top leaderboard accuracy, English only",
        "vram": "~6 GB",
        "wer": "5.63%",
        "rtfx": "~458x",
        "languages": "English only",
        "function": transcribe_canary_qwen_with_alignment,
        "params": {"language": "en"},
    },
    "distil-whisper": {
        "name": "Distil-Whisper",
        "description": "6x faster distilled Whisper",
        "vram": "~4 GB (fp16)",
        "wer": "~7.5%",
        "rtfx": "~3,600x",
        "languages": "99",
        "function": transcribe_distil_whisper,
        "params": {"model_size": "distil-large-v3", "device": "auto", "language": "en"},
    },
    "moonshine": {
        "name": "Moonshine",
        "description": "Ultra-lightweight for edge/CPU",
        "vram": "<1 GB",
        "wer": "~10%",
        "rtfx": "~80x",
        "languages": "8",
        "function": transcribe_moonshine,
        "params": {"model_name": "moonshine/base"},
    },
    "sensevoice": {
        "name": "SenseVoice",
        "description": "50+ languages, emotion detection",
        "vram": "~1 GB",
        "wer": "~5.5%",
        "rtfx": "~50x",
        "languages": "50+",
        "function": transcribe_sensevoice,
        "params": {"model_name": "iic/SenseVoiceSmall", "language": "auto"},
    },
    "vosk": {
        "name": "Vosk",
        "description": "Kaldi-based, minimal resources",
        "vram": "<500 MB",
        "wer": "~12%",
        "rtfx": "~100x",
        "languages": "20+",
        "function": transcribe_vosk,
        "params": {"model_path": "vosk-model-small-en-us-0.15"},
    },
    "whisper-original": {
        "name": "Whisper (Original)",
        "description": "The baseline, 99 languages",
        "vram": "1-10 GB",
        "wer": "~7.4%",
        "rtfx": "~1-10x",
        "languages": "99",
        "function": transcribe_whisper_original,
        "params": {"model_size": "large-v3"},
    },
    "whisper-cpp": {
        "name": "whisper.cpp",
        "description": "C++ port, runs on anything",
        "vram": "1-10 GB",
        "wer": "~7.4%",
        "rtfx": "~5-15x (CPU)",
        "languages": "99",
        "function": transcribe_whisper_cpp,
        "params": {"model_size": "large-v3"},
    },
}

MODEL_KEYS = list(MODEL_REGISTRY.keys())


# ========================= SUBTITLE STYLING =========================

FONT_CHOICES = [
    "Arial", "Helvetica", "Times New Roman", "Georgia",
    "Verdana", "Trebuchet MS", "Impact", "Comic Sans MS",
    "Courier New", "Lucida Console", "Tahoma", "Calibri",
]

COLOR_PRESETS = {
    "White": "&H00FFFFFF",
    "Yellow": "&H0000FFFF",
    "Cyan": "&H00FFFF00",
    "Green": "&H0000FF00",
    "Red": "&H000000FF",
    "Magenta": "&H00FF00FF",
    "Blue": "&H00FF0000",
    "Black": "&H00000000",
}

POSITION_PRESETS = {
    "Bottom center": 2,
    "Top center": 8,
    "Middle center": 5,
    "Bottom left": 1,
    "Bottom right": 3,
    "Top left": 7,
    "Top right": 9,
}


# ========================= GRADIO CALLBACKS =========================

def get_device_info():
    """Detect available compute device."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            return f"CUDA: {gpu_name} ({vram:.1f} GB VRAM)"
        else:
            return "CPU (no CUDA detected)"
    except ImportError:
        return "CPU (torch not installed)"


def _sanitize_font_name(font_name: str) -> str:
    """Sanitize font name to prevent ASS injection. Only allow alphanumeric, spaces, and common punctuation."""
    import re
    return re.sub(r"[^a-zA-Z0-9\s\-\(\)\.]", "", font_name)[:64]


def fix_grammar_punctuation(words, words_per_group=3, capitalize_sections=True):
    """Fix grammar and punctuation in transcribed words."""
    if not words:
        return words
    fixed = [dict(w) for w in words]
    i_map = {"i": "I", "i'm": "I'm", "i've": "I've", "i'll": "I'll", "i'd": "I'd"}
    conjunctions = {"and", "but", "or", "so", "because", "although", "however",
                    "therefore", "moreover", "furthermore", "nevertheless",
                    "meanwhile", "otherwise", "instead", "yet", "nor"}
    intro_words = {"well", "now", "so", "yes", "no", "oh", "hey", "okay",
                   "ok", "right", "look", "listen", "basically", "actually",
                   "honestly", "anyway", "anyways", "then",
                   "first", "second", "third", "finally", "also"}
    for i in range(len(fixed)):
        text = fixed[i].get("word", "").strip()
        if not text:
            continue
        lower = text.lower()
        if lower in i_map:
            fixed[i] = dict(fixed[i])
            fixed[i]["word"] = i_map[lower]
            continue
        if i > 0:
            prev = fixed[i - 1].get("word", "").strip()
            if prev and prev[-1] in ".!?":
                fixed[i] = dict(fixed[i])
                fixed[i]["word"] = text[0].upper() + text[1:]
    for i in range(1, len(fixed) - 1):
        text = fixed[i].get("word", "").strip().lower()
        prev = fixed[i - 1].get("word", "").strip()
        if prev and text in conjunctions and prev[-1] not in ",.;:!?":
            fixed[i] = dict(fixed[i])
            fixed[i - 1] = dict(fixed[i - 1])
            fixed[i - 1]["word"] = prev + ","
    for start in range(0, len(fixed), words_per_group):
        for j in range(start, min(start + words_per_group, len(fixed))):
            text = fixed[j].get("word", "").strip().lower()
            if text in intro_words and j + 1 < len(fixed):
                next_w = fixed[j + 1].get("word", "").strip()
                if next_w and next_w[-1] not in ",.;:!?":
                    fixed[j] = dict(fixed[j])
                    fixed[j + 1] = dict(fixed[j + 1])
                    fixed[j + 1]["word"] = next_w + ","
                break
    if fixed and fixed[0].get("word", "").strip():
        t = fixed[0]["word"].strip()
        fixed[0] = dict(fixed[0])
        fixed[0]["word"] = t[0].upper() + t[1:] if len(t) > 1 else t.upper()
    if capitalize_sections:
        for start in range(0, len(fixed), words_per_group):
            for j in range(start, min(start + words_per_group, len(fixed))):
                t = fixed[j].get("word", "").strip()
                if t:
                    fixed[j] = dict(fixed[j])
                    fixed[j]["word"] = t[0].upper() + t[1:] if len(t) > 1 else t.upper()
                    break
    for start in range(0, len(fixed), words_per_group):
        end = min(start + words_per_group, len(fixed)) - 1
        if end < 0 or end >= len(fixed):
            continue
        t = fixed[end].get("word", "").strip()
        if t and t[-1] not in "..,;:!?\"'":
            fixed[end] = dict(fixed[end])
            fixed[end]["word"] = t + "."
    return fixed


def _sanitize_error_message(exc: Exception) -> str:
    """Return a user-friendly error message without leaking internals."""
    error_type = type(exc).__name__
    safe_messages = {
        "FileNotFoundError": "The specified file was not found.",
        "PermissionError": "Permission denied — cannot read the specified file.",
        "TimeoutError": "Processing timed out. The file may be too large.",
        "MemoryError": "Insufficient memory to process this file.",
        "ValueError": "Invalid input provided.",
        "OSError": "An operating system error occurred.",
    }
    return safe_messages.get(error_type, f"An error occurred ({error_type}). Check logs for details.")


def on_transcribe(
    audio_file, audio_path, model_key, language, model_size,
    words_per_chunk, burn_into_video, font_name, font_color, font_size,
    position_name, outline, shadow, fix_grammar, capitalize_sections,
):
    """Run transcription and optionally burn subtitles into video."""
    tmp_dir = None
    try:
        # Determine input file
        if audio_file is not None:
            input_file = audio_file.name
        elif audio_path and audio_path.strip():
            input_file = audio_path.strip()
            # Validate path: must be absolute and exist
            if not os.path.isabs(input_file):
                raise gr.Error("Please provide an absolute file path (e.g. /home/user/video.mp4).")
            if not os.path.isfile(input_file):
                raise gr.Error("File not found. Check the path and try again.")
            # Check file size (warn if > 2GB)
            file_size_gb = os.path.getsize(input_file) / (1024**3)
            if file_size_gb > 2:
                raise gr.Error(f"File is {file_size_gb:.1f} GB — too large to process safely.")
        else:
            raise gr.Error("Please upload an audio/video file or enter a file path.")

        if not model_key:
            raise gr.Error("Please select a transcription model.")

        # Get model info
        model_info = MODEL_REGISTRY.get(model_key)
        if not model_info:
            raise gr.Error("Unknown model selected.")

        # Sanitize font name
        font_name = _sanitize_font_name(font_name)

        # Prepare parameters
        params = model_info["params"].copy()
        if language and "language" in params:
            params["language"] = language
        if model_size and "model_size" in params:
            params["model_size"] = model_size

        # Validate words_per_chunk
        words_per_chunk = max(1, min(8, int(words_per_chunk)))

        # Run transcription
        try:
            start_time = time.time()
            words = model_info["function"](input_file, **params)
            elapsed = time.time() - start_time
        except Exception as e:
            raise gr.Error(_sanitize_error_message(e))

        if not words:
            raise gr.Error("No speech detected in the audio.")

        # Apply grammar fix if requested
        if fix_grammar:
            words = fix_grammar_punctuation(words, words_per_chunk, capitalize_sections)

        # Create output directory
        tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
        os.makedirs(tmp_dir, exist_ok=True)

        # Generate SRT
        srt_path = os.path.join(tmp_dir, "subtitles.srt")
        words_to_srt(words, srt_path, words_per_chunk)

        # Generate ASS with styling
        ass_path = os.path.join(tmp_dir, "subtitles.ass")
        prefs = {
            "font": font_name,
            "color": COLOR_PRESETS.get(font_color, "&H00FFFFFF"),
            "color_name": font_color,
            "font_size": font_size,
            "position": POSITION_PRESETS.get(position_name, 2),
            "position_name": position_name,
            "outline": outline,
            "shadow": shadow,
        }
        words_to_ass(words, ass_path, prefs, words_per_chunk)

        # Optionally burn subtitles into video
        video_output = None
        if burn_into_video:
            try:
                video_out_path = os.path.join(tmp_dir, "output_with_subtitles.mp4")
                success = burn_subtitles_to_video(input_file, ass_path, video_out_path, prefs)
                if success:
                    video_output = gr.update(value=video_out_path, visible=True)
                else:
                    video_output = gr.update(value=None, visible=False)
            except Exception:
                video_output = gr.update(value=None, visible=False)

        # Build status message
        status = (
            f"**Transcription complete!**\n\n"
            f"- Model: {model_info['name']}\n"
            f"- Words detected: {len(words)}\n"
            f"- Time: {elapsed:.1f}s"
        )
        if burn_into_video:
            status += "\n- Subtitles burned into video"
        else:
            status += "\n- Subtitle files ready (SRT + ASS)"

        return (
            gr.update(value=status, visible=True),
            gr.update(value=srt_path, visible=True),
            gr.update(value=ass_path, visible=True),
            video_output if video_output is not None else gr.update(visible=False),
            words,
        )

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(_sanitize_error_message(e))


def on_burn_subtitles(
    video_file, ass_file, font_name, font_color, font_size,
    position_name, outline, shadow,
):
    """Burn subtitles onto video."""
    if video_file is None:
        raise gr.Error("Please upload a video file first.")
    if ass_file is None:
        raise gr.Error("Please generate subtitles first (run transcription).")

    # Sanitize font name
    font_name = _sanitize_font_name(font_name)

    # Create output path
    tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(tmp_dir, exist_ok=True)
    output_path = os.path.join(tmp_dir, "output_with_subtitles.mp4")

    # Build preferences
    prefs = {
        "font": font_name,
        "color": COLOR_PRESETS.get(font_color, "&H00FFFFFF"),
        "color_name": font_color,
        "font_size": font_size,
        "position": POSITION_PRESETS.get(position_name, 2),
        "position_name": position_name,
        "outline": outline,
        "shadow": shadow,
    }

    # Burn subtitles
    try:
        success = burn_subtitles_to_video(
            video_file.name, ass_file, output_path, prefs
        )
    except Exception as e:
        raise gr.Error(_sanitize_error_message(e))

    if not success:
        raise gr.Error("Failed to burn subtitles. Check that ffmpeg is installed and the video file is valid.")

    return gr.update(value=output_path, visible=True)


def _format_model_info(model_key):
    """Format model information for display."""
    info = MODEL_REGISTRY.get(model_key, {})
    if not info:
        return "Select a model to see details."
    return (
        f"**{info['name']}**\n\n"
        f"- VRAM: {info['vram']}\n"
        f"- WER: {info['wer']}\n"
        f"- Speed: {info['rtfx']}\n"
        f"- Languages: {info['languages']}"
    )


# ========================= BUILD UI =========================

DEVICE_INFO = get_device_info()

with gr.Blocks(
    title="Transcribix",
    theme=gr.themes.Soft(),
    css="""
    .main-title {
        text-align: center;
        margin-bottom: 10px;
    }
    .model-info {
        padding: 10px;
        border-radius: 5px;
        background: #f0f0f0;
    }
    """
) as demo:
    gr.Markdown(
        "# Transcribix\n"
        "Offline speech-to-text transcription with 11 local AI models.\n"
        "Generates styled subtitles and burns them directly onto video.\n\n"
        "*by C0m3b4ck • Apache License 2.0*",
        elem_classes=["main-title"]
    )

    gr.Markdown(f"**Device:** `{DEVICE_INFO}`")

    with gr.Row():
        # Left column: Input & Model Selection
        with gr.Column(scale=1):
            gr.Markdown("## Input")
            
            with gr.Tab("Upload File"):
                audio_upload = gr.File(
                    label="Upload Audio/Video",
                    file_types=[".mp4", ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".mkv", ".avi", ".mov", ".webm"],
                )
            
            with gr.Tab("Enter Path"):
                audio_path_input = gr.Textbox(
                    label="File Path",
                    placeholder="/path/to/your/video.mp4",
                    info="Enter the full path to your audio/video file",
                )
                path_status = gr.Markdown(visible=False)

            gr.Markdown("## Model Selection")
            model_dropdown = gr.Dropdown(
                label="Transcription Model",
                choices=[(f"{m['name']} — {m['description']}", k) for k, m in MODEL_REGISTRY.items()],
                value="faster-whisper",
                info="Select a speech-to-text model",
            )

            model_info_display = gr.Markdown(
                value=_format_model_info("faster-whisper"),
                elem_classes=["model-info"]
            )

            with gr.Row():
                language_input = gr.Textbox(
                    label="Language Code",
                    value="en",
                    info="e.g. en, es, fr, de (model-dependent)",
                )
                model_size_input = gr.Dropdown(
                    label="Model Size",
                    choices=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                    value="large-v3",
                    info="Larger = more accurate but slower",
                )

        # Right column: Subtitle Styling
        with gr.Column(scale=1):
            gr.Markdown("## Subtitle Styling")

            with gr.Row():
                font_dropdown = gr.Dropdown(
                    label="Font",
                    choices=FONT_CHOICES,
                    value="Arial",
                )
                font_size_slider = gr.Slider(
                    label="Font Size",
                    minimum=12,
                    maximum=72,
                    value=24,
                    step=1,
                )

            with gr.Row():
                color_dropdown = gr.Dropdown(
                    label="Color",
                    choices=list(COLOR_PRESETS.keys()),
                    value="White",
                )
                position_dropdown = gr.Dropdown(
                    label="Position",
                    choices=list(POSITION_PRESETS.keys()),
                    value="Bottom center",
                )

            with gr.Row():
                outline_slider = gr.Slider(
                    label="Outline Thickness",
                    minimum=0,
                    maximum=4,
                    value=2,
                    step=1,
                )
                shadow_slider = gr.Slider(
                    label="Shadow Depth",
                    minimum=0,
                    maximum=3,
                    value=1,
                    step=1,
                )

            words_per_chunk = gr.Slider(
                label="Words per Caption",
                minimum=1,
                maximum=8,
                value=3,
                step=1,
                info="Number of words per subtitle block",
            )

            burn_into_video = gr.Checkbox(
                label="Burn subtitles into video",
                value=False,
                info="Adds ~1-2 min processing time. Requires ffmpeg.",
            )

            gr.Markdown("## Text Fix")
            fix_grammar = gr.Checkbox(
                label="Fix Grammar & Punctuation",
                value=False,
                info="Adds commas/periods, capitalizes 'I' and sentence starts.",
            )
            capitalize_sections = gr.Checkbox(
                label="Capitalize Beginning of Each Section",
                value=True,
                info="Capitalizes first word of every subtitle chunk.",
            )

    gr.Markdown("---")

    # Transcription button
    with gr.Row():
        transcribe_btn = gr.Button(
            "Transcribe & Generate Subtitles",
            variant="primary",
            size="lg",
        )

    # Status and outputs
    with gr.Row():
        with gr.Column(scale=2):
            status_display = gr.Markdown(visible=False)
            srt_output = gr.File(label="SRT Subtitles", visible=False)
            ass_output = gr.File(label="ASS Subtitles (Styled)", visible=False)
            video_output = gr.File(label="Video with Subtitles", visible=False)

    # Event wiring
    model_dropdown.change(
        fn=_format_model_info,
        inputs=[model_dropdown],
        outputs=[model_info_display],
    )

    transcribe_btn.click(
        fn=on_transcribe,
        inputs=[
            audio_upload, audio_path_input, model_dropdown, language_input, model_size_input,
            words_per_chunk, burn_into_video, font_dropdown, color_dropdown, font_size_slider,
            position_dropdown, outline_slider, shadow_slider, fix_grammar, capitalize_sections,
        ],
        outputs=[status_display, srt_output, ass_output, video_output, gr.State([])],
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transcribix Web UI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on (default: 7860)")
    args = parser.parse_args()
    demo.launch(server_name=args.host, server_port=args.port)
