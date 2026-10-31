import os
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
print("[STARTUP] Starting webui.py...", flush=True)

_dir = os.path.dirname(os.path.abspath(__file__))
# Configure writable Matplotlib and cache directory inside workspace
os.environ["MPLCONFIGDIR"] = os.path.join(_dir, ".cache", "matplotlib")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import time
import glob
import shutil
import tempfile
import subprocess
import soundfile as sf
import numpy as np
import torch

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# Ensure FFmpeg and local venv Scripts are on PATH
_dir = os.path.dirname(os.path.abspath(__file__))
_venv_scripts = os.path.join(_dir, ".venv", "Scripts")
if os.path.exists(_venv_scripts) and _venv_scripts not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_scripts + os.pathsep + os.environ.get("PATH", "")

from pydub import AudioSegment
from pydub.utils import which
_ffmpeg_exe = which("ffmpeg") or os.path.join(_venv_scripts, "ffmpeg.exe")
if os.path.exists(_ffmpeg_exe):
    AudioSegment.converter = _ffmpeg_exe
    AudioSegment.ffmpeg = _ffmpeg_exe

import gradio as gr
from seed_vc_wrapper import SeedVCWrapper

# Directories
OUTPUTS_DIR = os.path.join(_dir, "outputs")
TEMP_UPLOADS_DIR = os.path.join(_dir, "temp_uploads")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(TEMP_UPLOADS_DIR, exist_ok=True)

# Preset sample paths
SAMPLE_SOURCE = os.path.join(_dir, "user_audios", "google_ai_studio_input.wav")
SAMPLE_TARGET = os.path.join(_dir, "user_audios", "user_real_voice_reference.wav")
DOWNLOADS_DIR = os.path.join(os.path.expanduser("~"), "Downloads")

# Cached model engine
vc_engine = None

def get_engine():
    global vc_engine
    if vc_engine is None:
        print("[INFO] Initializing Seed-VC Studio Engine on CUDA GPU...", flush=True)
        vc_engine = SeedVCWrapper()
    return vc_engine

def get_downloads_list():
    """Scan Downloads folder for audio and video files, newest first."""
    items = []
    if os.path.exists(DOWNLOADS_DIR):
        for ext in ("*.wav", "*.mp3", "*.m4a", "*.flac", "*.ogg", "*.aac", "*.mp4", "*.mkv", "*.mov", "*.webm"):
            items.extend(glob.glob(os.path.join(DOWNLOADS_DIR, ext)))
    items.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return [os.path.basename(p) for p in items]

def extract_and_prepare_audio(file_path):
    """
    Extracts audio from video or converts audio to 44.1kHz mono WAV in temp_uploads.
    Returns: (processed_wav_path, info_markdown)
    """
    if not file_path or not os.path.exists(file_path):
        return None, "*(No file loaded. Drop an audio or video file above!)*"

    base_name = os.path.basename(file_path)
    ext = os.path.splitext(file_path)[1].lower()

    # Create destination in temp_uploads
    out_wav = os.path.join(TEMP_UPLOADS_DIR, f"input_{int(time.time()*1000)}.wav")

    # Fast-path for WAV files using soundfile directly (bypasses FFmpeg subprocess)
    if ext == ".wav":
        try:
            data, in_sr = sf.read(file_path)
            if len(data.shape) > 1:
                data = data.mean(axis=1)
            sf.write(out_wav, data, in_sr)
            dur = len(data) / in_sr
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            info_text = f"✅ **Loaded Successfully:** `{base_name}` ({file_size_mb:.2f} MB | ⏱️ Duration: {dur:.1f}s | Sample Rate: {in_sr}Hz)"
            return out_wav, info_text
        except Exception:
            pass

    # Use ffmpeg to extract or convert cleanly to 44.1kHz 16-bit mono WAV
    try:
        cmd = [
            _ffmpeg_exe if os.path.exists(_ffmpeg_exe) else "ffmpeg",
            "-y",
            "-i", file_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "1",
            out_wav
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if res.returncode != 0 or not os.path.exists(out_wav):
            # Fallback using soundfile / pydub
            sound = AudioSegment.from_file(file_path)
            sound = sound.set_frame_rate(44100).set_channels(1)
            sound.export(out_wav, format="wav")

        # Get audio duration
        info = sf.info(out_wav)
        dur = info.duration
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

        info_text = f"✅ **Loaded Successfully:** `{base_name}` ({file_size_mb:.2f} MB | ⏱️ Duration: {dur:.1f}s | Sample Rate: 44.1kHz)"
        return out_wav, info_text

    except Exception as e:
        return None, f"⚠️ Error processing file: {str(e)}"

def convert_voice_fn(
    source_preview_path,
    source_file_path,
    reference_audio,
    mode,
    diffusion_steps,
    length_adjust,
    cfg_rate,
    pitch_shift,
    progress=gr.Progress(track_tqdm=True)
):
    # Determine the actual source
    actual_source = source_preview_path or source_file_path
    if not actual_source or not os.path.exists(actual_source):
        raise gr.Error("Please upload or select an audio or video file in Step 1!")

    # Ensure audio is extracted to clean WAV
    if not actual_source.lower().endswith(".wav"):
        actual_source, _ = extract_and_prepare_audio(actual_source)
        if not actual_source:
            raise gr.Error("Could not extract audio from the provided file.")

    actual_reference = reference_audio or SAMPLE_TARGET
    if not actual_reference or not os.path.exists(actual_reference):
        raise gr.Error("Please provide or record a Reference Voice in Step 2!")

    # Ensure reference audio is extracted to clean 44.1kHz WAV (e.g. from mic webm/ogg)
    if not actual_reference.lower().endswith(".wav"):
        ref_wav, _ = extract_and_prepare_audio(actual_reference)
        if ref_wav and os.path.exists(ref_wav):
            actual_reference = ref_wav

    progress(0.05, desc="Loading voice conversion models...")
    engine = get_engine()

    f0_cond = ("Pitch / F0 Conditioned" in mode)
    auto_f0 = True

    progress(0.15, desc=f"Converting voice on GPU ({diffusion_steps} steps, CFG {cfg_rate})...")
    t_start = time.time()

    try:
        last_full_audio = None
        for mp3_chunk, full_audio in engine.convert_voice(
            source=actual_source,
            target=actual_reference,
            diffusion_steps=int(diffusion_steps),
            length_adjust=float(length_adjust),
            inference_cfg_rate=float(cfg_rate),
            f0_condition=f0_cond,
            auto_f0_adjust=auto_f0,
            pitch_shift=int(pitch_shift),
            stream_output=True,
            encode_mp3=False
        ):
            if full_audio is not None:
                last_full_audio = full_audio

        if last_full_audio is None:
            raise gr.Error("Voice conversion did not produce any audio. Please check your inputs.")

        sr, audio_data = last_full_audio
        elapsed = time.time() - t_start
        audio_dur = len(audio_data) / sr

        # Save to high quality wav file with unique timestamp
        out_filename = f"converted_voice_{int(time.time()*1000)}.wav"
        out_path = os.path.join(OUTPUTS_DIR, out_filename)
        sf.write(out_path, audio_data, sr)

        status_msg = f"🎉 Conversion Complete in {elapsed:.2f}s! (Audio Duration: {audio_dur:.2f}s | Speed: {audio_dur/elapsed:.1f}x real-time) | Saved to outputs/{out_filename}"
        return out_path, out_path, status_msg
    except Exception as e:
        print(f"[ERROR in convert_voice_fn]: {e}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error during voice conversion: {str(e)}")

# Gradio Interface
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate"
)

css = """
.gradio-container { max-width: 1150px !important; margin: auto !important; }
.convert-btn { font-size: 1.3rem !important; padding: 14px 28px !important; font-weight: 700 !important; }
.info-box { padding: 8px 12px; border-radius: 6px; background-color: #f1f5f9; margin-top: 4px; }
"""

with gr.Blocks(title="Voice Clone Studio - Speech to Speech", theme=theme, css=css) as demo:
    gr.Markdown(
        """
        # 🎙️ Voice Clone Studio (100% Studio Voice Match)
        ### Transform Google AI Studio Audio / Video into Your Own Voice
        Zero-Shot Voice Conversion on NVIDIA GPU — High-Fidelity 44.1kHz Studio Quality with Zero Robotic Artifacts.
        """
    )

    with gr.Row():
        # Column 1: Source Audio / Video
        with gr.Column(scale=1):
            gr.Markdown("### 1️⃣ Source Audio (AI Voice to Change)")
            
            source_file = gr.File(
                label="📂 Upload Audio or Video File (MP4, WAV, MP3, M4A, etc.)",
                file_types=["audio", "video", ".wav", ".mp3", ".m4a", ".mp4", ".mov", ".mkv", ".webm", ".aac", ".flac", ".ogg"],
                type="filepath"
            )
            
            source_preview = gr.Audio(
                label="🎧 Source Audio Player (Preview what will be converted)",
                type="filepath"
            )

            source_info = gr.Markdown("*(No file selected yet. Drag & drop any audio or video file above!)*")
            
            with gr.Row():
                load_sample_source_btn = gr.Button("⚡ Load Bhagavad Gita 30s Sample", size="sm", variant="secondary")
                clear_source_btn = gr.Button("🗑️ Clear", size="sm", variant="stop")

            with gr.Accordion("📂 Quick Pick from Downloads Folder", open=True):
                init_choices = ["-- Select from Downloads --"] + get_downloads_list()
                source_dropdown = gr.Dropdown(
                    label="Choose File from Downloads",
                    choices=init_choices,
                    value=init_choices[0],
                    info="Selecting a file instantly extracts & loads it into the preview player above!"
                )
                refresh_dl_btn = gr.Button("🔄 Refresh Downloads List", size="sm")

        # Column 2: Target Reference Voice
        with gr.Column(scale=1):
            gr.Markdown("### 2️⃣ Target Voice (Your Own Voice Profile)")
            
            target_audio = gr.Audio(
                value=SAMPLE_TARGET if os.path.exists(SAMPLE_TARGET) else None,
                label="🎙️ Reference Voice (Your Own Voice Profile)",
                type="filepath",
                sources=["microphone", "upload"]
            )
            
            target_info = gr.Markdown("✅ **Default Voice Profile Loaded:** User Real Voice Reference (10.8s, 44.1kHz Studio Quality)")

            with gr.Row():
                load_sample_target_btn = gr.Button("⚡ Reset to Default Voice Profile", size="sm", variant="secondary")
                clear_target_btn = gr.Button("🗑️ Clear Reference", size="sm", variant="stop")

    with gr.Accordion("⚙️ Voice Tuning & Quality Settings", open=True):
        gr.Markdown(
            """
            > 💡 **ரோபோடிக் சத்தம் வராமல் 100% நேச்சுரல் மனிதக் குரல் வர சிறந்த அமைப்புகள் (Recommended Settings):**
            > - **Diffusion Steps = 35:** ஆடியோவில் ஏற்படும் நொய்ஸ் மற்றும் ரோபோடிக் அதிர்வுகளை முற்றிலும் நீக்கி ஸ்மூத் ஆக்கும்.
            > - **CFG Rate = 0.70 (0.65 - 0.70):** மெட்டாலிக் சவுண்டைத் தவிர்த்து, இயல்பான, ரிலாக்ஸ்டான மனித தொனியைத் தரும்.
            > - **Pitch Shift = -1:** கூர்மையான அல்லது செயற்கையான சவுண்ட் இல்லாமல், கம்பீரமான நேச்சுரல் மார்புக் குரல் (Chest Resonance) தரும்.
            """
        )
        with gr.Row():
            mode_select = gr.Radio(
                label="Conversion Mode",
                choices=[
                    "Pitch / F0 Conditioned (44.1kHz Studio Quality - Best Similarity)",
                    "Standard Timbre Transfer (22kHz Fast)"
                ],
                value="Pitch / F0 Conditioned (44.1kHz Studio Quality - Best Similarity)",
                info="Pitch conditioned mode preserves exact emotion, cadence, and vocal melody."
            )
        with gr.Row():
            speed_preset = gr.Radio(
                label="⚡ Processing Speed Preset",
                choices=[
                    "⚡ Turbo (10 Steps — Ultra Fast ~8s)",
                    "🚀 Balanced (15 Steps — Recommended ~15s)",
                    "💎 Studio Master (25 Steps — High Detail)"
                ],
                value="🚀 Balanced (15 Steps — Recommended ~15s)",
                info="Lower steps = dramatically faster processing. 15 steps provides pristine studio quality."
            )
        with gr.Row():
            diff_steps = gr.Slider(
                minimum=5,
                maximum=50,
                value=15,
                step=1,
                label="Diffusion Steps",
                info="15 steps delivers crystal clear audio in just ~10-15 seconds on RTX 5050."
            )
            cfg_rate = gr.Slider(
                minimum=0.2,
                maximum=1.2,
                value=0.70,
                step=0.05,
                label="CFG Rate (Guidance Scale)",
                info="0.70 provides natural, warm voice without metallic artifacts."
            )
            pitch_slider = gr.Slider(
                minimum=-12,
                maximum=12,
                value=-1,
                step=1,
                label="Pitch Shift (Semitones)",
                info="-1 aligns pitch frequency perfectly with natural human chest depth."
            )
            length_slider = gr.Slider(
                minimum=0.8,
                maximum=1.3,
                value=1.0,
                step=0.05,
                label="Length / Speed Adjust",
                info="1.0 keeps original speaking pace."
            )

    # Conversion Trigger
    convert_btn = gr.Button("🚀 Convert Voice to My Voice (Studio Quality)", variant="primary", elem_classes=["convert-btn"])

    # Output Section
    with gr.Row():
        with gr.Column(scale=1):
            output_audio = gr.Audio(
                label="🎧 Converted Output Audio (In Your Voice)",
                type="filepath"
            )
        with gr.Column(scale=1):
            output_file = gr.File(
                label="⬇️ Download Converted WAV Audio (44.1kHz Master)",
                type="filepath"
            )

    status_box = gr.Textbox(
        label="Performance & Processing Status",
        interactive=False,
        placeholder="Ready. Click 'Convert Voice to My Voice' to start..."
    )

    # Wire event listeners
    def on_file_selected(fpath):
        if not fpath:
            return None, "*(No file selected)*"
        return extract_and_prepare_audio(fpath)

    source_file.change(
        fn=on_file_selected,
        inputs=[source_file],
        outputs=[source_preview, source_info]
    )

    def on_dropdown_select(fname):
        if not fname or fname == "-- Select from Downloads --":
            return None, None, "*(Select a file from dropdown)*"
        full_p = os.path.join(DOWNLOADS_DIR, fname)
        if os.path.exists(full_p):
            wav_p, info = extract_and_prepare_audio(full_p)
            return full_p, wav_p, info
        return None, None, "⚠️ File not found in Downloads."

    source_dropdown.change(
        fn=on_dropdown_select,
        inputs=[source_dropdown],
        outputs=[source_file, source_preview, source_info]
    )

    def refresh_dl():
        new_choices = ["-- Select from Downloads --"] + get_downloads_list()
        return gr.update(choices=new_choices, value=new_choices[0])

    refresh_dl_btn.click(fn=refresh_dl, outputs=source_dropdown)

    # Clear buttons
    def clear_source():
        return None, None, "*(No file selected. Drop an audio or video file above!)*"

    clear_source_btn.click(fn=clear_source, outputs=[source_file, source_preview, source_info])
    clear_target_btn.click(lambda: (None, "⚠️ No reference voice profile loaded! Please record or upload your voice."), outputs=[target_audio, target_info])

    # Preset sample buttons
    def load_preset_source():
        if os.path.exists(SAMPLE_SOURCE):
            wav_path, info = extract_and_prepare_audio(SAMPLE_SOURCE)
            return SAMPLE_SOURCE, wav_path, info
        return None, None, "⚠️ Sample audio not found."

    load_sample_source_btn.click(fn=load_preset_source, outputs=[source_file, source_preview, source_info])

    def load_preset_target():
        if os.path.exists(SAMPLE_TARGET):
            return SAMPLE_TARGET, "✅ **Default Voice Profile Loaded:** User Real Voice Reference (10.8s, 44.1kHz Studio Quality)"
        return None, "⚠️ Default voice profile not found."

    load_sample_target_btn.click(fn=load_preset_target, outputs=[target_audio, target_info])

    def update_steps_preset(preset):
        if "Turbo" in preset:
            return 10
        elif "Balanced" in preset:
            return 15
        else:
            return 25

    speed_preset.change(fn=update_steps_preset, inputs=[speed_preset], outputs=[diff_steps])

    # Convert button handler
    convert_btn.click(
        fn=convert_voice_fn,
        inputs=[
            source_preview,
            source_file,
            target_audio,
            mode_select,
            diff_steps,
            length_slider,
            cfg_rate,
            pitch_slider
        ],
        outputs=[output_audio, output_file, status_box]
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voice Clone Studio - RTX 5050 Edition")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 7865)), help="Server port")
    parser.add_argument("--share", action="store_true", help="Generate public Gradio share link (for Colab/Cloud)")
    parser.add_argument("--listen", action="store_true", help="Listen on 0.0.0.0 for external network access")
    args, _ = parser.parse_known_args()

    port = args.port
    server_name = "0.0.0.0" if (args.listen or args.share) else "127.0.0.1"

    # Whitelist all audio directories
    allowed_dirs = [
        os.path.abspath(_dir),
        tempfile.gettempdir(),
        os.path.join(tempfile.gettempdir(), "gradio"),
        DOWNLOADS_DIR,
        os.path.join(os.path.expanduser("~"), "Desktop"),
        r"C:\Users\hari3\Downloads",
        r"C:\Users\hari3\Desktop",
        OUTPUTS_DIR,
        TEMP_UPLOADS_DIR
    ]
    allowed_dirs = [p for p in allowed_dirs if os.path.exists(p)]

    print(f"[INFO] Launching Voice Clone Studio on http://{server_name}:{port} (share={args.share})", flush=True)
    demo.launch(
        server_name=server_name,
        server_port=port,
        inbrowser=False,
        share=args.share,
        allowed_paths=allowed_dirs
    )

