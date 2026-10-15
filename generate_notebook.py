import json
import os

def create_colab_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# 🎙️ Voice Clone Studio (Seed-VC 44.1kHz F0 + BigVGAN)\n",
                    "### Complete 100% Zero-Robotic High-Fidelity Voice Cloning on Google Colab GPU\n",
                    "\n",
                    "> **Quick Instructions:**\n",
                    "> 1. Ensure GPU is active: **Runtime** -> **Change runtime type** -> select **T4 GPU** (or A100/V100/L4).\n",
                    "> 2. Click the Play button on each cell sequentially.\n",
                    "> 3. Once Step 3 starts, click the public **`gradio.live`** URL to open Voice Clone Studio in your browser!"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {
                    "cellView": "form"
                },
                "outputs": [],
                "source": [
                    "#@title 🎮 Check GPU Hardware\n",
                    "!nvidia-smi"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {
                    "cellView": "form"
                },
                "outputs": [],
                "source": [
                    "#@title 📦 1. Clone Repository & Install Dependencies\n",
                    "!git clone https://github.com/Plachtaa/seed-vc.git\n",
                    "%cd seed-vc\n",
                    "!pip install -q torchaudio torchvision soundfile pydub librosa transformers accelerate munch einops descript-audio-codec pyyaml huggingface-hub omegaconf gradio"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {
                    "cellView": "form"
                },
                "outputs": [],
                "source": [
                    "#@title 📥 2. Pre-download High-Fidelity AI Checkpoints\n",
                    "from hf_utils import load_custom_model_from_hf\n",
                    "print('[INFO] Downloading Seed-VC 44.1kHz F0 model checkpoint...')\n",
                    "load_custom_model_from_hf('Plachta/Seed-VC', 'DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema.pth', 'config_dit_mel_seed_uvit_whisper_base_f0_44k.yml')\n",
                    "load_custom_model_from_hf('funasr/campplus', 'campplus_cn_common.bin', None)\n",
                    "load_custom_model_from_hf('lj1995/VoiceConversionWebUI', 'rmvpe.pt', None)\n",
                    "\n",
                    "print('[INFO] Downloading BigVGAN 44.1kHz Vocoder...')\n",
                    "from modules.bigvgan import bigvgan\n",
                    "bigvgan.BigVGAN.from_pretrained('nvidia/bigvgan_v2_44khz_128band_512x', use_cuda_kernel=False)\n",
                    "print('✅ All Models Pre-cached and Ready!')"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {
                    "cellView": "form"
                },
                "outputs": [],
                "source": [
                    "#@title 🚀 3. Launch Voice Clone Studio WebUI\n",
                    "# Launch Gradio with public shareable link\n",
                    "!python webui.py --share"
                ]
            }
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {
                "provenance": [],
                "toc_visible": True
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }
    
    out_paths = [
        os.path.join(os.path.dirname(__file__), "Voice_Clone_Studio_Colab.ipynb"),
        os.path.join(os.path.expanduser("~"), "Desktop", "Voice_Clone_Studio_Colab.ipynb")
    ]
    
    for path in out_paths:
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(notebook, f, indent=2)
            print(f"[SUCCESS] Wrote notebook to: {path}")
        except Exception as e:
            print(f"[ERROR] Could not write to {path}: {e}")

if __name__ == "__main__":
    create_colab_notebook()
