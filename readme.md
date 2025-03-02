# Noise2Noise Audio Denoising

This project implements an audio denoising solution using the Noise2Noise approach.
## Prerequisites

Before running the `denoise_audio.py` script, ensure you have the required dependencies installed. Run the following commands in your terminal within your virtual environment:

```bash
pip install torch==2.4.1 torchaudio==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu121
pip install ffmpeg soundfile