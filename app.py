import streamlit as st
import torch
import torchaudio
import pickle
import os
from pathlib import Path
from model_def import DCUnet20  # Import your model definition

# Constants (same as in your script)
SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000  # 3072
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000  # 768

# Load the model (cached to avoid reloading on every run)
@st.cache_resource
def load_model():
    """Load the saved denoiser model"""
    with open('denoiser_model.pkl', 'rb') as f:
        save_dict = pickle.load(f)
    
    model = DCUnet20(n_fft=N_FFT, hop_length=HOP_LENGTH)
    model.load_state_dict(save_dict['state_dict'])
    model.eval()
    return model

def process_audio(model, input_audio_path, output_audio_path):
    """Process an audio file through the denoising model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load audio
    input_audio, sr = torchaudio.load(input_audio_path)
    if sr != SAMPLE_RATE:
        st.info(f"Resampling audio from {sr} Hz to {SAMPLE_RATE} Hz")
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        input_audio = resampler(input_audio)

    # Convert to mono if stereo
    if input_audio.size(0) > 1:
        input_audio = torch.mean(input_audio, dim=0, keepdim=True)
    
    # Process in chunks
    chunk_size = 165000
    output_chunks = []
    window = torch.hann_window(N_FFT).to(device)
    
    with torch.no_grad():
        for i in range(0, input_audio.size(1), chunk_size):
            chunk = input_audio[:, i:min(i + chunk_size, input_audio.size(1))]
            if chunk.size(1) < chunk_size:
                pad_size = chunk_size - chunk.size(1)
                chunk = torch.nn.functional.pad(chunk, (0, pad_size))
            
            chunk = chunk.to(device)
            chunk_stft = torch.stft(
                chunk, 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH, 
                window=window, 
                normalized=True, 
                return_complex=True, 
                onesided=True, 
                center=True
            )
            
            if chunk_stft.dim() == 3:
                chunk_stft = chunk_stft.unsqueeze(0)
            
            denoised_chunk = model(chunk_stft, is_istft=True, window=window, 
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
            denoised_chunk = denoised_chunk.cpu()
            
            if i + chunk_size > input_audio.size(1):
                denoised_chunk = denoised_chunk[:, :-(pad_size)]
            
            output_chunks.append(denoised_chunk)
    
    # Concatenate chunks
    denoised_audio = torch.cat(output_chunks, dim=1)
    
    # Save the output
    os.makedirs(os.path.dirname(output_audio_path) if os.path.dirname(output_audio_path) else '.', exist_ok=True)
    torchaudio.save(output_audio_path, denoised_audio, SAMPLE_RATE, bits_per_sample=16)
    return output_audio_path

def main():
    st.title("Audio Denoiser with DCUnet20")
    st.write("Upload a noisy audio file and download the denoised version.")

    # Load model
    with st.spinner("Loading the denoising model..."):
        model = load_model()
    st.success("Model loaded successfully!")

    # File upload
    uploaded_file = st.file_uploader("Choose a noisy audio file (e.g., .wav, .mp3)", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        input_path = f"temp_{uploaded_file.name}"
        with open(input_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(input_path, format="audio/wav")  # Preview the noisy audio
        
        # Process the audio
        if st.button("Denoise Audio"):
            with st.spinner("Processing audio..."):
                output_path = os.path.splitext(input_path)[0] + "_denoised.wav"
                try:
                    process_audio(model, input_path, output_path)
                    st.success(f"Denoised audio saved as {output_path}")

                    # Preview denoised audio
                    st.audio(output_path, format="audio/wav")

                    # Provide download link
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Denoised Audio",
                            data=f,
                            file_name=os.path.basename(output_path),
                            mime="audio/wav"
                        )
                except Exception as e:
                    st.error(f"Error processing audio: {str(e)}")
                finally:
                    # Clean up temporary files
                    if os.path.exists(input_path):
                        os.remove(input_path)
                    if os.path.exists(output_path):
                        os.remove(output_path)

if __name__ == "__main__":
    main()