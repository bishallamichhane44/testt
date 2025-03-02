import torch
import torchaudio
import pickle
import os
from pathlib import Path
from model_def import DCUnet20

# Constants
SAMPLE_RATE = 48000
N_FFT = (SAMPLE_RATE * 64) // 1000  # 3072
HOP_LENGTH = (SAMPLE_RATE * 16) // 1000  # 768

def load_model():
    """Load the saved denoiser model"""
    with open('denoiser_model.pkl', 'rb') as f:
        save_dict = pickle.load(f)
    
    # Create model with saved parameters, ensuring they match our constants
    model = DCUnet20(n_fft=N_FFT, hop_length=HOP_LENGTH)  # Override with our values
    model.load_state_dict(save_dict['state_dict'])
    model.eval()
    return model

def process_audio(model, input_path, output_path):
    """Process an audio file through the denoising model"""
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load audio
    input_audio, sr = torchaudio.load(input_path)
    if sr != SAMPLE_RATE:
        print(f"Resampling from {sr} to {SAMPLE_RATE}")
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        input_audio = resampler(input_audio)

    # Convert to mono if stereo
    if input_audio.size(0) > 1:
        input_audio = torch.mean(input_audio, dim=0, keepdim=True)
    
    # Process in chunks of 165000 samples (about 3.4 seconds at 48kHz)
    chunk_size = 165000
    output_chunks = []
    
    # Create Hann window for STFT and iSTFT
    window = torch.hann_window(N_FFT).to(device)
    
    for i in range(0, input_audio.size(1), chunk_size):
        # Get chunk
        chunk = input_audio[:, i:min(i + chunk_size, input_audio.size(1))]
        
        # Pad if necessary
        if chunk.size(1) < chunk_size:
            pad_size = chunk_size - chunk.size(1)
            chunk = torch.nn.functional.pad(chunk, (0, pad_size))
        
        # Move chunk to device
        chunk = chunk.to(device)
        
        # Convert to STFT
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
        
        # Add batch dimension if needed
        if chunk_stft.dim() == 3:
            chunk_stft = chunk_stft.unsqueeze(0)
        
        # Process through model, passing window and parameters
        with torch.no_grad():
            # Assuming DCUnet20 accepts these kwargs when is_istft=True
            denoised_chunk = model(chunk_stft, is_istft=True, window=window, 
                                 n_fft=N_FFT, hop_length=HOP_LENGTH)
        
        # Convert back to CPU
        denoised_chunk = denoised_chunk.cpu()
        
        # Remove padding if this was the last chunk
        if i + chunk_size > input_audio.size(1):
            denoised_chunk = denoised_chunk[:, :-(pad_size)]
        
        output_chunks.append(denoised_chunk)
    
    # Concatenate all chunks
    denoised_audio = torch.cat(output_chunks, dim=1)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    
    # Save the denoised audio
    torchaudio.save(output_path, denoised_audio, SAMPLE_RATE, bits_per_sample=16)
    print(f"Denoised audio saved to {output_path}")

def main():
    # Load the model
    print("Loading model...")
    model = load_model()
    print("Model loaded successfully")
    
    # Get input file path
    while True:
        input_path = input("Enter the path to your noisy audio file (or 'q' to quit): ")
        if input_path.lower() == 'q':
            break
            
        if not os.path.exists(input_path):
            print("File not found. Please try again.")
            continue
            
        # Generate output path
        output_path = os.path.splitext(input_path)[0] + "_denoised.wav"
        
        print(f"Processing {input_path}...")
        try:
            process_audio(model, input_path, output_path)
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            continue

if __name__ == "__main__":
    main()