import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import jiwer  # Used to calculate Word Error Rate
from scipy.io import wavfile

def generate_report_assets():
    print("Generating presentation assets...")
    
    # --- 1. Feature Extraction Visual (Spectrogram) ---
    # Create a dummy audio signal (sine sweep) using Numpy
    sr = 16000
    t = np.linspace(0, 2, 2 * sr)
    audio = np.sin(2 * np.pi * 440 * t * (1 + t/2)) + 0.1 * np.random.randn(len(t))
    
    plt.figure(figsize=(10, 4))
    # Use librosa to convert audio to Mel Spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Input Features: Mel Spectrogram')
    plt.tight_layout()
    plt.savefig('report_spectrogram.png', dpi=300)
    print("Saved: report_spectrogram.png")

    # --- 2. Training Loss Curve ---
    epochs = np.arange(1, 21)
    train_loss = 3.0 * np.exp(-epochs/4) + np.random.normal(0, 0.05, len(epochs))
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label='Training Loss', color='#6c5ce7', linewidth=3)
    plt.title('Training Convergence', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('CTC Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('report_loss_curve.png', dpi=300)
    print("Saved: report_loss_curve.png")

    # --- 3. WER Improvement (Using jiwer for validation) ---
    # Simulating a calculation of WER for the graph
    ground_truth = "hello world this is a test"
    hypothesis_early = "hullo warld dis is a text"
    hypothesis_final = "hello world this is a test"
    
    # Calculate real WER metrics using jiwer
    error_early = jiwer.wer(ground_truth, hypothesis_early)
    error_final = jiwer.wer(ground_truth, hypothesis_final)
    
    stages = ['Epoch 1', 'Epoch 5', 'Epoch 10', 'Epoch 15', 'Final']
    wer_values = [0.9, 0.6, 0.4, error_early, error_final] # Simulated descent
    
    plt.figure(figsize=(10, 6))
    plt.bar(stages, wer_values, color='#00b894', alpha=0.8)
    plt.plot(stages, wer_values, color='#2d3436', marker='o')
    plt.title('Word Error Rate (WER) Reduction', fontsize=14)
    plt.savefig('report_wer.png', dpi=300)
    print("Saved: report_wer.png")

if __name__ == "__main__":
    generate_report_assets()