import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import jiwer  # Used to calculate Word Error Rate
from scipy.io import wavfile
import pandas as pd
import os
from typing import Optional


def _load_or_generate_audio(path: Optional[str], sr: int = 16000):
    if path and os.path.exists(path):
        try:
            rate, data = wavfile.read(path)
            # Normalize and convert to float32
            if data.dtype.kind == 'i':
                maxv = np.iinfo(data.dtype).max
                audio = data.astype(np.float32) / maxv
            else:
                audio = data.astype(np.float32)
            if rate != sr:
                audio = librosa.resample(audio, orig_sr=rate, target_sr=sr)
            return audio, sr
        except Exception:
            pass

    # Fallback: synthetic audio (sine sweep + noise)
    t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
    audio = 0.8 * np.sin(2 * np.pi * 220 * t * (1 + t / 2))
    audio += 0.08 * np.random.randn(len(t))
    return audio.astype(np.float32), sr


def generate_report_assets(audio_path: Optional[str] = None, out_dir: str = '.'):
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating presentation assets into: {out_dir}")

    # Load audio
    audio, sr = _load_or_generate_audio(audio_path)

    # Prepare a summary dict for CSV
    summary = {
        'sample_rate': sr,
        'duration_sec': float(len(audio) / sr)
    }

    # 1) Waveform
    plt.figure(figsize=(10, 3))
    librosa.display.waveshow(audio, sr=sr)
    plt.title('Waveform')
    plt.tight_layout()
    waveform_path = os.path.join(out_dir, 'report_waveform.png')
    plt.savefig(waveform_path, dpi=300)
    plt.close()
    print('Saved:', waveform_path)

    # 2) STFT Spectrogram
    D = np.abs(librosa.stft(audio, n_fft=1024, hop_length=256))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(DB, sr=sr, hop_length=256, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('STFT Spectrogram (log freq)')
    stft_path = os.path.join(out_dir, 'report_stft.png')
    plt.tight_layout()
    plt.savefig(stft_path, dpi=300)
    plt.close()
    print('Saved:', stft_path)

    # 3) Mel Spectrogram
    S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    mel_path = os.path.join(out_dir, 'report_mel_spectrogram.png')
    plt.tight_layout()
    plt.savefig(mel_path, dpi=300)
    plt.close()
    print('Saved:', mel_path)

    # 4) MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    plt.figure(figsize=(10, 3))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    mfcc_path = os.path.join(out_dir, 'report_mfcc.png')
    plt.tight_layout()
    plt.savefig(mfcc_path, dpi=300)
    plt.close()
    print('Saved:', mfcc_path)

    # 5) Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    plt.figure(figsize=(10, 2.5))
    librosa.display.specshow(chroma, y_axis='chroma', x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('Chroma')
    chroma_path = os.path.join(out_dir, 'report_chroma.png')
    plt.tight_layout()
    plt.savefig(chroma_path, dpi=300)
    plt.close()
    print('Saved:', chroma_path)

    # 6) Zero-Crossing Rate and RMS Energy over time
    zcr = librosa.feature.zero_crossing_rate(audio, frame_length=1024, hop_length=256)[0]
    rms = librosa.feature.rms(y=audio, frame_length=1024, hop_length=256)[0]
    times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=256)

    plt.figure(figsize=(10, 3))
    plt.plot(times, zcr, label='ZCR')
    plt.plot(times, rms, label='RMS Energy')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.title('Zero-Crossing Rate & RMS Energy')
    zcr_rms_path = os.path.join(out_dir, 'report_zcr_rms.png')
    plt.tight_layout()
    plt.savefig(zcr_rms_path, dpi=300)
    plt.close()
    print('Saved:', zcr_rms_path)

    summary['zcr_mean'] = float(np.mean(zcr))
    summary['rms_mean'] = float(np.mean(rms))

    # 7) Pitch (using piptrack) summary
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sr, hop_length=256)
    pitch_values = []
    for i in range(pitches.shape[1]):
        idx = magnitudes[:, i].argmax()
        pitch = pitches[idx, i]
        if pitch > 0:
            pitch_values.append(pitch)
    if pitch_values:
        summary['pitch_mean'] = float(np.mean(pitch_values))
        summary['pitch_median'] = float(np.median(pitch_values))
    else:
        summary['pitch_mean'] = None
        summary['pitch_median'] = None

    # 8) Training & Validation Loss Curves (simulated example)
    epochs = np.arange(1, 31)
    train_loss = 2.5 * np.exp(-epochs / 8) + np.random.normal(0, 0.03, len(epochs))
    val_loss = 2.8 * np.exp(-epochs / 7.5) + np.random.normal(0, 0.04, len(epochs))
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_loss, label='Train Loss', linewidth=2)
    plt.plot(epochs, val_loss, label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    loss_path = os.path.join(out_dir, 'report_loss_curve.png')
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print('Saved:', loss_path)

    # 9) WER over epochs (simulated) and per-utterance WER scatter
    # Simulate decreasing WER
    wer_epochs = np.clip(np.linspace(0.9, 0.08, num=len(epochs)) + np.random.normal(0, 0.02, len(epochs)), 0, 1)
    plt.figure(figsize=(10, 4))
    plt.plot(epochs, wer_epochs, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('WER')
    plt.title('WER over Training')
    wer_path = os.path.join(out_dir, 'report_wer_epochs.png')
    plt.tight_layout()
    plt.savefig(wer_path, dpi=300)
    plt.close()
    print('Saved:', wer_path)

    # Per-utterance WER scatter (simulated dataset)
    n_utts = 40
    utt_lengths = np.random.randint(1, 20, size=n_utts)
    per_utt_wer = np.clip(0.4 - 0.02 * utt_lengths + np.random.normal(0, 0.05, n_utts), 0, 1)
    plt.figure(figsize=(8, 4))
    plt.scatter(utt_lengths, per_utt_wer, c=per_utt_wer, cmap='viridis', alpha=0.9)
    plt.colorbar(label='WER')
    plt.xlabel('Utterance length (words)')
    plt.ylabel('WER')
    plt.title('Per-utterance WER vs Utterance Length')
    perutt_path = os.path.join(out_dir, 'report_per_utt_wer.png')
    plt.tight_layout()
    plt.savefig(perutt_path, dpi=300)
    plt.close()
    print('Saved:', perutt_path)

    summary['avg_utterance_length_words'] = float(np.mean(utt_lengths))
    summary['avg_per_utt_wer'] = float(np.mean(per_utt_wer))

    # 10) Histogram of utterance lengths
    plt.figure(figsize=(6, 3))
    plt.hist(utt_lengths, bins=range(1, 22), color='#0984e3', alpha=0.8)
    plt.xlabel('Words')
    plt.ylabel('Count')
    plt.title('Histogram: Utterance Lengths')
    hist_path = os.path.join(out_dir, 'report_utt_length_hist.png')
    plt.tight_layout()
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print('Saved:', hist_path)

    # 11) Confusion Matrix (simulated) for a small token set
    labels = ['a', 'b', 'c', 'd', 'e']
    conf = np.random.randint(0, 50, size=(len(labels), len(labels))).astype(float)
    # Make diagonal larger to simulate correct predictions
    for i in range(len(labels)):
        conf[i, i] += 150
    plt.figure(figsize=(6, 5))
    im = plt.imshow(conf, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix (simulated)')
    plt.colorbar(im)
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    conf_path = os.path.join(out_dir, 'report_confusion_matrix.png')
    plt.tight_layout()
    plt.savefig(conf_path, dpi=300)
    plt.close()
    print('Saved:', conf_path)

    # 12) Write summary CSV
    summary_df = pd.DataFrame([summary])
    csv_path = os.path.join(out_dir, 'report_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print('Saved:', csv_path)

    print('All assets generated.')


if __name__ == '__main__':
    # Run with defaults (no external audio)
    generate_report_assets()