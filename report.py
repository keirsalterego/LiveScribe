import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import jiwer  # Used to calculate Word Error Rate
from scipy.io import wavfile
import pandas as pd
import os
import json
import argparse
from typing import Optional
from matplotlib.backends.backend_pdf import PdfPages


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


def generate_report_assets(audio_path: Optional[str] = None, 
                          out_dir: str = '.', 
                          predictions_data: Optional[dict] = None):
    """
    Generate comprehensive report assets.
    
    Args:
        audio_path: Path to audio file for analysis (optional)
        out_dir: Output directory for generated files
        predictions_data: Dictionary with model predictions and ground truth:
            {
                'y_true': list/array of ground truth transcriptions or labels,
                'y_pred': list/array of predicted transcriptions or labels,
                'train_losses': list of training losses per epoch (optional),
                'val_losses': list of validation losses per epoch (optional),
                'wer_per_epoch': list of WER values per epoch (optional),
                'labels': list of label names for confusion matrix (optional),
                'utterance_lengths': list of utterance lengths in words (optional)
            }
    """
    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating presentation assets into: {out_dir}")

    # Load audio
    audio, sr = _load_or_generate_audio(audio_path)

    # Prepare a summary dict for CSV
    summary = {
        'sample_rate': sr,
        'duration_sec': float(len(audio) / sr)
    }
    
    # Check if we have real predictions data
    has_predictions = predictions_data is not None and \
                     'y_true' in predictions_data and \
                     'y_pred' in predictions_data

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

        # Additional spectral metrics useful for presentations
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        spec_flatness = librosa.feature.spectral_flatness(y=audio)[0]
        spec_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)

        summary['spectral_centroid_mean'] = float(np.mean(spec_centroid))
        summary['spectral_centroid_median'] = float(np.median(spec_centroid))
        summary['spectral_bandwidth_mean'] = float(np.mean(spec_bandwidth))
        summary['spectral_rolloff_mean'] = float(np.mean(spec_rolloff))
        summary['spectral_flatness_mean'] = float(np.mean(spec_flatness))
        # store contrast as mean across bands
        summary['spectral_contrast_mean'] = float(np.mean(spec_contrast))

        # Plot spectral centroid and rolloff over time
        times_spec = librosa.frames_to_time(np.arange(len(spec_centroid)), sr=sr)
        plt.figure(figsize=(10, 3))
        plt.plot(times_spec, spec_centroid, label='Centroid')
        plt.plot(times_spec, spec_rolloff, label='Rolloff')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.title('Spectral Centroid & Rolloff')
        speccr_path = os.path.join(out_dir, 'report_spec_centroid_rolloff.png')
        plt.tight_layout()
        plt.savefig(speccr_path, dpi=300)
        plt.close()
        print('Saved:', speccr_path)

        # Tempo / Beat / Onset rate
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr, onset_envelope=onset_env)
        onset_rate = np.mean(librosa.onset.onset_strength(y=audio, sr=sr))
        summary['tempo_bpm'] = float(tempo)
        summary['onset_rate'] = float(onset_rate)

        # Onset envelope plot
        times_onset = librosa.times_like(onset_env, sr=sr)
        plt.figure(figsize=(10, 2.5))
        plt.plot(times_onset, onset_env)
        plt.vlines(librosa.frames_to_time(beats, sr=sr), 0, onset_env.max(), color='r', alpha=0.6, label='Beats')
        plt.title('Onset Strength & Beats')
        plt.legend()
        onset_path = os.path.join(out_dir, 'report_onset_beats.png')
        plt.tight_layout()
        plt.savefig(onset_path, dpi=300)
        plt.close()
        print('Saved:', onset_path)

        # Silence ratio and dynamic range
        # Use frame-wise RMS to determine silence frames
        rms_frames = rms
        silence_thresh = np.percentile(rms_frames, 10) * 0.5
        silence_ratio = float(np.mean(rms_frames < silence_thresh))
        dynamic_range_db = float(20.0 * np.log10((np.max(np.abs(audio)) + 1e-9) / (np.mean(np.abs(audio)) + 1e-9)))
        summary['silence_ratio'] = silence_ratio
        summary['dynamic_range_db'] = dynamic_range_db

        # SNR estimate: approximate noise floor as 10th percentile of RMS frames
        noise_floor = np.percentile(rms_frames, 10)
        signal_level = np.percentile(rms_frames, 90)
        if noise_floor > 0:
            snr_db = 20.0 * np.log10((signal_level + 1e-9) / (noise_floor + 1e-9))
        else:
            snr_db = None
        summary['snr_db_estimate'] = None if snr_db is None else float(snr_db)

        # Peak amplitude, RMS global
        summary['peak_amplitude'] = float(np.max(np.abs(audio)))
        summary['global_rms'] = float(np.mean(np.abs(audio)))


    # 8) Training & Validation Loss Curves
    if has_predictions and 'train_losses' in predictions_data and predictions_data['train_losses']:
        # Use real training data
        train_loss = predictions_data['train_losses']
        val_loss = predictions_data.get('val_losses', [])
        epochs = np.arange(1, len(train_loss) + 1)
        
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, train_loss, 'o-', label='Train Loss', linewidth=2)
        if val_loss:
            plt.plot(epochs, val_loss, 's-', label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss (Real Data)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        summary['final_train_loss'] = float(train_loss[-1])
        if val_loss:
            summary['final_val_loss'] = float(val_loss[-1])
            summary['min_val_loss'] = float(min(val_loss))
    else:
        # Simulated example for demonstration
        epochs = np.arange(1, 31)
        train_loss = 2.5 * np.exp(-epochs / 8) + np.random.normal(0, 0.03, len(epochs))
        val_loss = 2.8 * np.exp(-epochs / 7.5) + np.random.normal(0, 0.04, len(epochs))
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, train_loss, label='Train Loss (simulated)', linewidth=2)
        plt.plot(epochs, val_loss, label='Val Loss (simulated)', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss (Simulated - Provide real data!)')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    loss_path = os.path.join(out_dir, 'report_loss_curve.png')
    plt.tight_layout()
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print('Saved:', loss_path)

    # 9) WER over epochs and per-utterance WER analysis
    if has_predictions and 'wer_per_epoch' in predictions_data and predictions_data['wer_per_epoch']:
        # Use real WER data
        wer_epochs = predictions_data['wer_per_epoch']
        epochs_wer = np.arange(1, len(wer_epochs) + 1)
        plt.figure(figsize=(10, 4))
        plt.plot(epochs_wer, wer_epochs, 'o-', marker='o', linewidth=2, markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('WER')
        plt.title('Word Error Rate over Training (Real Data)')
        plt.grid(True, alpha=0.3)
        
        summary['final_wer'] = float(wer_epochs[-1])
        summary['best_wer'] = float(min(wer_epochs))
        summary['best_wer_epoch'] = int(np.argmin(wer_epochs) + 1)
    else:
        # Simulated WER for demonstration
        wer_epochs = np.clip(np.linspace(0.9, 0.08, num=len(epochs)) + np.random.normal(0, 0.02, len(epochs)), 0, 1)
        plt.figure(figsize=(10, 4))
        plt.plot(epochs, wer_epochs, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('WER')
        plt.title('WER over Training (Simulated - Provide real data!)')
        plt.grid(True, alpha=0.3)
    
    wer_path = os.path.join(out_dir, 'report_wer_epochs.png')
    plt.tight_layout()
    plt.savefig(wer_path, dpi=300)
    plt.close()
    print('Saved:', wer_path)

    # Per-utterance WER analysis
    if has_predictions:
        # Compute per-utterance WER from real data
        y_true_list = predictions_data['y_true']
        y_pred_list = predictions_data['y_pred']
        
        per_utt_wer = []
        utt_lengths = []
        
        for true_text, pred_text in zip(y_true_list, y_pred_list):
            # Compute WER for each utterance
            wer = jiwer.wer(true_text, pred_text)
            per_utt_wer.append(wer)
            # Count words in ground truth
            utt_lengths.append(len(true_text.split()))
        
        per_utt_wer = np.array(per_utt_wer)
        utt_lengths = np.array(utt_lengths)
        
        summary['avg_per_utterance_wer'] = float(np.mean(per_utt_wer))
        summary['median_per_utterance_wer'] = float(np.median(per_utt_wer))
    else:
        # Simulated per-utterance data
        n_utts = 40
        utt_lengths = np.random.randint(1, 20, size=n_utts)
        per_utt_wer = np.clip(0.4 - 0.02 * utt_lengths + np.random.normal(0, 0.05, n_utts), 0, 1)
    plt.figure(figsize=(8, 4))
    plt.scatter(utt_lengths, per_utt_wer, c=per_utt_wer, cmap='viridis', alpha=0.9, s=50)
    plt.colorbar(label='WER')
    plt.xlabel('Utterance length (words)')
    plt.ylabel('WER')
    data_type = 'Real Data' if has_predictions else 'Simulated'
    plt.title(f'Per-utterance WER vs Utterance Length ({data_type})')
    plt.grid(True, alpha=0.3)
    perutt_path = os.path.join(out_dir, 'report_per_utt_wer.png')
    plt.tight_layout()
    plt.savefig(perutt_path, dpi=300)
    plt.close()
    print('Saved:', perutt_path)

    summary['avg_utterance_length_words'] = float(np.mean(utt_lengths))
    if not has_predictions:
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

    # 11) Comprehensive Confusion Matrix Analysis
    if has_predictions:
        from sklearn.metrics import confusion_matrix, classification_report
        from collections import Counter
        
        y_true_list = predictions_data['y_true']
        y_pred_list = predictions_data['y_pred']
        
        # === Character-Level Confusion Matrix ===
        all_true_chars = []
        all_pred_chars = []
        
        for true_text, pred_text in zip(y_true_list, y_pred_list):
            max_len = max(len(true_text), len(pred_text))
            true_padded = true_text.ljust(max_len)
            pred_padded = pred_text.ljust(max_len)
            all_true_chars.extend(list(true_padded))
            all_pred_chars.extend(list(pred_padded))
        
        char_counts = Counter(all_true_chars)
        top_chars = [char for char, _ in char_counts.most_common(20)]
        
        filtered_true = [c for c in all_true_chars if c in top_chars]
        filtered_pred = [c for c in all_pred_chars if c in top_chars]
        
        char_to_idx = {c: i for i, c in enumerate(top_chars)}
        y_true_idx = [char_to_idx.get(c, -1) for c in filtered_true]
        y_pred_idx = [char_to_idx.get(c, -1) for c in filtered_pred]
        
        valid_pairs = [(t, p) for t, p in zip(y_true_idx, y_pred_idx) if t >= 0 and p >= 0]
        if valid_pairs:
            y_true_idx, y_pred_idx = zip(*valid_pairs)
            conf = confusion_matrix(y_true_idx, y_pred_idx)
            conf_normalized = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
            
            # Main confusion matrix (raw counts)
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Raw counts
            im1 = axes[0].imshow(conf, interpolation='nearest', cmap='Blues')
            axes[0].set_title('Character Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
            plt.colorbar(im1, ax=axes[0])
            
            tick_marks = np.arange(len(top_chars))
            display_labels = [repr(c) if c != ' ' else "'space'" for c in top_chars]
            axes[0].set_xticks(tick_marks)
            axes[0].set_xticklabels(display_labels, rotation=45, ha='right')
            axes[0].set_yticks(tick_marks)
            axes[0].set_yticklabels(display_labels)
            axes[0].set_ylabel('True Character', fontsize=12)
            axes[0].set_xlabel('Predicted Character', fontsize=12)
            
            # Add text annotations for high values
            thresh = conf.max() / 2
            for i in range(len(conf)):
                for j in range(len(conf)):
                    if conf[i, j] > conf.max() * 0.1:  # Only show significant values
                        axes[0].text(j, i, int(conf[i, j]),
                                   ha="center", va="center",
                                   color="white" if conf[i, j] > thresh else "black",
                                   fontsize=8)
            
            # Normalized confusion matrix (percentages)
            im2 = axes[1].imshow(conf_normalized, interpolation='nearest', cmap='Reds', vmin=0, vmax=1)
            axes[1].set_title('Normalized Confusion (% of True Label)', fontsize=14, fontweight='bold')
            plt.colorbar(im2, ax=axes[1], format='%.2f')
            
            axes[1].set_xticks(tick_marks)
            axes[1].set_xticklabels(display_labels, rotation=45, ha='right')
            axes[1].set_yticks(tick_marks)
            axes[1].set_yticklabels(display_labels)
            axes[1].set_ylabel('True Character', fontsize=12)
            axes[1].set_xlabel('Predicted Character', fontsize=12)
            
            plt.tight_layout()
            
            # Analyze confusion patterns
            confusion_pairs = []
            for i in range(len(conf)):
                for j in range(len(conf)):
                    if i != j and conf[i, j] > 0:
                        confusion_pairs.append({
                            'true': top_chars[i],
                            'pred': top_chars[j],
                            'count': int(conf[i, j]),
                            'rate': float(conf_normalized[i, j])
                        })
            confusion_pairs.sort(key=lambda x: x['count'], reverse=True)
            
            # Store top confusions in summary
            summary['top_10_character_confusions'] = [
                f"{p['true']}→{p['pred']}: {p['count']} ({p['rate']:.1%})" 
                for p in confusion_pairs[:10]
            ]
            summary['character_accuracy'] = float(np.trace(conf) / np.sum(conf))
            
            # Calculate per-character accuracy
            per_char_acc = {}
            for i, char in enumerate(top_chars):
                if conf[i].sum() > 0:
                    per_char_acc[char] = float(conf[i, i] / conf[i].sum())
            summary['worst_recognized_chars'] = sorted(per_char_acc.items(), key=lambda x: x[1])[:5]
            summary['best_recognized_chars'] = sorted(per_char_acc.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Save main confusion matrix
            conf_path = os.path.join(out_dir, 'report_confusion_matrix.png')
            plt.savefig(conf_path, dpi=300)
            plt.close()
            print(f'Saved: {conf_path} (with real data)')
            
            # === Additional Confusion-Related Visualizations ===
            
            # 1) Top confusion pairs bar chart
            top_confusions = confusion_pairs[:15]
            if top_confusions:
                fig, ax = plt.subplots(figsize=(10, 6))
                labels = [f"'{p['true']}' → '{p['pred']}'" for p in top_confusions]
                counts = [p['count'] for p in top_confusions]
                colors_bar = plt.cm.Reds(np.linspace(0.4, 0.9, len(top_confusions)))
                
                bars = ax.barh(labels, counts, color=colors_bar)
                ax.set_xlabel('Number of Confusions', fontsize=12)
                ax.set_title('Top 15 Character Confusions', fontsize=14, fontweight='bold')
                ax.invert_yaxis()
                ax.grid(axis='x', alpha=0.3)
                
                for bar, count in zip(bars, counts):
                    ax.text(count + max(counts) * 0.01, bar.get_y() + bar.get_height()/2, 
                           str(count), va='center', fontsize=9)
                
                plt.tight_layout()
                conf_bars_path = os.path.join(out_dir, 'report_confusion_top_pairs.png')
                plt.savefig(conf_bars_path, dpi=300)
                plt.close()
                print(f'Saved: {conf_bars_path}')
            
            # 2) Per-character accuracy bar chart
            if per_char_acc:
                sorted_chars = sorted(per_char_acc.items(), key=lambda x: x[1])
                chars_display = [repr(c) if c != ' ' else "'space'" for c, _ in sorted_chars]
                accuracies = [acc for _, acc in sorted_chars]
                
                fig, ax = plt.subplots(figsize=(12, max(6, len(chars_display) * 0.3)))
                colors_acc = ['#d63031' if acc < 0.7 else '#fdcb6e' if acc < 0.9 else '#00b894' 
                             for acc in accuracies]
                
                bars = ax.barh(chars_display, accuracies, color=colors_acc)
                ax.set_xlabel('Recognition Accuracy', fontsize=12)
                ax.set_title('Per-Character Recognition Accuracy', fontsize=14, fontweight='bold')
                ax.set_xlim(0, 1)
                ax.axvline(0.9, color='green', linestyle='--', alpha=0.5, label='90% threshold')
                ax.axvline(0.7, color='orange', linestyle='--', alpha=0.5, label='70% threshold')
                ax.legend()
                ax.grid(axis='x', alpha=0.3)
                
                for bar, acc in zip(bars, accuracies):
                    ax.text(acc + 0.02, bar.get_y() + bar.get_height()/2, 
                           f'{acc:.1%}', va='center', fontsize=8)
                
                plt.tight_layout()
                char_acc_path = os.path.join(out_dir, 'report_char_accuracy.png')
                plt.savefig(char_acc_path, dpi=300)
                plt.close()
                print(f'Saved: {char_acc_path}')
            
            # 3) Word-level error analysis
            word_errors = {'insertions': 0, 'deletions': 0, 'substitutions': 0, 'correct': 0}
            for true_text, pred_text in zip(y_true_list, y_pred_list):
                true_words = true_text.split()
                pred_words = pred_text.split()
                
                if len(pred_words) > len(true_words):
                    word_errors['insertions'] += len(pred_words) - len(true_words)
                elif len(pred_words) < len(true_words):
                    word_errors['deletions'] += len(true_words) - len(pred_words)
                
                for tw, pw in zip(true_words, pred_words):
                    if tw == pw:
                        word_errors['correct'] += 1
                    else:
                        word_errors['substitutions'] += 1
            
            # 4) Error breakdown pie chart
            error_types = ['Substitutions', 'Insertions', 'Deletions', 'Correct']
            error_counts = [word_errors['substitutions'], word_errors['insertions'], 
                          word_errors['deletions'], word_errors['correct']]
            
            if sum(error_counts) > 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                colors_pie = ['#e74c3c', '#e67e22', '#f39c12', '#27ae60']
                explode = (0.05, 0.05, 0.05, 0)
                
                wedges, texts, autotexts = ax.pie(error_counts, labels=error_types, autopct='%1.1f%%',
                                                   colors=colors_pie, explode=explode, startangle=90)
                ax.set_title('Word-Level Error Distribution', fontsize=14, fontweight='bold')
                
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                    autotext.set_fontsize(11)
                
                plt.tight_layout()
                error_dist_path = os.path.join(out_dir, 'report_error_distribution.png')
                plt.savefig(error_dist_path, dpi=300)
                plt.close()
                print(f'Saved: {error_dist_path}')
                
                summary['word_error_breakdown'] = word_errors
                summary['word_accuracy'] = float(word_errors['correct'] / sum(error_counts))
        else:
            plt.figure(figsize=(6, 5))
            plt.text(0.5, 0.5, 'Not enough data for\nconfusion matrix', 
                    ha='center', va='center', fontsize=14)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            conf_path = os.path.join(out_dir, 'report_confusion_matrix.png')
            plt.savefig(conf_path, dpi=300)
            plt.close()
            print(f'Saved: {conf_path} (insufficient data)')
    else:
        # Placeholder when no predictions available
        plt.figure(figsize=(6, 5))
        plt.text(0.5, 0.5, 
                'Confusion Matrix:\n\nProvide predictions_data to generate_report_assets()\n\n' +
                'predictions_data = {\n' +
                '    "y_true": [...],\n' +
                '    "y_pred": [...]\n' +
                '}',
                ha='center', va='center', fontsize=10, family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')
        plt.title('Confusion Matrix (Provide real data!)')
        conf_path = os.path.join(out_dir, 'report_confusion_matrix.png')
        plt.savefig(conf_path, dpi=300)
        plt.close()
        print(f'Saved: {conf_path} (placeholder - provide predictions_data)')

    # 12) Write summary CSV
    summary_df = pd.DataFrame([summary])
    csv_path = os.path.join(out_dir, 'report_summary.csv')
    summary_df.to_csv(csv_path, index=False)
    print('Saved:', csv_path)

    # Also write JSON summary for easy presentation consumption
    json_path = os.path.join(out_dir, 'report_summary.json')
    with open(json_path, 'w') as jf:
        json.dump(summary, jf, indent=2)
    print('Saved:', json_path)

    # Create a combined PDF with all generated PNGs for presentation
    try:
        png_files = [p for p in os.listdir(out_dir) if p.endswith('.png')]
        png_files = sorted(png_files)
        pdf_path = os.path.join(out_dir, 'report_all_plots.pdf')
        with PdfPages(pdf_path) as pdf:
            for png in png_files:
                fig = plt.figure()
                img = plt.imread(os.path.join(out_dir, png))
                plt.imshow(img)
                plt.axis('off')
                pdf.savefig(fig)
                plt.close(fig)
        print('Saved:', pdf_path)
    except Exception:
        print('Could not create combined PDF (missing backends or libraries).')

    print('All assets generated.')


def plot_confusion_matrix(y_true, y_pred, labels=None, out_dir='.', normalize=False):
    """
    Compute and plot a real confusion matrix from predictions.
    
    Args:
        y_true: ground truth labels (1D array or list)
        y_pred: predicted labels (1D array or list)
        labels: list of label names (optional, will use unique values if None)
        out_dir: output directory for saving the plot
        normalize: if True, normalize confusion matrix by row (show percentages)
    
    Returns:
        confusion matrix as numpy array
    
    Usage example:
        # After running inference on validation set
        y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
        y_pred = [0, 1, 1, 0, 1, 2, 0, 2, 2]
        plot_confusion_matrix(y_true, y_pred, labels=['class_a', 'class_b', 'class_c'])
    """
    from sklearn.metrics import confusion_matrix
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
        labels = [str(l) for l in labels]
    
    conf = confusion_matrix(y_true, y_pred)
    
    if normalize:
        conf = conf.astype('float') / conf.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        fmt = 'd'
        title = 'Confusion Matrix'
    
    plt.figure(figsize=(max(8, len(labels) * 0.8), max(6, len(labels) * 0.7)))
    im = plt.imshow(conf, interpolation='nearest', cmap='Blues')
    plt.title(title)
    plt.colorbar(im)
    
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha='right')
    plt.yticks(tick_marks, labels)
    
    # Add text annotations
    thresh = conf.max() / 2.
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            plt.text(j, i, format(conf[i, j], fmt),
                    ha="center", va="center",
                    color="white" if conf[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    conf_path = os.path.join(out_dir, 'report_confusion_matrix_real.png')
    plt.savefig(conf_path, dpi=300)
    plt.close()
    print('Saved real confusion matrix:', conf_path)
    
    return conf


def _parse_args():
    p = argparse.ArgumentParser(
        description='Generate detailed audio/model report assets',
        epilog="""
Examples:
  # Generate report with audio analysis only (simulated training data)
  python report.py --audio input.wav --outdir reports/

  # Generate report with real predictions (call from Python):
  from report import generate_report_assets
  
  predictions_data = {
      'y_true': ["hello world", "this is a test"],
      'y_pred': ["helo world", "this is test"],
      'train_losses': [2.5, 2.1, 1.8, 1.5, 1.3],
      'val_losses': [2.7, 2.3, 1.9, 1.6, 1.5],
      'wer_per_epoch': [0.85, 0.62, 0.45, 0.32, 0.18]
  }
  
  generate_report_assets(
      audio_path='sample.wav',
      out_dir='reports/',
      predictions_data=predictions_data
  )
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument('--audio', '-a', help='Path to WAV audio file to analyze', default=None)
    p.add_argument('--outdir', '-o', help='Output directory', default='reports')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    print("\n" + "="*70)
    print("NOTE: Running from command line generates audio analysis only.")
    print("To generate reports with REAL model predictions, call from Python:")
    print("="*70)
    print("""
from report import generate_report_assets

predictions_data = {
    'y_true': ["ground truth text 1", "ground truth text 2", ...],
    'y_pred': ["predicted text 1", "predicted text 2", ...],
    'train_losses': [loss values per epoch],
    'val_losses': [val loss values per epoch],
    'wer_per_epoch': [WER values per epoch]
}

generate_report_assets(
    audio_path='your_audio.wav',
    out_dir='reports/',
    predictions_data=predictions_data  # <-- This is the key!
)
    """)
    print("="*70 + "\n")
    
    generate_report_assets(audio_path=args.audio, out_dir=args.outdir)