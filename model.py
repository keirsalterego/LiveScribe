import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa

class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn_layers=3, n_rnn_layers=5, rnn_dim=512, n_class=29, n_feats=128, stride=2, dropout=0.3):
        super(SpeechRecognitionModel, self).__init__()
        
        # --- 1. CNN Feature Extraction (The "Ear") ---
        # Extracts spatial features from the Spectrogram
        self.cnn = nn.Conv2d(1, 32, 3, stride=stride, padding=1)
        self.cnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(32, 32, 3, stride=1, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(n_cnn_layers)
        ])
        
        # Calculate size to feed into RNN
        self.fully_connected = nn.Linear(n_feats // stride * 32, rnn_dim)
        
        # --- 2. RNN Sequence Modeling (The "Brain") ---
        # Processes features over time (Bi-Directional GRU)
        self.birnn_layers = nn.ModuleList([
            nn.GRU(rnn_dim if i==0 else rnn_dim*2, rnn_dim, 
                   batch_first=True, bidirectional=True, dropout=dropout if i < n_rnn_layers - 1 else 0)
            for i in range(n_rnn_layers)
        ])
        
        # Add dropout layers after each RNN
        self.rnn_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(n_rnn_layers)])
        
        # --- 3. Classifier (The "Mouth") ---
        self.classifier = nn.Sequential(
            nn.Linear(rnn_dim * 2, rnn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        # x shape: (batch, channel, feature, time)
        x = self.cnn(x)
        for layer in self.cnn_layers:
            x = layer(x)
        
        # Prepare for RNN: Flatten features, keep time
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3]) 
        x = x.transpose(1, 2) # (batch, time, feature)
        
        x = self.fully_connected(x)
        
        for i, rnn in enumerate(self.birnn_layers):
            x, _ = rnn(x)
            x = self.rnn_dropouts[i](x)
            
        x = self.classifier(x)
        return x


# ===== DATA AUGMENTATION FUNCTIONS =====
# Apply these during training to prevent overfitting

def add_noise(audio, noise_factor=0.005):
    """
    Add random Gaussian noise to audio.
    Args:
        audio: numpy array of audio samples
        noise_factor: amount of noise (default 0.005)
    Returns:
        augmented audio
    """
    noise = np.random.randn(len(audio))
    augmented = audio + noise_factor * noise
    return augmented.astype(audio.dtype)


def time_stretch(audio, rate=None):
    """
    Stretch or compress audio in time without changing pitch.
    Args:
        audio: numpy array of audio samples
        rate: stretch factor (0.8-1.2 recommended). If None, random in [0.9, 1.1]
    Returns:
        time-stretched audio
    """
    if rate is None:
        rate = np.random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio, sr=16000, n_steps=None):
    """
    Shift pitch of audio up or down.
    Args:
        audio: numpy array of audio samples
        sr: sample rate
        n_steps: semitones to shift (-2 to 2 recommended). If None, random in [-2, 2]
    Returns:
        pitch-shifted audio
    """
    if n_steps is None:
        n_steps = np.random.uniform(-2, 2)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def augment_audio(audio, sr=16000, apply_noise=True, apply_stretch=True, apply_pitch=True):
    """
    Apply multiple augmentations randomly to audio.
    Args:
        audio: numpy array of audio samples
        sr: sample rate
        apply_noise: whether to add noise
        apply_stretch: whether to apply time stretching
        apply_pitch: whether to apply pitch shifting
    Returns:
        augmented audio
    """
    if apply_noise and np.random.rand() > 0.5:
        audio = add_noise(audio)
    if apply_stretch and np.random.rand() > 0.5:
        audio = time_stretch(audio)
    if apply_pitch and np.random.rand() > 0.5:
        audio = pitch_shift(audio, sr=sr)
    return audio


# ===== EARLY STOPPING =====

class EarlyStopping:
    """
    Early stopping to prevent overfitting.
    Monitors validation loss and stops training when it stops improving.
    """
    def __init__(self, patience=3, min_delta=0.001, verbose=True):
        """
        Args:
            patience: how many epochs to wait after last improvement
            min_delta: minimum change to qualify as improvement
            verbose: print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, epoch, val_loss):
        """
        Call this after each epoch with validation loss.
        Returns True if training should stop.
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
            if self.verbose:
                print(f'EarlyStopping: Initial best loss = {val_loss:.4f}')
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.verbose:
                print(f'EarlyStopping: Loss improved to {val_loss:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping: No improvement for {self.counter} epoch(s)')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print(f'EarlyStopping: Stopping at epoch {epoch}. Best was epoch {self.best_epoch} with loss {self.best_loss:.4f}')
                    
        return self.early_stop