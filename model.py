import torch
import torch.nn as nn
import torch.nn.functional as F

class SpeechRecognitionModel(nn.Module):
    def __init__(self, n_cnn_layers=3, n_rnn_layers=5, rnn_dim=512, n_class=29, n_feats=128, stride=2, dropout=0.1):
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
                   batch_first=True, bidirectional=True)
            for i in range(n_rnn_layers)
        ])
        
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
        
        for rnn in self.birnn_layers:
            x, _ = rnn(x)
            
        x = self.classifier(x)
        return x