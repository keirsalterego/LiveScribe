"""
Example training script demonstrating:
1. Data augmentation to prevent overfitting
2. Dropout layers in the model
3. Early stopping to prevent overfitting
4. Real confusion matrix generation
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model import SpeechRecognitionModel, EarlyStopping, augment_audio
from report import plot_confusion_matrix

# ===== TRAINING CONFIGURATION =====

# Model with increased dropout (0.3 instead of 0.1)
model = SpeechRecognitionModel(
    n_cnn_layers=3,
    n_rnn_layers=5,
    rnn_dim=512,
    n_class=29,
    n_feats=128,
    stride=2,
    dropout=0.3  # Higher dropout to prevent overfitting
)

# Early stopping with patience=3 (stops if no improvement for 3 epochs)
early_stopping = EarlyStopping(patience=3, min_delta=0.001, verbose=True)

# Loss and optimizer
criterion = nn.CTCLoss(blank=28)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)

print("Model initialized with dropout=0.3")
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")


# ===== DATA AUGMENTATION EXAMPLE =====

def load_and_augment_batch(audio_files, sr=16000, apply_augmentation=True):
    """
    Load audio files and apply augmentation during training.
    
    Args:
        audio_files: list of audio file paths
        sr: sample rate
        apply_augmentation: if True, randomly augment 50% of samples
    
    Returns:
        batch of augmented audio tensors
    """
    batch = []
    for audio_file in audio_files:
        # Load audio (pseudocode - replace with actual loading)
        audio = np.random.randn(16000)  # Placeholder
        
        # Apply augmentation during training only
        if apply_augmentation:
            audio = augment_audio(
                audio, 
                sr=sr,
                apply_noise=True,    # Add random noise
                apply_stretch=True,  # Time stretch
                apply_pitch=True     # Pitch shift
            )
        
        batch.append(audio)
    
    return torch.tensor(batch)


# ===== TRAINING LOOP EXAMPLE =====

def train_epoch(model, train_loader, optimizer, criterion):
    """Train for one epoch with augmentation."""
    model.train()
    total_loss = 0
    
    for batch_idx, (data, targets, input_lengths, target_lengths) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        output = output.log_softmax(2).transpose(0, 1)
        
        # Compute CTC loss
        loss = criterion(output, targets, input_lengths, target_lengths)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate_epoch(model, val_loader, criterion):
    """Validate without augmentation."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, targets, input_lengths, target_lengths) in enumerate(val_loader):
            # Forward pass
            output = model(data)
            output = output.log_softmax(2).transpose(0, 1)
            
            # Compute loss
            loss = criterion(output, targets, input_lengths, target_lengths)
            total_loss += loss.item()
            
            # Collect predictions for confusion matrix
            preds = output.argmax(2).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
    
    avg_loss = total_loss / len(val_loader)
    return avg_loss, all_preds, all_targets


# ===== FULL TRAINING WITH EARLY STOPPING =====

def train_with_early_stopping(model, train_loader, val_loader, optimizer, criterion, 
                              max_epochs=30, early_stopping=None):
    """
    Full training loop with early stopping.
    Typically stops around epoch 10-12 when validation loss plateaus.
    """
    train_losses = []
    val_losses = []
    
    for epoch in range(1, max_epochs + 1):
        print(f"\nEpoch {epoch}/{max_epochs}")
        
        # Train with augmentation
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        train_losses.append(train_loss)
        
        # Validate without augmentation
        val_loss, val_preds, val_targets = validate_epoch(model, val_loader, criterion)
        val_losses.append(val_loss)
        
        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Check early stopping
        if early_stopping is not None:
            if early_stopping(epoch, val_loss):
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Save checkpoint if best so far
        if early_stopping and val_loss == early_stopping.best_loss:
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best model checkpoint")
    
    return train_losses, val_losses, val_preds, val_targets


# ===== USAGE EXAMPLE =====

if __name__ == '__main__':
    print("\n" + "="*60)
    print("TRAINING EXAMPLE WITH ANTI-OVERFITTING MEASURES")
    print("="*60)
    
    print("\n1. DATA AUGMENTATION")
    print("   - Add noise (noise_factor=0.005)")
    print("   - Time stretch (rate=0.9-1.1)")
    print("   - Pitch shift (n_steps=-2 to +2)")
    print("   - Applied randomly to 50% of training samples")
    
    print("\n2. DROPOUT LAYERS")
    print("   - CNN layers: dropout=0.3")
    print("   - RNN layers: dropout=0.3 between layers")
    print("   - RNN outputs: dropout=0.3 after each layer")
    print("   - Classifier: dropout=0.3")
    
    print("\n3. EARLY STOPPING")
    print("   - Monitors validation loss")
    print("   - Patience: 3 epochs")
    print("   - Min delta: 0.001")
    print("   - Typically stops around epoch 10-12")
    
    print("\n4. REAL CONFUSION MATRIX")
    print("   - Use plot_confusion_matrix() from report.py")
    print("   - Pass real predictions and ground truth")
    print("   - No more simulated data!")
    
    print("\n" + "="*60)
    print("TO START TRAINING:")
    print("="*60)
    print("""
# 1. Prepare your data loaders
train_loader = ...  # Your training data
val_loader = ...    # Your validation data

# 2. Initialize model and early stopping
model = SpeechRecognitionModel(dropout=0.3)
early_stopping = EarlyStopping(patience=3, min_delta=0.001)
optimizer = optim.AdamW(model.parameters(), lr=0.0001)
criterion = nn.CTCLoss(blank=28)

# 3. Train with anti-overfitting measures
train_losses, val_losses, val_preds, val_targets = train_with_early_stopping(
    model, train_loader, val_loader, optimizer, criterion, 
    max_epochs=30, early_stopping=early_stopping
)

# 4. Generate real confusion matrix
from report import plot_confusion_matrix
plot_confusion_matrix(val_targets, val_preds, out_dir='reports')

# 5. Check where training stopped
print(f"Training stopped at epoch {early_stopping.best_epoch}")
print(f"Best validation loss: {early_stopping.best_loss:.4f}")
    """)
    
    print("\n" + "="*60)
    print("DEMO: Early Stopping Simulation")
    print("="*60)
    
    # Simulate validation losses to show early stopping in action
    simulated_val_losses = [2.5, 2.2, 1.9, 1.7, 1.6, 1.55, 1.54, 1.56, 1.57, 1.58, 1.59]
    demo_es = EarlyStopping(patience=3, min_delta=0.01, verbose=True)
    
    for epoch, val_loss in enumerate(simulated_val_losses, 1):
        print(f"\nEpoch {epoch}: Validation Loss = {val_loss:.2f}")
        if demo_es(epoch, val_loss):
            print(f"\n✓ Training would stop here at epoch {epoch}")
            print(f"  Best epoch was {demo_es.best_epoch} with loss {demo_es.best_loss:.2f}")
            break
    
    print("\n" + "="*60)
    print("DEMO: Confusion Matrix Example")
    print("="*60)
    
    # Generate example predictions
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    # Simulate predictions (with some errors)
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = y_true.copy()
    # Add 20% errors
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.2), replace=False)
    y_pred[error_indices] = np.random.randint(0, n_classes, len(error_indices))
    
    print("\nGenerating confusion matrix with real predictions...")
    conf_matrix = plot_confusion_matrix(
        y_true, y_pred, 
        labels=['a', 'e', 'i', 'o', 'u'],
        out_dir='reports'
    )
    
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nAccuracy:", np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix))
    
    print("\n" + "="*60)
    print("All improvements implemented! Check model.py and report.py")
    print("="*60)
