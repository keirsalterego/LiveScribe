import sounddevice as sd
import numpy as np
import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import queue
import sys

# --- CONFIGURATION ---
MODEL_ID = "facebook/wav2vec2-base-960h" # Pre-trained model for demo reliability
SAMPLE_RATE = 16000
BLOCK_SIZE = 16000 * 4 # Process 4 seconds chunks (Latency vs Accuracy trade-off)

class RealTimeASR:
    def __init__(self):
        print("Loading Model... (this might take a minute)")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Pre-trained Transformer
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(self.device)
        
        self.q = queue.Queue()
        print(f"Model loaded on {self.device.upper()}")

    def audio_callback(self, indata, frames, time, status):
        """Called by sounddevice for every audio block"""
        if status:
            print(status, file=sys.stderr)
        # Add a copy of the audio block to the queue
        self.q.put(indata.copy())

    def run(self):
        # Open microphone stream
        with sd.InputStream(callback=self.audio_callback,
                            channels=1,
                            samplerate=SAMPLE_RATE,
                            blocksize=BLOCK_SIZE):
            
            print("\n" + "="*40)
            print("LIVE TRANSCRIPTION STARTED")
            print("Speak into your microphone...")
            print("="*40 + "\n")
            
            while True:
                try:
                    # Get audio from queue
                    audio_data = self.q.get()
                    
                    # Flatten to 1D array
                    audio_input = audio_data.flatten()
                    
                    # Preprocess inputs
                    inputs = self.processor(audio_input, sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True)
                    input_values = inputs.input_values.to(self.device)
                    
                    # Inference
                    with torch.no_grad():
                        logits = self.model(input_values).logits
                    
                    # Decode
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = self.processor.batch_decode(predicted_ids)[0]
                    
                    # Only print if speech is detected
                    if len(transcription.strip()) > 0:
                        print(f"Detected: {transcription}")
                        
                except KeyboardInterrupt:
                    print("\nStopping...")
                    break

if __name__ == "__main__":
    asr = RealTimeASR()
    asr.run()