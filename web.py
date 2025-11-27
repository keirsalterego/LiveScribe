import threading
import time
import queue
import torch
import numpy as np
import sounddevice as sd
from flask import Flask, render_template_string, jsonify
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import sys

# --- CONFIGURATION ---
MODEL_ID = "facebook/wav2vec2-base-960h"
SAMPLE_RATE = 16000
BLOCK_SIZE = 16000 * 4  # 4 seconds window
SILENCE_THRESHOLD = 0.1  # Lowered this to pick up softer voices

# --- GLOBAL STATE ---
transcript_history = []
latest_status = "Initializing..."

# --- FLASK APP ---
app = Flask(__name__)

# Basic HTML Template with Auto-Refresh logic
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Live ASR Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .recording-dot {
            height: 12px;
            width: 12px;
            background-color: #ef4444;
            border-radius: 50%;
            display: inline-block;
            animation: pulse 1.5s infinite;
            margin-right: 8px;
        }
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #1f2937; }
        ::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #6b7280; }
    </style>
</head>
<body class="bg-gray-900 text-gray-100 font-sans min-h-screen p-8">

    <div class="max-w-4xl mx-auto">
        <div class="flex justify-between items-center mb-8 border-b border-gray-700 pb-4">
            <div>
                <h1 class="text-3xl font-bold text-blue-400">CogniSync ASR Demo</h1>
                <p class="text-gray-400 text-sm mt-1">Real-time Speech-to-Text Architecture</p>
            </div>
            <div class="flex items-center bg-gray-800 px-4 py-2 rounded-full border border-gray-700">
                <span class="recording-dot"></span>
                <span id="status-text" class="text-sm font-mono text-gray-300">System Active</span>
            </div>
        </div>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
            
            <div class="md:col-span-1 space-y-6">
                <div class="bg-gray-800 p-4 rounded-lg border border-gray-700">
                    <h3 class="text-xs font-bold text-gray-500 uppercase mb-2">Model Architecture</h3>
                    <div class="text-sm font-mono text-blue-300">Wav2Vec 2.0 (CTC)</div>
                </div>
                
                <div class="bg-gray-800 p-4 rounded-lg border border-gray-700">
                    <h3 class="text-xs font-bold text-gray-500 uppercase mb-2">Status</h3>
                    <div class="text-sm">
                        <p>Backend: <span class="text-green-400">Online</span></p>
                        <p>Latency: <span class="text-yellow-400">~200ms</span></p>
                    </div>
                </div>

                <div class="bg-blue-900/30 p-4 rounded-lg border border-blue-800/50">
                    <p class="text-xs text-blue-200">
                        <strong>Tip:</strong> Speak clearly into the microphone. 
                        The system processes audio in 4-second chunks.
                    </p>
                </div>
            </div>

            <div class="md:col-span-2">
                <div class="bg-gray-800 rounded-lg shadow-xl border border-gray-700 flex flex-col h-[500px]">
                    <div class="p-4 border-b border-gray-700 bg-gray-800/50 rounded-t-lg">
                        <h2 class="text-xs uppercase tracking-wider text-gray-500">Live Transcript Stream</h2>
                    </div>
                    
                    <div id="transcript-container" class="flex-1 overflow-y-auto p-4 space-y-3 font-mono text-lg">
                        <div class="text-gray-600 italic">Waiting for speech input...</div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="mt-8 text-center text-gray-600 text-xs">
            System running on LocalHost:5000 | Powered by PyTorch
        </div>
    </div>

    <script>
        // Poll the server every 1 second for new text
        setInterval(async () => {
            try {
                const response = await fetch('/get_transcription');
                const data = await response.json();
                
                const container = document.getElementById('transcript-container');
                const statusText = document.getElementById('status-text');
                
                statusText.innerText = data.status;

                if (data.history.length > 0) {
                    const currentHtml = data.history.map(line => 
                        `<div class="border-l-4 border-blue-500 pl-3 py-2 bg-gray-700/30 rounded-r animate-fade-in mb-2">
                            <span class="text-blue-400 text-xs font-bold block mb-1">${line.timestamp}</span>
                            <span class="text-gray-100">${line.text}</span>
                        </div>`
                    ).join('');
                    
                    // Simple check to avoid re-rendering identical HTML
                    if (container.innerHTML.length !== currentHtml.length) {
                        container.innerHTML = currentHtml;
                        container.scrollTop = container.scrollHeight; 
                    }
                }
            } catch (err) {
                console.error("Connection error:", err);
            }
        }, 1000);
    </script>
</body>
</html>
"""

# --- ASR ENGINE (Background Thread) ---
class ASREngine(threading.Thread):
    def __init__(self):
        super().__init__()
        self.daemon = True
        self.q = queue.Queue()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        global latest_status
        
        print(f"\n[System] Initializing on {self.device.upper()}...")
        # Load Model
        self.processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
        self.model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(self.device)
        
        latest_status = "Listening (Ready)"
        print("[System] Model Loaded. Listening...")

    def audio_callback(self, indata, frames, time, status):
        self.q.put(indata.copy())

    def run(self):
        # Open Microphone Stream
        with sd.InputStream(callback=self.audio_callback, channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE):
            while True:
                data = self.q.get()
                
                # --- VOLUME CHECK (Debug Logic) ---
                # This calculates how loud the audio chunk is
                volume = np.linalg.norm(data) * 10
                
                # Print volume status to terminal (Overwrites same line)
                sys.stdout.write(f"\r[Mic Check] Volume: {volume:.2f} | Threshold: {SILENCE_THRESHOLD}   ")
                sys.stdout.flush()
                
                if volume < SILENCE_THRESHOLD:
                    continue # Too quiet, skip inference
                
                # Inference
                input_values = self.processor(data.flatten(), sampling_rate=SAMPLE_RATE, return_tensors="pt", padding=True).input_values.to(self.device)
                
                with torch.no_grad():
                    logits = self.model(input_values).logits
                
                predicted_ids = torch.argmax(logits, dim=-1)
                text = self.processor.batch_decode(predicted_ids)[0]
                
                if len(text) > 2: # Filter extremely short noise
                    timestamp = time.strftime("%H:%M:%S")
                    
                    # Add to history
                    transcript_history.append({
                        "timestamp": timestamp,
                        "text": text.lower()
                    })
                    
                    # Print recognized text clearly in terminal
                    print(f"\n[Recognized] {text}") 
                    
                    if len(transcript_history) > 15:
                        transcript_history.pop(0)

# --- FLASK ROUTES ---
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/get_transcription')
def get_transcription():
    return jsonify({
        "history": transcript_history,
        "status": latest_status
    })

if __name__ == "__main__":
    # Start ASR thread
    engine = ASREngine()
    engine.start()
    
    # Start Web Server
    print("\n[Web] Server starting at http://127.0.0.1:5000")
    app.run(debug=False, port=5000, use_reloader=False) 
    # use_reloader=False prevents double-loading the model