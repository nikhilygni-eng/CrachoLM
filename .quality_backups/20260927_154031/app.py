#!/usr/bin/env python3
"""
CrachoLM Web Studio Backend Server
==================================
Serves the web application UI and handles model generation API requests.
Built with Python standard library HTTP server (no extra dependencies required).
"""

import json
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
import torch

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.device import get_device
from src.tokenizer import CrachoTokenizer
from src.model import CrachoLM
from src.generator import generate_text
from src.checkpoint import load_checkpoint_file, get_model_config_from_checkpoint

# Global State for Model & Tokenizer
MODEL = None
TOKENIZER = None
DEVICE = None
MODEL_INFO = {}


def init_model():
    global MODEL, TOKENIZER, DEVICE, MODEL_INFO
    DEVICE = get_device()
    tokenizer_path = os.path.abspath("checkpoints/tokenizer.json")
    checkpoint_path = os.path.abspath("checkpoints/best_model.pt")

    if not os.path.exists(tokenizer_path) or not os.path.exists(checkpoint_path):
        print("[X] ERROR: Missing checkpoint or tokenizer file. Please train model first.")
        sys.exit(1)

    print(f"[*] Loading Tokenizer from {tokenizer_path}...")
    TOKENIZER = CrachoTokenizer.load(tokenizer_path)

    print(f"[*] Loading Checkpoint from {checkpoint_path}...")
    checkpoint = load_checkpoint_file(checkpoint_path, DEVICE)
    model_config = get_model_config_from_checkpoint(checkpoint)

    MODEL = CrachoLM(model_config).to(DEVICE)
    MODEL.load_state_dict(checkpoint["model_state_dict"])
    MODEL.eval()

    n_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
    MODEL_INFO = {
        "model_name": "CrachoLM-0.1",
        "parameters": n_params,
        "parameters_m": round(n_params / 1e6, 2),
        "vocab_size": TOKENIZER.vocab_size,
        "checkpoint_epoch": checkpoint.get("epoch", "unknown"),
        "val_loss": checkpoint.get("val_loss", None),
        "device": str(DEVICE),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }
    print(f"[✓] CrachoLM loaded successfully! ({MODEL_INFO['parameters_m']}M params on {MODEL_INFO['device']})")


class CrachoLMHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve files from web directory
        super().__init__(*args, directory=os.path.join(os.path.dirname(__file__), "web"), **kwargs)

    def do_GET(self):
        if self.path == "/api/info":
            self.send_json_response(200, MODEL_INFO)
        else:
            if self.path == "/":
                self.path = "/index.html"
            super().do_GET()

    def do_POST(self):
        if self.path == "/api/generate":
            content_length = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_length)
            
            try:
                body = json.loads(post_data.decode("utf-8"))
                prompt = body.get("prompt", "First Citizen:")
                max_new_tokens = int(body.get("max_new_tokens", 100))
                temperature = float(body.get("temperature", 0.7))
                top_k = int(body.get("top_k", 40))
                greedy = bool(body.get("greedy", False))

                # Generate Text
                generated_text = generate_text(
                    model=MODEL,
                    tokenizer=TOKENIZER,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    greedy=greedy,
                    device=DEVICE
                )

                response_payload = {
                    "status": "success",
                    "prompt": prompt,
                    "generated_text": generated_text,
                    "parameters_used": {
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                        "top_k": top_k,
                        "greedy": greedy
                    }
                }
                self.send_json_response(200, response_payload)

            except Exception as e:
                self.send_json_response(500, {"status": "error", "message": str(e)})
        else:
            self.send_error(404, "Endpoint Not Found")

    def send_json_response(self, code, data):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))


def run_server(port=7860):
    init_model()
    server_address = ("", port)
    httpd = HTTPServer(server_address, CrachoLMHandler)
    print("\n" + "=" * 68)
    print(f"  🚀 CrachoLM Web Studio is Live at: http://localhost:{port}")
    print("=" * 68 + "\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Server shutting down cleanly...")
        httpd.server_close()


if __name__ == "__main__":
    port = 7860
    if len(sys.argv) > 1:
        port = int(sys.argv[1])
    run_server(port)
