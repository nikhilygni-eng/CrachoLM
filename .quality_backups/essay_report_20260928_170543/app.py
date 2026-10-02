#!/usr/bin/env python3
"""
CrachoLM Web Studio Backend Server
==================================
Serves the web application UI and handles model generation API requests.
Built with Python standard library HTTP server (no extra dependencies required).
"""

import argparse
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
from src.assistant_response import generate_assistant_reply
from src.checkpoint import load_checkpoint_file, get_model_config_from_checkpoint
from src.inference import load_model_for_inference

# Global State for Model & Tokenizer
MODEL = None
TOKENIZER = None
DEVICE = None
MODEL_INFO = {}


def init_model(checkpoint_path=None, tokenizer_path=None):
    global MODEL, TOKENIZER, DEVICE, MODEL_INFO
    torch.set_num_threads(2)
    DEVICE = get_device()
    MODEL, TOKENIZER, MODEL_INFO = load_model_for_inference(
        checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path, device=DEVICE
    )
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
                prompt = body.get("prompt", "")
                if not isinstance(prompt, str) or not prompt.strip():
                    raise ValueError("Enter a message.")
                mode = body.get("mode", "chat")
                if mode not in ("chat", "completion"):
                    raise ValueError("Mode must be chat or completion.")
                max_new_tokens = max(1, min(300, int(body.get("max_new_tokens", 80))))
                temperature = float(body.get("temperature", 0.7))
                top_k = int(body.get("top_k", 40))
                greedy = bool(body.get("greedy", True))
                use_tools = body.get("use_tools", True)
                if not isinstance(use_tools, bool):
                    raise ValueError("use_tools must be true or false.")

                if mode == "chat":
                    reply, max_new_tokens, source = generate_assistant_reply(
                        MODEL, TOKENIZER, prompt, use_tools=use_tools,
                        max_new_tokens=max_new_tokens, temperature=temperature,
                        top_k=top_k, greedy=greedy, device=DEVICE,
                    )
                    generated_text = prompt + "\n" + reply
                else:
                    source = "model"
                    generated_text = generate_text(
                        model=MODEL, tokenizer=TOKENIZER, prompt=prompt,
                        max_new_tokens=max_new_tokens, temperature=temperature,
                        top_k=top_k, greedy=greedy, device=DEVICE,
                    )
                    reply = generated_text[len(prompt):]

                response_payload = {
                    "status": "success",
                    "mode": mode,
                    "source": source,
                    "reply": reply,
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

            except (ValueError, TypeError) as e:
                self.send_json_response(400, {"status": "error", "message": str(e)})
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


def run_server(port=7860, checkpoint_path=None, tokenizer_path=None):
    init_model(checkpoint_path, tokenizer_path)
    server_address = ("127.0.0.1", port)
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
    parser = argparse.ArgumentParser(description="CrachoLM Web Studio")
    parser.add_argument("port", nargs="?", type=int, default=7860)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--tokenizer", default=None)
    args = parser.parse_args()
    run_server(args.port, args.checkpoint, args.tokenizer)
