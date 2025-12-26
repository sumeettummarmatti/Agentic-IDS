"""Local LLM (Ollama) utilities"""

import os
import subprocess
from typing import Optional

class OllamaManager:
    """Manage local Ollama instance"""
    
    def __init__(self, model: str = "deepseek-v3:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
    
    @staticmethod
    def is_running(base_url: str = "http://localhost:11434") -> bool:
        """Check if Ollama is running"""
        import requests
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def start_ollama():
        """Start Ollama service (macOS)"""
        try:
            subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✓ Ollama started")
        except Exception as e:
            print(f"⚠️  Could not auto-start Ollama: {e}")
            print("   Run manually: ollama serve")
    
    @staticmethod
    def pull_model(model: str):
        """Pull model from Ollama registry"""
        try:
            subprocess.run(["ollama", "pull", model], check=True)
            print(f"✓ Model {model} pulled")
        except Exception as e:
            print(f"⚠️  Failed to pull {model}: {e}")

if __name__ == "__main__":
    manager = OllamaManager()
    if not manager.is_running():
        print("Ollama not running. Starting...")
        manager.start_ollama()
    manager.pull_model("deepseek-v3:7b")
