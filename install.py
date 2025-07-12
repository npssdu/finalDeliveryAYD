#!/usr/bin/env python3
"""
Installation script for Drawing with LLMs GUI Simulator
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages."""
    print("Installing basic requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    
def install_optional_llm_packages():
    """Install optional LLM packages based on user choice."""
    print("\nOptional LLM integrations:")
    print("1. Ollama (Recommended - Free, local)")
    print("2. OpenAI (Requires API key)")
    print("3. HuggingFace Transformers (Local models)")
    print("4. Skip LLM installation")
    
    choice = input("Choose an option (1-4): ").strip()
    
    if choice == "1":
        print("Ollama setup: Please download and install Ollama from https://ollama.ai")
        print("Then run: ollama pull llama3.1:8b")
    elif choice == "2":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openai>=1.3.0"])
        print("OpenAI installed. Set OPENAI_API_KEY environment variable.")
    elif choice == "3":
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers>=4.20.0", "torch>=2.0.0"])
        print("HuggingFace Transformers installed.")
    else:
        print("Skipping LLM installation.")

def main():
    """Main installation process."""
    print("=== Drawing with LLMs - GUI Simulator Installation ===")
    
    # Change to gui_simulator directory if not already there
    if os.path.basename(os.getcwd()) != "gui_simulator":
        if os.path.exists("gui_simulator"):
            os.chdir("gui_simulator")
        else:
            print("Error: Please run this script from the project root or gui_simulator directory")
            return
    
    try:
        install_requirements()
        install_optional_llm_packages()
        
        print("\n=== Installation Complete ===")
        print("To run the application:")
        print("  python -m gui_simulator.main")
        print("\nOr if you're in the gui_simulator directory:")
        print("  python main.py")
        
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
