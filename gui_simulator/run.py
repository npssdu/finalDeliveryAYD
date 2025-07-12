#!/usr/bin/env python3
"""
Launch script for Drawing with LLMs GUI Simulator
"""

import sys
import os

def main():
    """Launch the GUI application."""
    # Add current directory to path for imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    try:
        from main import main as gui_main
        gui_main()
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you're running this from the gui_simulator directory")
        return 1
    except Exception as e:
        print(f"Error launching application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
