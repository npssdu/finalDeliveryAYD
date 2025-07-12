"""
GUI Simulator Package for Drawing with LLMs Project

Este paquete contiene una aplicación GUI independiente que simula
el proceso completo del proyecto Drawing with LLMs.

Componentes principales:
- main.py: Interfaz gráfica principal
- svg_generator.py: Generador de SVG
- data_processor.py: Procesador de datos
- performance_analyzer.py: Analizador de rendimiento

Uso:
    python -m gui_simulator.main
    
    o 
    
    from gui_simulator.main import DrawingWithLLMsGUI
    import tkinter as tk
    
    root = tk.Tk()
    app = DrawingWithLLMsGUI(root)
    root.mainloop()
"""

__version__ = "1.0.0"
__author__ = "Drawing with LLMs Project"
__description__ = "GUI Simulator for Drawing with LLMs Project"

# Importaciones principales
from .main import DrawingWithLLMsGUI
from .svg_generator import SVGGenerator
from .data_processor import DataProcessor
from .performance_analyzer import PerformanceAnalyzer

__all__ = [
    'DrawingWithLLMsGUI',
    'SVGGenerator', 
    'DataProcessor',
    'PerformanceAnalyzer'
]
