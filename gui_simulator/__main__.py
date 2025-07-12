"""
Punto de entrada para ejecutar el GUI Simulator como módulo.
Permite ejecutar: python -m gui_simulator
"""

import tkinter as tk
from .main import DrawingWithLLMsGUI

def main():
    """Función principal para ejecutar la aplicación GUI."""
    root = None
    try:
        # Crear la ventana principal
        root = tk.Tk()
        
        # Crear la aplicación
        app = DrawingWithLLMsGUI(root)
        
        # Ejecutar el bucle principal
        root.mainloop()
        
    except KeyboardInterrupt:
        print("\nAplicación cerrada por el usuario")
    except Exception as e:
        print(f"Error ejecutando la aplicación: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Limpiar recursos
        if root is not None:
            try:
                root.destroy()
            except:
                pass

if __name__ == "__main__":
    main()
