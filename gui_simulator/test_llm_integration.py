"""
Script de prueba para verificar la integración LLM
"""

import sys
import os

# Agregar el directorio padre al path para poder importar gui_simulator
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_llm_connector():
    """Prueba básica del conector LLM."""
    print("🧪 TESTING LLM CONNECTOR")
    print("=" * 30)
    
    try:
        from gui_simulator.llm_connector import LLMManager
        
        # Inicializar manager
        print("1. Inicializando LLM Manager...")
        manager = LLMManager()
        
        # Verificar conectores disponibles
        available = manager.get_available_connectors()
        print(f"2. Conectores disponibles: {available}")
        
        if available:
            print("✅ LLM Manager inicializado correctamente")
            
            # Prueba de análisis
            print("\n3. Probando análisis de descripción...")
            test_description = "a purple forest at dusk"
            analysis = manager.analyze_description(test_description)
            print(f"Análisis resultado: {analysis.get('success', False)}")
            
            # Prueba de generación (solo si hay conectores)
            print("\n4. Probando generación SVG...")
            result = manager.generate_svg_enhanced(test_description)
            print(f"Generación exitosa: {result.get('success', False)}")
            print(f"Fuente: {result.get('source', 'unknown')}")
            
            return True
        else:
            print("⚠️ No hay conectores LLM disponibles")
            print("Configura OpenAI API key o instala transformers")
            return False
            
    except ImportError as e:
        print(f"❌ Error importando módulos: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en prueba: {e}")
        return False

def test_gui_integration():
    """Prueba la integración con la GUI."""
    print("\n🖥️ TESTING GUI INTEGRATION")
    print("=" * 30)
    
    try:
        # Verificar que se puede importar la clase principal
        from gui_simulator.main import DrawingWithLLMsGUI
        print("✅ Clase principal importada correctamente")
        
        # Verificar que tkinter está disponible
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()  # No mostrar ventana
        
        print("✅ Tkinter disponible")
        
        # Verificar que se puede crear la instancia (sin mostrar)
        # Esto verificará que todas las importaciones están correctas
        print("1. Verificando inicialización...")
        
        # Esto debería funcionar sin errores
        gui = DrawingWithLLMsGUI(root)
        print("✅ GUI inicializada correctamente")
        
        # Verificar que el LLM manager está disponible
        if hasattr(gui, 'llm_manager'):
            print("✅ LLM Manager integrado en GUI")
        else:
            print("❌ LLM Manager no encontrado en GUI")
            
        root.destroy()
        return True
        
    except ImportError as e:
        print(f"❌ Error importando GUI: {e}")
        return False
    except Exception as e:
        print(f"❌ Error en integración GUI: {e}")
        return False

def test_dependencies():
    """Verifica las dependencias necesarias."""
    print("\n📦 TESTING DEPENDENCIES")
    print("=" * 30)
    
    dependencies = {
        'tkinter': 'GUI framework',
        'pandas': 'Data processing',
        'numpy': 'Numerical operations',
        'openai': 'OpenAI API (opcional)',
        'requests': 'HTTP requests',
        'transformers': 'HuggingFace models (opcional)'
    }
    
    results = {}
    
    for dep, description in dependencies.items():
        try:
            __import__(dep)
            results[dep] = "✅ Disponible"
        except ImportError:
            if dep in ['openai', 'transformers']:
                results[dep] = "⚠️ Opcional - No instalado"
            else:
                results[dep] = "❌ Requerido - No disponible"
    
    print("Estado de dependencias:")
    for dep, status in results.items():
        print(f"  {dep}: {status}")
    
    # Verificar si hay al menos una opción LLM
    has_llm = results.get('openai', '').startswith('✅') or results.get('transformers', '').startswith('✅')
    
    if has_llm:
        print("\n✅ Al menos una opción LLM está disponible")
    else:
        print("\n⚠️ Ninguna opción LLM disponible - solo modo tradicional")
    
    return has_llm

def main():
    """Ejecuta todas las pruebas."""
    print("🚀 TESTING GUI SIMULATOR CON LLM INTEGRATION")
    print("=" * 50)
    
    # Test 1: Dependencias
    deps_ok = test_dependencies()
    
    # Test 2: LLM Connector
    llm_ok = test_llm_connector()
    
    # Test 3: GUI Integration
    gui_ok = test_gui_integration()
    
    print("\n" + "=" * 50)
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 50)
    
    print(f"Dependencias: {'✅ OK' if deps_ok else '⚠️ Parcial'}")
    print(f"LLM Connector: {'✅ OK' if llm_ok else '⚠️ Limitado'}")
    print(f"GUI Integration: {'✅ OK' if gui_ok else '❌ Error'}")
    
    if gui_ok:
        print("\n🎉 Sistema listo para usar!")
        print("Ejecuta: python -m gui_simulator.main")
        
        if not llm_ok:
            print("\n💡 Para habilitar LLM:")
            print("1. pip install openai")
            print("2. export OPENAI_API_KEY=tu_key")
    else:
        print("\n❌ Hay problemas que resolver antes de usar el sistema")
    
    return gui_ok and (llm_ok or deps_ok)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
