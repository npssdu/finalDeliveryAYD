"""
Conector LLM para el GUI Simulator
Integra modelos de lenguaje para mejorar la generación de SVG
"""

import os
import json
import time
import requests
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

# Importaciones opcionales
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    openai = None

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class LLMConnector(ABC):
    """Clase base abstracta para conectores LLM."""
    
    @abstractmethod
    def generate_svg_enhanced(self, description: str) -> Dict:
        """Genera SVG mejorado usando LLM."""
        pass
    
    @abstractmethod
    def analyze_description(self, description: str) -> Dict:
        """Analiza la descripción para extraer características."""
        pass


class OpenAIConnector(LLMConnector):
    """Conector para OpenAI GPT-3.5/GPT-4."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """
        Inicializa el conector OpenAI.
        
        Args:
            api_key: Clave API de OpenAI (se toma de variable de entorno si no se proporciona)
            model: Modelo a usar (gpt-3.5-turbo, gpt-4, etc.)
        """
        if not OPENAI_AVAILABLE:
            self.client = None
            self.api_key = None
            self.model = model
            return
            
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.model = model
        if OPENAI_AVAILABLE and openai and self.api_key:
            self.client = openai.OpenAI(api_key=self.api_key)
        else:
            self.client = None
        
        # Prompts optimizados para SVG generation
        self.svg_system_prompt = """Eres un experto en generación de SVG. Tu tarea es:
1. Analizar descripciones textuales
2. Generar código SVG válido y optimizado
3. Asegurar que el SVG sea visualmente coherente con la descripción

Reglas importantes:
- Usa solo elementos SVG estándar (rect, circle, path, polygon, etc.)
- Incluye viewBox="0 0 400 400" 
- Usa colores apropiados y formas geométricas
- El SVG debe ser completo y renderizable
- Mantén el código limpio y estructurado"""
        
        self.analysis_system_prompt = """Eres un analizador de texto especializado en contenido visual.
Extrae características clave de descripciones textuales para generación de imágenes SVG.

Devuelve un JSON con:
- colors: lista de colores mencionados
- shapes: formas geométricas identificadas  
- objects: objetos principales
- scene_type: tipo de escena (landscape, geometric, abstract, etc.)
- complexity: score de 1-10 de complejidad visual
- keywords: palabras clave importantes"""

    def generate_svg_enhanced(self, description: str) -> Dict:
        """
        Genera SVG mejorado usando OpenAI.
        
        Args:
            description: Descripción textual
            
        Returns:
            Dict con SVG generado y metadatos
        """
        if not self.client:
            return self._fallback_generation(description)
        
        try:
            start_time = time.time()
            
            # Prompt para generación SVG
            svg_prompt = f"""
Genera un SVG completo y válido para esta descripción:
"{description}"

Devuelve SOLO el código SVG, sin explicaciones adicionales.
El SVG debe empezar con <svg> y terminar con </svg>.
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.svg_system_prompt},
                    {"role": "user", "content": svg_prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            svg_code = response.choices[0].message.content.strip()
            generation_time = time.time() - start_time
            
            # Validar que sea SVG válido
            if not self._validate_svg(svg_code):
                return self._fallback_generation(description)
            
            return {
                'svg_code': svg_code,
                'generation_time': generation_time,
                'source': 'openai',
                'model': self.model,
                'success': True,
                'tokens_used': response.usage.total_tokens if response.usage else 0
            }
            
        except Exception as e:
            print(f"Error con OpenAI: {e}")
            return self._fallback_generation(description)

    def analyze_description(self, description: str) -> Dict:
        """
        Analiza descripción textual para extraer características.
        
        Args:
            description: Descripción a analizar
            
        Returns:
            Dict con características extraídas
        """
        if not self.client:
            return self._fallback_analysis(description)
        
        try:
            analysis_prompt = f"""
Analiza esta descripción y devuelve un JSON con las características visuales:
"{description}"

Formato de respuesta esperado:
{{
    "colors": ["color1", "color2"],
    "shapes": ["shape1", "shape2"], 
    "objects": ["object1", "object2"],
    "scene_type": "landscape|geometric|abstract|clothing|other",
    "complexity": 5,
    "keywords": ["keyword1", "keyword2"],
    "mood": "calm|energetic|neutral",
    "style": "modern|classic|abstract"
}}
"""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.analysis_system_prompt},
                    {"role": "user", "content": analysis_prompt}
                ],
                max_tokens=500,
                temperature=0.3
            )
            
            # Extraer JSON de la respuesta
            content = response.choices[0].message.content.strip()
            
            # Intentar parsear JSON
            try:
                analysis = json.loads(content)
                analysis['source'] = 'openai'
                analysis['success'] = True
                return analysis
            except json.JSONDecodeError:
                return self._fallback_analysis(description)
                
        except Exception as e:
            print(f"Error analizando con OpenAI: {e}")
            return self._fallback_analysis(description)

    def _validate_svg(self, svg_code: str) -> bool:
        """Valida que el código SVG sea básicamente correcto."""
        if not svg_code:
            return False
        
        svg_lower = svg_code.lower()
        return (
            '<svg' in svg_lower and 
            '</svg>' in svg_lower and
            'viewbox' in svg_lower or 'width' in svg_lower
        )

    def _fallback_generation(self, description: str) -> Dict:
        """Generación de respaldo cuando falla el LLM."""
        return {
            'svg_code': f'<svg viewBox="0 0 400 400"><text x="200" y="200" text-anchor="middle">Error: {description[:20]}...</text></svg>',
            'generation_time': 0.001,
            'source': 'fallback',
            'model': 'none',
            'success': False,
            'tokens_used': 0
        }

    def _fallback_analysis(self, description: str) -> Dict:
        """Análisis de respaldo cuando falla el LLM."""
        words = description.lower().split()
        
        # Análisis básico por palabras clave
        colors = [w for w in words if w in ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white']]
        shapes = [w for w in words if w in ['circle', 'square', 'triangle', 'rectangle', 'line']]
        
        return {
            'colors': colors,
            'shapes': shapes,
            'objects': words[:3],  # primeras 3 palabras como objetos
            'scene_type': 'other',
            'complexity': min(len(words), 10),
            'keywords': words,
            'mood': 'neutral',
            'style': 'modern',
            'source': 'fallback',
            'success': False
        }


class HuggingFaceConnector(LLMConnector):
    """Conector para modelos de Hugging Face (local)."""
    
    def __init__(self, model_name: str = "microsoft/CodeT5-base"):
        """
        Inicializa conector Hugging Face.
        
        Args:
            model_name: Nombre del modelo de HF a usar
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Carga el modelo de Hugging Face."""
        if not TRANSFORMERS_AVAILABLE:
            print("❌ Transformers no está disponible")
            self.model = None
            self.tokenizer = None
            return
            
        try:
            print(f"Cargando modelo {self.model_name}...")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            print("Modelo cargado exitosamente")
        except Exception as e:
            print(f"Error cargando modelo HF: {e}")
            self.model = None
            self.tokenizer = None

    def generate_svg_enhanced(self, description: str) -> Dict:
        """Genera SVG usando modelo local de HF."""
        if not self.model or not self.tokenizer:
            return self._fallback_generation(description)
        
        try:
            start_time = time.time()
            
            # Prompt optimizado para generación SVG
            prompt = f"Generate SVG code for: {description}"
            
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            outputs = self.model.generate(**inputs, max_length=1024, num_return_sequences=1)
            svg_code = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            generation_time = time.time() - start_time
            
            return {
                'svg_code': svg_code,
                'generation_time': generation_time,
                'source': 'huggingface',
                'model': self.model_name,
                'success': True,
                'tokens_used': len(inputs['input_ids'][0])
            }
            
        except Exception as e:
            print(f"Error con HuggingFace: {e}")
            return self._fallback_generation(description)

    def analyze_description(self, description: str) -> Dict:
        """Análisis básico sin modelo específico."""
        return self._fallback_analysis(description)

    def _fallback_generation(self, description: str) -> Dict:
        """Generación de respaldo."""
        return {
            'svg_code': f'<svg viewBox="0 0 400 400"><text x="200" y="200" text-anchor="middle">HF Error: {description[:20]}...</text></svg>',
            'generation_time': 0.001,
            'source': 'fallback',
            'model': 'none',
            'success': False,
            'tokens_used': 0
        }

    def _fallback_analysis(self, description: str) -> Dict:
        """Análisis de respaldo."""
        words = description.lower().split()
        
        return {
            'colors': [w for w in words if w in ['red', 'blue', 'green', 'yellow', 'purple']],
            'shapes': [w for w in words if w in ['circle', 'square', 'triangle']],
            'objects': words[:3],
            'scene_type': 'other',
            'complexity': min(len(words), 10),
            'keywords': words,
            'mood': 'neutral',
            'style': 'modern',
            'source': 'fallback_hf',
            'success': False
        }


class LLMManager:
    """Gestor principal para múltiples conectores LLM."""
    
    def __init__(self):
        """Inicializa el gestor con conectores disponibles."""
        self.connectors = {}
        self.current_connector = None
        self._initialize_connectors()
    
    def _initialize_connectors(self):
        """Inicializa conectores disponibles."""
        # Try Ollama first (recommended - free)
        try:
            try:
                from .ollama_connector import OllamaConnector
            except ImportError:
                from ollama_connector import OllamaConnector
            ollama_connector = OllamaConnector()
            if ollama_connector.is_available():
                self.connectors['ollama'] = ollama_connector
                self.current_connector = 'ollama'
                print("Ollama connector initialized (RECOMMENDED)")
            else:
                print("Ollama is not running locally")
        except Exception as e:
            print(f"Error initializing Ollama: {e}")
        
        # Try OpenAI
        if OPENAI_AVAILABLE:
            try:
                openai_connector = OpenAIConnector()
                if openai_connector.client:
                    self.connectors['openai'] = openai_connector
                    if not self.current_connector:
                        self.current_connector = 'openai'
                    print("✅ OpenAI connector inicializado")
                else:
                    print("OpenAI available but no API key configured")
            except Exception as e:
                print(f"Error initializing OpenAI: {e}")
        else:
            print("OpenAI not available (pip install openai)")
        
        # Try Hugging Face
        if TRANSFORMERS_AVAILABLE:
            try:
                hf_connector = HuggingFaceConnector()
                if hf_connector.model:
                    self.connectors['huggingface'] = hf_connector
                    if not self.current_connector:
                        self.current_connector = 'huggingface'
                    print("HuggingFace connector initialized")
                else:
                    print("HuggingFace available but model could not be loaded")
            except Exception as e:
                print(f"Error initializing HuggingFace: {e}")
        else:
            print("HuggingFace not available (pip install transformers)")
        
        if not self.connectors:
            print("Could not initialize any LLM connector")
            print("   System will work in traditional mode only")
            print("\nTo enable free LLM:")
            print("   1. Install Ollama: https://ollama.ai")
            print("   2. Run: ollama pull llama3.1:8b")
            print("   3. Restart this application")

    def set_connector(self, connector_name: str) -> bool:
        """
        Cambia el conector activo.
        
        Args:
            connector_name: Nombre del conector ('openai', 'huggingface')
            
        Returns:
            True si el cambio fue exitoso
        """
        if connector_name in self.connectors:
            self.current_connector = connector_name
            print(f"✅ Cambiado a conector: {connector_name}")
            return True
        else:
            print(f"❌ Conector no disponible: {connector_name}")
            return False

    def get_available_connectors(self) -> List[str]:
        """Retorna lista de conectores disponibles."""
        return list(self.connectors.keys())

    def generate_svg_enhanced(self, description: str) -> Dict:
        """Genera SVG usando el conector activo."""
        if self.current_connector and self.current_connector in self.connectors:
            return self.connectors[self.current_connector].generate_svg_enhanced(description)
        else:
            return {
                'svg_code': f'<svg viewBox="0 0 400 400"><text x="200" y="200" text-anchor="middle">No LLM: {description[:20]}...</text></svg>',
                'generation_time': 0.001,
                'source': 'no_llm',
                'model': 'none',
                'success': False,
                'tokens_used': 0
            }

    def analyze_description(self, description: str) -> Dict:
        """Analiza descripción usando el conector activo."""
        if self.current_connector and self.current_connector in self.connectors:
            return self.connectors[self.current_connector].analyze_description(description)
        else:
            return {
                'colors': [],
                'shapes': [],
                'objects': [],
                'scene_type': 'unknown',
                'complexity': 0,
                'keywords': [],
                'mood': 'neutral',
                'style': 'unknown',
                'source': 'no_llm',
                'success': False
            }

    def get_connector_status(self) -> Dict:
        """Retorna estado de todos los conectores."""
        status = {
            'current': self.current_connector,
            'available': list(self.connectors.keys()),
            'details': {}
        }
        
        for name, connector in self.connectors.items():
            if hasattr(connector, '__class__') and 'OllamaConnector' in str(connector.__class__):
                status['details'][name] = {
                    'type': 'Ollama (GRATUITO)',
                    'model': connector.model_name,
                    'available': connector.is_available()
                }
            elif isinstance(connector, OpenAIConnector):
                status['details'][name] = {
                    'type': 'OpenAI',
                    'model': connector.model,
                    'has_api_key': bool(connector.api_key)
                }
            elif isinstance(connector, HuggingFaceConnector):
                status['details'][name] = {
                    'type': 'HuggingFace', 
                    'model': connector.model_name,
                    'loaded': bool(connector.model)
                }
        
        return status
