"""
Conector Ollama para el GUI Simulator
Integra modelos Ollama gratuitos para generación de SVG
"""

import requests
import json
import time
from typing import Dict, List, Optional


class OllamaConnector:
    """Conector para modelos Ollama ejecutándose localmente."""
    
    def __init__(self, model_name: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        """
        Inicializa el conector Ollama.
        
        Args:
            model_name: Nombre del modelo Ollama (llama3.1:8b, codellama:7b, etc.)
            base_url: URL base del servidor Ollama
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.api_url = f"{self.base_url}/api/generate"
        self.available = self._check_availability()
        
        # Prompts optimizados para SVG
        self.svg_system_prompt = """Eres un experto en generación de SVG. Tu tarea es generar código SVG válido basado en descripciones textuales.

REGLAS IMPORTANTES:
1. Genera SOLO código SVG válido, nada más
2. Usa viewBox="0 0 400 400" siempre
3. Incluye elementos apropiados: rect, circle, path, polygon, etc.
4. Usa colores que coincidan con la descripción
5. El SVG debe ser completo y renderizable
6. NO incluyas explicaciones, solo el código SVG

Ejemplo de formato de salida:
<svg viewBox="0 0 400 400" xmlns="http://www.w3.org/2000/svg">
  <!-- elementos SVG aquí -->
</svg>"""

    def _check_availability(self) -> bool:
        """Verifica si Ollama está disponible y funcionando."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [model['name'] for model in models]
                
                # Verificar si el modelo está disponible
                if self.model_name in model_names:
                    print(f"✅ Ollama disponible con modelo {self.model_name}")
                    return True
                else:
                    print(f"⚠️ Ollama disponible pero modelo {self.model_name} no encontrado")
                    print(f"   Modelos disponibles: {model_names}")
                    
                    # Intentar usar el primer modelo disponible
                    if model_names:
                        self.model_name = model_names[0]
                        print(f"   Usando {self.model_name} en su lugar")
                        return True
                    return False
            else:
                return False
        except Exception as e:
            print(f"❌ Ollama no disponible: {e}")
            return False

    def generate_svg_enhanced(self, description: str) -> Dict:
        """
        Genera SVG usando Ollama.
        
        Args:
            description: Descripción textual
            
        Returns:
            Dict con SVG generado y metadatos
        """
        if not self.available:
            return self._fallback_generation(description)
        
        try:
            start_time = time.time()
            
            # Crear prompt específico para SVG
            prompt = f"""Descripción: "{description}"

Genera código SVG válido para esta descripción. Responde SOLO con el código SVG, sin explicaciones adicionales."""

            # Hacer request a Ollama
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "system": self.svg_system_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 1000
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                svg_code = result.get('response', '').strip()
                generation_time = time.time() - start_time
                
                # Limpiar y validar SVG
                svg_code = self._clean_svg_response(svg_code)
                
                if self._validate_svg(svg_code):
                    return {
                        'svg_code': svg_code,
                        'generation_time': generation_time,
                        'source': 'ollama',
                        'model': self.model_name,
                        'success': True,
                        'tokens_used': len(svg_code.split())
                    }
                else:
                    print("⚠️ SVG generado no es válido, usando fallback")
                    return self._fallback_generation(description)
            else:
                print(f"❌ Error en Ollama API: {response.status_code}")
                return self._fallback_generation(description)
                
        except Exception as e:
            print(f"❌ Error generando con Ollama: {e}")
            return self._fallback_generation(description)

    def analyze_description(self, description: str) -> Dict:
        """
        Analiza descripción textual para extraer características.
        
        Args:
            description: Descripción a analizar
            
        Returns:
            Dict con características extraídas
        """
        if not self.available:
            return self._fallback_analysis(description)
        
        try:
            analysis_prompt = f"""Analiza esta descripción visual y extrae características clave: "{description}"

Devuelve un JSON con este formato exacto:
{{
    "colors": ["color1", "color2"],
    "shapes": ["shape1", "shape2"],
    "objects": ["object1", "object2"],
    "scene_type": "landscape|geometric|abstract|clothing|other",
    "complexity": 5,
    "keywords": ["word1", "word2"],
    "mood": "calm|energetic|mysterious|neutral",
    "style": "modern|classic|abstract|naturalistic"
}}

Responde SOLO con el JSON, nada más."""

            payload = {
                "model": self.model_name,
                "prompt": analysis_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 300
                }
            }
            
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get('response', '').strip()
                
                # Intentar extraer JSON de la respuesta
                try:
                    # Buscar JSON en la respuesta
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        analysis = json.loads(json_str)
                        analysis['source'] = 'ollama'
                        analysis['success'] = True
                        return analysis
                    else:
                        return self._fallback_analysis(description)
                        
                except json.JSONDecodeError:
                    return self._fallback_analysis(description)
            else:
                return self._fallback_analysis(description)
                
        except Exception as e:
            print(f"❌ Error analizando con Ollama: {e}")
            return self._fallback_analysis(description)

    def _clean_svg_response(self, svg_text: str) -> str:
        """Limpia la respuesta SVG de Ollama."""
        # Remover texto antes y después del SVG
        svg_text = svg_text.strip()
        
        # Buscar el inicio del SVG
        start_markers = ['<svg', '<?xml']
        start_idx = -1
        
        for marker in start_markers:
            idx = svg_text.lower().find(marker)
            if idx >= 0:
                start_idx = idx
                break
        
        if start_idx >= 0:
            svg_text = svg_text[start_idx:]
        
        # Buscar el final del SVG
        end_idx = svg_text.lower().rfind('</svg>')
        if end_idx >= 0:
            svg_text = svg_text[:end_idx + 6]
        
        return svg_text.strip()

    def _validate_svg(self, svg_code: str) -> bool:
        """Valida que el código SVG sea básicamente correcto."""
        if not svg_code:
            return False
        
        svg_lower = svg_code.lower()
        return (
            '<svg' in svg_lower and 
            '</svg>' in svg_lower and
            ('viewbox' in svg_lower or 'width' in svg_lower or 'height' in svg_lower)
        )

    def _fallback_generation(self, description: str) -> Dict:
        """Generación de respaldo cuando falla Ollama."""
        # SVG básico basado en palabras clave
        words = description.lower().split()
        
        # Detectar colores
        color_map = {
            'purple': '#800080', 'red': '#FF0000', 'blue': '#0000FF',
            'green': '#008000', 'yellow': '#FFFF00', 'orange': '#FFA500',
            'black': '#000000', 'white': '#FFFFFF', 'gray': '#808080',
            'pink': '#FFC0CB', 'brown': '#A52A2A'
        }
        
        color = '#808080'  # default gray
        for word in words:
            if word in color_map:
                color = color_map[word]
                break
        
        # Detectar formas
        if any(shape in words for shape in ['circle', 'round']):
            svg = f'<svg viewBox="0 0 400 400"><circle cx="200" cy="200" r="80" fill="{color}"/></svg>'
        elif any(shape in words for shape in ['square', 'rectangle']):
            svg = f'<svg viewBox="0 0 400 400"><rect x="160" y="160" width="80" height="80" fill="{color}"/></svg>'
        else:
            # Forma por defecto
            svg = f'<svg viewBox="0 0 400 400"><circle cx="200" cy="200" r="60" fill="{color}"/><text x="200" y="350" text-anchor="middle" font-size="12">{description[:20]}...</text></svg>'
        
        return {
            'svg_code': svg,
            'generation_time': 0.001,
            'source': 'fallback',
            'model': 'basic',
            'success': False,
            'tokens_used': 0
        }

    def _fallback_analysis(self, description: str) -> Dict:
        """Análisis de respaldo básico."""
        words = description.lower().split()
        
        # Análisis básico por palabras clave
        colors = [w for w in words if w in ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 'gray', 'pink', 'brown']]
        shapes = [w for w in words if w in ['circle', 'square', 'triangle', 'rectangle', 'line', 'oval']]
        
        # Detectar tipo de escena
        scene_type = 'other'
        if any(w in words for w in ['forest', 'tree', 'mountain', 'ocean', 'sky', 'landscape']):
            scene_type = 'landscape'
        elif any(w in words for w in ['shirt', 'pants', 'coat', 'dress', 'clothing']):
            scene_type = 'clothing'
        elif any(w in words for w in ['triangle', 'circle', 'square', 'geometric']):
            scene_type = 'geometric'
        
        return {
            'colors': colors,
            'shapes': shapes,
            'objects': words[:3],  # primeras 3 palabras como objetos
            'scene_type': scene_type,
            'complexity': min(len(words), 10),
            'keywords': words,
            'mood': 'neutral',
            'style': 'simple',
            'source': 'fallback',
            'success': False
        }

    def get_available_models(self) -> List[str]:
        """Obtiene lista de modelos disponibles en Ollama."""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                return [model['name'] for model in models]
            return []
        except:
            return []

    def is_available(self) -> bool:
        """Verifica si Ollama está disponible."""
        return self.available
