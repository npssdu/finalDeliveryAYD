"""
Generador de SVG para el simulador Drawing with LLMs
Versión simplificada del modelo original para la interfaz gráfica
"""

import re
import random
from typing import Dict, List, Optional


class SVGGenerator:
    """Genera SVGs a partir de descripciones textuales."""
    
    def __init__(self):
        """Inicializa el generador con patrones y plantillas."""
        self.width = 400
        self.height = 400
        
        # Patrones de colores
        self.colors = {
            'purple': '#800080',
            'gray': '#808080', 
            'grey': '#808080',
            'green': '#008000',
            'orange': '#FFA500',
            'burgundy': '#800020',
            'crimson': '#DC143C',
            'magenta': '#FF00FF',
            'black': '#000000',
            'white': '#FFFFFF',
            'blue': '#0000FF',
            'red': '#FF0000',
            'yellow': '#FFFF00',
            'brown': '#A52A2A',
            'silver': '#C0C0C0',
            'bronze': '#CD7F32',
            'teal': '#008080',
            'azure': '#007FFF',
            'khaki': '#F0E68C',
            'maroon': '#800000',
            'pink': '#FFC0CB',
            'cyan': '#00FFFF',
            'lime': '#00FF00',
            'navy': '#000080',
            'olive': '#808000',
            'coral': '#FF7F50',
            'gold': '#FFD700',
            'indigo': '#4B0082',
            'violet': '#EE82EE'
        }
        
        # Formas básicas
        self.shapes = {
            'circle': self._create_circle,
            'rectangle': self._create_rectangle,
            'square': self._create_square,
            'triangle': self._create_triangle,
            'pyramid': self._create_pyramid,
            'trapezoid': self._create_trapezoid,
            'dodecahedron': self._create_dodecahedron,
            'ellipse': self._create_ellipse,
            'polygon': self._create_polygon,
            'crescent': self._create_crescent
        }
        
        # Plantillas de escenas
        self.scene_templates = {
            'forest': self._create_forest_scene,
            'ocean': self._create_ocean_scene,
            'lighthouse': self._create_lighthouse_scene,
            'lagoon': self._create_lagoon_scene,
            'snow': self._create_snow_scene,
            'night': self._create_night_scene,
            'sky': self._create_sky_scene,
            'plain': self._create_plain_scene
        }
        
        # Patrones especiales
        self.patterns = {
            'checkered': self._create_checkered_pattern,
            'grid': self._create_grid_pattern,
            'stripes': self._create_stripes_pattern,
            'dots': self._create_dots_pattern
        }
    
    def generate_svg(self, description: str) -> str:
        """
        Genera un SVG basado en la descripción.
        
        Args:
            description: Descripción textual
            
        Returns:
            Código SVG válido
        """
        description_lower = description.lower()
        
        # Extraer información de la descripción
        colors_found = self._extract_colors(description_lower)
        shapes_found = self._extract_shapes(description_lower)
        scenes_found = self._extract_scenes(description_lower)
        patterns_found = self._extract_patterns(description_lower)
        
        # Determinar estrategia de generación
        if patterns_found:
            return self._generate_pattern_svg(patterns_found, colors_found, description_lower)
        elif scenes_found:
            return self._generate_scene_svg(scenes_found, colors_found, description_lower)
        elif shapes_found:
            return self._generate_geometric_svg(shapes_found, colors_found, description_lower)
        elif any(word in description_lower for word in ['coat', 'pants', 'scarf', 'overalls', 'clothing']):
            return self._generate_clothing_svg(colors_found, description_lower)
        else:
            return self._generate_abstract_svg(colors_found, description_lower)
    
    def _extract_colors(self, description: str) -> List[str]:
        """Extrae colores de la descripción."""
        found_colors = []
        for color in self.colors.keys():
            if color in description:
                found_colors.append(color)
        return found_colors if found_colors else ['black']
    
    def _extract_shapes(self, description: str) -> List[str]:
        """Extrae formas de la descripción."""
        found_shapes = []
        for shape in self.shapes.keys():
            if shape in description:
                found_shapes.append(shape)
        return found_shapes
    
    def _extract_scenes(self, description: str) -> List[str]:
        """Extrae escenas de la descripción."""
        found_scenes = []
        scene_keywords = {
            'forest': ['forest', 'tree', 'woods', 'dusk'],
            'ocean': ['ocean', 'sea', 'waves', 'water'],
            'lighthouse': ['lighthouse', 'beacon', 'shore'],
            'lagoon': ['lagoon', 'pond', 'lake'],
            'snow': ['snow', 'snowy', 'winter', 'peaks'],
            'night': ['night', 'starlit', 'stars', 'evening'],
            'sky': ['sky', 'cloudy', 'clouds', 'air'],
            'plain': ['plain', 'field', 'meadow', 'grass']
        }
        
        for scene, keywords in scene_keywords.items():
            if any(keyword in description for keyword in keywords):
                found_scenes.append(scene)
        
        return found_scenes
    
    def _extract_patterns(self, description: str) -> List[str]:
        """Extrae patrones de la descripción."""
        found_patterns = []
        pattern_keywords = {
            'checkered': ['checkered', 'checker', 'chess'],
            'grid': ['grid', 'mesh', 'network'],
            'stripes': ['stripes', 'striped', 'bands'],
            'dots': ['dots', 'dotted', 'spots', 'polka']
        }
        
        for pattern, keywords in pattern_keywords.items():
            if any(keyword in description for keyword in keywords):
                found_patterns.append(pattern)
        
        return found_patterns
    
    def _get_color(self, color_name: str, default: str = '#000000') -> str:
        """Obtiene código hexadecimal de un color."""
        return self.colors.get(color_name, default)
    
    def _create_svg_header(self) -> str:
        """Crea el header SVG."""
        return f'<svg width="{self.width}" height="{self.height}" xmlns="http://www.w3.org/2000/svg">'
    
    def _create_svg_footer(self) -> str:
        """Crea el footer SVG."""
        return '</svg>'
    
    def _create_circle(self, cx: int, cy: int, r: int, color: str, **kwargs) -> str:
        """Crea un círculo."""
        opacity = kwargs.get('opacity', 1.0)
        stroke = kwargs.get('stroke', 'none')
        stroke_width = kwargs.get('stroke_width', 0)
        
        return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{color}" opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_width}" />'
    
    def _create_rectangle(self, x: int, y: int, width: int, height: int, color: str, **kwargs) -> str:
        """Crea un rectángulo."""
        opacity = kwargs.get('opacity', 1.0)
        stroke = kwargs.get('stroke', 'none')
        stroke_width = kwargs.get('stroke_width', 0)
        
        return f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{color}" opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_width}" />'
    
    def _create_square(self, x: int, y: int, size: int, color: str, **kwargs) -> str:
        """Crea un cuadrado."""
        return self._create_rectangle(x, y, size, size, color, **kwargs)
    
    def _create_triangle(self, x1: int, y1: int, x2: int, y2: int, x3: int, y3: int, color: str, **kwargs) -> str:
        """Crea un triángulo."""
        opacity = kwargs.get('opacity', 1.0)
        stroke = kwargs.get('stroke', 'none')
        stroke_width = kwargs.get('stroke_width', 0)
        
        points = f"{x1},{y1} {x2},{y2} {x3},{y3}"
        return f'<polygon points="{points}" fill="{color}" opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_width}" />'
    
    def _create_pyramid(self, x: int, y: int, size: int, color: str, **kwargs) -> str:
        """Crea una pirámide."""
        x1, y1 = x, y + size
        x2, y2 = x + size, y + size
        x3, y3 = x + size//2, y
        return self._create_triangle(x1, y1, x2, y2, x3, y3, color, **kwargs)
    
    def _create_trapezoid(self, x: int, y: int, width: int, height: int, color: str, **kwargs) -> str:
        """Crea un trapecio."""
        offset = width // 4
        x1, y1 = x + offset, y
        x2, y2 = x + width - offset, y
        x3, y3 = x + width, y + height
        x4, y4 = x, y + height
        
        opacity = kwargs.get('opacity', 1.0)
        stroke = kwargs.get('stroke', 'none')
        stroke_width = kwargs.get('stroke_width', 0)
        
        points = f"{x1},{y1} {x2},{y2} {x3},{y3} {x4},{y4}"
        return f'<polygon points="{points}" fill="{color}" opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_width}" />'
    
    def _create_dodecahedron(self, cx: int, cy: int, radius: int, color: str, **kwargs) -> str:
        """Crea un dodecágono (aproximación 2D de dodecahedron)."""
        import math
        
        points = []
        for i in range(12):
            angle = 2 * math.pi * i / 12
            x = cx + radius * math.cos(angle)
            y = cy + radius * math.sin(angle)
            points.append(f"{x:.1f},{y:.1f}")
        
        opacity = kwargs.get('opacity', 1.0)
        stroke = kwargs.get('stroke', 'none')
        stroke_width = kwargs.get('stroke_width', 0)
        
        points_str = " ".join(points)
        return f'<polygon points="{points_str}" fill="{color}" opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_width}" />'
    
    def _create_ellipse(self, cx: int, cy: int, rx: int, ry: int, color: str, **kwargs) -> str:
        """Crea una elipse."""
        opacity = kwargs.get('opacity', 1.0)
        stroke = kwargs.get('stroke', 'none')
        stroke_width = kwargs.get('stroke_width', 0)
        
        return f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="{color}" opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_width}" />'
    
    def _create_polygon(self, points: List[tuple], color: str, **kwargs) -> str:
        """Crea un polígono a partir de puntos."""
        opacity = kwargs.get('opacity', 1.0)
        stroke = kwargs.get('stroke', 'none')
        stroke_width = kwargs.get('stroke_width', 0)
        
        points_str = " ".join([f"{x},{y}" for x, y in points])
        return f'<polygon points="{points_str}" fill="{color}" opacity="{opacity}" stroke="{stroke}" stroke-width="{stroke_width}" />'
    
    def _create_crescent(self, cx: int, cy: int, radius: int, color: str, **kwargs) -> str:
        """Crea una media luna."""
        opacity = kwargs.get('opacity', 1.0)
        
        # Crear path para media luna
        path = f'M {cx-radius} {cy} A {radius} {radius} 0 1 1 {cx+radius} {cy} A {radius//2} {radius//2} 0 1 0 {cx-radius} {cy} Z'
        
        return f'<path d="{path}" fill="{color}" opacity="{opacity}" />'
    
    def _generate_pattern_svg(self, patterns: List[str], colors: List[str], description: str) -> str:
        """Genera SVG basado en patrones."""
        svg_parts = [self._create_svg_header()]
        
        primary_color = self._get_color(colors[0] if colors else 'black')
        secondary_color = self._get_color(colors[1] if len(colors) > 1 else 'white')
        
        for pattern in patterns:
            if pattern == 'checkered':
                svg_parts.append(self._create_checkered_pattern(primary_color, secondary_color))
            elif pattern == 'grid':
                svg_parts.append(self._create_grid_pattern(primary_color, secondary_color))
            elif pattern == 'stripes':
                svg_parts.append(self._create_stripes_pattern(primary_color, secondary_color))
            elif pattern == 'dots':
                svg_parts.append(self._create_dots_pattern(primary_color, secondary_color))
        
        svg_parts.append(self._create_svg_footer())
        return ''.join(svg_parts)
    
    def _generate_scene_svg(self, scenes: List[str], colors: List[str], description: str) -> str:
        """Genera SVG basado en escenas."""
        svg_parts = [self._create_svg_header()]
        
        for scene in scenes:
            if scene in self.scene_templates:
                svg_parts.append(self.scene_templates[scene](colors))
        
        svg_parts.append(self._create_svg_footer())
        return ''.join(svg_parts)
    
    def _generate_geometric_svg(self, shapes: List[str], colors: List[str], description: str) -> str:
        """Genera SVG basado en formas geométricas."""
        svg_parts = [self._create_svg_header()]
        
        # Fondo
        bg_color = self._get_color(colors[0] if colors else 'white', '#f0f0f0')
        svg_parts.append(self._create_rectangle(0, 0, self.width, self.height, bg_color))
        
        # Generar formas
        for i, shape in enumerate(shapes):
            color = self._get_color(colors[i % len(colors)] if colors else 'black')
            
            if shape == 'circle':
                svg_parts.append(self._create_circle(
                    self.width//2, self.height//2, 80, color, opacity=0.8
                ))
            elif shape == 'rectangle':
                svg_parts.append(self._create_rectangle(
                    self.width//2-60, self.height//2-40, 120, 80, color, opacity=0.8
                ))
            elif shape == 'triangle':
                svg_parts.append(self._create_triangle(
                    self.width//2, self.height//2-40,
                    self.width//2-60, self.height//2+40,
                    self.width//2+60, self.height//2+40,
                    color, opacity=0.8
                ))
            elif shape == 'pyramid':
                svg_parts.append(self._create_pyramid(
                    self.width//2-40, self.height//2-40, 80, color, opacity=0.8
                ))
            elif shape == 'trapezoid':
                svg_parts.append(self._create_trapezoid(
                    self.width//2-60, self.height//2-40, 120, 80, color, opacity=0.8
                ))
            elif shape == 'dodecahedron':
                svg_parts.append(self._create_dodecahedron(
                    self.width//2, self.height//2, 60, color, opacity=0.8
                ))
        
        svg_parts.append(self._create_svg_footer())
        return ''.join(svg_parts)
    
    def _generate_clothing_svg(self, colors: List[str], description: str) -> str:
        """Genera SVG basado en ropa."""
        svg_parts = [self._create_svg_header()]
        
        primary_color = self._get_color(colors[0] if colors else 'gray')
        secondary_color = self._get_color(colors[1] if len(colors) > 1 else 'white')
        
        # Fondo
        svg_parts.append(self._create_rectangle(0, 0, self.width, self.height, '#f8f8f8'))
        
        if 'coat' in description:
            # Dibujar abrigo
            svg_parts.append(self._create_rectangle(120, 100, 160, 200, primary_color))
            svg_parts.append(self._create_rectangle(130, 120, 140, 20, secondary_color))  # Cuello
            svg_parts.append(self._create_circle(160, 180, 5, '#333333'))  # Botón
            svg_parts.append(self._create_circle(160, 220, 5, '#333333'))  # Botón
        elif 'pants' in description:
            # Dibujar pantalones
            svg_parts.append(self._create_rectangle(150, 120, 80, 180, primary_color))
            svg_parts.append(self._create_rectangle(150, 130, 80, 20, secondary_color))  # Cinturón
        elif 'scarf' in description:
            # Dibujar bufanda
            svg_parts.append(self._create_rectangle(100, 180, 200, 40, primary_color))
            svg_parts.append(self._create_rectangle(80, 190, 20, 20, secondary_color))  # Borla
            svg_parts.append(self._create_rectangle(300, 190, 20, 20, secondary_color))  # Borla
        elif 'overalls' in description:
            # Dibujar overol
            svg_parts.append(self._create_rectangle(140, 120, 120, 180, primary_color))
            svg_parts.append(self._create_rectangle(160, 100, 80, 40, primary_color))  # Pechera
            svg_parts.append(self._create_rectangle(170, 110, 10, 60, secondary_color))  # Tirante
            svg_parts.append(self._create_rectangle(220, 110, 10, 60, secondary_color))  # Tirante
        
        svg_parts.append(self._create_svg_footer())
        return ''.join(svg_parts)
    
    def _generate_abstract_svg(self, colors: List[str], description: str) -> str:
        """Genera SVG abstracto."""
        svg_parts = [self._create_svg_header()]
        
        # Fondo gradiente
        bg_color = self._get_color(colors[0] if colors else 'blue', '#e0e0e0')
        svg_parts.append(self._create_rectangle(0, 0, self.width, self.height, bg_color))
        
        # Elementos abstractos
        for i in range(3):
            color = self._get_color(colors[i % len(colors)] if colors else 'black')
            x = random.randint(50, self.width - 50)
            y = random.randint(50, self.height - 50)
            size = random.randint(20, 80)
            
            if i % 3 == 0:
                svg_parts.append(self._create_circle(x, y, size, color, opacity=0.7))
            elif i % 3 == 1:
                svg_parts.append(self._create_rectangle(x, y, size, size, color, opacity=0.7))
            else:
                svg_parts.append(self._create_triangle(x, y, x+size, y+size, x, y+size, color, opacity=0.7))
        
        svg_parts.append(self._create_svg_footer())
        return ''.join(svg_parts)
    
    def _create_checkered_pattern(self, color1: str, color2: str) -> str:
        """Crea un patrón de cuadros."""
        pattern = []
        square_size = 25
        
        for row in range(0, self.height, square_size):
            for col in range(0, self.width, square_size):
                # Alternar colores
                is_even = (row // square_size + col // square_size) % 2 == 0
                color = color1 if is_even else color2
                
                pattern.append(self._create_rectangle(
                    col, row, square_size, square_size, color
                ))
        
        return ''.join(pattern)
    
    def _create_grid_pattern(self, color1: str, color2: str) -> str:
        """Crea un patrón de rejilla."""
        pattern = []
        
        # Fondo
        pattern.append(self._create_rectangle(0, 0, self.width, self.height, color2))
        
        # Líneas de rejilla
        grid_size = 30
        for i in range(0, self.width, grid_size):
            pattern.append(self._create_rectangle(i, 0, 2, self.height, color1))
        
        for i in range(0, self.height, grid_size):
            pattern.append(self._create_rectangle(0, i, self.width, 2, color1))
        
        return ''.join(pattern)
    
    def _create_stripes_pattern(self, color1: str, color2: str) -> str:
        """Crea un patrón de rayas."""
        pattern = []
        stripe_width = 20
        
        for i in range(0, self.width, stripe_width * 2):
            pattern.append(self._create_rectangle(
                i, 0, stripe_width, self.height, color1
            ))
            pattern.append(self._create_rectangle(
                i + stripe_width, 0, stripe_width, self.height, color2
            ))
        
        return ''.join(pattern)
    
    def _create_dots_pattern(self, color1: str, color2: str) -> str:
        """Crea un patrón de puntos."""
        pattern = []
        
        # Fondo
        pattern.append(self._create_rectangle(0, 0, self.width, self.height, color2))
        
        # Puntos
        dot_spacing = 40
        dot_radius = 8
        
        for x in range(dot_spacing, self.width, dot_spacing):
            for y in range(dot_spacing, self.height, dot_spacing):
                pattern.append(self._create_circle(x, y, dot_radius, color1))
        
        return ''.join(pattern)
    
    def _create_forest_scene(self, colors: List[str]) -> str:
        """Crea una escena de bosque."""
        scene = []
        
        # Cielo
        sky_color = self._get_color('purple' if 'purple' in colors else 'blue', '#87CEEB')
        scene.append(self._create_rectangle(0, 0, self.width, self.height//2, sky_color))
        
        # Suelo
        ground_color = self._get_color('brown', '#8B4513')
        scene.append(self._create_rectangle(0, self.height//2, self.width, self.height//2, ground_color))
        
        # Árboles
        tree_color = self._get_color('green', '#228B22')
        for i in range(5):
            x = 50 + i * 70
            # Tronco
            scene.append(self._create_rectangle(x, self.height//2-30, 20, 60, '#8B4513'))
            # Copa
            scene.append(self._create_circle(x+10, self.height//2-40, 30, tree_color))
        
        return ''.join(scene)
    
    def _create_ocean_scene(self, colors: List[str]) -> str:
        """Crea una escena del océano."""
        scene = []
        
        # Cielo
        sky_color = self._get_color('blue', '#87CEEB')
        scene.append(self._create_rectangle(0, 0, self.width, self.height//2, sky_color))
        
        # Océano
        ocean_color = self._get_color('blue', '#006994')
        scene.append(self._create_rectangle(0, self.height//2, self.width, self.height//2, ocean_color))
        
        # Olas
        wave_color = self._get_color('white', '#FFFFFF')
        for i in range(3):
            y = self.height//2 + i * 30
            scene.append(self._create_ellipse(self.width//2, y, 100, 10, wave_color, opacity=0.7))
        
        return ''.join(scene)
    
    def _create_lighthouse_scene(self, colors: List[str]) -> str:
        """Crea una escena de faro."""
        scene = []
        
        # Cielo
        scene.append(self._create_rectangle(0, 0, self.width, self.height//2, '#87CEEB'))
        
        # Océano
        scene.append(self._create_rectangle(0, self.height//2, self.width, self.height//2, '#006994'))
        
        # Faro
        lighthouse_color = self._get_color('white', '#FFFFFF')
        scene.append(self._create_rectangle(self.width//2-15, self.height//2-80, 30, 100, lighthouse_color))
        
        # Techo del faro
        scene.append(self._create_triangle(
            self.width//2-20, self.height//2-80,
            self.width//2+20, self.height//2-80,
            self.width//2, self.height//2-100,
            '#FF0000'
        ))
        
        # Luz
        scene.append(self._create_circle(self.width//2, self.height//2-70, 8, '#FFFF00'))
        
        return ''.join(scene)
    
    def _create_lagoon_scene(self, colors: List[str]) -> str:
        """Crea una escena de laguna."""
        scene = []
        
        # Cielo
        sky_color = self._get_color('blue', '#87CEEB')
        scene.append(self._create_rectangle(0, 0, self.width, self.height//2, sky_color))
        
        # Laguna
        lagoon_color = self._get_color('green', '#20B2AA')
        scene.append(self._create_ellipse(self.width//2, self.height//2+50, 150, 80, lagoon_color))
        
        # Orilla
        shore_color = self._get_color('brown', '#D2B48C')
        scene.append(self._create_rectangle(0, self.height//2, self.width, self.height//2, shore_color))
        
        # Nubes
        cloud_color = self._get_color('white', '#FFFFFF')
        for i in range(3):
            x = 80 + i * 120
            scene.append(self._create_circle(x, 60, 25, cloud_color, opacity=0.8))
            scene.append(self._create_circle(x+30, 60, 20, cloud_color, opacity=0.8))
        
        return ''.join(scene)
    
    def _create_snow_scene(self, colors: List[str]) -> str:
        """Crea una escena nevada."""
        scene = []
        
        # Cielo
        sky_color = self._get_color('gray', '#B0C4DE')
        scene.append(self._create_rectangle(0, 0, self.width, self.height//2, sky_color))
        
        # Planicie nevada
        snow_color = self._get_color('white', '#FFFFFF')
        scene.append(self._create_rectangle(0, self.height//2, self.width, self.height//2, snow_color))
        
        # Montañas
        mountain_color = self._get_color('gray', '#708090')
        for i in range(3):
            x = i * 150
            scene.append(self._create_triangle(
                x, self.height//2,
                x+100, self.height//2,
                x+50, self.height//2-80,
                mountain_color
            ))
        
        # Copos de nieve
        for i in range(20):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height//2)
            scene.append(self._create_circle(x, y, 2, snow_color))
        
        return ''.join(scene)
    
    def _create_night_scene(self, colors: List[str]) -> str:
        """Crea una escena nocturna."""
        scene = []
        
        # Cielo nocturno
        night_color = self._get_color('black', '#191970')
        scene.append(self._create_rectangle(0, 0, self.width, self.height//2, night_color))
        
        # Suelo
        ground_color = self._get_color('gray', '#2F4F4F')
        scene.append(self._create_rectangle(0, self.height//2, self.width, self.height//2, ground_color))
        
        # Estrellas
        star_color = self._get_color('white', '#FFFFFF')
        for i in range(15):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height//2)
            scene.append(self._create_circle(x, y, 1, star_color))
        
        # Luna
        moon_color = self._get_color('yellow', '#FFFF00')
        scene.append(self._create_circle(self.width-80, 80, 30, moon_color))
        
        return ''.join(scene)
    
    def _create_sky_scene(self, colors: List[str]) -> str:
        """Crea una escena del cielo."""
        scene = []
        
        # Cielo
        sky_color = self._get_color('blue', '#87CEEB')
        scene.append(self._create_rectangle(0, 0, self.width, self.height, sky_color))
        
        # Nubes
        cloud_color = self._get_color('white', '#FFFFFF')
        for i in range(4):
            x = 50 + i * 80
            y = 80 + i * 40
            scene.append(self._create_circle(x, y, 30, cloud_color, opacity=0.9))
            scene.append(self._create_circle(x+40, y, 25, cloud_color, opacity=0.9))
            scene.append(self._create_circle(x+20, y-15, 20, cloud_color, opacity=0.9))
        
        return ''.join(scene)
    
    def _create_plain_scene(self, colors: List[str]) -> str:
        """Crea una escena de planicie."""
        scene = []
        
        # Cielo
        sky_color = self._get_color('blue', '#87CEEB')
        scene.append(self._create_rectangle(0, 0, self.width, self.height//2, sky_color))
        
        # Planicie
        plain_color = self._get_color('green', '#9ACD32')
        scene.append(self._create_rectangle(0, self.height//2, self.width, self.height//2, plain_color))
        
        # Hierba
        grass_color = self._get_color('green', '#228B22')
        for i in range(50):
            x = random.randint(0, self.width)
            y = random.randint(self.height//2, self.height)
            scene.append(self._create_rectangle(x, y, 2, 10, grass_color))
        
        return ''.join(scene)
