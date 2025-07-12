"""
Módulo de procesamiento de datos para el simulador GUI.
Maneja la carga y análisis de datos de entrada del proyecto drawing-with-llms.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import re
from collections import Counter

class DataProcessor:
    """Procesador de datos para analizar las descripciones de texto y extraer características."""
    
    def __init__(self):
        self.data = None
        self.processed_features = {}
        self.word_frequencies = {}
        self.pattern_analysis = {}
        
    def load_data(self, file_path: str) -> bool:
        """
        Carga los datos desde un archivo CSV.
        
        Args:
            file_path: Ruta al archivo CSV
            
        Returns:
            bool: True si la carga fue exitosa, False en caso contrario
        """
        try:
            self.data = pd.read_csv(file_path)
            return True
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return False
    
    def get_basic_stats(self) -> Dict:
        """
        Obtiene estadísticas básicas del dataset.
        
        Returns:
            Dict: Estadísticas básicas del dataset
        """
        if self.data is None:
            return {}
        
        stats = {
            'total_samples': len(self.data),
            'columns': list(self.data.columns),
            'missing_values': self.data.isnull().sum().to_dict(),
            'data_types': self.data.dtypes.to_dict()
        }
        
        # Análisis de la columna de texto (asumiendo que hay una columna 'text' o 'description')
        text_column = None
        for col in self.data.columns:
            if 'text' in col.lower() or 'description' in col.lower() or 'prompt' in col.lower():
                text_column = col
                break
        
        if text_column:
            stats['text_stats'] = {
                'avg_length': self.data[text_column].str.len().mean(),
                'min_length': self.data[text_column].str.len().min(),
                'max_length': self.data[text_column].str.len().max(),
                'unique_texts': self.data[text_column].nunique()
            }
        
        return stats
    
    def analyze_text_features(self, text_column: Optional[str] = None) -> Dict:
        """
        Analiza las características del texto en el dataset.
        
        Args:
            text_column: Nombre de la columna de texto a analizar
            
        Returns:
            Dict: Análisis de características del texto
        """
        if self.data is None:
            return {}
        
        # Encontrar columna de texto si no se especifica
        if text_column is None:
            for col in self.data.columns:
                if 'text' in col.lower() or 'description' in col.lower() or 'prompt' in col.lower():
                    text_column = col
                    break
        
        if text_column is None or text_column not in self.data.columns:
            return {}
        
        texts = self.data[text_column].dropna()
        
        # Análisis de palabras clave
        all_words = []
        color_words = []
        shape_words = []
        object_words = []
        
        color_patterns = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 
                         'gray', 'grey', 'brown', 'pink', 'cyan', 'magenta', 'crimson', 'azure']
        
        shape_patterns = ['circle', 'square', 'rectangle', 'triangle', 'oval', 'diamond', 
                         'polygon', 'line', 'curve', 'arc', 'ellipse']
        
        object_patterns = ['tree', 'house', 'car', 'person', 'flower', 'mountain', 'cloud', 
                          'sun', 'moon', 'star', 'forest', 'ocean', 'river', 'lighthouse']
        
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            all_words.extend(words)
            
            # Categorizar palabras
            for word in words:
                if word in color_patterns:
                    color_words.append(word)
                elif word in shape_patterns:
                    shape_words.append(word)
                elif word in object_patterns:
                    object_words.append(word)
        
        # Contar frecuencias
        word_freq = Counter(all_words)
        color_freq = Counter(color_words)
        shape_freq = Counter(shape_words)
        object_freq = Counter(object_words)
        
        analysis = {
            'total_words': len(all_words),
            'unique_words': len(word_freq),
            'most_common_words': word_freq.most_common(20),
            'color_analysis': {
                'total_colors': len(color_words),
                'unique_colors': len(color_freq),
                'most_common_colors': color_freq.most_common(10)
            },
            'shape_analysis': {
                'total_shapes': len(shape_words),
                'unique_shapes': len(shape_freq),
                'most_common_shapes': shape_freq.most_common(10)
            },
            'object_analysis': {
                'total_objects': len(object_words),
                'unique_objects': len(object_freq),
                'most_common_objects': object_freq.most_common(10)
            }
        }
        
        self.word_frequencies = word_freq
        self.pattern_analysis = analysis
        
        return analysis
    
    def extract_features(self, text: str) -> Dict:
        """
        Extrae características específicas de un texto individual.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Dict: Características extraídas
        """
        features = {
            'length': len(text),
            'word_count': len(text.split()),
            'colors': [],
            'shapes': [],
            'objects': [],
            'adjectives': [],
            'complexity_score': 0
        }
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Patrones de detección
        color_patterns = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'black', 'white', 
                         'gray', 'grey', 'brown', 'pink', 'cyan', 'magenta', 'crimson', 'azure']
        
        shape_patterns = ['circle', 'square', 'rectangle', 'triangle', 'oval', 'diamond', 
                         'polygon', 'line', 'curve', 'arc', 'ellipse']
        
        object_patterns = ['tree', 'house', 'car', 'person', 'flower', 'mountain', 'cloud', 
                          'sun', 'moon', 'star', 'forest', 'ocean', 'river', 'lighthouse']
        
        adjective_patterns = ['beautiful', 'large', 'small', 'bright', 'dark', 'smooth', 
                             'rough', 'tall', 'short', 'wide', 'narrow', 'deep', 'shallow']
        
        # Extraer características
        for word in words:
            if word in color_patterns:
                features['colors'].append(word)
            elif word in shape_patterns:
                features['shapes'].append(word)
            elif word in object_patterns:
                features['objects'].append(word)
            elif word in adjective_patterns:
                features['adjectives'].append(word)
        
        # Calcular puntuación de complejidad
        complexity_factors = [
            len(features['colors']) * 0.2,
            len(features['shapes']) * 0.3,
            len(features['objects']) * 0.4,
            len(features['adjectives']) * 0.1,
            min(features['word_count'] / 20, 1.0) * 0.3
        ]
        
        features['complexity_score'] = sum(complexity_factors)
        
        return features
    
    def get_sample_data(self, n_samples: int = 10) -> List[Dict]:
        """
        Obtiene una muestra de datos procesados.
        
        Args:
            n_samples: Número de muestras a devolver
            
        Returns:
            List[Dict]: Lista de muestras con sus características
        """
        if self.data is None:
            return []
        
        # Encontrar columna de texto
        text_column = None
        for col in self.data.columns:
            if 'text' in col.lower() or 'description' in col.lower() or 'prompt' in col.lower():
                text_column = col
                break
        
        if text_column is None:
            return []
        
        sample_data = []
        sample_df = self.data.sample(n=min(n_samples, len(self.data)))
        
        for _, row in sample_df.iterrows():
            text = row[text_column] if pd.notna(row[text_column]) else "Sin descripción"
            features = self.extract_features(text)
            
            sample_data.append({
                'text': text,
                'features': features,
                'row_data': row.to_dict()
            })
        
        return sample_data
    
    def get_category_distribution(self) -> Dict:
        """
        Obtiene la distribución de categorías en el dataset.
        
        Returns:
            Dict: Distribución de categorías
        """
        if self.pattern_analysis:
            return {
                'colors': dict(self.pattern_analysis['color_analysis']['most_common_colors']),
                'shapes': dict(self.pattern_analysis['shape_analysis']['most_common_shapes']),
                'objects': dict(self.pattern_analysis['object_analysis']['most_common_objects'])
            }
        return {}
    
    def generate_summary_report(self) -> str:
        """
        Genera un reporte resumen del análisis de datos.
        
        Returns:
            str: Reporte de resumen
        """
        if self.data is None:
            return "No hay datos cargados"
        
        stats = self.get_basic_stats()
        
        report = f"""
=== REPORTE DE ANÁLISIS DE DATOS ===

Estadísticas Básicas:
- Total de muestras: {stats.get('total_samples', 0):,}
- Columnas: {', '.join(stats.get('columns', []))}

"""
        
        if 'text_stats' in stats:
            text_stats = stats['text_stats']
            report += f"""Análisis de Texto:
- Longitud promedio: {text_stats['avg_length']:.1f} caracteres
- Longitud mínima: {text_stats['min_length']} caracteres
- Longitud máxima: {text_stats['max_length']} caracteres
- Textos únicos: {text_stats['unique_texts']:,}

"""
        
        if self.pattern_analysis:
            pa = self.pattern_analysis
            report += f"""Análisis de Patrones:
- Total de palabras: {pa['total_words']:,}
- Palabras únicas: {pa['unique_words']:,}
- Colores detectados: {pa['color_analysis']['total_colors']}
- Formas detectadas: {pa['shape_analysis']['total_shapes']}
- Objetos detectados: {pa['object_analysis']['total_objects']}

Colores más comunes:
"""
            for color, count in pa['color_analysis']['most_common_colors'][:5]:
                report += f"  - {color}: {count} veces\n"
        
        return report
    
    def analyze_dataset(self, descriptions: List[str]) -> Dict:
        """
        Analiza un conjunto de descripciones y devuelve estadísticas agregadas.
        
        Args:
            descriptions: Lista de descripciones de texto
            
        Returns:
            Dict: Análisis agregado del dataset
        """
        if not descriptions:
            return {}
        
        all_features = []
        all_colors = []
        all_shapes = []
        all_objects = []
        complexity_scores = []
        
        for desc in descriptions:
            features = self.extract_features(desc)
            all_features.append(features)
            
            all_colors.extend(features['colors'])
            all_shapes.extend(features['shapes'])
            all_objects.extend(features['objects'])
            complexity_scores.append(features['complexity_score'])
        
        # Calcular estadísticas agregadas
        from collections import Counter
        
        color_counts = Counter(all_colors)
        shape_counts = Counter(all_shapes)
        object_counts = Counter(all_objects)
        
        analysis = {
            'total_descriptions': len(descriptions),
            'avg_complexity': sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0,
            'color_distribution': dict(color_counts.most_common(10)),
            'shape_distribution': dict(shape_counts.most_common(10)),
            'object_distribution': dict(object_counts.most_common(10)),
            'total_unique_colors': len(color_counts),
            'total_unique_shapes': len(shape_counts),
            'total_unique_objects': len(object_counts),
            'avg_word_count': sum(f['word_count'] for f in all_features) / len(all_features),
            'features_per_description': all_features
        }
        
        return analysis
