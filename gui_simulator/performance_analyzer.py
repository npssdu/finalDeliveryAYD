"""
Performance analysis module for GUI simulator.
Handles metrics, reports and system performance analysis.
"""

import time
import json
import numpy as np
import os
import psutil  # For system resource monitoring
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import threading
from collections import defaultdict

try:
    import GPUtil  # For GPU monitoring
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

class PerformanceAnalyzer:
    """Performance analyzer for the simulation system."""
    
    def __init__(self):
        self.metrics = {
            'llm_response_times': [],  # LLM response times in seconds
            'svg_generation_times': [],  # Total SVG generation times
            'cpu_usage_samples': [],  # CPU usage percentage samples
            'memory_usage_samples': [],  # Memory usage in MB samples
            'gpu_usage_samples': [],  # GPU usage percentage samples
            'gpu_memory_samples': [],  # GPU memory usage in MB
            'disk_io_samples': [],  # Disk I/O operations per second
            'network_io_samples': [],  # Network I/O bytes per second
            'error_count': 0,
            'total_operations': 0,
            'complexity_scores': [],
            'success_rate': 0.0  # Success rate percentage
        }
        
        # GPU monitoring configuration
        self.gpu_info = self._detect_gpus()
        self.primary_gpu_index = self._find_primary_gpu()
        self.gpu_history = {i: [] for i in range(len(self.gpu_info))}  # History for each GPU
        
        self.session_stats = {
            'start_time': None,
            'end_time': None,
            'total_svgs_generated': 0,
            'total_errors': 0,
            'average_response_time': 0.0,
            'average_cpu_usage': 0.0,
            'average_memory_usage': 0.0,
            'peak_memory_usage': 0.0,
            'peak_cpu_usage': 0.0
        }
        
        self.operation_log = []
        self.performance_history = []
        self._lock = threading.Lock()
        self._start_cpu_time = None
        self._start_memory = None
        self._monitoring_active = False
        
        # Initialize baseline measurements
        try:
            self._baseline_disk_io = psutil.disk_io_counters()
            self._baseline_network_io = psutil.net_io_counters()
        except:
            self._baseline_disk_io = None
            self._baseline_network_io = None
        self._last_measurement_time = time.time()
        
    def start_session(self):
        """Start a new analysis session."""
        with self._lock:
            self.session_stats['start_time'] = datetime.now()
            self.session_stats['total_svgs_generated'] = 0
            self.session_stats['total_errors'] = 0
            self.operation_log = []
            self._start_cpu_time = time.process_time()
            self._start_memory = self._get_memory_usage()
            
            # Start resource monitoring
            self.start_resource_monitoring()
    
    def end_session(self):
        """End the current session."""
        with self._lock:
            self.session_stats['end_time'] = datetime.now()
            self._calculate_session_metrics()
            
            # Stop resource monitoring
            self.stop_resource_monitoring()
    
    def record_svg_generation(self, generation_time: float, success: bool, 
                            complexity_score: float = 0.0, error_msg: Optional[str] = None):
        """
        Registra métricas de generación de SVG.
        
        Args:
            generation_time: Tiempo de generación en segundos
            success: Si la generación fue exitosa
            complexity_score: Puntuación de complejidad del SVG
            error_msg: Mensaje de error si la generación falló
        """
        with self._lock:
            self.metrics['total_operations'] += 1
            
            if success:
                self.metrics['svg_generation_times'].append(generation_time)
                self.metrics['complexity_scores'].append(complexity_score)
                self.session_stats['total_svgs_generated'] += 1
            else:
                self.metrics['error_count'] += 1
                self.session_stats['total_errors'] += 1
            
            # Actualizar tasa de éxito
            self.metrics['success_rate'] = (
                (self.metrics['total_operations'] - self.metrics['error_count']) /
                self.metrics['total_operations']
            ) * 100
            
            # Registrar operación
            self.operation_log.append({
                'timestamp': datetime.now(),
                'operation': 'svg_generation',
                'duration': generation_time,
                'success': success,
                'complexity': complexity_score,
                'error': error_msg
            })
    
    def record_processing_time(self, operation: str, processing_time: float):
        """
        Registra tiempo de procesamiento para una operación.
        
        Args:
            operation: Nombre de la operación
            processing_time: Tiempo de procesamiento en segundos
        """
        with self._lock:
            self.metrics['processing_times'].append({
                'operation': operation,
                'time': processing_time,
                'timestamp': datetime.now()
            })
    
    def get_real_time_metrics(self) -> Dict:
        """
        Get real-time metrics.
        
        Returns:
            Dict: Current system metrics
        """
        with self._lock:
            # LLM response time metrics
            if not self.metrics['llm_response_times']:
                avg_response_time = 0.0
                min_response_time = 0.0
                max_response_time = 0.0
            else:
                avg_response_time = np.mean(self.metrics['llm_response_times'])
                min_response_time = np.min(self.metrics['llm_response_times'])
                max_response_time = np.max(self.metrics['llm_response_times'])
            
            # Resource usage metrics
            avg_cpu_usage = np.mean(self.metrics['cpu_usage_samples']) if self.metrics['cpu_usage_samples'] else 0.0
            avg_memory_usage = np.mean(self.metrics['memory_usage_samples']) if self.metrics['memory_usage_samples'] else 0.0
            avg_gpu_usage = np.mean(self.metrics['gpu_usage_samples']) if self.metrics['gpu_usage_samples'] else 0.0
            avg_gpu_memory = np.mean(self.metrics['gpu_memory_samples']) if self.metrics['gpu_memory_samples'] else 0.0
            
            return {
                'total_operations': self.metrics['total_operations'],
                'error_count': self.metrics['error_count'],
                'avg_response_time': avg_response_time,
                'min_response_time': min_response_time,
                'max_response_time': max_response_time,
                'avg_cpu_usage': avg_cpu_usage,
                'peak_cpu_usage': self.session_stats['peak_cpu_usage'],
                'avg_memory_usage': avg_memory_usage,
                'peak_memory_usage': self.session_stats['peak_memory_usage'],
                'avg_gpu_usage': avg_gpu_usage,
                'avg_gpu_memory': avg_gpu_memory,
                'total_llm_requests': len(self.metrics['llm_response_times']),
                'avg_complexity': np.mean(self.metrics['complexity_scores']) if self.metrics['complexity_scores'] else 0.0,
                'session_duration': self._get_session_duration()
            }
    
    def get_performance_trends(self) -> Dict:
        """
        Obtiene tendencias de rendimiento.
        
        Returns:
            Dict: Tendencias de rendimiento
        """
        with self._lock:
            trends = {
                'generation_times_trend': [],
                'success_rate_trend': [],
                'complexity_trend': [],
                'hourly_performance': defaultdict(list)
            }
            
            # Análisis de tendencias por ventana de tiempo
            window_size = 10
            times = self.metrics['svg_generation_times']
            
            if len(times) >= window_size:
                for i in range(window_size, len(times)):
                    window = times[i-window_size:i]
                    trends['generation_times_trend'].append(np.mean(window))
            
            # Análisis por hora
            for log_entry in self.operation_log:
                hour = log_entry['timestamp'].hour
                if log_entry['success']:
                    trends['hourly_performance'][hour].append(log_entry['duration'])
            
            return trends
    
    def calculate_efficiency_score(self) -> float:
        """
        Calcula una puntuación de eficiencia del sistema.
        
        Returns:
            float: Puntuación de eficiencia (0-100)
        """
        with self._lock:
            if self.metrics['total_operations'] == 0:
                return 0.0
            
            # Factores de eficiencia
            success_factor = self.metrics['success_rate'] / 100
            
            # Factor de velocidad (inversamente proporcional al tiempo promedio)
            if self.metrics['svg_generation_times']:
                avg_time = np.mean(self.metrics['svg_generation_times'])
                speed_factor = max(0, 1 - (avg_time / 5.0))  # Normalizar a 5 segundos como referencia
            else:
                speed_factor = 0.0
            
            # Factor de consistencia (basado en desviación estándar)
            if len(self.metrics['svg_generation_times']) > 1:
                std_time = np.std(self.metrics['svg_generation_times'])
                consistency_factor = max(0.0, 1.0 - (float(std_time) / 2.0))  # Normalizar a 2 segundos como referencia
            else:
                consistency_factor = 1.0
            
            # Puntuación final ponderada
            efficiency_score = (
                success_factor * 0.5 +
                speed_factor * 0.3 +
                consistency_factor * 0.2
            ) * 100
            
            return min(100, max(0, efficiency_score))
    
    def generate_performance_report(self) -> str:
        """
        Genera un reporte detallado de rendimiento.
        
        Returns:
            str: Reporte de rendimiento
        """
        metrics = self.get_real_time_metrics()
        efficiency = self.calculate_efficiency_score()
        
        report = f"""
=== REPORTE DE RENDIMIENTO ===

Métricas Generales:
- Operaciones totales: {metrics['total_operations']:,}
- Tasa de éxito: {metrics['success_rate']:.1f}%
- Errores: {metrics['error_count']:,}
- SVGs generados: {metrics['total_svgs_generated']:,}

Rendimiento de Generación:
- Tiempo promedio: {metrics['avg_generation_time']:.2f}s
- Tiempo mínimo: {metrics['min_generation_time']:.2f}s
- Tiempo máximo: {metrics['max_generation_time']:.2f}s
- Complejidad promedio: {metrics['avg_complexity']:.2f}

Sesión Actual:
- Duración: {metrics['session_duration']}
- Puntuación de eficiencia: {efficiency:.1f}/100

"""
        
        # Análisis de errores recientes
        recent_errors = [log for log in self.operation_log[-20:] if not log['success']]
        if recent_errors:
            report += "Errores Recientes:\n"
            for error in recent_errors[-5:]:
                report += f"- {error['timestamp'].strftime('%H:%M:%S')}: {error['error']}\n"
        
        # Recomendaciones
        report += "\nRecomendaciones:\n"
        if metrics['success_rate'] < 90:
            report += "- Revisar la lógica de generación SVG para mejorar la tasa de éxito\n"
        
        if metrics['avg_generation_time'] > 3:
            report += "- Optimizar el tiempo de generación SVG\n"
        
        if efficiency < 70:
            report += "- Mejorar la eficiencia general del sistema\n"
        
        return report
    
    def export_metrics(self, filepath: str) -> bool:
        """
        Exporta métricas a un archivo JSON.
        
        Args:
            filepath: Ruta del archivo para exportar
            
        Returns:
            bool: True si la exportación fue exitosa
        """
        try:
            with self._lock:
                export_data = {
                    'metrics': self.metrics,
                    'session_stats': self.session_stats,
                    'operation_log': [
                        {
                            'timestamp': log['timestamp'].isoformat(),
                            'operation': log['operation'],
                            'duration': log['duration'],
                            'success': log['success'],
                            'complexity': log['complexity'],
                            'error': log['error']
                        }
                        for log in self.operation_log
                    ],
                    'export_timestamp': datetime.now().isoformat()
                }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error exportando métricas: {e}")
            return False
    
    def get_statistics_summary(self) -> Dict:
        """
        Obtiene un resumen estadístico detallado.
        
        Returns:
            Dict: Resumen estadístico
        """
        with self._lock:
            if not self.metrics['svg_generation_times']:
                return {
                    'generation_stats': {},
                    'error_analysis': {},
                    'performance_percentiles': {}
                }
            
            times = np.array(self.metrics['svg_generation_times'])
            complexities = np.array(self.metrics['complexity_scores'])
            
            return {
                'generation_stats': {
                    'mean': float(np.mean(times)),
                    'median': float(np.median(times)),
                    'std': float(np.std(times)),
                    'min': float(np.min(times)),
                    'max': float(np.max(times)),
                    'total_samples': len(times)
                },
                'complexity_stats': {
                    'mean': float(np.mean(complexities)) if len(complexities) > 0 else 0.0,
                    'median': float(np.median(complexities)) if len(complexities) > 0 else 0.0,
                    'std': float(np.std(complexities)) if len(complexities) > 0 else 0.0,
                    'min': float(np.min(complexities)) if len(complexities) > 0 else 0.0,
                    'max': float(np.max(complexities)) if len(complexities) > 0 else 0.0
                },
                'error_analysis': {
                    'error_rate': self.metrics['error_count'] / self.metrics['total_operations'] * 100 if self.metrics['total_operations'] > 0 else 0.0,
                    'total_errors': self.metrics['error_count'],
                    'success_count': self.metrics['total_operations'] - self.metrics['error_count']
                },
                'performance_percentiles': {
                    'p25': float(np.percentile(times, 25)),
                    'p50': float(np.percentile(times, 50)),
                    'p75': float(np.percentile(times, 75)),
                    'p90': float(np.percentile(times, 90)),
                    'p95': float(np.percentile(times, 95)),
                    'p99': float(np.percentile(times, 99))
                }
            }
    
    def _calculate_session_metrics(self):
        """Calcula métricas de la sesión actual."""
        with self._lock:
            if self.metrics['llm_response_times']:
                self.session_stats['average_response_time'] = np.mean(self.metrics['llm_response_times'])
            
            if self.metrics['cpu_usage_samples']:
                self.session_stats['average_cpu_usage'] = np.mean(self.metrics['cpu_usage_samples'])
            
            if self.metrics['memory_usage_samples']:
                self.session_stats['average_memory_usage'] = np.mean(self.metrics['memory_usage_samples'])
            
            self.session_stats['end_time'] = datetime.now()

    def get_performance_summary(self) -> Dict:
        """
        Get comprehensive performance summary.
        
        Returns:
            Dict: Complete performance analysis
        """
        with self._lock:
            summary = {
                'session_info': {
                    'start_time': self.session_stats.get('start_time'),
                    'end_time': self.session_stats.get('end_time'),
                    'duration_seconds': self._get_session_duration(),
                    'total_operations': self.metrics['total_operations'],
                    'error_count': self.metrics['error_count']
                },
                'response_time_metrics': {
                    'average': np.mean(self.metrics['llm_response_times']) if self.metrics['llm_response_times'] else 0.0,
                    'median': np.median(self.metrics['llm_response_times']) if self.metrics['llm_response_times'] else 0.0,
                    'min': np.min(self.metrics['llm_response_times']) if self.metrics['llm_response_times'] else 0.0,
                    'max': np.max(self.metrics['llm_response_times']) if self.metrics['llm_response_times'] else 0.0,
                    'std_dev': np.std(self.metrics['llm_response_times']) if self.metrics['llm_response_times'] else 0.0
                },
                'resource_usage': {
                    'avg_cpu_usage': np.mean(self.metrics['cpu_usage_samples']) if self.metrics['cpu_usage_samples'] else 0.0,
                    'peak_cpu_usage': self.session_stats['peak_cpu_usage'],
                    'avg_memory_usage': np.mean(self.metrics['memory_usage_samples']) if self.metrics['memory_usage_samples'] else 0.0,
                    'peak_memory_usage': self.session_stats['peak_memory_usage']
                }
            }
            return summary

    def export_performance_log(self, filename: Optional[str] = None) -> str:
        """
        Export performance data to JSON file.
        
        Args:
            filename: Optional filename, if not provided will use timestamp
            
        Returns:
            str: Path to exported file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"performance_log_{timestamp}.json"
        
        performance_data = {
            'summary': self.get_performance_summary(),
            'detailed_metrics': {
                'llm_response_times': self.metrics['llm_response_times'],
                'cpu_usage_samples': self.metrics['cpu_usage_samples'],
                'memory_usage_samples': self.metrics['memory_usage_samples'],
                'operation_log': [
                    {
                        'timestamp': entry['timestamp'].isoformat(),
                        'operation': entry.get('operation', ''),
                        'response_time': entry.get('response_time', 0.0),
                        'success': entry.get('success', False),
                        'cpu_usage': entry.get('cpu_usage', 0.0),
                        'memory_usage': entry.get('memory_usage', 0.0),
                        'error': entry.get('error', None)
                    }
                    for entry in self.operation_log
                ]
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(performance_data, f, indent=2, default=str)
        
        return filename

    def _get_session_duration(self) -> str:
        """Obtiene la duración de la sesión actual como string."""
        if self.session_stats['start_time'] is None:
            return "0:00:00"
        
        end_time = self.session_stats['end_time'] or datetime.now()
        duration = end_time - self.session_stats['start_time']
        
        hours = int(duration.total_seconds() // 3600)
        minutes = int((duration.total_seconds() % 3600) // 60)
        seconds = int(duration.total_seconds() % 60)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def clear_metrics(self):
        """Limpia todas las métricas acumuladas."""
        with self._lock:
            self.metrics = {
                'svg_generation_times': [],
                'processing_times': [],
                'success_rate': 0.0,
                'error_count': 0,
                'total_operations': 0,
                'memory_usage': [],
                'complexity_scores': []
            }
            self.operation_log = []
            self.session_stats = {
                'start_time': None,
                'end_time': None,
                'total_svgs_generated': 0,
                'total_errors': 0,
                'average_generation_time': 0.0
            }
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB using psutil."""
        try:
            memory = psutil.virtual_memory()
            return memory.used / (1024 * 1024)  # Convert to MB
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage using psutil."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except:
            return 0.0
    
    def _get_gpu_usage(self) -> Tuple[float, float]:
        """Get GPU usage and memory. Returns (usage_percent, memory_mb)."""
        if not GPU_AVAILABLE:
            return 0.0, 0.0
        
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus and self.primary_gpu_index < len(gpus):
                gpu = gpus[self.primary_gpu_index]  # Use primary GPU (RTX 4070)
                usage = gpu.load * 100
                memory = gpu.memoryUsed
                
                # Store usage for this specific GPU
                if self.primary_gpu_index in self.gpu_history:
                    self.gpu_history[self.primary_gpu_index].append(usage)
                    # Keep only last 100 samples
                    if len(self.gpu_history[self.primary_gpu_index]) > 100:
                        self.gpu_history[self.primary_gpu_index] = self.gpu_history[self.primary_gpu_index][-100:]
                
                return usage, memory
        except Exception as e:
            print(f"GPU monitoring error: {e}")
        return 0.0, 0.0
    
    def _detect_gpus(self) -> List[Dict]:
        """Detect all available GPUs and return their information."""
        gpu_info = []
        if not GPU_AVAILABLE:
            return gpu_info
            
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                info = {
                    'index': i,
                    'name': gpu.name,
                    'id': gpu.id,
                    'uuid': getattr(gpu, 'uuid', f'gpu_{i}'),
                    'memory_total': gpu.memoryTotal,
                    'driver_version': getattr(gpu, 'driver', 'Unknown')
                }
                gpu_info.append(info)
                print(f"GPU {i}: {gpu.name} (Memory: {gpu.memoryTotal}MB)")
        except Exception as e:
            print(f"GPU detection error: {e}")
            
        return gpu_info
    
    def _find_primary_gpu(self) -> int:
        """Find the primary GPU (RTX 4070) for monitoring."""
        for i, gpu_info in enumerate(self.gpu_info):
            # Look for NVIDIA RTX 4070
            if 'rtx' in gpu_info['name'].lower() and '4070' in gpu_info['name'].lower():
                print(f"Primary GPU selected: {gpu_info['name']} (Index: {i})")
                return i
        
        # Fallback: use first NVIDIA GPU
        for i, gpu_info in enumerate(self.gpu_info):
            if 'nvidia' in gpu_info['name'].lower():
                print(f"Primary GPU selected (fallback): {gpu_info['name']} (Index: {i})")
                return i
        
        # Final fallback: use first GPU
        if self.gpu_info:
            print(f"Primary GPU selected (default): {self.gpu_info[0]['name']} (Index: 0)")
            return 0
        
        return 0
    
    def get_all_gpu_usage(self) -> Dict[int, Dict]:
        """Get usage information for all GPUs."""
        all_gpu_usage = {}
        if not GPU_AVAILABLE:
            return all_gpu_usage
            
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                all_gpu_usage[i] = {
                    'name': gpu.name,
                    'usage_percent': gpu.load * 100,
                    'memory_used_mb': gpu.memoryUsed,
                    'memory_total_mb': gpu.memoryTotal,
                    'memory_percent': (gpu.memoryUsed / gpu.memoryTotal * 100) if gpu.memoryTotal > 0 else 0,
                    'temperature': getattr(gpu, 'temperature', None)
                }
        except Exception as e:
            print(f"GPU monitoring error: {e}")
            
        return all_gpu_usage
    
    def start_resource_monitoring(self):
        """Start continuous resource monitoring."""
        self._monitoring_active = True
        self._monitor_resources()
        
    def stop_resource_monitoring(self):
        """Stop resource monitoring."""
        self._monitoring_active = False
        
    def _monitor_resources(self):
        """Monitor system resources in a separate thread."""
        def monitor():
            while self._monitoring_active:
                try:
                    with self._lock:
                        # Sample system resources
                        cpu_usage = self._get_cpu_usage()
                        memory_usage = self._get_memory_usage()
                        gpu_usage, gpu_memory = self._get_gpu_usage()
                        
                        self.metrics['cpu_usage_samples'].append(cpu_usage)
                        self.metrics['memory_usage_samples'].append(memory_usage)
                        self.metrics['gpu_usage_samples'].append(gpu_usage)
                        self.metrics['gpu_memory_samples'].append(gpu_memory)
                        
                        # Update session peaks
                        if cpu_usage > self.session_stats['peak_cpu_usage']:
                            self.session_stats['peak_cpu_usage'] = cpu_usage
                        if memory_usage > self.session_stats['peak_memory_usage']:
                            self.session_stats['peak_memory_usage'] = memory_usage
                except Exception as e:
                    print(f"Resource monitoring error: {e}")
                
                time.sleep(1)  # Sample every second
        
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def record_llm_response(self, response_time: float, success: bool, 
                           operation: str = "llm_request", error_msg: Optional[str] = None):
        """
        Record LLM response metrics.
        
        Args:
            response_time: Response time in seconds
            success: Whether the request was successful
            operation: Type of LLM operation
            error_msg: Error message if the request failed
        """
        with self._lock:
            self.metrics['total_operations'] += 1
            
            if success:
                self.metrics['llm_response_times'].append(response_time)
            else:
                self.metrics['error_count'] += 1
                self.session_stats['total_errors'] += 1
            
            # Record resource usage
            cpu_usage = self._get_cpu_usage()
            memory_usage = self._get_memory_usage()
            
            self.metrics['cpu_usage_samples'].append(cpu_usage)
            self.metrics['memory_usage_samples'].append(memory_usage)
            
            # Update peaks
            if cpu_usage > self.session_stats['peak_cpu_usage']:
                self.session_stats['peak_cpu_usage'] = cpu_usage
            if memory_usage > self.session_stats['peak_memory_usage']:
                self.session_stats['peak_memory_usage'] = memory_usage
            
            # Log operation
            self.operation_log.append({
                'timestamp': datetime.now(),
                'operation': operation,
                'response_time': response_time,
                'success': success,
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'error': error_msg
            })
