"""
GUI Simulator for Drawing with LLMs Project
Graphical interface application for simulating the complete project process
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import time
import json
import os
from datetime import datetime
import webbrowser
import tempfile
from typing import Dict, List, Tuple, Optional

# Try relative imports first, fall back to direct imports
try:
    from .svg_generator import SVGGenerator
    from .data_processor import DataProcessor
    from .performance_analyzer import PerformanceAnalyzer
    from .llm_connector import LLMManager
except ImportError:
    from svg_generator import SVGGenerator
    from data_processor import DataProcessor
    from performance_analyzer import PerformanceAnalyzer
    from llm_connector import LLMManager


class DrawingWithLLMsGUI:
    """Main graphical interface for the Drawing with LLMs simulator."""
    
    def __init__(self, root: tk.Tk):
        """Initialize the graphical interface."""
        self.root = root
        self.root.title("Drawing with LLMs - Simulator")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # System components
        self.svg_generator = SVGGenerator()
        self.data_processor = DataProcessor()
        self.performance_analyzer = PerformanceAnalyzer()
        self.llm_manager = LLMManager()
        
        # State variables
        self.is_running = False
        self.current_results = []
        self.current_svg = ""
        self.llm_enabled = len(self.llm_manager.get_available_connectors()) > 0
        
        # Setup interface
        self._setup_styles()
        self._create_widgets()
        self._load_sample_data()
        
    def _setup_styles(self):
        """Configure interface styles."""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure custom colors
        style.configure('Title.TLabel', 
                       font=('Arial', 16, 'bold'),
                       foreground='#2c3e50')
        
        style.configure('Header.TLabel',
                       font=('Arial', 12, 'bold'),
                       foreground='#34495e')
        
        style.configure('Status.TLabel',
                       font=('Arial', 10),
                       foreground='#27ae60')
        
        style.configure('Success.TButton',
                       background='#27ae60',
                       foreground='white')
        
        style.configure('Warning.TButton',
                       background='#f39c12',
                       foreground='white')
        
        style.configure('Danger.TButton',
                       background='#e74c3c',
                       foreground='white')
    
    def _create_widgets(self):
        """Create all interface widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(main_frame, text="Drawing with LLMs - Simulator", 
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Control panel
        self._create_control_panel(main_frame)
        
        # Input data panel
        self._create_input_panel(main_frame)
        
        # Results panel
        self._create_results_panel(main_frame)
        
        # Metrics panel
        self._create_metrics_panel(main_frame)
        
        # Status bar
        self._create_status_bar(main_frame)
    
    def _create_control_panel(self, parent):
        """Create the main control panel."""
        control_frame = ttk.LabelFrame(parent, text="Control Panel", padding="10")
        control_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 10))
        
        # First row - Main buttons
        ttk.Button(control_frame, text="Run Complete Simulation",
                  command=self._run_full_simulation,
                  style='Success.TButton').grid(row=0, column=0, padx=5)
        
        ttk.Button(control_frame, text="Analyze Dataset",
                  command=self._analyze_dataset,
                  style='Warning.TButton').grid(row=0, column=1, padx=5)
        
        ttk.Button(control_frame, text="Clear Results",
                  command=self._clear_results,
                  style='Danger.TButton').grid(row=0, column=2, padx=5)
        
        ttk.Button(control_frame, text="Export Results",
                  command=self._export_results).grid(row=0, column=3, padx=5)
        
        ttk.Button(control_frame, text="Load Dataset",
                  command=self._load_dataset).grid(row=0, column=4, padx=5)
        
        # Second row - LLM controls
        llm_frame = ttk.Frame(control_frame)
        llm_frame.grid(row=1, column=0, columnspan=5, sticky="ew", pady=(10, 0))
        
        # LLM status
        llm_status = "LLM Active" if self.llm_enabled else "LLM Not Available"
        ttk.Label(llm_frame, text=llm_status, 
                 style='Status.TLabel').grid(row=0, column=0, padx=5)
        
        # LLM info (Ollama only)
        if self.llm_enabled:
            current_llm = self.llm_manager.current_connector or "None"
            ttk.Label(llm_frame, text=f"LLM: {current_llm.title()}", 
                     style='Status.TLabel').grid(row=0, column=1, padx=5)
        
            # Toggle LLM
            self.use_llm = tk.BooleanVar(value=True)
            ttk.Checkbutton(llm_frame, text="Use LLM", 
                           variable=self.use_llm).grid(row=0, column=3, padx=5)
            
            # Connector status
            ttk.Button(llm_frame, text="LLM Status", 
                      command=self._show_llm_status).grid(row=0, column=4, padx=5)
        
        # Tercera fila - Barra de progreso
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, 
                                           variable=self.progress_var,
                                           maximum=100)
        self.progress_bar.grid(row=2, column=0, columnspan=5, sticky="ew", pady=(10, 0))
        
        control_frame.columnconfigure(0, weight=1)
        control_frame.columnconfigure(1, weight=1)
        control_frame.columnconfigure(2, weight=1)
        control_frame.columnconfigure(3, weight=1)
        control_frame.columnconfigure(4, weight=1)
    
    def _create_input_panel(self, parent):
        """Create the data input panel."""
        input_frame = ttk.LabelFrame(parent, text="Input Data", padding="10")
        input_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 5))
        
        # Text area for individual description
        ttk.Label(input_frame, text="Individual Description:", 
                 style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.description_entry = tk.Entry(input_frame, font=('Arial', 10))
        self.description_entry.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        self.description_entry.insert(0, "a purple forest at dusk")
        
        ttk.Button(input_frame, text="Generate SVG",
                  command=self._generate_single_svg).grid(row=2, column=0, pady=(0, 10))
        
        # Complete dataset
        ttk.Label(input_frame, text="Complete dataset:", 
                 style='Header.TLabel').grid(row=3, column=0, sticky=tk.W, pady=(10, 5))
        
        self.dataset_text = scrolledtext.ScrolledText(input_frame, height=15, width=50)
        self.dataset_text.grid(row=4, column=0, sticky="nsew", pady=(0, 5))
        
        input_frame.columnconfigure(0, weight=1)
        input_frame.rowconfigure(4, weight=1)
    
    def _create_results_panel(self, parent):
        """Create the results panel."""
        results_frame = ttk.LabelFrame(parent, text="Results", padding="10")
        results_frame.grid(row=2, column=1, sticky="nsew", padx=5)
        
        # Current SVG
        ttk.Label(results_frame, text="Generated SVG:", 
                 style='Header.TLabel').grid(row=0, column=0, sticky=tk.W, pady=(0, 5))
        
        self.svg_text = scrolledtext.ScrolledText(results_frame, height=8, width=50)
        self.svg_text.grid(row=1, column=0, sticky="ew", pady=(0, 5))
        
        # Button to visualize SVG
        ttk.Button(results_frame, text="Visualize SVG",
                  command=self._preview_svg).grid(row=2, column=0, pady=(0, 10))
        
        # Processing analysis
        ttk.Label(results_frame, text="Processing Analysis:", 
                 style='Header.TLabel').grid(row=3, column=0, sticky=tk.W, pady=(10, 5))
        
        self.analysis_text = scrolledtext.ScrolledText(results_frame, height=10, width=50)
        self.analysis_text.grid(row=4, column=0, sticky="nsew", pady=(0, 5))
        
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(4, weight=1)
    
    def _create_metrics_panel(self, parent):
        """Create the metrics panel."""
        metrics_frame = ttk.LabelFrame(parent, text="Performance Metrics", padding="10")
        metrics_frame.grid(row=2, column=2, sticky="nsew", padx=(5, 0))
        
        # Real-time metrics
        self.metrics_labels = {}
        metrics_data = [
            ("Processed Cases", "0"),
            ("Success Rate", "0%"),
            ("Average LLM Time", "0.0s"),
            ("Throughput", "0 pred/s"),
            ("CPU Usage", "0%"),
            ("Primary GPU", "N/A"),
            ("GPU Usage", "0%"),
            ("GPU Memory", "0 MB"),
            ("RAM Usage", "0 MB"),
            ("Average Elements", "0")
        ]
        
        for i, (label, value) in enumerate(metrics_data):
            ttk.Label(metrics_frame, text=f"{label}:", 
                     style='Header.TLabel').grid(row=i, column=0, sticky=tk.W, pady=2)
            
            self.metrics_labels[label] = ttk.Label(metrics_frame, text=value, 
                                                  style='Status.TLabel')
            self.metrics_labels[label].grid(row=i, column=1, sticky=tk.W, padx=(10, 0), pady=2)
        
        # Performance chart (simplified)
        ttk.Label(metrics_frame, text="Performance History:", 
                 style='Header.TLabel').grid(row=len(metrics_data), column=0, 
                                           columnspan=2, sticky=tk.W, pady=(20, 5))
        
        self.performance_text = scrolledtext.ScrolledText(metrics_frame, height=8, width=40)
        self.performance_text.grid(row=len(metrics_data)+1, column=0, columnspan=2, 
                                  sticky="nsew", pady=(0, 5))
        
        metrics_frame.columnconfigure(1, weight=1)
        metrics_frame.rowconfigure(len(metrics_data)+1, weight=1)
    
    def _create_status_bar(self, parent):
        """Crea la barra de estado."""
        status_frame = ttk.Frame(parent)
        status_frame.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        self.status_var = tk.StringVar()
        self.status_var.set("System ready. Load a dataset or run simulation.")
        
        ttk.Label(status_frame, textvariable=self.status_var, 
                 style='Status.TLabel').grid(row=0, column=0, sticky=tk.W)
        
        # Time information
        self.time_var = tk.StringVar()
        self.time_var.set(f"{datetime.now().strftime('%H:%M:%S')}")
        
        ttk.Label(status_frame, textvariable=self.time_var).grid(row=0, column=1, sticky=tk.E)
        
        status_frame.columnconfigure(0, weight=1)
        
        # Update time every second
        self._update_time()
    
    def _update_time(self):
        """Updates the time in the status bar."""
        self.time_var.set(f"{datetime.now().strftime('%H:%M:%S')}")
        self.root.after(1000, self._update_time)
    
    def _load_sample_data(self):
        """Loads sample data into the dataset."""
        sample_data = [
            "a purple forest at dusk",
            "gray wool coat with a faux fur collar",
            "a lighthouse overlooking the ocean",
            "burgundy corduroy pants with patch pockets",
            "orange corduroy overalls",
            "a purple silk scarf with tassel trim",
            "a green lagoon under a cloudy sky",
            "crimson rectangles forming a chaotic grid",
            "purple pyramids spiraling around a bronze cone",
            "magenta trapezoids layered on a silver sheet",
            "a snowy plain",
            "black and white checkered pants",
            "a starlit night over snow-covered peaks",
            "khaki triangles and azure crescents",
            "a maroon dodecahedron interwoven with teal threads"
        ]
        
        self.dataset_text.delete(1.0, tk.END)
        for i, desc in enumerate(sample_data, 1):
            self.dataset_text.insert(tk.END, f"{i:02d}. {desc}\n")
    
    def _generate_single_svg(self):
        """Genera un SVG para una descripción individual."""
        description = self.description_entry.get().strip()
        if not description:
            messagebox.showwarning("Warning", "Please enter a description.")
            return
        
        self.status_var.set("Generating SVG...")
        self.root.update()
        
        try:
            start_time = time.time()
            analysis = {}  # Inicializar analysis
            
            # Decidir si usar LLM o generador tradicional
            if self.llm_enabled and hasattr(self, 'use_llm') and self.use_llm.get():
                # Usar LLM para generación mejorada
                self.status_var.set("Generating with LLM...")
                self.root.update()
                
                llm_start_time = time.time()
                llm_result = self.llm_manager.generate_svg_enhanced(description)
                llm_end_time = time.time()
                llm_response_time = llm_end_time - llm_start_time
                
                svg_code = llm_result['svg_code']
                llm_analysis = self.llm_manager.analyze_description(description)
                
                # Register LLM response in performance analyzer
                if hasattr(self, 'performance_analyzer') and self.performance_analyzer:
                    success = llm_result.get('success', False)
                    self.performance_analyzer.record_llm_response(
                        response_time=llm_response_time,
                        success=success,
                        operation='svg_generation'
                    )
                
                # Combinar análisis tradicional con LLM
                analysis = self.data_processor.extract_features(description)
                analysis.update({
                    'llm_used': True,
                    'llm_source': llm_result.get('source', 'unknown'),
                    'llm_model': llm_result.get('model', 'unknown'),
                    'llm_success': llm_result.get('success', False),
                    'llm_tokens': llm_result.get('tokens_used', 0),
                    'llm_generation_time': llm_response_time,
                    'llm_analysis': llm_analysis
                })
                
            else:
                # Usar generador tradicional
                svg_code = self.svg_generator.generate_svg(description)
                analysis = self.data_processor.extract_features(description)
                analysis['llm_used'] = False
            
            end_time = time.time()
            
            # Calcular métricas finales
            analysis['total_generation_time'] = end_time - start_time
            analysis['svg_size'] = len(svg_code)
            analysis['svg_elements'] = self._count_svg_elements(svg_code)
            
            # Mostrar resultados
            self.current_svg = svg_code
            self.svg_text.delete(1.0, tk.END)
            self.svg_text.insert(tk.END, svg_code)
            
            # Mostrar análisis
            self.analysis_text.delete(1.0, tk.END)
            analysis_text = self._format_analysis_with_llm(analysis)
            self.analysis_text.insert(tk.END, analysis_text)
            
            # Actualizar métricas
            self._update_single_metrics(analysis)
            
            status_msg = f"✅ SVG generado en {analysis['total_generation_time']:.3f}s"
            if analysis.get('llm_used'):
                status_msg += f" (LLM: {analysis['llm_source']})"
            self.status_var.set(status_msg)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating SVG: {str(e)}")
            self.status_var.set("Error generating SVG")
    
    def _run_full_simulation(self):
        """Runs the complete system simulation."""
        if self.is_running:
            messagebox.showwarning("Warning", "Simulation is already running.")
            return
        
        # Get descriptions from dataset
        dataset_content = self.dataset_text.get(1.0, tk.END).strip()
        if not dataset_content:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        
        descriptions = self._parse_dataset(dataset_content)
        if not descriptions:
            messagebox.showwarning("Warning", "No valid descriptions found.")
            return
        
        # Initialize performance analyzer
        if hasattr(self, 'performance_analyzer') and self.performance_analyzer:
            self.performance_analyzer.start_session()
        
        # Run simulation in separate thread
        self.is_running = True
        thread = threading.Thread(target=self._run_simulation_thread, args=(descriptions,))
        thread.daemon = True
        thread.start()
    
    def _run_simulation_thread(self, descriptions: List[str]):
        """Executes simulation in a separate thread."""
        try:
            # Initialize performance analyzer session
            if hasattr(self, 'performance_analyzer') and self.performance_analyzer:
                self.performance_analyzer.start_session()
            
            self.root.after(0, lambda: self.status_var.set("Starting full simulation..."))
            
            results = []
            total_descriptions = len(descriptions)
            
            for i, description in enumerate(descriptions):
                if not self.is_running:
                    break
                
                # Update progress
                progress = (i / total_descriptions) * 100
                self.root.after(0, lambda p=progress: self.progress_var.set(p))
                self.root.after(0, lambda d=description: self.status_var.set(f"Processing: {d[:50]}..."))
                
                # Generar SVG
                start_time = time.time()
                svg_code = self.svg_generator.generate_svg(description)
                end_time = time.time()
                
                # Analizar resultado
                analysis = self.data_processor.extract_features(description)
                analysis['generation_time'] = end_time - start_time
                analysis['svg_size'] = len(svg_code)
                analysis['svg_elements'] = self._count_svg_elements(svg_code)
                analysis['svg_code'] = svg_code
                analysis['description'] = description
                analysis['index'] = i + 1
                
                results.append(analysis)
                
                # Update metrics in real time
                self.root.after(0, lambda r=results: self._update_simulation_metrics(r))
                
                # Pequeña pausa para actualizar interfaz
                time.sleep(0.1)
            
            # Finalizar simulación
            self.current_results = results
            self.root.after(0, lambda: self.progress_var.set(100))
            self.root.after(0, lambda: self._finalize_simulation(results))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Simulation error: {str(e)}"))
        finally:
            self.is_running = False
    
    def _analyze_dataset(self):
        """Analiza el dataset actual."""
        dataset_content = self.dataset_text.get(1.0, tk.END).strip()
        if not dataset_content:
            messagebox.showwarning("Warning", "Please load a dataset first.")
            return
        
        descriptions = self._parse_dataset(dataset_content)
        if not descriptions:
            messagebox.showwarning("Warning", "No valid descriptions found.")
            return
        
        # Analizar dataset
        analysis = self.data_processor.analyze_dataset(descriptions)
        
        # Mostrar análisis
        self.analysis_text.delete(1.0, tk.END)
        analysis_text = self._format_dataset_analysis(analysis)
        self.analysis_text.insert(tk.END, analysis_text)
        
        self.status_var.set(f"Dataset analyzed: {len(descriptions)} descriptions")
    
    def _clear_results(self):
        """Limpia todos los resultados."""
        self.svg_text.delete(1.0, tk.END)
        self.analysis_text.delete(1.0, tk.END)
        self.performance_text.delete(1.0, tk.END)
        
        # Reset metrics
        for label in self.metrics_labels:
            if "Usage" in label:
                self.metrics_labels[label].config(text="0%" if "%" in self.metrics_labels[label].cget("text") else "0 MB")
            elif "Average LLM Time" in label:
                self.metrics_labels[label].config(text="0.0s")
            else:
                self.metrics_labels[label].config(text="0")
        
        self.progress_var.set(0)
        self.current_results = []
        self.current_svg = ""
        
        self.status_var.set("Results cleared")
    
    def _export_results(self):
        """Exporta los resultados actuales."""
        if not self.current_results:
            messagebox.showwarning("Warning", "No results to export.")
            return
        
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                export_data = {
                    'timestamp': datetime.now().isoformat(),
                    'total_cases': len(self.current_results),
                    'summary': self._generate_summary(),
                    'results': self.current_results
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("Success", f"Results exported to {filename}")
                self.status_var.set(f"Results exported to {os.path.basename(filename)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Export error: {str(e)}")
    
    def _load_dataset(self):
        """Carga un dataset desde archivo."""
        filename = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        
        if filename:
            try:
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                self.dataset_text.delete(1.0, tk.END)
                self.dataset_text.insert(tk.END, content)
                
                self.status_var.set(f"Dataset loaded from {os.path.basename(filename)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading dataset: {str(e)}")
    
    def _preview_svg(self):
        """Abre una ventana para previsualizar el SVG actual."""
        if not self.current_svg:
            messagebox.showwarning("Warning", "No SVG to preview.")
            return
        
        try:
            # Crear archivo temporal
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as f:
                f.write(self.current_svg)
                temp_filename = f.name
            
            # Abrir en navegador
            webbrowser.open(f'file://{temp_filename}')
            
            self.status_var.set("SVG opened in browser")
            
        except Exception as e:
            messagebox.showerror("Error", f"SVG preview error: {str(e)}")
    
    def _parse_dataset(self, content: str) -> List[str]:
        """Parsea el contenido del dataset."""
        descriptions = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detectar formato numerado
            if line.startswith(tuple(f"{i:02d}." for i in range(1, 100))):
                description = line.split('.', 1)[1].strip()
                descriptions.append(description)
            # Detectar formato CSV
            elif ',' in line and not line.startswith('id,'):
                parts = line.split(',')
                if len(parts) >= 2:
                    description = parts[1].strip(' "')
                    descriptions.append(description)
            # Línea simple
            elif not line.startswith('id,') and not line.startswith('"id"'):
                descriptions.append(line)
        
        return descriptions
    
    def _count_svg_elements(self, svg_code: str) -> int:
        """Cuenta los elementos SVG en el código."""
        elements = ['<rect', '<circle', '<ellipse', '<polygon', '<path', '<line', '<text']
        count = 0
        for element in elements:
            count += svg_code.count(element)
        return count
    
    def _format_analysis(self, analysis: Dict) -> str:
        """Format analysis for display."""
        lines = []
        lines.append(f"Description: {analysis.get('description', 'N/A')}")
        lines.append(f"Generation time: {analysis.get('generation_time', 0):.4f}s")
        lines.append(f"SVG size: {analysis.get('svg_size', 0)} characters")
        lines.append(f"SVG elements: {analysis.get('svg_elements', 0)}")
        lines.append(f"Colors detected: {', '.join(analysis.get('colors', []))}")
        lines.append(f"Shapes detected: {', '.join(analysis.get('shapes', []))}")
        lines.append(f"Category: {analysis.get('category', 'N/A')}")
        lines.append(f"Complexity: {analysis.get('complexity', 0):.2f}")
        lines.append(f"Keywords: {', '.join(analysis.get('keywords', []))}")
        
        return '\n'.join(lines)
    
    def _format_dataset_analysis(self, analysis: Dict) -> str:
        """Formatea el análisis del dataset."""
        lines = []
        lines.append(f"📊 ANÁLISIS DEL DATASET")
        lines.append(f"=" * 50)
        lines.append(f"Total de descripciones: {analysis.get('total_descriptions', 0)}")
        lines.append(f"Average length: {analysis.get('avg_length', 0):.1f} characters")
        lines.append(f"Longitud mínima: {analysis.get('min_length', 0)} caracteres")
        lines.append(f"Longitud máxima: {analysis.get('max_length', 0)} caracteres")
        lines.append(f"")
        lines.append(f"Distribución por categorías:")
        for category, count in analysis.get('categories', {}).items():
            lines.append(f"  • {category}: {count} casos")
        lines.append(f"")
        lines.append(f"Colores más frecuentes:")
        for color, count in analysis.get('common_colors', [])[:5]:
            lines.append(f"  • {color}: {count} apariciones")
        lines.append(f"")
        lines.append(f"Formas más frecuentes:")
        for shape, count in analysis.get('common_shapes', [])[:5]:
            lines.append(f"  • {shape}: {count} apariciones")
        
        return '\n'.join(lines)
    
    def _update_single_metrics(self, analysis: Dict):
        """Updates metrics for a single case."""
        self.metrics_labels["Processed Cases"].config(text="1")
        self.metrics_labels["Success Rate"].config(text="100%")
        
        # Use LLM generation time specifically for Average LLM Time
        llm_time = analysis.get('llm_generation_time', analysis.get('generation_time', 0))
        self.metrics_labels["Average LLM Time"].config(text=f"{llm_time:.4f}s")
        
        self.metrics_labels["Throughput"].config(text=f"{1/llm_time if llm_time > 0 else 0:.1f} pred/s")
        self.metrics_labels["Average Elements"].config(text=f"{analysis.get('svg_elements', 0)}")
        
        # Update with current resource usage from performance analyzer
        self._update_metrics_from_analyzer()
    
    def _update_metrics_from_analyzer(self):
        """Updates interface metrics from performance analyzer data."""
        if not hasattr(self, 'performance_analyzer') or not self.performance_analyzer:
            return
        
        try:
            # Get real-time metrics from performance analyzer
            metrics = self.performance_analyzer.get_real_time_metrics()
            
            # Update interface with real metrics
            self.metrics_labels["Processed Cases"].config(text=str(metrics.get('total_operations', 0)))
            
            # Calculate success rate
            total_ops = metrics.get('total_operations', 0)
            error_count = metrics.get('error_count', 0)
            success_rate = ((total_ops - error_count) / total_ops * 100) if total_ops > 0 else 0
            self.metrics_labels["Success Rate"].config(text=f"{success_rate:.1f}%")
            
            # LLM response time metrics (specific to LLM, not total generation)
            avg_llm_time = metrics.get('avg_response_time', 0)
            self.metrics_labels["Average LLM Time"].config(text=f"{avg_llm_time:.4f}s")
            
            # Throughput calculation
            throughput = (1 / avg_llm_time) if avg_llm_time > 0 else 0
            self.metrics_labels["Throughput"].config(text=f"{throughput:.1f} pred/s")
            
            # Resource usage metrics
            cpu_usage = metrics.get('avg_cpu_usage', 0)
            memory_usage = metrics.get('avg_memory_usage', 0)
            
            self.metrics_labels["CPU Usage"].config(text=f"{cpu_usage:.1f}%")
            self.metrics_labels["RAM Usage"].config(text=f"{memory_usage:.0f} MB")
            
            # GPU metrics with detailed information
            if hasattr(self.performance_analyzer, 'gpu_info') and self.performance_analyzer.gpu_info:
                primary_gpu_idx = self.performance_analyzer.primary_gpu_index
                if primary_gpu_idx < len(self.performance_analyzer.gpu_info):
                    primary_gpu_name = self.performance_analyzer.gpu_info[primary_gpu_idx]['name']
                    # Truncate GPU name for display
                    if len(primary_gpu_name) > 20:
                        primary_gpu_name = primary_gpu_name[:17] + "..."
                    self.metrics_labels["Primary GPU"].config(text=primary_gpu_name)
                
                # Get all GPU usage data
                all_gpu_usage = self.performance_analyzer.get_all_gpu_usage()
                if primary_gpu_idx in all_gpu_usage:
                    gpu_data = all_gpu_usage[primary_gpu_idx]
                    gpu_usage = gpu_data['usage_percent']
                    gpu_memory = gpu_data['memory_used_mb']
                    gpu_memory_total = gpu_data['memory_total_mb']
                    
                    self.metrics_labels["GPU Usage"].config(text=f"{gpu_usage:.1f}%")
                    self.metrics_labels["GPU Memory"].config(text=f"{gpu_memory:.0f}/{gpu_memory_total:.0f} MB")
                    
                    # Update performance history with GPU details
                    self._update_gpu_performance_history(all_gpu_usage)
                else:
                    # Fallback to legacy method
                    gpu_usage = metrics.get('avg_gpu_usage', 0)
                    self.metrics_labels["GPU Usage"].config(text=f"{gpu_usage:.1f}%")
                    self.metrics_labels["GPU Memory"].config(text="N/A")
            else:
                self.metrics_labels["Primary GPU"].config(text="Not detected")
                self.metrics_labels["GPU Usage"].config(text="0%")
                self.metrics_labels["GPU Memory"].config(text="N/A")
            
            # Complexity metrics
            complexity = metrics.get('avg_complexity', 0)
            self.metrics_labels["Average Elements"].config(text=f"{complexity:.1f}")
            
        except Exception as e:
            print(f"Error updating metrics from analyzer: {e}")
    
    def _update_gpu_performance_history(self, all_gpu_usage: Dict):
        """Update performance history with detailed GPU information."""
        try:
            current_time = datetime.now().strftime("%H:%M:%S")
            
            # Clear and update performance text with GPU details
            self.performance_text.delete(1.0, tk.END)
            
            self.performance_text.insert(tk.END, f"GPU Performance Update - {current_time}\n")
            self.performance_text.insert(tk.END, "=" * 40 + "\n")
            
            for gpu_idx, gpu_data in all_gpu_usage.items():
                gpu_name = gpu_data['name']
                usage = gpu_data['usage_percent']
                memory_used = gpu_data['memory_used_mb']
                memory_total = gpu_data['memory_total_mb']
                memory_percent = gpu_data['memory_percent']
                
                # Highlight high usage
                status = "⚠️ HIGH" if usage > 80 else "🟢 Normal" if usage > 0 else "💤 Idle"
                
                self.performance_text.insert(tk.END, f"\nGPU {gpu_idx}: {gpu_name}\n")
                self.performance_text.insert(tk.END, f"  Usage: {usage:.1f}% {status}\n")
                self.performance_text.insert(tk.END, f"  Memory: {memory_used:.0f}/{memory_total:.0f} MB ({memory_percent:.1f}%)\n")
                
                # Show temperature if available
                if gpu_data.get('temperature') is not None:
                    temp = gpu_data['temperature']
                    temp_status = "🔥 Hot" if temp > 80 else "🌡️ Normal"
                    self.performance_text.insert(tk.END, f"  Temperature: {temp}°C {temp_status}\n")
            
            # Scroll to bottom
            self.performance_text.see(tk.END)
            
        except Exception as e:
            print(f"Error updating GPU performance history: {e}")

    def _update_simulation_metrics(self, results: List[Dict]):
        """Updates metrics during simulation."""
        if not results:
            return
        
        total_cases = len(results)
        success_cases = sum(1 for r in results if r.get('svg_size', 0) > 0)
        
        # Calculate average LLM time specifically (not total generation time)
        llm_times = [r.get('llm_generation_time', r.get('generation_time', 0)) for r in results]
        avg_llm_time = sum(llm_times) / total_cases if total_cases > 0 else 0
        
        avg_elements = sum(r.get('svg_elements', 0) for r in results) / total_cases
        throughput = 1 / avg_llm_time if avg_llm_time > 0 else 0
        
        # Update labels with translated text
        self.metrics_labels["Processed Cases"].config(text=str(total_cases))
        self.metrics_labels["Success Rate"].config(text=f"{(success_cases/total_cases)*100:.1f}%")
        self.metrics_labels["Average LLM Time"].config(text=f"{avg_llm_time:.4f}s")
        self.metrics_labels["Throughput"].config(text=f"{throughput:.1f} pred/s")
        self.metrics_labels["Average Elements"].config(text=f"{avg_elements:.1f}")
        
        # Register metrics in performance analyzer
        if hasattr(self, 'performance_analyzer') and self.performance_analyzer:
            # Record the latest generation
            if results:
                latest_result = results[-1]
                generation_time = latest_result.get('llm_generation_time', latest_result.get('generation_time', 0))
                svg_size = latest_result.get('svg_size', 0)
                success = svg_size > 0
                
                # Record this generation in performance analyzer
                self.performance_analyzer.record_svg_generation(
                    generation_time=generation_time,
                    success=success,
                    complexity_score=latest_result.get('svg_elements', 0)
                )
                
                # Update metrics from performance analyzer
                self._update_metrics_from_analyzer()
        
        # Update history display
        if total_cases <= 15:  # Only show first 15 to avoid saturation
            self.performance_text.delete(1.0, tk.END)
            for i, result in enumerate(results, 1):
                line = f"{i:02d}. {result.get('description', '')[:30]}... "
                line += f"({result.get('generation_time', 0):.3f}s, "
                line += f"{result.get('svg_size', 0)} chars)\n"
                self.performance_text.insert(tk.END, line)
    
    def _finalize_simulation(self, results: List[Dict]):
        """Finalizes simulation and shows final results."""
        if not results:
            return
        
        # End performance analyzer session
        if hasattr(self, 'performance_analyzer') and self.performance_analyzer:
            self.performance_analyzer.end_session()
        
        # Final analysis
        summary = self._generate_summary()
        
        # Show complete analysis
        self.analysis_text.delete(1.0, tk.END)
        final_analysis = self._format_final_analysis(summary)
        self.analysis_text.insert(tk.END, final_analysis)
        
        # Update status
        self.metrics_labels["System Status"].config(text="Completed")
        self.status_var.set(f"Simulation completed: {len(results)} cases processed")
        
        # Mostrar mensaje de éxito
        messagebox.showinfo("Simulation Completed", 
                           f"Simulation completed successfully!\n\n"
                           f"Casos procesados: {len(results)}\n"
                           f"Tasa de éxito: {summary['success_rate']:.1f}%\n"
                           f"Average time: {summary['avg_time']:.4f}s")
    
    def _generate_summary(self) -> Dict:
        """Genera un resumen de los resultados."""
        if not self.current_results:
            return {}
        
        results = self.current_results
        total_cases = len(results)
        success_cases = sum(1 for r in results if r.get('svg_size', 0) > 0)
        
        return {
            'total_cases': total_cases,
            'success_cases': success_cases,
            'success_rate': (success_cases / total_cases) * 100 if total_cases > 0 else 0,
            'avg_time': sum(r.get('generation_time', 0) for r in results) / total_cases if total_cases > 0 else 0,
            'avg_size': sum(r.get('svg_size', 0) for r in results) / total_cases if total_cases > 0 else 0,
            'avg_elements': sum(r.get('svg_elements', 0) for r in results) / total_cases if total_cases > 0 else 0,
            'total_time': sum(r.get('generation_time', 0) for r in results),
            'throughput': total_cases / sum(r.get('generation_time', 0) for r in results) if sum(r.get('generation_time', 0) for r in results) > 0 else 0
        }
    
    def _format_final_analysis(self, summary: Dict) -> str:
        """Formatea el análisis final."""
        lines = []
        lines.append(f"🎯 RESUMEN FINAL DE LA SIMULACIÓN")
        lines.append(f"=" * 60)
        lines.append(f"")
        lines.append(f"GENERAL METRICS:")
        lines.append(f"  • Total cases: {summary.get('total_cases', 0)}")
        lines.append(f"  • Successful cases: {summary.get('success_cases', 0)}")
        lines.append(f"  • Success rate: {summary.get('success_rate', 0):.1f}%")
        lines.append(f"  • Total time: {summary.get('total_time', 0):.4f}s")
        lines.append(f"")
        lines.append(f"PERFORMANCE:")
        lines.append(f"  • Average time: {summary.get('avg_time', 0):.4f}s")
        lines.append(f"  • Throughput: {summary.get('throughput', 0):.1f} pred/s")
        lines.append(f"  • Average SVG size: {summary.get('avg_size', 0):.0f} chars")
        lines.append(f"  • Average elements: {summary.get('avg_elements', 0):.1f}")
        lines.append(f"")
        lines.append(f"STATUS: SIMULATION COMPLETED SUCCESSFULLY")
        lines.append(f"")
        lines.append(f"Results can be exported using the 'Export Results' button")
        lines.append(f"Individual SVGs can be visualized by selecting specific cases")
        
        return '\n'.join(lines)
    
    def _show_llm_status(self):
        """Show detailed status of LLM connectors."""
        status = self.llm_manager.get_connector_status()
        
        status_text = "🤖 ESTADO DE CONECTORES LLM\n"
        status_text += "=" * 40 + "\n\n"
        
        status_text += f"Conector Activo: {status['current'] or 'Ninguno'}\n"
        status_text += f"Conectores Disponibles: {', '.join(status['available']) or 'Ninguno'}\n\n"
        
        if status['details']:
            status_text += "DETALLES POR CONECTOR:\n"
            status_text += "-" * 25 + "\n"
            
            for name, details in status['details'].items():
                status_text += f"\n📡 {name.upper()}:\n"
                status_text += f"   Tipo: {details['type']}\n"
                status_text += f"   Modelo: {details['model']}\n"
                
                if details['type'] == 'OpenAI':
                    api_status = "✅ Configurada" if details['has_api_key'] else "❌ Falta API Key"
                    status_text += f"   API Key: {api_status}\n"
                elif details['type'] == 'HuggingFace':
                    model_status = "✅ Cargado" if details['loaded'] else "❌ Error al cargar"
                    status_text += f"   Modelo: {model_status}\n"
        else:
            status_text += "❌ No hay conectores disponibles\n"
            status_text += "\nPara usar LLM:\n"
            status_text += "1. Instala: pip install openai\n"
            status_text += "2. Configura: export OPENAI_API_KEY=tu_key\n"
            status_text += "3. Reinicia la aplicación\n"
        
        # Mostrar en ventana popup
        status_window = tk.Toplevel(self.root)
        status_window.title("Estado LLM")
        status_window.geometry("500x400")
        status_window.configure(bg='#f0f0f0')
        
        text_widget = scrolledtext.ScrolledText(status_window, wrap=tk.WORD, 
                                               font=('Courier', 10))
        text_widget.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        text_widget.insert(tk.END, status_text)
        text_widget.config(state=tk.DISABLED)
    
    def _format_analysis_with_llm(self, analysis: Dict) -> str:
        """Formatea el análisis incluyendo información del LLM."""
        text = self._format_analysis(analysis)
        
        if analysis.get('llm_used'):
            text += "\n\n🤖 ANÁLISIS LLM:\n"
            text += "=" * 20 + "\n"
            text += f"Fuente: {analysis.get('llm_source', 'desconocida')}\n"
            text += f"Modelo: {analysis.get('llm_model', 'desconocido')}\n"
            text += f"Éxito: {'✅ Sí' if analysis.get('llm_success') else '❌ No'}\n"
            text += f"Tokens usados: {analysis.get('llm_tokens', 0)}\n"
            text += f"Tiempo LLM: {analysis.get('llm_generation_time', 0):.3f}s\n"
            
            llm_analysis = analysis.get('llm_analysis', {})
            if llm_analysis and llm_analysis.get('success'):
                text += "\nAnálisis semántico LLM:\n"
                text += f"• Colores: {', '.join(llm_analysis.get('colors', []))}\n"
                text += f"• Formas: {', '.join(llm_analysis.get('shapes', []))}\n"
                text += f"• Objetos: {', '.join(llm_analysis.get('objects', []))}\n"
                text += f"• Tipo escena: {llm_analysis.get('scene_type', 'desconocido')}\n"
                text += f"• Complejidad: {llm_analysis.get('complexity', 0)}/10\n"
                text += f"• Estilo: {llm_analysis.get('style', 'desconocido')}\n"
        
        return text
    

def main():
    """Función principal para ejecutar la aplicación."""
    root = tk.Tk()
    app = DrawingWithLLMsGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
