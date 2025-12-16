import curses
import os
import time
import threading
import subprocess
import psutil
import signal
import datetime
import shutil
from pathlib import Path
import sys 
import pickle 

# ================= CONFIGURATION =================
# Directorio por defecto para los modelos.
DEFAULT_MODELS_DIR = "./"

# Nombre del binario del servidor.
LLAMA_SERVER_BIN = "llama-server" 

# Directorio por defecto para el binario de llama-server.
DEFAULT_LLAMA_SERVER_DIR = "./"

# --- CONFIGURACIÓN DE PERSISTENCIA ---
SETTINGS_FOLDER_NAME = ".llama_tui"
SETTINGS_FILE_NAME = "config.dat"
# =================================================

class LlamaTUI:
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.running = True
        self.server_process = None
        self.running_model_path = None 
        
        # LOGGING STATE
        self.log_buffer = [] 
        self.log_thread = None
        self.show_log = False 
        self.log_max_lines = 1000 
        
        # NVIDIA-SMI VIEW
        self.show_nvidia_smi = False
        
        # Settings: Inicialización con valores por defecto
        # Usamos un diccionario de configuración para simplificar la persistencia
        self.config = {
            'model_dir': os.path.abspath(DEFAULT_MODELS_DIR),
            'backend': "CPU", 
            'host_ip': "0.0.0.0",
            'port': "8080",
            'allow_lan': False,
            'gpu_layers': 0, 
            'n_threads': psutil.cpu_count(logical=True), # Nuevo: Hilos de CPU (-t)
            'context_size': 4096,                      # Nuevo: Longitud del contexto (-c)
            'schedule_time': "",
            'schedule_active': False,
            'llama_server_dir': os.path.abspath(DEFAULT_LLAMA_SERVER_DIR),
        }

        # Sincronizar variables de instancia con el diccionario por compatibilidad
        self.sync_config_to_attributes()
        
        # --- Cargar la configuración persistente ---
        self.settings_path = self._get_settings_path()
        self.load_settings()
        
        # UI State
        self.active_field = 0 
        self.selected_file_idx = 0
        self.file_offset = 0
        self.files = []
        if not hasattr(self, 'msg_log') or self.msg_log == "":
            self.msg_log = "Welcome. Select a model and press 'S' to serve."
        
        # Initialize File List
        self.refresh_file_list()
        
        # Colors
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLUE)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_RED, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)
        curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLUE)
        curses.init_pair(6, curses.COLOR_YELLOW, -1)

        curses.curs_set(0) 
        self.stdscr.nodelay(True) 

    def sync_config_to_attributes(self):
        """Sincroniza el diccionario de configuración con los atributos de la instancia."""
        for key, value in self.config.items():
            setattr(self, key, value)
    
    def sync_attributes_to_config(self):
        """Sincroniza los atributos de la instancia con el diccionario de configuración."""
        for key in self.config.keys():
            self.config[key] = getattr(self, key)
            
    # --- MÉTODOS DE PERSISTENCIA ---
    def _get_settings_path(self):
        """Calcula y asegura la existencia del directorio de configuración en ~."""
        try:
            home_dir = Path.home()
            settings_dir = home_dir / SETTINGS_FOLDER_NAME
            settings_dir.mkdir(parents=True, exist_ok=True)
            return settings_dir / SETTINGS_FILE_NAME
        except Exception:
            return Path(SETTINGS_FILE_NAME)

    def load_settings(self):
        """Carga la configuración persistente desde el archivo."""
        self.msg_log = ""
        if self.settings_path.exists():
            try:
                with open(self.settings_path, 'rb') as f:
                    settings = pickle.load(f)
                
                # Cargar desde el diccionario y usar valores por defecto si no existen
                for key, default_value in self.config.items():
                    self.config[key] = settings.get(key, default_value)

                self.sync_config_to_attributes()
                self.msg_log = f"Settings loaded from {self.settings_path.name}."
            except Exception:
                self.msg_log = "Error loading settings. Using defaults."
            
        else:
            self.msg_log = "No persistent settings found. Using defaults."
            
        self.save_settings()

    def save_settings(self):
        """Guarda la configuración actual en el archivo."""
        self.sync_attributes_to_config() # Asegurar que el diccionario esté actualizado
        try:
            with open(self.settings_path, 'wb') as f:
                pickle.dump(self.config, f)
        except Exception:
            pass 
    # ---------------------------------------------------

    def refresh_file_list(self):
        path = Path(self.model_dir)
        self.files = []
        if path.exists() and path.is_dir():
            self.files = sorted([f for f in path.glob("*.gguf") if f.is_file()], key=lambda f: f.name)
            self.selected_file_idx = 0
            self.file_offset = 0
        else:
            self.msg_log = f"Error: Directory not found: {self.model_dir}"

    def get_ram_info(self):
        try:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            avail_gb = mem.available / (1024**3)
            percent = mem.percent
            return f"RAM: {avail_gb:.1f}GB Free / {total_gb:.1f}GB Total ({percent}%)"
        except:
            return "RAM: N/A"

    def get_file_size(self, path):
        try:
            size_gb = path.stat().st_size / (1024**3)
            return f"{size_gb:.2f} GB"
        except:
            return "?.?? GB"
    
    # --- MÉTODO: Ejecutar nvidia-smi ---
    def get_nvidia_smi_output(self):
        """Ejecuta nvidia-smi y devuelve la salida o un mensaje de error."""
        try:
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                check=True,
                timeout=5
            )
            return result.stdout.strip().split('\n')
        except FileNotFoundError:
            return ["Error: 'nvidia-smi' command not found. (NVIDIA drivers not installed or not in PATH)"]
        except subprocess.CalledProcessError as e:
            return [f"Error executing nvidia-smi: {e.stderr.strip()}"]
        except subprocess.TimeoutExpired:
            return ["Error: 'nvidia-smi' command timed out."]
        except Exception as e:
            return [f"An unexpected error occurred: {e}"]
            
    # --- LECTOR DE LOGS EN SEGUNDO PLANO ---
    def _log_reader_thread(self):
        if self.server_process and self.server_process.stderr:
            for line in iter(self.server_process.stderr.readline, b''):
                try:
                    line_str = line.decode(sys.getdefaultencoding(), errors='replace').strip()
                    if line_str:
                        self.log_buffer.append(line_str)
                        if len(self.log_buffer) > self.log_max_lines:
                            self.log_buffer.pop(0)
                except Exception:
                    pass

    def kill_server(self):
        if self.server_process:
            if self.log_thread and self.log_thread.is_alive():
                pass 

            try:
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
                self.server_process.wait(timeout=5)
                
                self.server_process = None
                self.running_model_path = None 
                self.log_buffer = []
                return "Server stopped and cache cleared."
            except Exception as e:
                try:
                    self.server_process.terminate()
                    self.server_process.wait(timeout=5)
                    self.server_process = None
                    self.running_model_path = None
                    self.log_buffer = []
                    return "Server stopped (fallback)."
                except:
                    return f"Error stopping: {e}"
        return "No server running."
        
    def start_server(self):
        if self.server_process:
            return "Server already running. Press 'K' first."
        if not self.files:
            return "No models found in current directory."
            
        llama_server_path = os.path.join(self.llama_server_dir, LLAMA_SERVER_BIN)

        if not os.path.exists(llama_server_path) or not os.path.isfile(llama_server_path):
            return f"Error: '{llama_server_path}' not found or is not a file."

        current_file_obj = self.files[self.selected_file_idx]
        model_path = str(current_file_obj.resolve())
        
        cmd = [llama_server_path, "-m", model_path, "--port", self.port]
        
        if self.allow_lan:
            cmd.extend(["--host", self.host_ip])
        else:
            cmd.extend(["--host", "127.0.0.1"])

        # LÓGICA -NGL (CAPAS GPU)
        if (self.backend == "CUDA" or self.backend == "VULKAN") and self.gpu_layers > 0:
            cmd.extend(["-ngl", str(self.gpu_layers)]) 
            
        # LÓGICA -T (HILOS CPU) -> Crucial para el rendimiento de la CPU
        if self.n_threads > 0:
            cmd.extend(["-t", str(self.n_threads)])

        # LÓGICA -C (TAMAÑO DEL CONTEXTO)
        if self.context_size > 0:
            cmd.extend(["-c", str(self.context_size)])
            
        try:
            self.server_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                start_new_session=True,
                bufsize=1
            )
            
            self.log_buffer = []
            self.log_thread = threading.Thread(target=self._log_reader_thread, daemon=True)
            self.log_thread.start()
            self.running_model_path = str(current_file_obj)
            return f"SERVING: {current_file_obj.name} - Press 'V' to view log."
        except FileNotFoundError:
            return f"Error: {llama_server_path} binary not found."
        except Exception as e:
            return f"Error: {str(e)}"

    def draw_box(self, y, x, h, w, title=""):
        try:
            self.stdscr.addstr(y, x, "┌" + "─" * (w-2) + "┐")
            for i in range(1, h-1):
                self.stdscr.addstr(y+i, x, "│" + " " * (w-2) + "│")
            self.stdscr.addstr(y+h-1, x, "└" + "─" * (w-2) + "┘")
            if title:
                self.stdscr.addstr(y, x+2, f" {title} ", curses.A_BOLD)
        except curses.error:
            pass

    def input_string(self, y, x, prompt, default_val):
        curses.echo()
        curses.curs_set(1)
        self.stdscr.move(y, 0)
        self.stdscr.clrtoeol()
        prompt_str = f"{prompt} [{default_val}]: "
        self.stdscr.addstr(y, x, prompt_str, curses.color_pair(4))
        self.stdscr.refresh()
        win = curses.newwin(1, 100, y, x + len(prompt_str))
        try:
            box = win.getstr().decode('utf-8').strip()
        except:
            box = ""
        curses.noecho()
        curses.curs_set(0)
        self.stdscr.move(y, 0)
        self.stdscr.clrtoeol()
        return box if box else default_val

    def check_schedule(self):
        if self.schedule_active and not self.server_process:
            now = datetime.datetime.now().strftime("%H:%M")
            if now == self.schedule_time:
                self.msg_log = self.start_server()
                time.sleep(60) 

    # --- FUNCIÓN: DIBUJAR EL LOG ---
    def draw_log_view(self, h, w):
        self.stdscr.clear()
        self.draw_box(0, 0, h, w, "Server Log (Press ANY Key to return)")
        log_lines = self.log_buffer
        max_display_lines = h - 2
        
        if len(log_lines) > max_display_lines:
            log_lines = log_lines[-max_display_lines:]
            
        for i, line in enumerate(log_lines):
            try:
                display_line = line[:w-3]
                self.stdscr.addstr(1 + i, 2, display_line, curses.color_pair(6))
            except curses.error:
                pass
        
        self.stdscr.refresh()
        
    # --- MÉTODO: DIBUJAR NVIDIA-SMI ---
    def draw_smi_view(self, h, w):
        self.stdscr.clear()
        self.draw_box(0, 0, h, w, "NVIDIA System Management Interface (Press ANY Key to return)")
        
        smi_lines = self.get_nvidia_smi_output()
        max_display_lines = h - 2
        
        for i, line in enumerate(smi_lines):
            if i >= max_display_lines:
                break
            try:
                display_line = line[:w-3]
                self.stdscr.addstr(1 + i, 2, display_line)
            except curses.error:
                pass
        
        self.stdscr.refresh()

    def run(self):
        while self.running:
            h, w = self.stdscr.getmaxyx()
            
            # 1. Modo NVIDIA-SMI
            if self.show_nvidia_smi:
                self.draw_smi_view(h, w)
                try:
                    key = self.stdscr.getch()
                except:
                    key = -1
                
                if key != -1:
                    self.show_nvidia_smi = False
                
                time.sleep(0.05)
                continue
                
            # 2. Modo Log
            if self.show_log:
                self.draw_log_view(h, w)
                try:
                    key = self.stdscr.getch()
                except:
                    key = -1
                
                if key != -1:
                    self.show_log = False
                
                time.sleep(0.05)
                continue
            
            # --- Lógica de Dibujo Normal (TUI) ---
            self.stdscr.clear()
            
            if h < 20 or w < 80:
                self.stdscr.addstr(0,0, "Window too small. Resize > 80x20")
                self.stdscr.refresh()
                time.sleep(0.5)
                continue

            header_text = f" LLAMA-WEB-SERVER TUI by JANS | {self.get_ram_info()} "
            self.stdscr.attron(curses.color_pair(1))
            self.stdscr.addstr(0, 0, header_text)
            try:
                self.stdscr.addstr(0, len(header_text), " " * (w - len(header_text)))
            except: pass
            self.stdscr.attroff(curses.color_pair(1))

            list_h = h - 8
            list_w = int(w * 0.65)
            sett_x = list_w + 1
            sett_w = w - list_w - 2

            self.draw_box(2, 0, list_h, list_w, f"GGUF Models List ({len(self.files)})")
            
            max_items = list_h - 2
            if self.selected_file_idx < self.file_offset:
                self.file_offset = self.selected_file_idx
            elif self.selected_file_idx >= self.file_offset + max_items:
                self.file_offset = self.selected_file_idx - max_items + 1
                
            
            for i in range(max_items):
                file_idx = self.file_offset + i
                if file_idx >= len(self.files):
                    break
                
                f = self.files[file_idx]
                y_pos = 3 + i
                is_selected = (file_idx == self.selected_file_idx)
                is_running = (str(f) == self.running_model_path)
                display_str = f"{f.name[:list_w-16]} [{self.get_file_size(f)}]"
                
                if is_selected and is_running:
                    self.stdscr.attron(curses.color_pair(5) | curses.A_BOLD)
                    self.stdscr.addstr(y_pos, 2, f"{display_str:<{list_w-4}}")
                    self.stdscr.attroff(curses.color_pair(5) | curses.A_BOLD)
                elif is_selected:
                    self.stdscr.attron(curses.color_pair(1))
                    if self.active_field != 0: self.stdscr.attroff(curses.color_pair(1))
                    else: self.stdscr.attron(curses.A_BOLD)
                        
                    self.stdscr.addstr(y_pos, 2, f"{display_str:<{list_w-4}}")
                    self.stdscr.attroff(curses.color_pair(1) | curses.A_BOLD)
     
                elif is_running:
                    self.stdscr.attron(curses.color_pair(2) | curses.A_BOLD)
                    self.stdscr.addstr(y_pos, 2, f"{display_str:<{list_w-4}}")
                    self.stdscr.attroff(curses.color_pair(2) | curses.A_BOLD)
                else:
                    self.stdscr.addstr(y_pos, 2, display_str)

            self.draw_box(2, sett_x, list_h, sett_w, "Configuration (Tab,Enter to change):")
            
            def draw_setting(idx, y_off, label, value):
                style = curses.color_pair(1) if self.active_field == idx else curses.A_NORMAL
                self.stdscr.addstr(3 + y_off, sett_x + 2, label, style)
                self.stdscr.addstr(3 + y_off, sett_x + 2 + len(label), str(value))

            disp_dir = (self.model_dir[:sett_w-10] + '..') if len(self.model_dir) > sett_w-10 else self.model_dir
            disp_server_dir = (self.llama_server_dir[:sett_w-10] + '..') if len(self.llama_server_dir) > sett_w-10 else self.llama_server_dir
            
            # Mapeo de campos activos: 0 (Lista) y 1-9 (Configuración)
            
            draw_setting(1, 0, "Model Dir:", disp_dir)
            draw_setting(2, 2, "Backend:CPU/CUDA/VULKAN:", f"[{self.backend}]")
            
            # 3. GPU Layers
            draw_setting(3, 4, "GPU Layers (-ngl):", str(self.gpu_layers)) 
            
            # 4. CPU Threads
            draw_setting(4, 5, "CPU Threads (-t):", str(self.n_threads))
            
            # 5. Context Size
            draw_setting(5, 6, "Context Size (-c):", str(self.context_size))

            # Ajuste de IP/Port (índices reajustados)
            draw_setting(6, 8, "IP:", self.host_ip) # y_off = 8
            draw_setting(7, 9, "Port:", self.port) # y_off = 9
            
            # Ajuste de LAN (active_field = 8)
            lan_txt = "ENABLED" if self.allow_lan else "DISABLED"
            lan_col = curses.color_pair(2) if self.allow_lan else curses.color_pair(3)
            style_lbl = curses.color_pair(1) if self.active_field == 8 else curses.A_NORMAL
            self.stdscr.addstr(3 + 10, sett_x + 2, "LAN:", style_lbl) # y_off = 10
            self.stdscr.addstr(3 + 10, sett_x + 2 + len("LAN:") + 1, lan_txt, lan_col)
            
            # Ajuste de Schedule (active_field = 9)
            sched_txt = f"ON ({self.schedule_time})" if self.schedule_active else "OFF"
            draw_setting(9, 11, "Schedule Starting:", sched_txt) # y_off = 11

            # 10. Llama Server Bin Dir
            draw_setting(10, 13, "Llama-Server Binary Dir:", disp_server_dir)

            self.stdscr.addstr(list_h - 2, sett_x + 2, "Server Status:", curses.A_UNDERLINE)
            if self.server_process:
                self.stdscr.addstr(list_h - 1, sett_x + 2, "● RUNNING", curses.color_pair(2) | curses.A_BOLD)
                self.stdscr.addstr(list_h, sett_x + 2, f"PID: {self.server_process.pid}")
            else:
                self.stdscr.addstr(list_h - 1, sett_x + 2, "● STOPPED", curses.color_pair(3))

            self.draw_box(h-6, 0, 6, w, "Controls")
            
            keys_hint = "TAB:Switch Section | ENTER:Edit/Select | ARROWS:Navigate"
            cmds_hint = "[S] START | [K] KILL | [V] VIEW LOG | [N] NVIDIA-SMI | [D] SERVER DIR | [Q] QUIT"
            
            self.stdscr.addstr(h-4, 2, keys_hint)
            self.stdscr.addstr(h-3, 2, cmds_hint, curses.A_BOLD)
            
            self.stdscr.addstr(h-2, 2, f"> {self.msg_log}", curses.color_pair(4))

            self.stdscr.refresh()
            self.check_schedule()

            try:
                key = self.stdscr.getch()
            except:
                key = -1

            
            if key == ord('q') or key == ord('Q'):
                self.kill_server()
                self.save_settings()
                self.running = False
            
            elif key == ord('k') or key == ord('K'):
                self.msg_log = self.kill_server()

            elif key == ord('s') or key == ord('S'):
                self.msg_log = self.start_server()
                
            elif key == ord('v') or key == ord('V'):
                if self.server_process:
                    self.show_log = True
                else:
                    self.msg_log = "Server is not running. Start it first."
            
            elif key == ord('n') or key == ord('N'):
                self.show_nvidia_smi = True
                
            elif key == ord('d') or key == ord('D'):
                new_server_dir = self.input_string(h-2, 2, "Full Path to llama-server binary directory:", self.llama_server_dir)
                if os.path.isdir(new_server_dir):
                    self.llama_server_dir = os.path.abspath(new_server_dir)
                    self.msg_log = f"Llama-server directory updated to: {self.llama_server_dir}"
                    self.save_settings()
                else:
                    self.msg_log = "Invalid directory path."

            elif key == 9: 
                # Navegación con TAB (11 campos activos: 0 a 10)
                self.active_field = (self.active_field + 1) % 11 
            
            elif key == curses.KEY_UP:
                if self.active_field == 0 and self.selected_file_idx > 0:
                    self.selected_file_idx -= 1
            
            elif key == curses.KEY_DOWN:
                if self.active_field == 0 and self.selected_file_idx < len(self.files) - 1:
                    self.selected_file_idx += 1

            elif key == 10: # ENTER
                if self.active_field == 1: # Model Dir
                    new_dir = self.input_string(h-2, 2, "Full Path to Models:", self.model_dir)
                    if os.path.isdir(new_dir):
                        self.model_dir = os.path.abspath(new_dir)
                        self.refresh_file_list()
                        self.msg_log = "Directory updated."
                        self.save_settings()
                    else:
                        self.msg_log = "Invalid directory path."

                elif self.active_field == 2: # Backend
                    modes = ["CPU", "CUDA", "VULKAN"]
                    curr_idx = modes.index(self.backend)
                    self.backend = modes[(curr_idx + 1) % len(modes)]
                    self.save_settings()
                
                elif self.active_field == 3: # GPU Layers (-ngl)
                    val = self.input_string(h-2, 2, "Number of GPU Layers (-ngl):", str(self.gpu_layers))
                    try:
                        new_ngl = int(val)
                        if new_ngl >= 0:
                            self.gpu_layers = new_ngl
                            self.msg_log = f"GPU Layers set to {self.gpu_layers} (0=auto/CPU only)."
                            self.save_settings()
                        else:
                            self.msg_log = "Value must be 0 or greater."
                    except ValueError:
                        self.msg_log = "Invalid number format."
                        
                elif self.active_field == 4: # CPU Threads (-t)
                    val = self.input_string(h-2, 2, f"Number of CPU Threads (-t) [Default: {psutil.cpu_count(logical=True)}]:", str(self.n_threads))
                    try:
                        new_threads = int(val)
                        if new_threads >= 1:
                            self.n_threads = new_threads
                            self.msg_log = f"CPU Threads set to {self.n_threads}."
                            self.save_settings()
                        else:
                            self.msg_log = "Value must be 1 or greater."
                    except ValueError:
                        self.msg_log = "Invalid number format."
                
                elif self.active_field == 5: # Context Size (-c)
                    val = self.input_string(h-2, 2, "Context Size (-c) [4096 is common]:", str(self.context_size))
                    try:
                        new_context = int(val)
                        if new_context >= 512:
                            self.context_size = new_context
                            self.msg_log = f"Context Size set to {self.context_size}."
                            self.save_settings()
                        else:
                            self.msg_log = "Value must be at least 512."
                    except ValueError:
                        self.msg_log = "Invalid number format."
                
                elif self.active_field == 6: # IP
                    self.host_ip = self.input_string(h-2, 2, "Host IP:", self.host_ip)
                    self.save_settings()
                
                elif self.active_field == 7: # Port
                    self.port = self.input_string(h-2, 2, "Port:", self.port)
                    self.save_settings()

                elif self.active_field == 8: # LAN
                    self.allow_lan = not self.allow_lan
                    self.save_settings()
                
                elif self.active_field == 9: # Schedule
                    if self.schedule_active:
                        self.schedule_active = False
                        self.msg_log = "Schedule disabled."
                        self.save_settings()
                    else:
                        t = self.input_string(h-2, 2, "Start Time (HH:MM):", "08:00")
                        try:
                            datetime.datetime.strptime(t, "%H:%M")
                            self.schedule_time = t
                            self.schedule_active = True
                            self.msg_log = f"Scheduled for {t}"
                            self.save_settings()
                        except ValueError:
                            self.msg_log = "Invalid time format. Use HH:MM"

                elif self.active_field == 10: # Llama Server Bin Dir
                    new_server_dir = self.input_string(h-2, 2, "Full Path to llama-server binary directory:", self.llama_server_dir)
                    if os.path.isdir(new_server_dir):
                        self.llama_server_dir = os.path.abspath(new_server_dir)
                        self.msg_log = f"Llama-server directory updated to: {self.llama_server_dir}"
                        self.save_settings()
                    else:
                        self.msg_log = "Invalid directory path."

            time.sleep(0.05)

def main():
    if not os.path.exists(DEFAULT_MODELS_DIR):
        try:
            os.makedirs(DEFAULT_MODELS_DIR)
        except:
            pass 

    try:
        curses.wrapper(lambda stdscr: LlamaTUI(stdscr).run())
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Fatal Error: {e}")
        # Opcional: Escribir el error en un archivo si la TUI falla

if __name__ == "__main__":
    main()