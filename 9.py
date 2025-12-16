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
import pickle # Para serializar la configuración

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
        
        # Settings: Inicialización con valores por defecto
        self.model_dir = os.path.abspath(DEFAULT_MODELS_DIR)
        self.backend = "CPU" 
        self.host_ip = "0.0.0.0"
        self.port = "8080"
        self.allow_lan = False
    
        self.schedule_time = "" 
        self.schedule_active = False
        self.llama_server_dir = os.path.abspath(DEFAULT_LLAMA_SERVER_DIR) 
        
        # --- NUEVO: Estado de los LEDs de actividad ---
        # Pares de colores de curses (1: Blanco/Azul, 2: Verde, 3: Rojo, 4: Cyan, 5: Verde/Azul, 6: Amarillo)
        self.led_colors = [1, 2, 3, 4, 5, 6] 
        self.active_led_idx = 0 
        
        # --- NUEVO: Cargar la configuración persistente ---
        self.settings_path = self._get_settings_path()
        self.load_settings()
        
        # UI State
        self.active_field = 0 
  
        self.selected_file_idx = 0
        self.file_offset = 0
        self.files = []
        # El mensaje de bienvenida se establece en load_settings o aquí si no se carga.
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

    # --- MÉTODOS DE PERSISTENCIA ---
    def _get_settings_path(self):
        """Calcula y asegura la existencia del directorio de configuración en ~."""
        try:
            home_dir = Path.home()
            settings_dir = home_dir / SETTINGS_FOLDER_NAME
      
            # Crea el directorio si no existe
            settings_dir.mkdir(parents=True, exist_ok=True)
            return settings_dir / SETTINGS_FILE_NAME
        except Exception:
            # Fallback si no puede acceder al directorio home (aunque raro)
            return Path(SETTINGS_FILE_NAME)

    def load_settings(self):
        """Carga la configuración persistente desde el archivo."""
        self.msg_log = "" # Inicializa el log para la carga
        if self.settings_path.exists():
            try:
                with open(self.settings_path, 'rb') as f:
                    settings = pickle.load(f)
               
                # Cargar las variables que deben persistir, usando valores por defecto
                self.model_dir = settings.get('model_dir', self.model_dir)
                self.backend = settings.get('backend', self.backend)
      
                self.host_ip = settings.get('host_ip', self.host_ip)
                self.port = settings.get('port', self.port)
                self.allow_lan = settings.get('allow_lan', self.allow_lan)
                self.schedule_time = settings.get('schedule_time', self.schedule_time)
                self.schedule_active = settings.get('schedule_active', self.schedule_active)
           
                self.llama_server_dir = settings.get('llama_server_dir', self.llama_server_dir)
                
                self.msg_log = f"Settings loaded from {self.settings_path.name}."
            except Exception as e:
                self.msg_log = f"Error loading settings ({e}). Using defaults."
        else:
            self.msg_log = "No persistent settings found. Using defaults."
        
        # Guarda los valores por defecto en la primera ejecución
        if not self.settings_path.exists():
             self.save_settings() 

    def save_settings(self):
        """Guarda la configuración actual en el archivo."""
        settings = {
            'model_dir': self.model_dir,
            'backend': self.backend,
            'host_ip': self.host_ip,
            'port': self.port,
            'allow_lan': self.allow_lan,
            'schedule_time': self.schedule_time,
            'schedule_active': self.schedule_active,
            'llama_server_dir': self.llama_server_dir,
        }
        try:
            with open(self.settings_path, 'wb') as f:
                pickle.dump(settings, f)
        
        # --- CORRECCIÓN DEL ERROR DE SINTAXIS ---
        except Exception: 
            # Manejamos el error de forma silenciosa si no podemos guardar (I/O, permisos, etc.)
            pass
        # ----------------------------------------
    # ---------------------------------------------------


    def refresh_file_list(self):
        path = Path(self.model_dir)
        self.files = []
        if path.exists() and path.is_dir():
            # Get all .gguf files
    
            self.files = sorted([f for f in path.glob("*.gguf") if f.is_file()], key=lambda f: f.name)
            self.selected_file_idx = 0
            self.file_offset = 0
        else:
            self.msg_log = f"Error: Directory not found: {self.model_dir}"
    
    # --- LECTOR DE LOGS EN SEGUNDO PLANO ---
    def _log_reader_thread(self):
      
        if self.server_process and self.server_process.stderr:
            for line in iter(self.server_process.stderr.readline, b''):
                try:
                    line_str = line.decode(sys.getdefaultencoding(), errors='replace').strip()
                    if line_str:
                   
                        self.log_buffer.append(line_str)
                        # Mantener el buffer limpio
                        if len(self.log_buffer) > self.log_max_lines:
                            self.log_buffer.pop(0)
            
                except Exception:
                    pass

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

    def kill_server(self):
        if self.server_process:
            if self.log_thread and self.log_thread.is_alive():
                pass 

            try:
                # Intento de killpg
                os.killpg(os.getpgid(self.server_process.pid), signal.SIGTERM)
 
                self.server_process.wait(timeout=5) 
                
                # Limpieza de estado
                self.server_process = None
                self.running_model_path = None 
            
                self.log_buffer = [] 
                return "Server stopped and cache cleared."
            except Exception as e:
                # Fallback
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
            
        # Construir la ruta completa al ejecutable
        llama_server_path = os.path.join(self.llama_server_dir, LLAMA_SERVER_BIN)

        # Verificar que el ejecutable existe
        if not os.path.exists(llama_server_path) or not os.path.isfile(llama_server_path):
 
            return f"Error: '{llama_server_path}' not found or is not a file."

        current_file_obj = self.files[self.selected_file_idx]
        model_path = str(current_file_obj.resolve())
        
        # Usar la ruta completa del binario en el comando
        cmd = [llama_server_path, "-m", model_path, "--port", self.port]
        
        if self.allow_lan:
            cmd.extend(["--host", self.host_ip])
        else:
            cmd.extend(["--host", "127.0.0.1"])

        if self.backend == "CUDA":
            cmd.extend(["-ngl", "99"]) 
        elif self.backend == "VULKAN":
            cmd.extend(["-ngl", "99"]) 

        try:
        
            # Redirigimos stdout y stderr a PIPES para poder leerlos
            self.server_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                start_new_session=True,
       
                bufsize=1 
            )
            
            # Iniciamos el thread de lectura de logs
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
            
    # --- FUNCIÓN: Dibujar los LEDs de actividad ---
    def draw_activity_leds(self, y, x):
        """Dibuja 5 indicadores (LEDs) de colores en movimiento para indicar actividad."""
        if self.server_process:
            # Dibujamos "RUNNING" en verde
            self.stdscr.addstr(y, x, "● RUNNING ", curses.color_pair(2) | curses.A_BOLD)
            
            # Dibujamos un bucle de 5 LEDs de colores en movimiento
            start_x = x + len("● RUNNING ")
            
            for i in range(5):
                # Usamos el índice de color para seleccionar un par de colores
                color_index = self.led_colors[(self.active_led_idx + i) % len(self.led_colors)]
                
                # Usamos un carácter de punto para el LED
                char = "●"
                
                # El LED principal (basado en el índice de animación) estará en negrita para resaltarlo
                # Se añade A_REVERSE para que el color del fondo cambie y parezca un movimiento
                if i == 0:
                    style = curses.color_pair(color_index) | curses.A_BOLD | curses.A_REVERSE
                else:
                    style = curses.color_pair(color_index) | curses.A_BOLD
                    
                self.stdscr.addstr(y, start_x + (i*2), char, style)
                self.stdscr.addstr(y, start_x + (i*2) + 1, " ") # Espacio entre LEDs
        
        else:
            # Muestra "STOPPED" en rojo
            self.stdscr.addstr(y, x, "● STOPPED", curses.color_pair(3))
            self.stdscr.clrtoeol() # Limpia la línea si se detuvo el servidor
    # ---------------------------------------------------


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
        
        # Mostrar solo las últimas líneas que quepan en la pantalla
        if len(log_lines) > max_display_lines:
            log_lines = log_lines[-max_display_lines:]
            
       
        for i, line in enumerate(log_lines):
            try:
                # Cortar la línea si es demasiado larga para evitar errores de curses
                display_line = line[:w-3]
                self.stdscr.addstr(1 + i, 2, display_line, curses.color_pair(6))
            except curses.error:
   
                # Ignorar si se sale del área de dibujo
                pass
        
        self.stdscr.refresh()

    def run(self):
        while self.running:
            
            if self.show_log:
        
                # Si estamos en modo LOG, dibujamos el log y esperamos la tecla
                h, w = self.stdscr.getmaxyx()
                self.draw_log_view(h, w)
                
                try:
            
                    key = self.stdscr.getch()
                except:
                    key = -1
                
                if key != -1:
                 
                    # Cualquier tecla vuelve a la interfaz principal
                    self.show_log = False
                
                time.sleep(0.05)
                continue
            
        
            # --- Lógica de Dibujo Normal (TUI) ---
            self.stdscr.clear()
            h, w = self.stdscr.getmaxyx()
            
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

            self.draw_box(2, sett_x, list_h, sett_w, "Configuration ,Tab,Enter to change:")
            
            def draw_setting(idx, y_off, label, value):
                style = curses.color_pair(1) if self.active_field == idx else curses.A_NORMAL
             
                self.stdscr.addstr(3 + y_off, sett_x + 2, label, style)
                self.stdscr.addstr(3 + y_off, sett_x + 2 + len(label), str(value))

            disp_dir = (self.model_dir[:sett_w-10] + '..') if len(self.model_dir) > sett_w-10 else self.model_dir
            disp_server_dir = (self.llama_server_dir[:sett_w-10] + '..') if len(self.llama_server_dir) > sett_w-10 else self.llama_server_dir
            
          
            draw_setting(1, 0, "Model Dir:", disp_dir)
            draw_setting(2, 2, "Backend:CPU/CUDA/VULKAN:", f"[{self.backend}]")
            draw_setting(3, 4, "IP:", self.host_ip)
            draw_setting(4, 5, "Port:", self.port)
            
            # Nuevo campo de configuración del directorio del binario (idx=7)
            draw_setting(7, 7, "Llama-Server Binary Dir:", disp_server_dir) 

            lan_txt = "ENABLED" if self.allow_lan else "DISABLED"
            lan_col = curses.color_pair(2) if self.allow_lan else curses.color_pair(3)
            style_lbl = curses.color_pair(1) if self.active_field == 5 else curses.A_NORMAL
            self.stdscr.addstr(3 + 6, sett_x + 2, "LAN:", style_lbl) 
            self.stdscr.addstr(3 + 6, sett_x + 2 + len("LAN:") + 1, lan_txt, lan_col)
            
            sched_txt = f"ON ({self.schedule_time})" if self.schedule_active else "OFF"
            draw_setting(6, 8, "Schedule Starting:", sched_txt) 

            self.stdscr.addstr(13, sett_x + 2, "Server Status:", curses.A_UNDERLINE)
            
            # --- LLAMADA A LA FUNCIÓN NUEVA: INDICADORES LEDS ---
            self.draw_activity_leds(14, sett_x + 2)
            
            if self.server_process:
                # Se muestra el PID debajo de los indicadores
                self.stdscr.addstr(15, sett_x + 2, f"PID: {self.server_process.pid}")
            # ----------------------------------------------------

            self.draw_box(h-6, 0, 6, w, "Controls")
            
            keys_hint = "TAB:Switch Section | ENTER:Edit/Select | ARROWS:Navigate"
            cmds_hint = "[S] START | [K] KILL | [V] VIEW LOG | [D] SERVER DIR | [Q] QUIT"
            
            self.stdscr.addstr(h-4, 2, keys_hint)
            self.stdscr.addstr(h-3, 2, cmds_hint, curses.A_BOLD)
            
            
            self.stdscr.addstr(h-2, 2, f"> {self.msg_log}", curses.color_pair(4))

            self.stdscr.refresh()
            self.check_schedule()
            
            # --- Lógica de la animación de los LEDs (solo si está corriendo) ---
            if self.server_process:
                # Cambia el LED activo cada iteración (cada 0.05 segundos)
                self.active_led_idx = (self.active_led_idx + 1) % len(self.led_colors)
            # -------------------------------------------------------------------------


            try:
                key = self.stdscr.getch()
            except:
                key = -1

            
            if key == ord('q') or key == ord('Q'):
                self.kill_server()
                self.save_settings() # GUARDAR ANTES DE SALIR
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
            
            # Lógica para la tecla D/d (cambiar directorio del servidor)
            elif key == ord('d') or key == ord('D'):
                new_server_dir = self.input_string(h-2, 2, "Full Path to llama-server binary directory:", self.llama_server_dir)
                if os.path.isdir(new_server_dir):
   
                    self.llama_server_dir = os.path.abspath(new_server_dir)
                    self.msg_log = f"Llama-server directory updated to: {self.llama_server_dir}"
                    self.save_settings() # GUARDAR CAMBIO
                else:
               
                    self.msg_log = "Invalid directory path."

            elif key == 9: 
                # Ajustar la navegación con TAB (hay 8 campos de configuración activos: 0 a 7)
                self.active_field = (self.active_field + 1) % 8 
            
           
            elif key == curses.KEY_UP:
                if self.active_field == 0 and self.selected_file_idx > 0:
                    self.selected_file_idx -= 1
            
            elif key == curses.KEY_DOWN:
                if self.active_field == 0 and self.selected_file_idx < len(self.files) - 1:
                    self.selected_file_idx += 1

            elif key == 10: 
                if self.active_field == 1: 
                    new_dir = self.input_string(h-2, 2, "Full Path to Models:", self.model_dir)
             
                    if os.path.isdir(new_dir):
                        self.model_dir = os.path.abspath(new_dir)
                        self.refresh_file_list()
                        self.msg_log = "Directory updated."
               
                        self.save_settings() # GUARDAR CAMBIO
                    else:
                        self.msg_log = "Invalid directory path."

                elif self.active_field == 2: 
                   
                    modes = ["CPU", "CUDA", "VULKAN"]
                    curr_idx = modes.index(self.backend)
                    self.backend = modes[(curr_idx + 1) % len(modes)]
                    self.save_settings() # GUARDAR CAMBIO
                
       
                elif self.active_field == 3: 
                    self.host_ip = self.input_string(h-2, 2, "Host IP:", self.host_ip)
                    self.save_settings() # GUARDAR CAMBIO
                
                elif self.active_field == 4: 
 
                    self.port = self.input_string(h-2, 2, "Port:", self.port)
                    self.save_settings() # GUARDAR CAMBIO

                elif self.active_field == 5: 
                    self.allow_lan = not self.allow_lan
         
                    self.save_settings() # GUARDAR CAMBIO
                
                elif self.active_field == 6: 
                    if self.schedule_active:
                        self.schedule_active = False
  
                        self.msg_log = "Schedule disabled."
                        self.save_settings() # GUARDAR CAMBIO
                    else:
                        t = self.input_string(h-2, 2, "Start Time (HH:MM):", "08:00")
                        try:
                            datetime.datetime.strptime(t, "%H:%M")
                            self.schedule_time = t
             
                            self.schedule_active = True
                            self.msg_log = f"Scheduled for {t}"
                            self.save_settings() # GUARDAR CAMBIO
                   
                        except ValueError:
                            self.msg_log = "Invalid time format. Use HH:MM"

                # Lógica para ENTER en el campo Server Bin Dir (active_field = 7)
                elif self.active_field == 7: 
                    new_server_dir = self.input_string(h-2, 2, "Full Path to llama-server binary directory:", self.llama_server_dir)
                    if os.path.isdir(new_server_dir):
                        self.llama_server_dir = os.path.abspath(new_server_dir)
                        self.msg_log = f"Llama-server directory updated to: {self.llama_server_dir}"
                        self.save_settings() # GUARDAR CAMBIO
                
                    else:
                        self.msg_log = "Invalid directory path."

            time.sleep(0.05)

def main():
    # Intenta crear el directorio por defecto si no existe
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
        # Si falla curses, imprime el error
        print(f"Fatal Error: {e}")

if __name__ == "__main__":
    main()