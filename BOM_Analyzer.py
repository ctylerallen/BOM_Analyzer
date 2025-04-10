import os
import sys
import json
import webbrowser
import requests
import pandas as pd
import numpy as np
import time
import logging
import tkinter as tk
from tkinter import filedialog, ttk, messagebox, scrolledtext
from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
from pathlib import Path
from dotenv import load_dotenv
import openai
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
import csv
import ssl
import re # For regex parsing
import threading # For checking main thread

# --- Dependency Check ---
try:
    from prophet import Prophet
except ImportError:
    try: # Try using tk directly if root doesn't exist
        import tkinter as tk
        from tkinter import messagebox
        root = tk.Tk()
        root.withdraw() # Hide the empty window
        messagebox.showerror("Dependency Error",
                             "Prophet library not found. Please install it: \n"
                             "pip install prophet")
        root.destroy()
    except Exception as e:
        print("CRITICAL ERROR: Prophet library not found AND Tkinter failed. Please install Prophet: pip install prophet")
    sys.exit(1)

# --- Logging Setup ---
# Use DEBUG initially for detailed setup info
logging.basicConfig(level=logging.DEBUG, # Changed to INFO for less verbosity, can set back to DEBUG if needed
                    format='%(asctime)s - %(levelname)s - %(threadName)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)

# Quieten noisy libraries (can be adjusted based on debugging needs)
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING) # Add matplotlib if used

logging.basicConfig(level=logging.DEBUG)

# Log messages at different levels
logging.debug("This is a debug message")
logging.info("This is an info message")
logging.warning("This is a warning message")
logging.error("This is an error message")
logging.critical("This is a critical message")

# --- Configuration & Constants ---
SCRIPT_DIR = Path(__file__).parent # Get the directory where the script resides
CACHE_DIR = SCRIPT_DIR / 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)
CERT_FILE = SCRIPT_DIR / "localhost.pem" # Expected cert file location

# File Paths
TOKEN_FILE = CACHE_DIR / 'digikey_oauth2_token.json'
MOUSER_COUNTER_FILE = CACHE_DIR / 'mouser_request_counter.json'
NEXAR_TOKEN_FILE = CACHE_DIR / 'nexar_oauth2_token.json' # Renamed for clarity
HISTORICAL_DATA_FILE = SCRIPT_DIR / 'bom_historical_data.csv'
PREDICTION_FILE = SCRIPT_DIR / 'supply_chain_predictions.csv'

# API Endpoints
NEXAR_TOKEN_URL = "https://identity.nexar.com/connect/token"
NEXAR_API_URL = "https://api.nexar.com/graphql"

# Other Constants
DEFAULT_TARIFF_RATE = 0.035
API_TIMEOUT_SECONDS = 20 # Increased slightly
MAX_API_WORKERS = 8
APP_NAME = "NPI BOM Analyzer"
APP_VERSION = "1.1.0" # Updated Version

# --- Load Environment Variables ---
env_path = SCRIPT_DIR / 'keys.env'
logger.info(f"Attempting to load environment variables from: {env_path}")
if not load_dotenv(env_path):
    logger.warning(f"Could not find {env_path.name} file. API features requiring keys may be disabled.")
else:
    logger.info(f"Successfully loaded environment variables from: {env_path}")

# --- API Key Validation and Loading ---
# Standardize key names in .env if possible (e.g., OPENAI_API_KEY)
DIGIKEY_CLIENT_ID = os.getenv('DIGIKEY_CLIENT_ID')
DIGIKEY_CLIENT_SECRET = os.getenv('DIGIKEY_CLIENT_SECRET')
MOUSER_API_KEY = os.getenv('MOUSER_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or os.getenv('CHATGPT_API_KEY') # Allow both names
NEXAR_CLIENT_ID = os.getenv('NEXAR_CLIENT_ID')
NEXAR_CLIENT_SECRET = os.getenv('NEXAR_CLIENT_SECRET')
ARROW_API_KEY = os.getenv('ARROW_API_KEY')
AVNET_API_KEY = os.getenv('AVNET_API_KEY')

API_KEYS = {
    "DigiKey": bool(DIGIKEY_CLIENT_ID and DIGIKEY_CLIENT_SECRET),
    "Mouser": bool(MOUSER_API_KEY),
    "OpenAI": bool(OPENAI_API_KEY),
    "Octopart (Nexar)": bool(NEXAR_CLIENT_ID and NEXAR_CLIENT_SECRET),
    "Arrow": bool(ARROW_API_KEY),
    "Avnet": bool(AVNET_API_KEY),
}

# Log API Key Status
for api_name, is_set in API_KEYS.items():
    status = "Set" if is_set else "Not set"
    if api_name in ["Arrow", "Avnet"] and not is_set:
        status += " "
    elif api_name == "OpenAI" and not is_set:
        status += " (Optional - Summary Disabled)"
    logger.info(f"{api_name} Keys/API Key: {status}")

# Configure OpenAI Client (Updated SDK Usage)
if API_KEYS["OpenAI"]:
    try:
        openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        # Test connection (optional, but good practice)
        # openai_client.models.list()
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
        API_KEYS["OpenAI"] = False # Disable if init fails
        openai_client = None
else:
    logger.warning("OPENAI_API_KEY not set - AI analysis features will be disabled.")
    openai_client = None

# --- Utility Functions ---

def is_main_thread():
    """Checks if the current thread is the main thread."""
    return threading.current_thread() is threading.main_thread()

@lru_cache(maxsize=128)
def call_chatgpt(prompt, model="gpt-4o", max_tokens=2000): # Updated model, increased tokens
    """Calls the OpenAI API with caching, retry logic, and updated SDK usage."""
    if not openai_client: # Check the client instance
        logger.warning("OpenAI client not available. Skipping ChatGPT call.")
        return "OpenAI client not configured."

    max_retries = 3
    base_wait_time = 5 # Increased base wait time

    for attempt in range(max_retries + 1):
        try:
            logger.debug(f"Attempt {attempt + 1} calling OpenAI model {model}...")
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a strategic supply chain advisor specializing in electronic components, providing concise, actionable insights for executive review. Focus on risk, cost optimization, and build readiness."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.6 # Slightly increased for potentially more nuanced analysis
            )
            logger.debug(f"OpenAI call successful on attempt {attempt + 1}.")
            # Check for content before accessing
            if response.choices and response.choices[0].message:
                 return response.choices[0].message.content.strip()
            else:
                 logger.error("OpenAI response missing expected content.")
                 return "OpenAI response format error."

        except openai.RateLimitError as e:
            logger.warning(f"OpenAI Rate Limit Error (Attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                wait_time = base_wait_time * (2 ** attempt)
                logger.info(f"Waiting {wait_time} seconds before retrying OpenAI call...")
                time.sleep(wait_time)
            else:
                logger.error("OpenAI Rate Limit Error: Max retries exceeded.")
                return f"OpenAI Rate Limit Error: Max retries exceeded. ({e})"

        except openai.AuthenticationError as e:
            logger.error(f"OpenAI Authentication Error: {e}. Check your API key.")
            # Disable OpenAI key in the app state?
            API_KEYS["OpenAI"] = False
            # Schedule GUI update? Needs access to app instance. Pass app or use callback.
            # For now, just log and return error.
            return "OpenAI Authentication Error. Check Key."

        except Exception as e:
            logger.error(f"ChatGPT API error (Attempt {attempt + 1}): {e}", exc_info=True) # Log traceback
            # Attempt to extract more details from the exception if available
            error_message = f"ChatGPT API error: {str(e)}"
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                 error_message += f" | Response: {e.response.text[:200]}" # Add first 200 chars of response
            return error_message # Return detailed error

    return "ChatGPT call failed after multiple attempts." # Fallback

def init_csv_file(filepath, header):
    """Initializes a CSV file with a header if it doesn't exist or is empty."""
    try:
        # Check if file exists and has content beyond header
        file_exists = filepath.exists()
        needs_header = not file_exists
        if file_exists:
             with open(filepath, 'r', encoding='utf-8') as f:
                  # Check if file has more than just maybe a header line
                  try:
                      first_line = f.readline().strip()
                      second_line = f.readline().strip()
                      if not first_line or not second_line: # Empty or only header
                           needs_header = True
                           logger.info(f"CSV file {filepath.name} exists but is empty or only has header. Will rewrite header.")
                      # Optional: Check if header matches exactly
                      elif first_line != ','.join(header):
                            logger.warning(f"Header mismatch in {filepath.name}. Expected: {header}. Found: {first_line.split(',')}. Rewriting header.")
                            needs_header = True # Overwrite if header doesn't match
                  except Exception:
                      needs_header = True # Treat read errors as needing header

        if needs_header:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            logger.info(f"Initialized/Reset CSV file: {filepath.name}")

    except IOError as e:
        logger.error(f"Failed to initialize/check CSV file {filepath}: {e}")
        # Attempt GUI popup (handle potential early Tkinter issues)
        try:
            import tkinter as tk; from tkinter import messagebox
            temp_root = tk.Tk(); temp_root.withdraw()
            messagebox.showerror("File Error", f"Could not create/access required data file:\n{filepath.name}\n\nError: {e}\n\nApplication might not function correctly.")
            temp_root.destroy()
        except Exception:
            print(f"CRITICAL FILE ERROR: Could not create/access required data file: {filepath}. Error: {e}")


def append_to_csv(filepath, data_rows):
    """Appends rows of data to a CSV file, converting items to strings."""
    if not data_rows: return
    try:
        cleaned_rows = []
        for row in data_rows:
            if isinstance(row, (list, tuple)):
                # Convert all elements to string, handle None/NaN explicitly
                cleaned_rows.append([
                    '' if item is None or pd.isna(item) else str(item)
                    for item in row
                ])
            else:
                logger.warning(f"Skipping invalid row type during CSV append: {type(row)}")

        if not cleaned_rows:
             logger.warning(f"No valid rows to append to {filepath.name} after cleaning.")
             return

        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL) # Use minimal quoting
            writer.writerows(cleaned_rows)
    except IOError as e:
        logger.error(f"Failed to append to CSV file {filepath.name}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error writing to CSV {filepath.name}: {e}", exc_info=True)


def safe_float(value, default=np.nan):
    """Safely convert value to float, handling common invalid inputs and types."""
    if value is None or isinstance(value, bool): return default
    if isinstance(value, (int, float)):
        return float(value) if not np.isinf(value) else default # Handle infinity
    try:
        s_val = str(value).strip().replace('$', '').replace(',', '').replace('%', '').lower()
        if not s_val or s_val in ['n/a', 'none', 'inf', '-inf', 'na', 'nan', '']:
            return default
        return float(s_val)
    except (ValueError, TypeError):
        return default

def convert_lead_time_to_days(lead_time_str):
    """Converts various lead time strings (weeks, days, numbers) to integer days."""
    if lead_time_str is None or pd.isna(lead_time_str): return np.nan

    # Handle direct numeric input (assume weeks if > reasonable number of days?)
    # Let's treat direct numbers as days for simplicity, as APIs often return days.
    if isinstance(lead_time_str, (int, float)):
        if np.isinf(lead_time_str) or pd.isna(lead_time_str): return np.nan
        return int(round(lead_time_str)) # Treat number as days directly, round first

    s = str(lead_time_str).lower().strip()
    if s in ['n/a', 'unknown', '', 'na', 'none', 'stock']: # Treat 'stock' as 0 days
        return 0 if s == 'stock' else np.nan

    try:
        # Improved parsing: find number, check for units
        match = re.search(r'(\d+(\.\d+)?)', s) # Find integer or float
        if not match: return np.nan
        num = float(match.group(1))

        if 'week' in s:
            return int(round(num * 7))
        elif 'day' in s:
            return int(round(num))
        else:
            # Assume days if no unit is present and number seems like days (e.g., < 100?)
            # Assume weeks if number is smaller (e.g. <= 20?) - this is ambiguous
            # Let's default to assuming DAYS if no unit. Many APIs return days.
            logger.debug(f"No unit in lead time '{lead_time_str}', assuming days.")
            return int(round(num))

    except Exception as e:
        logger.warning(f"Failed to convert lead time '{lead_time_str}': {e}")
        return np.nan

# --- OAuth Handler (for DigiKey - Uses HTTPS/SSL) ---
class OAuthHandler(BaseHTTPRequestHandler):
    """Handles the OAuth callback from DigiKey via HTTPS."""
    def do_GET(self):
        auth_code = None
        server_instance = getattr(self.server, 'app_instance', None) # Get app instance if passed
        status_message = "OAuth Error: Unknown"
        status_level = "error"

        try:
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            code = params.get('code', [None])[0]
            # state = params.get('state', [None])[0] # Optional: Validate state if used

            if code:
                auth_code = code # Set code for the server instance
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<html><head><title>Authentication Successful</title></head>"
                                 b"<body style='font-family: sans-serif; text-align: center; padding-top: 50px;'>"
                                 b"<h1>Authentication Successful!</h1>"
                                 b"<p>Authorization code received.</p>"
                                 b"<p>You can close this window and return to the BOM Analyzer.</p>"
                                 b"<script>window.close();</script>" # Try to auto-close
                                 b"</body></html>")
                logger.info("OAuth code received successfully via HTTPS.")
                status_message = "OAuth code received, exchanging for token..."
                status_level = "info"
            else:
                error = params.get('error', ['Unknown error'])[0]
                error_desc = params.get('error_description', ['No description provided'])[0]
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f"<html><head><title>Authentication Failed</title></head>"
                                 f"<body style='font-family: sans-serif; text-align: center; padding-top: 50px;'>"
                                 f"<h1>Authentication Failed</h1>"
                                 f"<p><b>Error:</b> {error}</p>"
                                 f"<p><b>Description:</b> {error_desc}</p>"
                                 f"<p>Please close this window and check the application logs.</p>"
                                 f"</body></html>".encode('utf-8'))
                logger.error(f"OAuth failed via HTTPS. Error: {error}, Description: {error_desc}")
                status_message = f"OAuth Error: {error_desc}"
                status_level = "error"

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(b"<html><body>Internal Server Error during OAuth callback. Check application logs.</body></html>")
            logger.error(f"Error in OAuthHandler: {e}", exc_info=True)
            status_message = f"OAuth Server Error: {e}"
            status_level = "error"
        finally:
            # Signal the auth code (or None on failure) back to the waiting thread
            self.server.auth_code = auth_code
            # Update status via app instance if available
            if server_instance and hasattr(server_instance, 'update_status_threadsafe'):
                 server_instance.update_status_threadsafe(status_message, status_level)


    def log_message(self, format, *args):
        # Quieten server logging unless debugging OAuth specifically
        # logger.debug(f"OAuthServer: {format % args}")
        pass

# --- Main Application Class ---
class BOMAnalyzerApp:
    # --- Risk Configuration Constants (Can be externalized to JSON/config file later) ---
    RISK_WEIGHTS = {'Sourcing': 0.30, 'Stock': 0.15, 'LeadTime': 0.15, 'Lifecycle': 0.30, 'Geographic': 0.10}
    GEO_RISK_TIERS = {
        # Higher Risk
        "China": 7, "Russia": 9,
        # Moderate Risk
        "Taiwan": 5, "Malaysia": 4, "Vietnam": 4, "India": 5, "Philippines": 4,
        "Thailand": 4, "South Korea": 3,
        # Lower Risk
        "USA": 1, "United States": 1, "Mexico": 2, "Canada": 1, "Japan": 1,
        "Germany": 1, "France": 1, "UK": 1, "Ireland": 1, "Switzerland": 1, "EU": 1,
        # Default / Slightly Elevated Penalty for Unknown
        "Unknown": 4, "N/A": 4, "_DEFAULT_": 4
    }
    RISK_CATEGORIES = {'high': (6.6, 10.0), 'moderate': (3.6, 6.5), 'low': (0.0, 3.5)}
    # --- End Risk Configuration ---

    def __init__(self, root):
        self.root = root
        self.root.title(f"{APP_NAME} - v{APP_VERSION}")
        self.root.geometry("1500x900") # Adjusted size
        self.root.minsize(1100, 700)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0) 
        self.root.grid_columnconfigure(0, weight=1) # Allow content to expand horizontally
        # Set a modern theme
        self.style = ttk.Style()
        available_themes = self.style.theme_names()
        logger.debug(f"Available themes: {available_themes}")
        # Prefer modern themes if available
        for theme in ['clam', 'alt', 'default']: # Prioritize clam
            if theme in available_themes:
                 self.style.theme_use(theme)
                 logger.info(f"Using theme: {theme}")
                 break
        else:
             logger.warning("Could not find preferred themes (clam, alt, default). Using system default.")

        # Configure base styles for a cleaner look
        self.root.configure(bg='#e1e1e1') # Light gray background
        self.style.configure("TFrame", background='#e1e1e1')
        self.style.configure("TLabel", background='#e1e1e1', font=("Segoe UI", 9))
        self.style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=(8, 4))
        self.style.configure("Treeview", font=("Segoe UI", 9), rowheight=25, fieldbackground="#ffffff")
        self.style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"), background="#c1c1c1", relief="groove")
        self.style.map("Treeview.Heading", relief=[('active','groove'),('pressed','sunken')])
        self.style.configure("TNotebook", background='#e1e1e1', borderwidth=0)
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[12, 6], background="#d0d0d0", foreground="#333")
        self.style.map("TNotebook.Tab", background=[("selected", "#a1a1a1")], foreground=[("selected", "#ffffff")])
        self.style.configure("TLabelframe", background='#e1e1e1', relief="solid", borderwidth=1, padding=10)
        self.style.configure("TLabelframe.Label", background='#e1e1e1', font=("Segoe UI", 10, "bold"), padding=(0,0,0,5))
        self.style.configure("TScrollbar", background='#e1e1e1', troughcolor='#f0f0f0')
        self.style.configure("TProgressbar", troughcolor='#f0f0f0', background='#0078d4') # Use a distinct progress bar color

        logger.info("Initializing GUI...")

        # --- Tooltip Setup ---
        self._tooltips = {} # Store {widget_id: Tooltip instance
        self.tooltip_texts = {} # Store {widget: text} for universal tooltip 

        # --- Define Headers ---
        # Historical Data Header
        self.hist_header = ['Component', 'Manufacturer', 'Part_Number', 'Distributor',
               'Lead_Time_Days', 'Cost', 'Inventory', 'Stock_Probability', 'Fetch_Timestamp']
        # Prediction Data Header (Ensure all used columns are here in desired order)
        self.pred_header = ['Component', 'Date',
                       'Prophet_Lead', 'Prophet_Cost',
                       'RAG_Lead', 'RAG_Cost',
                       'AI_Lead', 'AI_Cost',
                       'Stock_Probability',
                       'Real_Lead', 'Real_Cost', 'Real_Stock',
                       'Prophet_Ld_Acc', 'Prophet_Cost_Acc',
                       'RAG_Ld_Acc', 'RAG_Cost_Acc',
                       'AI_Ld_Acc', 'AI_Cost_Acc']

        # --- Main Layout ---
        self.main_paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned_window.grid(row=0, column=0, sticky="nsew", padx=10, pady=(0, 5))
        logger.debug("Main paned window gridded onto root window.")

        # --- Universal Status Bar (Creation & Packing) ---
        # Create the Frame (child of root)
        STATUS_BAR_HEIGHT = 35,
        self.universal_status_bar = ttk.Frame(self.root, relief='sunken', style="StatusBar.TFrame", height=STATUS_BAR_HEIGHT)
                                                #borderwidth=2, background='red') # TEMPORARY DEBUG

        # Create the Label (child of the status bar frame)
        self.universal_tooltip_label = ttk.Label(
            self.universal_status_bar,
            text=" ",
            anchor='w',
            wraplength=self.root.winfo_screenwidth() - 50, # Adjust as needed
            font=("Segoe UI", 8),
            style="Status.TLabel"
        )
        # Pack the LABEL *inside* the STATUS BAR FRAME
        self.universal_tooltip_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=(2,2))

        # Pack the STATUS BAR FRAME itself onto the root window (BOTTOM, LAST)
        # This line MUST come AFTER the main content is packed (Step 1D)
        self.universal_status_bar.grid(row=1, column=0, sticky="ew", padx=0, pady=(1, 0))
        self.root.grid_rowconfigure(1, weight=0, uniform='statusbar') # Added uniform group
        self.universal_status_bar.grid_propagate(False)

        # --- Optional: Define Status Bar Styles ---
        # Place these lines earlier in __init__ with your other style configurations
        self.style.configure("StatusBar.TFrame", background="#e0e0e0", borderwidth=1)
        self.style.configure("Status.TLabel", background="#e0e0e0", foreground="#333333")
        
        # --- Left Pane: Configuration ---
        self.config_frame_outer = ttk.Frame(self.main_paned_window, padding=0, width=450) 
        self.main_paned_window.add(self.config_frame_outer, weight=2)

        # Make the config frame scrollable
        self.config_scroll_canvas = tk.Canvas(self.config_frame_outer, borderwidth=0, background="#e1e1e1", highlightthickness=0)
        self.config_scrollbar = ttk.Scrollbar(self.config_frame_outer, orient="vertical", command=self.config_scroll_canvas.yview)
        self.config_frame = ttk.Frame(self.config_scroll_canvas, padding=(15, 15)) # Inner frame for content

        self.config_frame.bind("<Configure>", lambda e: self.config_scroll_canvas.configure(scrollregion=self.config_scroll_canvas.bbox("all")))
        self.config_scroll_canvas.create_window((0, 0), window=self.config_frame, anchor="nw")
        self.config_scroll_canvas.configure(yscrollcommand=self.config_scrollbar.set)

        self.config_scroll_canvas.pack(side="left", fill="both", expand=True)
        self.config_scrollbar.pack(side="right", fill="y")

        # Bind mouse wheel scrolling for config frame (platform specific)
        def _on_mousewheel_config(event):
            # Determine scroll direction (platform dependent)
            if event.num == 4 or event.delta > 0: # Linux up / Windows up
                self.config_scroll_canvas.yview_scroll(-1, "units")
            elif event.num == 5 or event.delta < 0: # Linux down / Windows down
                self.config_scroll_canvas.yview_scroll(1, "units")
        # Bind for Linux and Windows/Mac
        self.config_frame.bind_all("<MouseWheel>", _on_mousewheel_config) # Windows/Mac
        self.config_frame.bind_all("<Button-4>", _on_mousewheel_config) # Linux Scroll Up
        self.config_frame.bind_all("<Button-5>", _on_mousewheel_config) # Linux Scroll Down

        # --- Configuration Widgets ---
        ttk.Label(self.config_frame, text="Configuration", font=("Segoe UI", 14, "bold")).pack(fill="x", pady=(0, 15), anchor='w')

        # Load BOM Section
        load_bom_frame = ttk.Frame(self.config_frame)
        load_bom_frame.pack(fill="x", pady=(0, 10))
        self.load_button = ttk.Button(load_bom_frame, text="Load BOM...", command=self.load_bom)
        self.load_button.pack(side=tk.LEFT, padx=(0, 5))
        self.create_tooltip(self.load_button, "Load a Bill of Materials (BOM) in CSV format.\nRequires columns like 'Part Number' and 'Quantity'.")
        self.file_label = ttk.Label(load_bom_frame, text="No BOM loaded.", style="Hint.TLabel", wraplength=250) # Use custom style
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5,0))
        self.style.configure("Hint.TLabel", foreground="#555555") # Define hint style

        # Analysis Controls Section
        run_frame = ttk.LabelFrame(self.config_frame, text="Analysis Controls", padding=10)
        run_frame.pack(fill="x", pady=(10, 10))
        self.run_button = ttk.Button(run_frame, text="Run Analysis", command=self.validate_and_run_analysis, state="disabled", style="Accent.TButton") # Use accent style
        self.create_tooltip(self.run_button, "Run the full analysis using current BOM and configuration.\nFetches data from suppliers, calculates risk, and determines strategies.")
        self.run_button.pack(side=tk.LEFT, padx=(0,5), ipady=2) # Add internal padding
        self.style.configure("Accent.TButton", background="#0078d4", foreground="white")
        self.style.map("Accent.TButton", background=[('active', '#005a9e')])

        self.predict_button = ttk.Button(run_frame, text="Run Predictions", command=self.run_predictive_analysis_gui, state="disabled")
        self.predict_button.pack(side=tk.LEFT, padx=5)
        self.create_tooltip(self.predict_button, "Generate future cost/lead time predictions based on historical data.\nRequires historical data from previous analysis runs.")

        self.ai_summary_button = ttk.Button(run_frame, text="AI Summary", command=self.generate_ai_summary_gui, state="disabled")
        self.ai_summary_button.pack(side=tk.LEFT)
        self.create_tooltip(self.ai_summary_button, "Generate an executive summary and recommendations using OpenAI.\nRequires analysis results and an OpenAI API key.")

        # Optimized Strategy Config Section
        optimized_strategy_frame = ttk.LabelFrame(self.config_frame, text="Optimized Strategy Configuration", padding=10)
        optimized_strategy_frame.pack(fill="x", pady=(10, 5))

        config_entries = [
            ("Total Units to Build:", "total_units", "100", "Number of finished units to build (calculates total quantity needed per part)."),
            ("Max Cost Premium (%):", "max_premium", "15", "Maximum percentage increase over the absolute lowest total cost allowed for a part in the Optimized Strategy."),
            ("Target Lead Time (days):", "target_lead_time_days", "56", "Maximum acceptable lead time (days) for any part chosen in the Optimized Strategy."),
            ("Cost Weight (0-1):", "cost_weight", "0.5", "Priority for minimizing cost (0=ignore, 1=only cost). Must sum to 1 with Lead Time Weight."),
            ("Lead Time Weight (0-1):", "lead_time_weight", "0.5", "Priority for minimizing lead time (0=ignore, 1=only LT). Must sum to 1 with Cost Weight."),
            ("Buy-Up Threshold (%):", "buy_up_threshold", "1", "Allow buying more parts (e.g., next price break) if total cost increases by no more than this percentage compared to buying the exact needed amount (or MOQ). Set to 0 to disable."),
        ]

        self.config_vars = {}
        # Use grid inside the LabelFrame for alignment
        optimized_strategy_frame.columnconfigure(1, weight=1) # Make entry column expand slightly

        for i, (label, attr, default, hint) in enumerate(config_entries):
            lbl = ttk.Label(optimized_strategy_frame, text=label)
            lbl.grid(row=i, column=0, sticky="w", padx=(0, 5), pady=2)
            entry = ttk.Entry(optimized_strategy_frame, width=8) # Reduced width
            entry.grid(row=i, column=1, sticky="w", pady=2)
            entry.insert(0, default)
            self.config_vars[attr] = entry
            entry.bind("<KeyRelease>", self.validate_inputs) # Validate on key release

            # Add tooltip to both label and entry
            self.create_tooltip(lbl, hint)
            self.create_tooltip(entry, hint)

        # Tariff Config Section
        self.tariff_frame = ttk.LabelFrame(self.config_frame, text="Custom Tariff Rates (%)", padding=10)
        self.tariff_frame.pack(fill="x", pady=(10, 5))
        self.tariff_entries = {}
        # Sorted list of common COOs
        top_countries = sorted(["China", "Mexico", "India", "Vietnam", "Taiwan", "Japan", "Malaysia", "Germany", "USA", "Philippines", "Thailand", "South Korea"])
        self.tariff_frame.columnconfigure((1, 3), weight=1) # Configure columns for entries
        for i, country in enumerate(top_countries):
            row, col = divmod(i, 2) # 2 columns layout
            frame = ttk.Frame(self.tariff_frame) # Use a subframe for each label/entry pair
            frame.grid(row=row, column=col*2, columnspan=2, sticky="ew", padx=5, pady=2)
            frame.columnconfigure(1, weight=1) # Allow entry to expand slightly

            lbl = ttk.Label(frame, text=f"{country}:", width=12) # Fixed width label
            lbl.pack(side=tk.LEFT)
            entry = ttk.Entry(frame, width=6)
            entry.insert(0, "")  # Default blank = use default/predicted
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            self.tariff_entries[country] = entry
            hint_tariff = f"Custom tariff rate (%) for parts from '{country}'.\nLeave blank to use USITC lookup or default/predicted rate."
            self.create_tooltip(lbl, hint_tariff)
            self.create_tooltip(entry, hint_tariff)
            entry.bind("<KeyRelease>", self.validate_inputs)

        ttk.Label(self.tariff_frame, text="(Blank uses default/predicted)", style="Hint.TLabel").grid(row=(len(top_countries)+1)//2, column=0, columnspan=4, pady=(8,0), sticky='w')

        # Validation Label (placed consistently)
        self.validation_label = ttk.Label(self.config_frame, text="", foreground="red", wraplength=350, font=("Segoe UI", 8))
        self.validation_label.pack(fill="x", pady=(5, 10), anchor='w')

        # API Status Section
        api_status_frame = ttk.LabelFrame(self.config_frame, text="API Status", padding=10)
        api_status_frame.pack(fill="x", pady=(10, 5), anchor='w')
        self.api_status_labels = {}
    
        api_status_frame.columnconfigure(1, weight=1) # Allow status text to expand
        for i, (api_name, is_set) in enumerate(API_KEYS.items()):
             # Determine Status Text based ONLY on whether the key is set
             if is_set:
                 status_text = "OK"
                 color = "#008000" # Dark Green
             elif api_name == "OpenAI":
                  status_text = "Not Set (Optional)"
                  color = "#ff8c00" # Orange
             else: # Key is not set for required APIs (DigiKey, Mouser, Nexar) or potential ones (Arrow, Avnet)
                  status_text = "Not Set"
                  color = "#e60000" # Dark Red

             lbl_name = ttk.Label(api_status_frame, text=f"{api_name}:", width=15)
             lbl_name.grid(row=i, column=0, sticky='w', padx=(0,5))
             lbl_status = ttk.Label(api_status_frame, text=status_text, foreground=color, anchor='w')
             lbl_status.grid(row=i, column=1, sticky='ew')
             self.api_status_labels[api_name] = lbl_status # Store the status label for updates

        # --- Right Pane: Results ---
        self.results_frame = ttk.Frame(self.main_paned_window, padding=(10, 0, 10, 0)) 
        self.main_paned_window.add(self.results_frame, weight=3) 
        self.results_frame.grid_rowconfigure(1, weight=1)    # Notebook takes most space
        self.results_frame.grid_columnconfigure(0, weight=1)

        # --- Status Bar --- (Improved Layout)
        status_progress_frame = ttk.Frame(self.results_frame, padding=(5, 5))
        status_progress_frame.grid(row=0, column=0, sticky="ew")
        status_progress_frame.grid_columnconfigure(0, weight=3) # Status label gets more space
        status_progress_frame.grid_columnconfigure(1, weight=1) # Progress bar
        status_progress_frame.grid_columnconfigure(2, weight=0) # Percentage label
        status_progress_frame.grid_columnconfigure(3, weight=2) # Rate limit label

        self.status_label = ttk.Label(status_progress_frame, text="Ready", anchor="w")
        self.status_label.grid(row=0, column=0, padx=(0, 5), sticky="ew")

        self.progress = ttk.Progressbar(status_progress_frame, orient="horizontal", length=150, mode="determinate")
        self.progress.grid(row=0, column=1, padx=5, sticky="ew")
        self.progress_label = ttk.Label(status_progress_frame, text="0%", width=5)
        self.progress_label.grid(row=0, column=2, padx=(0, 5), sticky="w")

        self.rate_label = ttk.Label(status_progress_frame, text="API Rates: -", anchor="e", style="Hint.TLabel")
        self.rate_label.grid(row=0, column=3, padx=(10, 0), sticky="ew")

        # --- Results Notebook ---
        self.results_notebook = ttk.Notebook(self.results_frame)
        self.results_notebook.grid(row=1, column=0, sticky="nsew", pady=(5,0))

        # --- Tab 1: BOM Analysis Summary ---
        self.analysis_tab = ttk.Frame(self.results_notebook, padding=(0, 10, 0, 0)) # Padding for content inside tab
        self.results_notebook.add(self.analysis_tab, text=" BOM Analysis ") # Add spaces for padding
        self.analysis_tab.grid_columnconfigure(0, weight=1)
        self.analysis_tab.grid_rowconfigure(0, weight=3) # Main Treeview
        self.analysis_tab.grid_rowconfigure(3, weight=1) # Summary Table

        # -- Parts Treeview --
        tree_frame_outer = ttk.Frame(self.analysis_tab) # Outer frame for tree and scrollbars
        tree_frame_outer.grid(row=0, column=0, sticky="nsew")
        tree_frame_outer.grid_rowconfigure(0, weight=1)
        tree_frame_outer.grid_columnconfigure(0, weight=1)

        columns = [
            "PartNumber", "Manufacturer", "MfgPN", "QtyNeed", "Status", "Sources", "StockAvail",
            "COO", "RiskScore", "TariffPct",
            "BestCostPer", "BestTotalCost", "ActualBuyQty", "BestCostLT", "BestCostSrc", # Added ActualBuyQty
            "FastestLT", "FastestCost", "FastestLTSrc",
            "Alternates", "Notes"
        ]
        headings = [
            "BOM P/N", "Manufacturer", "Mfg P/N", "Need", "Lifecycle", "Sources", "Stock",
            "COO", "Risk", "Tariff (%)",
            "Unit Cost ($)", "Total Cost ($)", "Buy Qty", "LT (d)", "Src", # Updated headers
            "Fastest LT (d)", "Cost ($)", "Src",
            "Alts?", "Notes/Flags"
        ]
        col_widths = {
            "PartNumber": 140, "Manufacturer": 110, "MfgPN": 140, "QtyNeed": 50, "Status": 70, "Sources": 50, "StockAvail": 70,
            "COO": 50, "RiskScore": 45, "TariffPct": 55,
            "BestCostPer": 70, "BestTotalCost": 75, "ActualBuyQty": 55, "BestCostLT": 40, "BestCostSrc": 40, # Adjusted widths
            "FastestLT": 40, "FastestCost": 70, "FastestLTSrc": 40,
            "Alternates": 40, "Notes": 150 # Wider notes
        }
        col_align = { # Alignment for cell content
            "QtyNeed": 'center', "Status": 'center', "Sources": 'center', "StockAvail": 'e',
            "COO": 'center', "RiskScore": 'center', "TariffPct": 'e',
            "BestCostPer": 'e', "BestTotalCost": 'e', "ActualBuyQty": 'center', "BestCostLT": 'center', "BestCostSrc": 'center',
            "FastestLT": 'center', "FastestCost": 'e', "FastestLTSrc": 'center',
            "Alternates": 'center',
        }
        col_tooltips = { # Tooltips for column headers
            "PartNumber": "Part number from the input BOM.",
            "Manufacturer": "Consolidated Manufacturer Name.",
            "MfgPN": "Consolidated Manufacturer Part Number.",
            "QtyNeed": "Total quantity needed (BOM Qty/Unit * Total Units).",
            "Status": "Lifecycle status (Active, EOL, Discontinued, NRND).",
            "Sources": "Number of suppliers found with data.",
            "StockAvail": "Total stock across all valid sources.",
            "COO": "Consolidated Country of Origin.",
            "RiskScore": "Overall Risk Score (0-10). Higher=More Risk.\nRed(>6.5), Yellow(3.6-6.5), Green(<=3.5).\nFactors: Sourcing, Stock, LeadTime, Lifecycle, Geo.",
            "TariffPct": "Estimated Tariff Rate (%) based on COO/HTS.",
            "BestCostPer": "Lowest Unit Cost ($) found for the chosen 'Actual Buy Qty'.",
            "BestTotalCost": "Lowest Total Cost ($) for the 'Actual Buy Qty' (may include price break optimization).",
            "ActualBuyQty": "Quantity chosen for the 'Best Total Cost' calculation (may be > QtyNeed due to MOQ or price breaks).",
            "BestCostLT": "Lead Time (days) for the Best Total Cost option.",
            "BestCostSrc": "Supplier for the Best Total Cost option.",
            "FastestLT": "Shortest Lead Time (days) found.",
            "FastestCost": "Total Cost ($) for the Fastest Lead Time option.",
            "FastestLTSrc": "Supplier for the Fastest Lead Time option.",
            "Alternates": "Indicates if potential alternates were found (via DigiKey). Double-click row to view.",
            "Notes": "Additional notes: Stock Gap, EOL/Discontinued flags, Buy-up reasons."
        }

        self.tree = ttk.Treeview(tree_frame_outer, columns=columns, show="headings", height=18, selectmode="browse")

        # Setup Treeview Columns and Headings with Tooltips
        for col, heading in zip(columns, headings):
            width = col_widths.get(col, 90)
            align = col_align.get(col, 'w') # Default left alignment for content
            self.tree.heading(col, text=heading, command=lambda c=col: self.sort_treeview(self.tree, c, False), anchor='center') # Center align headings
            self.tree.column(col, width=width, minwidth=40, stretch=True, anchor=align) # Align content
            
            self.tree_column_tooltips = {} # Store tooltips for tree columns
        # Setup Treeview Columns and Headings with Tooltips stored for later lookup
        for col, heading in zip(columns, headings):
            width = col_widths.get(col, 90)
            align = col_align.get(col, 'w') # Default left alignment for content
            self.tree.heading(col, text=heading, command=lambda c=col: self.sort_treeview(self.tree, c, False), anchor='center') # Center align headings
            self.tree.column(col, width=width, minwidth=40, stretch=True, anchor=align) # Align content

            # Store the tooltip text associated with the column identifier ('PartNumber', 'Manufacturer', etc.)
            tooltip_text = col_tooltips.get(col, heading)
            self.tree_column_tooltips[col] = tooltip_text

        # Treeview Scrollbars
        self.tree_vsb = ttk.Scrollbar(tree_frame_outer, orient="vertical", command=self.tree.yview)
        self.tree_hsb = ttk.Scrollbar(tree_frame_outer, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.tree_vsb.set, xscrollcommand=self.tree_hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree_vsb.grid(row=0, column=1, sticky="ns")
        self.tree_hsb.grid(row=1, column=0, sticky="ew")

        # Risk Color Tags
        self.tree.tag_configure('high_risk', background='#ffdddd') # Lighter red
        self.tree.tag_configure('moderate_risk', background='#ffffcc') # Light yellow
        self.tree.tag_configure('low_risk', background='#ddffdd') # Lighter green
        self.tree.tag_configure('na_risk', background='#f0f0f0') # Gray for N/A

        # Instructions / Export Button Frame Below Tree
        tree_actions_frame = ttk.Frame(self.analysis_tab)
        tree_actions_frame.grid(row=1, column=0, sticky="ew", pady=(5, 10))
        alt_instruct_label = ttk.Label(tree_actions_frame, text="Double-click row for alternates.", style="Hint.TLabel")
        alt_instruct_label.pack(side=tk.LEFT, padx=(5, 0))
        self.export_parts_list_btn = ttk.Button(tree_actions_frame, text="Export Parts List", command=self.export_treeview_data, state="disabled")
        self.export_parts_list_btn.pack(side=tk.RIGHT, padx=(0, 5))
        self.create_tooltip(self.export_parts_list_btn, "Export the current data shown in the main BOM Analysis parts list table to a CSV file.")
        
        self.tree.bind("<Motion>", self._on_treeview_motion)
        self.tree.bind("<Leave>", self._on_treeview_leave)
        self.tree.bind("<Double-Button-1>", self.show_alternates_popup)

        # -- Analysis Summary Table --
        self.analysis_table_frame = ttk.LabelFrame(self.analysis_tab, text="BOM Summary Metrics", padding=(10, 5)) # Add padding
        self.analysis_table_frame.grid(row=3, column=0, sticky="nsew", pady=(10, 0))
        self.analysis_table_frame.grid_columnconfigure(0, weight=1)
        self.analysis_table_frame.grid_rowconfigure(0, weight=1) # Allow table to resize vertically

        self.analysis_table = ttk.Treeview(self.analysis_table_frame, columns=["Metric", "Value"], show="headings", height=7, selectmode="none") # Increased height
        self.analysis_table.heading("Metric", text="Metric", anchor='w')
        self.analysis_table.heading("Value", text="Value", anchor='w')
        self.analysis_table.column("Metric", width=300, stretch=False, anchor='w')
        self.analysis_table.column("Value", width=450, stretch=True, anchor='w') # Allow value to stretch
        self.analysis_table_scrollbar = ttk.Scrollbar(self.analysis_table_frame, orient="vertical", command=self.analysis_table.yview)
        self.analysis_table.configure(yscrollcommand=self.analysis_table_scrollbar.set)
        self.analysis_table.grid(row=0, column=0, sticky="nsew")
        self.analysis_table_scrollbar.grid(row=0, column=1, sticky="ns")

        # Bind hover events for tooltips on the summary table rows
        self.analysis_table.bind("<Enter>", self._on_widget_enter, add='+') # Reuse universal enter/leave
        self.analysis_table.bind("<Leave>", self._on_widget_leave, add='+')
        self.analysis_table.bind("<Motion>", self._on_summary_table_motion, add='+') # Track motion for specific row

        # -- Export Strategy Buttons -- (Use a dedicated frame)
        export_strategy_frame = ttk.Frame(self.analysis_table_frame, padding=(0, 10, 0, 5))
        export_strategy_frame.grid(row=1, column=0, columnspan=2, sticky='ew')

        ttk.Label(export_strategy_frame, text="Export Strategy:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 10))

        self.lowest_cost_btn = ttk.Button(export_strategy_frame, text="Lowest Cost", command=lambda: self.export_strategy_gui("Lowest Cost"), state="disabled")
        self.lowest_cost_btn.pack(side=tk.LEFT, padx=3)
        self.create_tooltip(self.lowest_cost_btn, "Export CSV for 'Lowest Cost' strategy (prioritizes lowest total cost per part, considering price breaks/MOQ).")

        self.fastest_btn = ttk.Button(export_strategy_frame, text="Fastest", command=lambda: self.export_strategy_gui("Fastest"), state="disabled")
        self.fastest_btn.pack(side=tk.LEFT, padx=3)
        self.create_tooltip(self.fastest_btn, "Export CSV for 'Fastest' strategy (prioritizes shortest lead time per part).")

        self.optimized_strategy_btn = ttk.Button(export_strategy_frame, text="Optimized", command=lambda: self.export_strategy_gui("Optimized Strategy"), state="disabled")
        self.optimized_strategy_btn.pack(side=tk.LEFT, padx=3)
        self.create_tooltip(self.optimized_strategy_btn, "Export CSV for 'Optimized Strategy' (balances cost, lead time, constraints, and potential buy-ups).")

        # --- Tab 2: AI & Predictive Analysis ---
        self.predictive_tab = ttk.Frame(self.results_notebook, padding=10)
        self.results_notebook.add(self.predictive_tab, text=" AI & Predictions ")
        self.predictive_tab.grid_rowconfigure(1, weight=1) # Prediction Table Frame takes weight
        self.predictive_tab.grid_rowconfigure(2, weight=0) # Accuracy Frame fixed size
        self.predictive_tab.grid_columnconfigure(0, weight=1)

        # -- AI Summary Text Area --
        ai_frame = ttk.LabelFrame(self.predictive_tab, text="AI Analysis & Recommendations", padding=5)
        ai_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        ai_frame.grid_rowconfigure(0, weight=1)
        ai_frame.grid_columnconfigure(0, weight=1)
        self.ai_summary_text = scrolledtext.ScrolledText(
            ai_frame,
            wrap=tk.WORD,
            height=12,
            font=("Segoe UI", 9),
            relief="solid",
            borderwidth=1,
            state='disabled',
            background="#f8f8f8",  # Keep light background
            foreground="black"      # Explicitly set text color to black
        )
        self.ai_summary_text.grid(row=0, column=0, sticky="nsew")
        self.ai_summary_text.insert(tk.END, "Run analysis and then click 'AI Summary' (requires OpenAI key).")

        # -- Predictions and Human Input Table --
        pred_update_frame = ttk.LabelFrame(self.predictive_tab, text="Predictions vs Actuals", padding=5)
        pred_update_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 10))
        pred_update_frame.grid_columnconfigure(0, weight=1)
        pred_update_frame.grid_rowconfigure(1, weight=1) # Table takes available space

        # Treeview for Predictions
        pred_tree_frame = ttk.Frame(pred_update_frame) # Frame for tree + scrollbars
        pred_tree_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(5,5))
        pred_tree_frame.grid_rowconfigure(0, weight=1)
        pred_tree_frame.grid_columnconfigure(0, weight=1)

        # Define Prediction Table Columns
        pred_col_widths = {c: 75 for c in self.pred_header} # Default width
        pred_col_widths.update({ # Specific widths
            'Component': 180, 'Date': 80, 'Stock_Probability': 65,
            'Real_Lead': 60, 'Real_Cost': 70, 'Real_Stock': 60,
            'Prophet_Ld_Acc': 60, 'Prophet_Cost_Acc': 60,
            'RAG_Ld_Acc': 60, 'RAG_Cost_Acc': 60,
            'AI_Ld_Acc': 60, 'AI_Cost_Acc': 60,
        })
        pred_col_align = {c: 'center' for c in self.pred_header} # Default center
        pred_col_align.update({'Component': 'w'}) # Left align component name

        pred_col_tooltips = { # Tooltips for prediction table columns
            'Component': 'Consolidated Component Name (Mfg + MPN)',
            'Date': 'Date the prediction was generated.',
            'Prophet_Lead': 'Lead time prediction (days) from Prophet model.',
            'Prophet_Cost': 'Unit cost prediction ($) from Prophet model.',
            'RAG_Lead': 'Lead time prediction range (days) from RAG model (mock).',
            'RAG_Cost': 'Unit cost prediction range ($) from RAG model (mock).',
            'AI_Lead': 'Combined AI lead time prediction (days) (mock).',
            'AI_Cost': 'Combined AI unit cost prediction ($) (mock).',
            'Stock_Probability': 'Predicted probability (%) of finding sufficient stock.',
            'Real_Lead': 'ACTUAL observed lead time (days). Enter value here.',
            'Real_Cost': 'ACTUAL unit cost ($) paid. Enter value here.',
            'Real_Stock': 'Was sufficient stock ACTUALLY available? Select True/False here.',
            'Prophet_Ld_Acc': 'Accuracy (%) of Prophet Lead Time vs Actual.',
            'Prophet_Cost_Acc': 'Accuracy (%) of Prophet Cost vs Actual.',
            'RAG_Ld_Acc': 'Accuracy (%) of RAG Lead Time vs Actual.',
            'RAG_Cost_Acc': 'Accuracy (%) of RAG Cost vs Actual.',
            'AI_Ld_Acc': 'Accuracy (%) of AI Lead Time vs Actual.',
            'AI_Cost_Acc': 'Accuracy (%) of AI Cost vs Actual.',
        }

        self.predictions_tree = ttk.Treeview(pred_tree_frame, columns=self.pred_header, show="headings", height=10, selectmode="browse")

        self.pred_column_tooltips = {}
        for col in self.pred_header:
            width = pred_col_widths.get(col, 75)
            align = pred_col_align.get(col, 'center')
            heading_text = col.replace('_',' ')
            self.predictions_tree.heading(col, text=heading_text, anchor='center') # Set heading text
            self.predictions_tree.column(col, width=width, minwidth=40, stretch=False, anchor=align) # Align cell content
            
            tooltip_text = pred_col_tooltips.get(col, heading_text) # Get tooltip text
            self.pred_column_tooltips[col] = tooltip_text # Store it


        # Prediction Treeview Scrollbars
        pred_vsb = ttk.Scrollbar(pred_tree_frame, orient="vertical", command=self.predictions_tree.yview)
        pred_hsb = ttk.Scrollbar(pred_tree_frame, orient="horizontal", command=self.predictions_tree.xview)
        self.predictions_tree.configure(yscrollcommand=pred_vsb.set, xscrollcommand=pred_hsb.set)
        pred_vsb.grid(row=0, column=1, sticky="ns")
        pred_hsb.grid(row=1, column=0, sticky="ew")
        self.predictions_tree.grid(row=0, column=0, sticky="nsew")

        # Action Buttons Below Prediction Table
        pred_actions_frame = ttk.Frame(pred_update_frame)
        pred_actions_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(10, 5))
        load_pred_button = ttk.Button(pred_actions_frame, text="Load / Refresh Predictions", command=self.load_predictions_to_gui)
        load_pred_button.pack(side=tk.LEFT, padx=(0, 10))
        self.create_tooltip(load_pred_button, f"Load/Reload prediction data from {PREDICTION_FILE.name} into the table above.")

        # Frame for entering actuals
        update_inputs_frame = ttk.Frame(pred_actions_frame)
        update_inputs_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        ttk.Label(update_inputs_frame, text="Update Actuals for Selected Row ->").pack(side=tk.LEFT, padx=(0,5))

        lbl_actual_lead = ttk.Label(update_inputs_frame, text="Lead (d):"); lbl_actual_lead.pack(side=tk.LEFT, padx=(0,2))
        self.real_lead_entry = ttk.Entry(update_inputs_frame, width=6); self.real_lead_entry.pack(side=tk.LEFT, padx=(0,5))
        self.create_tooltip(lbl_actual_lead, "Enter the ACTUAL observed lead time (in days).")
        self.create_tooltip(self.real_lead_entry, "Enter the ACTUAL observed lead time (in days).")

        lbl_actual_cost = ttk.Label(update_inputs_frame, text="Cost ($):"); lbl_actual_cost.pack(side=tk.LEFT, padx=(0,2))
        self.real_cost_entry = ttk.Entry(update_inputs_frame, width=8); self.real_cost_entry.pack(side=tk.LEFT, padx=(0,5))
        self.create_tooltip(lbl_actual_cost, "Enter the ACTUAL unit cost ($) paid.")
        self.create_tooltip(self.real_cost_entry, "Enter the ACTUAL unit cost ($) paid.")

        lbl_actual_stock = ttk.Label(update_inputs_frame, text="Stock OK?:"); lbl_actual_stock.pack(side=tk.LEFT, padx=(0,2))
        self.real_stock_var = tk.StringVar(value="?") # ?, True, False
        self.real_stock_combo = ttk.Combobox(update_inputs_frame, textvariable=self.real_stock_var, values=["?", "True", "False"], width=5, state='readonly'); self.real_stock_combo.pack(side=tk.LEFT, padx=(0,10))
        self.create_tooltip(lbl_actual_stock, "Select if sufficient stock was ACTUALLY available.")
        self.create_tooltip(self.real_stock_combo, "Select if sufficient stock was ACTUALLY available (True/False).")

        self.save_pred_update_btn = ttk.Button(update_inputs_frame, text="Save Actuals", command=self.save_prediction_updates, state="disabled")
        self.save_pred_update_btn.pack(side=tk.LEFT)
        self.create_tooltip(self.save_pred_update_btn, "Save the entered Actual values to the predictions CSV for the selected row.")

        # Label to show selected prediction row ID (for debugging/info)
        self.selected_pred_id_label = ttk.Label(pred_actions_frame, text=" ", style="Hint.TLabel")
        self.selected_pred_id_label.pack(side=tk.RIGHT, padx=(5, 0))

        # Bind selection event
        self.predictions_tree.bind("<Motion>", self._on_predictions_tree_motion)
        self.predictions_tree.bind("<Leave>", self._on_predictions_tree_leave)
        self.predictions_tree.bind('<<TreeviewSelect>>', self.on_prediction_select)

        # -- Average Accuracy Display --
        avg_frame = ttk.LabelFrame(self.predictive_tab, text="Average Prediction Accuracy (%)", padding=5)
        avg_frame.grid(row=2, column=0, sticky='nsew', pady=(5, 0))
        avg_frame.columnconfigure((1, 2, 3, 4, 5, 6), weight=1) # Configure columns to expand

        self.avg_acc_labels = {} # Holds {'Prophet_Ld': label, 'Prophet_Cost': label, ...}

        # Headers for accuracy table
        headers = ["Model", "Ld Acc", "Cost Acc", "# Points", "Ld Acc", "Cost Acc", "# Points"]
        col_widths = [8, 8, 6, 8, 8, 6]
        models = ["Prophet", "RAG", "AI"]
        ttk.Label(avg_frame, text=" ", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky='w', padx=5) # Spacer
        ttk.Label(avg_frame, text="Lead Time", font=("Segoe UI", 9, "bold"), anchor='center').grid(row=0, column=1, columnspan=3, sticky='ew')
        ttk.Label(avg_frame, text="Cost", font=("Segoe UI", 9, "bold"), anchor='center').grid(row=0, column=4, columnspan=3, sticky='ew')
        ttk.Label(avg_frame, text="Model", font=("Segoe UI", 8, "bold")).grid(row=1, column=0, sticky='w', padx=5, pady=(0,3))
        ttk.Label(avg_frame, text="Avg Acc%", font=("Segoe UI", 8, "bold")).grid(row=1, column=1, sticky='ew', pady=(0,3))
        ttk.Label(avg_frame, text="# Pts", font=("Segoe UI", 8, "bold")).grid(row=1, column=2, sticky='ew', pady=(0,3))
        ttk.Label(avg_frame, text="Avg Acc%", font=("Segoe UI", 8, "bold")).grid(row=1, column=3, sticky='ew', pady=(0,3))
        ttk.Label(avg_frame, text="# Pts", font=("Segoe UI", 8, "bold")).grid(row=1, column=4, sticky='ew', pady=(0,3))

        # Create labels for each model
        for i, model in enumerate(models):
            row_num = i + 2
            ttk.Label(avg_frame, text=f"{model}:").grid(row=row_num, column=0, sticky='w', padx=5)

            # Lead Time Accuracy
            ld_key = f"{model}_Ld"
            ld_label = ttk.Label(avg_frame, text="N/A", width=col_widths[1], anchor='e', relief='sunken', background="#f0f0f0")
            ld_label.grid(row=row_num, column=1, sticky='ew', padx=2)
            self.avg_acc_labels[ld_key] = ld_label
            self.create_tooltip(ld_label, f"Average accuracy of {model} lead time predictions vs actuals.")

            # Lead Time Count
            ld_count_key = f"{model}_Ld_Count"
            ld_count_label = ttk.Label(avg_frame, text="0", width=col_widths[2], anchor='e', relief='sunken', background="#f0f0f0")
            ld_count_label.grid(row=row_num, column=2, sticky='ew', padx=2)
            self.avg_acc_labels[ld_count_key] = ld_count_label
            self.create_tooltip(ld_count_label, f"Number of data points used for {model} Lead Time accuracy.")

            # Cost Accuracy
            cost_key = f"{model}_Cost"
            cost_label = ttk.Label(avg_frame, text="N/A", width=col_widths[3], anchor='e', relief='sunken', background="#f0f0f0")
            cost_label.grid(row=row_num, column=3, sticky='ew', padx=2)
            self.avg_acc_labels[cost_key] = cost_label
            self.create_tooltip(cost_label, f"Average accuracy of {model} cost predictions vs actuals.")

            # Cost Count
            cost_count_key = f"{model}_Cost_Count"
            cost_count_label = ttk.Label(avg_frame, text="0", width=col_widths[4], anchor='e', relief='sunken', background="#f0f0f0")
            cost_count_label.grid(row=row_num, column=4, sticky='ew', padx=2)
            self.avg_acc_labels[cost_count_key] = cost_count_label
            self.create_tooltip(cost_count_label, f"Number of data points used for {model} Cost accuracy.")

        # --- Instance Variables ---
        self.bom_df = None
        self.bom_filepath = None
        self.analysis_results = {} # Stores {'config': {}, 'part_summaries': [], 'strategies': {}, 'summary_metrics': [], 'gui_entries': [], 'part_near_misses'[]}
        self.strategies_for_export = {} # Separate storage populated by calculate_summary_metrics
        self.historical_data_df = None
        self.predictions_df = None
        self.digikey_token_data = None
        self.nexar_token_data = None # Added for Nexar
        self.mouser_requests_today = 0
        self.mouser_last_reset_date = None
        self.mouser_daily_limit = 1000 # Default, can be adjusted
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_API_WORKERS, thread_name_prefix="BOMWorker")
        self.running_analysis = False # Flag to prevent concurrent runs
        self._hts_cache = {} # HTS cache per analysis run
        self.prediction_tree_row_map = {} # Map Treeview item ID -> original DataFrame index
        self.tree_item_data_map = {} # Map Analysis Treeview item ID -> original dict data
        self._active_tooltip_widget = None # Track widget for universal tooltip

        # --- Initial Setup Calls ---
        self.load_mouser_request_counter()
        self.update_rate_limit_display() # Call after loading counter
        self.load_digikey_token_from_cache()
        self.load_nexar_token_from_cache()
        self.initialize_data_files() # Call before loading predictions
        self.load_predictions_to_gui() # Load existing predictions on start
        self.validate_inputs() # Initial validation and button state update
        
        initial_sash_position = 480
        self.root.after(100, lambda: self.main_paned_window.sashpos(0, initial_sash_position))
        logger.info("GUI initialization complete.")


    def _on_predictions_tree_motion(self, event):
        """Handle mouse motion over the predictions treeview for tooltips."""
        tooltip_text = ""
        tree = self.predictions_tree # Reference the correct tree
        region = tree.identify_region(event.x, event.y)

        if region == "heading":
            column_id = tree.identify_column(event.x) # Returns '#1', '#2', etc.
            try:
                # Convert column ID to index
                col_index = int(column_id.replace('#', '')) - 1
                cols = tree['columns']
                if 0 <= col_index < len(cols):
                    col_name = cols[col_index]
                    # Look up stored tooltip for this tree's columns
                    tooltip_text = self.pred_column_tooltips.get(col_name, "")
            except (ValueError, IndexError, tk.TclError) as e:
                logger.warning(f"Could not identify predictions tree heading column: {e}")
        else:
            tooltip_text = "" # Clear tooltip when not over heading

        self._show_universal_tooltip(tooltip_text) # Show in the status bar label
        

    def _on_predictions_tree_leave(self, event):
        """Handle mouse leaving the predictions treeview."""
        # Only hide if the tooltip was triggered by *this* tree
        if event.widget == self.predictions_tree:
             self._hide_universal_tooltip()

    
    def _on_treeview_motion(self, event):
        """Handle mouse motion over the main parts treeview for tooltips."""
        tooltip_text = ""
        region = self.tree.identify_region(event.x, event.y)

        if region == "heading":
            column_id = self.tree.identify_column(event.x) # Returns '#1', '#2', etc.
            try:
                # Convert column ID to index (subtract 1 because it's 1-based)
                col_index = int(column_id.replace('#', '')) - 1
                cols = self.tree['columns']
                if 0 <= col_index < len(cols):
                    col_name = cols[col_index]
                    tooltip_text = self.tree_column_tooltips.get(col_name, "") # Look up stored tooltip
            except (ValueError, IndexError, tk.TclError) as e:
                logger.warning(f"Could not identify treeview heading column: {e}")

        elif region == "cell":
            # Optional: Implement tooltips for cell content here if desired
            # row_id = self.tree.identify_row(event.y)
            # column_id = self.tree.identify_column(event.x)
            # if row_id and column_id:
            #    # ... get cell value and maybe show tooltip ...
            #    pass
            tooltip_text = "" # Clear tooltip when over cells for now

        else: # Mouse is over separator, empty space, etc.
             tooltip_text = ""

        self._show_universal_tooltip(tooltip_text) # Show in the status bar label

    def _on_treeview_leave(self, event):
        """Handle mouse leaving the main parts treeview."""
        self._hide_universal_tooltip() # Hide tooltip when mouse leaves the tree

    # --- Tooltip Handling ---
    def create_tooltip(self, widget, text):
        """Stores tooltip text and binds events for the universal status bar tooltip."""
        if widget and text:
            try:
                if not widget.winfo_exists():
                     logger.warning(f"Attempted to register tooltip for non-existent widget: {widget}")
                     return
            except tk.TclError:
                 logger.warning(f"TclError checking existence for widget during tooltip registration: {widget}")
                 return

            # Store the text associated with the widget instance in the new dictionary
            self.tooltip_texts[widget] = text

            # Bind standard Enter/Leave events handled by the app itself
            widget.bind("<Enter>", self._on_widget_enter, add='+')
            widget.bind("<Leave>", self._on_widget_leave, add='+')
            # Optional: Bind FocusIn/FocusOut as well if needed
            # widget.bind("<FocusIn>", self._on_widget_enter, add='+')
            # widget.bind("<FocusOut>", self._on_widget_leave, add='+')

    def _show_universal_tooltip(self, text):
        """Internal method to display text in the universal tooltip label."""
        if not text or not hasattr(self, 'universal_tooltip_label') or not self.universal_tooltip_label.winfo_exists():
            self._hide_universal_tooltip()
            return
        try:
            self.universal_tooltip_label.config(text=text)
        except tk.TclError: pass

    def _hide_universal_tooltip(self):
        """Internal method to clear the universal tooltip label."""
        try:
            if hasattr(self, 'universal_tooltip_label') and self.universal_tooltip_label.winfo_exists():
                 self.universal_tooltip_label.config(text="") # Just clear the text
        except tk.TclError: pass

    def _on_widget_leave(self, event):
        """Callback when mouse leaves a widget - clears universal tooltip if needed."""
        # Hide only if leaving the widget that *triggered* the current tooltip
        if event.widget == self._active_tooltip_widget:
            self._hide_universal_tooltip()
            self._active_tooltip_widget = None
        # Also hide if leaving treeviews or summary table? Add specific checks if needed.
        elif event.widget in [self.tree, self.predictions_tree, self.analysis_table]:
             self._hide_universal_tooltip()
             self._active_tooltip_widget = None

    def _on_widget_enter(self, event):
        """Callback when mouse enters a widget - updates universal tooltip."""
        widget = event.widget
        tooltip_text = "" # Default to empty
        self._active_tooltip_widget = None # Reset active widget initially

        # --- Look up text for the specific widget ---
        if widget in self.tooltip_texts:
            tooltip_text = self.tooltip_texts[widget]
            self._active_tooltip_widget = widget # Track which widget triggered it

        # --- Special Handling for Treeview Headers ---
        # Check main parts tree
        if widget == self.tree:
            region = widget.identify_region(event.x, event.y)
            if region == "heading":
                column_id = widget.identify_column(event.x)
                try:
                    col_index = int(column_id.replace('#', '')) - 1
                    cols = widget['columns']
                    if 0 <= col_index < len(cols):
                        col_name = cols[col_index]
                        # Use the specific dict for tree tooltips
                        tooltip_text = self.tree_column_tooltips.get(col_name, "")
                        if tooltip_text: self._active_tooltip_widget = widget # Track if tooltip found
                except: pass # Ignore errors during identification

        # Check predictions tree
        elif widget == self.predictions_tree:
            region = widget.identify_region(event.x, event.y)
            if region == "heading":
                column_id = widget.identify_column(event.x)
                try:
                    col_index = int(column_id.replace('#', '')) - 1
                    cols = widget['columns']
                    if 0 <= col_index < len(cols):
                        col_name = cols[col_index]
                        tooltip_text = self.pred_column_tooltips.get(col_name, "")
                        if tooltip_text: self._active_tooltip_widget = widget # Track if tooltip found
                except: pass

        # --- Special Handling for Summary Table (if motion binding used) ---
        # If you have _on_summary_table_motion, this elif might not be needed,
        # but it doesn't hurt to ensure _active_tooltip_widget is set.
        elif widget == self.analysis_table:
             if self.universal_tooltip_label.cget("text") != "": # Check if motion handler set text
                 self._active_tooltip_widget = widget

        # --- Show text in the universal status bar label ---
        self._show_universal_tooltip(tooltip_text)

    def _on_summary_table_motion(self, event):
        """Show tooltip for the specific row under the mouse in the summary table."""
        widget = self.analysis_table
        row_id = widget.identify_row(event.y)
        tooltip_text = ""
        if row_id:
            item_values = widget.item(row_id, 'values')
            if item_values and len(item_values) > 0:
                metric_name = item_values[0]
                tooltip_text = self.get_summary_metric_description(metric_name)

        self._show_universal_tooltip(tooltip_text)
        # Set active widget so tooltip hides correctly on leave
        self._active_tooltip_widget = widget if tooltip_text else None

    # --- GUI Update Helpers (Thread-Safe) ---
    def update_status_threadsafe(self, message, level="info"):
        """ Safely updates the status bar from any thread. """
        if is_main_thread():
             self._update_status_gui(message, level)
        else:
             # Schedule the update to run on the main thread
             self.root.after(0, self._update_status_gui, message, level)

    def _update_status_gui(self, message, level="info"):
        """ GUI update part of status update (runs only on main thread). """
        if not hasattr(self, 'status_label') or not self.status_label.winfo_exists(): return
        try:
            color_map = {"info": "#000000", "warning": "#e67e00", "error": "#e60000", "success": "#008000"}
            self.status_label.config(text=message, foreground=color_map.get(level, "#000000"))
            # Optionally log here too, or keep logging separate
            # log_func = getattr(logger, level, logger.info)
            # log_func(f"Status Update: {message}")
        except tk.TclError:
            logger.warning("Ignoring Tkinter status update error, likely during shutdown.")
        except Exception as e:
            logger.error(f"Error updating status label: {e}", exc_info=True)

    def update_progress_threadsafe(self, value, maximum, label_text=""):
        """ Safely updates the progress bar from any thread. """
        if is_main_thread():
             self._update_progress_gui(value, maximum, label_text)
        else:
             self.root.after(0, self._update_progress_gui, value, maximum, label_text)

    def _update_progress_gui(self, value, maximum, label_text=""):
        """ GUI update part of progress bar update (runs only on main thread). """
        if not hasattr(self, 'progress') or not hasattr(self, 'progress_label') or \
           not self.progress.winfo_exists() or not self.progress_label.winfo_exists(): return
        try:
            if maximum > 0:
                self.progress["maximum"] = maximum
                # Clamp value to ensure it doesn't exceed maximum visually
                display_value = min(value, maximum)
                self.progress["value"] = display_value
                percentage = min(100.0, (value / maximum) * 100.0) # Calculate percentage based on actual value
                self.progress_label.config(text=f"{percentage:.0f}%")
                # Update status label alongside progress
                self.status_label.config(text=label_text)
            else:
                self.progress["value"] = 0
                self.progress_label.config(text="0%")
            # Ensure 100% is shown clearly at the end
            if value >= maximum and maximum > 0:
                 self.progress["value"] = maximum
                 self.progress_label.config(text="100%")

        except tk.TclError:
            logger.warning("Ignoring Tkinter progress update error, likely during shutdown.")
        except Exception as e:
            logger.error(f"Error updating progress bar: {e}", exc_info=True)

    def update_rate_limit_display(self):
        """Updates the API rate limit label. Should be called from main thread or scheduled."""
        # Schedule if not on main thread (though often called after GUI action)
        if not is_main_thread():
             self.root.after(0, self.update_rate_limit_display)
             return

        if not hasattr(self, 'rate_label') or not self.rate_label.winfo_exists(): return
        try:
            # DigiKey
            dk_remain = self.digikey_token_data.get("rate_limit_remaining", "NA") if self.digikey_token_data else "NA"
            dk_limit = self.digikey_token_data.get("rate_limit", "NA") if self.digikey_token_data else "NA"
            dk_str = f"DK:{dk_remain}/{dk_limit}" if API_KEYS["DigiKey"] else "DK: Off"

            # Mouser
            m_remain = max(0, self.mouser_daily_limit - self.mouser_requests_today)
            m_limit = self.mouser_daily_limit
            m_str = f"M:{m_remain}/{m_limit}" if API_KEYS["Mouser"] else "M: Off"

            # Nexar (No rate limit info easily available via token)
            nx_str = "NX: OK" if API_KEYS["Octopart (Nexar)"] else "NX: Off"
            if API_KEYS["Octopart (Nexar)"] and hasattr(self, 'nexar_rate_limit_info'): # Placeholder if info is added
                 nx_str = self.nexar_rate_limit_info

            # Mocked APIs
            ar_str = "AR: N/A" if not API_KEYS["Arrow"] else "AR: OK"
            av_str = "AV: N/A" if not API_KEYS["Avnet"] else "AV: OK"

            self.rate_label.config(text=f"API Limits: {dk_str} | {m_str} | {nx_str} | {ar_str} | {av_str}")
        except tk.TclError:
            logger.warning("Ignoring Tkinter rate limit update error, likely during shutdown.")
        except Exception as e:
            logger.error(f"Error updating rate limit display: {e}", exc_info=True)

    def update_analysis_controls_state(self, is_running):
        """ Enables/disables analysis control buttons based on running state and prerequisites. """
        if not is_main_thread():
            self.root.after(0, self.update_analysis_controls_state, is_running)
            return

        # Determine button states based on is_running and other conditions
        config_valid = self._is_config_valid() # Internal check without GUI update
        bom_loaded = self.bom_df is not None and not self.bom_df.empty
        hist_loaded = self.historical_data_df is not None and not self.historical_data_df.empty
        analysis_done = bool(self.analysis_results and self.analysis_results.get("summary_metrics"))
        openai_ready = API_KEYS["OpenAI"]

        can_run = config_valid and bom_loaded and not is_running
        can_predict = hist_loaded and not is_running
        can_summarize = analysis_done and openai_ready and not is_running

        try:
            if hasattr(self, 'run_button') and self.run_button.winfo_exists():
                 self.run_button.config(state="normal" if can_run else "disabled")
            if hasattr(self, 'predict_button') and self.predict_button.winfo_exists():
                 self.predict_button.config(state="normal" if can_predict else "disabled")
            if hasattr(self, 'ai_summary_button') and self.ai_summary_button.winfo_exists():
                 self.ai_summary_button.config(state="normal" if can_summarize else "disabled")
        except tk.TclError: pass # Ignore during shutdown
        except Exception as e: logger.error(f"Error updating control button states: {e}", exc_info=True)

    def update_export_buttons_state(self):
        """Enables/disables export buttons based on analysis results. Runs on main thread."""
        if not is_main_thread():
            self.root.after(0, self.update_export_buttons_state)
            return

        has_results = False
        strategies_valid = False
        lowest_cost_valid = False
        fastest_valid = False
        optimized_valid = False

        try:
            # Check main analysis results exist
            has_results = bool(self.analysis_results and self.analysis_results.get("summary_metrics"))

            # Check if specific strategy data exists in the export dictionary
            strategies_dict = self.strategies_for_export # Use the dedicated dict
            if isinstance(strategies_dict, dict):
                strategies_valid = bool(strategies_dict)
                lowest_cost_valid = "Lowest Cost" in strategies_dict and bool(strategies_dict["Lowest Cost"])
                fastest_valid = "Fastest" in strategies_dict and bool(strategies_dict["Fastest"])
                # Optimized strategy also needs a non-N/A cost in the summary table
                optimized_present = "Optimized Strategy" in strategies_dict and bool(strategies_dict["Optimized Strategy"])
                optimized_summary_value = "N/A"
                if isinstance(self.analysis_results.get("summary_metrics"), list):
                    summary_as_dict = dict(self.analysis_results["summary_metrics"])
                    optimized_summary_value = summary_as_dict.get("Balanced (Optimized Strategy) Cost / LT ($ / Days)", "N/A")

                optimized_valid = optimized_present and "N/A" not in optimized_summary_value

            # Configure button states
            if hasattr(self, 'lowest_cost_btn') and self.lowest_cost_btn.winfo_exists():
                 self.lowest_cost_btn.config(state="normal" if lowest_cost_valid else "disabled")
            if hasattr(self, 'fastest_btn') and self.fastest_btn.winfo_exists():
                 self.fastest_btn.config(state="normal" if fastest_valid else "disabled")
            if hasattr(self, 'optimized_strategy_btn') and self.optimized_strategy_btn.winfo_exists():
                 self.optimized_strategy_btn.config(state="normal" if optimized_valid else "disabled")
            if hasattr(self, 'export_parts_list_btn') and self.export_parts_list_btn.winfo_exists():
                 parts_in_tree = bool(self.tree.get_children())
                 self.export_parts_list_btn.config(state="normal" if parts_in_tree else "disabled")

        except tk.TclError: pass # Ignore errors during shutdown
        except Exception as e: logger.error(f"Error updating export button states: {e}", exc_info=True)

    # --- Treeview Sorting ---
    def sort_treeview(self, tree, col, reverse):
        """Sorts a treeview column, attempting numeric sort first."""
        try:
            # Get data for sorting: (sort_key, item_id)
            data = []
            for item_id in tree.get_children(''):
                val_str = tree.set(item_id, col)
                # Attempt numeric conversion
                num_val = safe_float(val_str, default=None)

                if num_val is not None:
                    sort_key = num_val
                else:
                    # Use lowercase string for non-numeric, place N/A etc. consistently
                    str_val_lower = val_str.lower()
                    if str_val_lower in ['n/a', '', 'unknown', 'none']:
                        # Sort these consistently (e.g., at the end when ascending)
                        sort_key = float('inf') if not reverse else float('-inf')
                    else:
                        sort_key = str_val_lower # Use string itself for sorting

                data.append((sort_key, item_id))

            # Sort the data based on the determined key type
            data.sort(key=lambda x: x[0], reverse=reverse)

        except (tk.TclError, ValueError, TypeError) as e:
            logger.error(f"Error preparing data for sort on column {col}: {e}", exc_info=True)
            return # Abort sort on error

        # Rearrange items in the treeview
        try:
            for index, (sort_key, item) in enumerate(data):
                tree.move(item, '', index)
        except tk.TclError as e:
            logger.error(f"TclError moving items during sort for {col}: {e}")
            return

        # Update the heading command to reverse sort direction
        tree.heading(col, command=lambda: self.sort_treeview(tree, col, not reverse))

    # --- Summary Metric Tooltips ---
    def get_summary_metric_description(self, metric_name):
        """Returns a description for a given summary metric name."""
        # Base descriptions
        descriptions = {
            "Total Parts Analyzed": "Total number of unique BOM line items processed.",
            "Immediate Stock Availability": "Indicates if ALL parts have sufficient stock (>= Qty Needed) available from at least one supplier.",
            "Est. Time to Full Kit (Days)": "Estimated maximum number of days required to have ALL parts in hand, considering current stock (0 days) or the minimum lead time if stock is insufficient.",
            "Parts with Stock Gaps": "Lists specific parts where no single supplier has enough stock to fulfill the required quantity.",
            "Potential Cost Range ($)": "The absolute lowest and highest potential total BOM cost based on available supplier pricing for the calculated 'Buy Qty' for each part.",
            "Lowest Cost Strategy / LT ($ / Days)": "Total BOM cost / Max lead time if systematically choosing the option with the lowest TOTAL cost for the calculated 'Buy Qty' for each part.",
            "Fastest Strategy Cost / LT ($ / Days)": "Total BOM cost / Max lead time if systematically choosing the option with the shortest LEAD TIME for each part.",
            "Balanced (Optimized Strategy) Cost / LT ($ / Days)": "Total BOM cost / Max lead time using the calculated 'Optimized' strategy balancing cost, lead time, and constraints (including potential buy-ups).",
            "Est. Total Tariff Cost": "Estimated total tariff cost based on the parts chosen in the indicated strategy (Optimized or Lowest Cost).", # Base key
            "Est. Total Tariff %": "Estimated total tariff as a percentage of the total BOM cost for the indicated strategy.", # Base key
        }

        # Match dynamically named metrics (like tariffs)
        for key_base, desc in descriptions.items():
            if metric_name.startswith(key_base):
                 # Add the strategy context if present in the metric name
                 if "(Optimized Strategy)" in metric_name: return f"{desc} (Calculated using Optimized Strategy parts)"
                 if "(Lowest Cost Strategy)" in metric_name: return f"{desc} (Calculated using Lowest Cost Strategy parts)"
                 if "(N/A)" in metric_name: return f"{desc} (Could not be calculated - N/A)"
                 return desc # Return base description if no context found

        return "No description available." # Fallback

    # --- Helper Functions ---
    def infer_coo_from_hts(self, hts_code):
        """Infers likely Country of Origin from HTS code using a basic mapping (can be expanded)."""
        if not hts_code or pd.isna(hts_code): return "Unknown"
        hts_clean = str(hts_code).strip().replace(".", "").replace(" ","")[:4] # Use first 4 digits

        # Basic HTS Prefix -> Likely COO Mapping (Example - Needs significant expansion for accuracy)
        hts_map = {
            '8542': 'Taiwan',   # Integrated Circuits
            '8541': 'China',    # Diodes, Transistors
            '8533': 'Japan',    # Resistors
            '8532': 'Malaysia', # Capacitors
            '8504': 'Germany',  # Inductors, Transformers
            '8536': 'Mexico',   # Connectors, Switches
            '9030': 'USA',      # Measuring Instruments
            # ... add more ...
        }
        return hts_map.get(hts_clean, "Unknown")

    def get_digikey_substitutions(self, product_number):
        """Attempts to find substitutions using the DigiKey API."""
        if not API_KEYS["DigiKey"]: return [] # Return empty list if no key
        logger.debug(f"Checking DigiKey substitutions for {product_number}...")
        access_token = self.get_digikey_token() # Assumes called from background thread context
        if not access_token:
            logger.warning("Failed to get DigiKey token for substitution check.")
            return []

        # Use the specific substitutions endpoint
        # Ensure the product number is URL-encoded
        encoded_product_number = urllib.parse.quote(product_number)
        url = f"https://api.digikey.com/products/v4/search/{encoded_product_number}/substitutions"
        headers = {
            'Authorization': f"Bearer {access_token}",
            'X-DIGIKEY-Client-Id': DIGIKEY_CLIENT_ID,
            'X-DIGIKEY-Locale-Site': 'US', # Add necessary locale headers
            'X-DIGIKEY-Locale-Language': 'en',
            'X-DIGIKEY-Locale-Currency': 'USD',
            'Accept': 'application/json' # Explicitly accept JSON
        }

        try:
            # Use GET for this endpoint according to docs
            logger.debug(f"Calling DigiKey Substitutions API: GET {url}")
            response = self._make_api_request("GET", url, headers=headers)
            data = response.json()

            # Update rate limit display after successful call (if possible)
            if self.digikey_token_data and is_main_thread():
                 self.digikey_token_data["rate_limit_remaining"] = response.headers.get('X-RateLimit-Remaining', 'NA')
                 self.digikey_token_data["rate_limit"] = response.headers.get('X-RateLimit-Limit', 'NA')
                 self.update_rate_limit_display()

            substitutes = data.get("ProductSubstitutes", [])
            if isinstance(substitutes, list):
                logger.info(f"Found {len(substitutes)} potential DigiKey substitutes for {product_number}.")
                # Return the list of substitute dicts
                # Ensure nested structures are handled if accessing deeper fields later
                return substitutes
            else:
                 logger.warning(f"Unexpected format for ProductSubstitutes for {product_number}: {type(substitutes)}")
                 return []

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                logger.debug(f"No DigiKey substitutions found (404) for {product_number}.")
            else:
                 logger.error(f"DigiKey Substitutions API HTTP Error for {product_number}: {e}", exc_info=True)
            return []
        except Exception as e:
            logger.error(f"Failed to get DigiKey substitutions for {product_number}: {e}", exc_info=True)
            return []

    def calculate_stock_probability_simple(self, options_list, qty_needed):
        """Calculates a simple stock probability score based on availability and lead times."""
        if not options_list: return 0.0

        suppliers_with_stock = 0
        total_stock = 0
        min_lead_with_stock = np.inf
        min_lead_no_stock = np.inf

        for option in options_list:
            stock = option.get('stock', 0)
            lead = option.get('lead_time', np.inf)
            if stock >= qty_needed:
                suppliers_with_stock += 1
                total_stock += stock
                min_lead_with_stock = min(min_lead_with_stock, lead if lead != np.inf else 0) # Treat inf lead as 0 if stock exists
            elif lead != np.inf: # Only consider lead time if finite
                min_lead_no_stock = min(min_lead_no_stock, lead)

        # Base score on having stock
        if suppliers_with_stock >= 2: score = 90.0
        elif suppliers_with_stock == 1: score = 70.0
        else: score = 15.0 # Low base score if no one has full stock

        # Adjust based on lead times
        if suppliers_with_stock > 0:
            if min_lead_with_stock <= 1: score += 10.0 # Bonus for immediate stock
            elif min_lead_with_stock > 45: score -= 15.0 # Penalty for long lead time even *with* stock
        elif min_lead_no_stock <= 28 : score += 5.0 # Small bonus if LT isn't terrible even without stock
        elif min_lead_no_stock > 90 : score -= 10.0 # Penalty for very long LT when no stock

        # Adjust based on total stock amount (simple ratio)
        if total_stock > qty_needed * 5 and suppliers_with_stock > 0: score += 5.0
        elif total_stock < qty_needed * 1.2 and suppliers_with_stock > 0: score -= 5.0

        return round(max(0.0, min(100.0, score)), 1)

    def create_strategy_entry(self, option_dict):
        """Creates a standardized dictionary for strategy storage and export."""
        if not isinstance(option_dict, dict):
            logger.warning(f"create_strategy_entry called with invalid type: {type(option_dict)}")
            # Return a dict with default N/A values matching export headers
            return {
                'source': 'N/A', 'cost': np.nan, 'lead_time': np.nan, 'stock': 0,
                'unit_cost': np.nan, 'moq': 0, 'actual_order_qty': 0,
                'discontinued': False, 'eol': False, 'bom_pn': 'N/A',
                'original_qty_per_unit': 0, 'total_qty_needed': 0,
                'Manufacturer': 'N/A', 'ManufacturerPartNumber': 'N/A',
                'SourcePartNumber': 'N/A', 'tariff_rate': np.nan,
                'optimized_strategy_score': '', 'CountryOfOrigin': 'N/A',
                'TariffCode': 'N/A', 'notes': 'Invalid Option Data'
            }

        # Extract and clean values using .get() and defaults
        return {
            'source': option_dict.get('source', 'N/A'),
            'cost': option_dict.get('cost', np.nan), # This is TOTAL cost from get_optimal_cost
            'lead_time': option_dict.get('lead_time', np.nan),
            'stock': option_dict.get('stock', 0),
            'unit_cost': option_dict.get('unit_cost', np.nan), # Unit cost corresponding to total cost
            'moq': option_dict.get('moq', 0),
            'actual_order_qty': option_dict.get('actual_order_qty', 0), # Crucial for export
            'discontinued': option_dict.get('discontinued', False),
            'eol': option_dict.get('eol', False),
            'bom_pn': option_dict.get('bom_pn', 'N/A'),
            'original_qty_per_unit': option_dict.get('original_qty_per_unit', 0),
            'total_qty_needed': option_dict.get('total_qty_needed', 0),
            'Manufacturer': option_dict.get('Manufacturer', 'N/A'),
            'ManufacturerPartNumber': option_dict.get('ManufacturerPartNumber', 'N/A'),
            'SourcePartNumber': option_dict.get('SourcePartNumber', 'N/A'),
            'tariff_rate': option_dict.get('tariff_rate', np.nan),
            'optimized_strategy_score': option_dict.get('optimized_strategy_score', ''),
            'CountryOfOrigin': option_dict.get('CountryOfOrigin', 'N/A'),
            'TariffCode': option_dict.get('TariffCode', 'N/A'),
            'notes': option_dict.get('notes', ''), # Include notes field
        }

    # --- Prediction Tab Event Handlers ---
    def on_prediction_select(self, event):
        """Handles selection change in the predictions Treeview."""
        selected_items = self.predictions_tree.selection()
        if not selected_items:
            self.selected_pred_id_label.config(text=" ") # Clear label
            if hasattr(self, 'real_lead_entry'): self.real_lead_entry.delete(0, tk.END)
            if hasattr(self, 'real_cost_entry'): self.real_cost_entry.delete(0, tk.END)
            if hasattr(self, 'real_stock_var'): self.real_stock_var.set("?")
            if hasattr(self, 'save_pred_update_btn'): self.save_pred_update_btn.config(state="disabled")
            return

        selected_item_id = selected_items[0]
        # Display simplified ID (e.g., Component + Date)
        try:
            item_values = self.predictions_tree.item(selected_item_id, 'values')
            cols = self.pred_header
            comp_idx = cols.index('Component')
            date_idx = cols.index('Date')
            display_id = f"{item_values[comp_idx]} ({item_values[date_idx]})"
            self.selected_pred_id_label.config(text=display_id[:40]) # Limit length
        except (ValueError, IndexError, tk.TclError):
             self.selected_pred_id_label.config(text="...") # Fallback

        # Populate entries only if widgets exist
        if not hasattr(self, 'real_lead_entry') or not hasattr(self, 'real_cost_entry') or not hasattr(self, 'real_stock_var') or not hasattr(self, 'save_pred_update_btn'):
            logger.error("Prediction input widgets not fully initialized. Cannot populate.")
            return

        try:
            item_values = self.predictions_tree.item(selected_item_id, 'values')
            cols = self.pred_header

            def get_value(col_name, default=''):
                 try: idx = cols.index(col_name); return item_values[idx] if idx < len(item_values) else default
                 except ValueError: return default

            self.real_lead_entry.delete(0, tk.END)
            self.real_cost_entry.delete(0, tk.END)

            self.real_lead_entry.insert(0, get_value('Real_Lead'))
            self.real_cost_entry.insert(0, get_value('Real_Cost'))

            real_stock_val = get_value('Real_Stock', '?').strip().lower()
            if real_stock_val == 'true': self.real_stock_var.set("True")
            elif real_stock_val == 'false': self.real_stock_var.set("False")
            else: self.real_stock_var.set("?")

            self.save_pred_update_btn.config(state="normal")

        except Exception as e:
            logger.error(f"Error loading selected prediction data: {e}", exc_info=True)
            messagebox.showerror("Error", f"Could not load data for selected row.\n{e}")
            self.selected_pred_id_label.config(text="Error")
            if hasattr(self, 'save_pred_update_btn'): self.save_pred_update_btn.config(state="disabled")

    def save_prediction_updates(self):
        """Saves the human/actual inputs back to the predictions CSV."""
        selected_items = self.predictions_tree.selection()
        if not selected_items:
            messagebox.showwarning("No Selection", "Please select a prediction row in the table first.")
            return

        selected_item_id = selected_items[0] # Unique ID generated by treeview.insert

        if not hasattr(self, 'prediction_tree_row_map') or selected_item_id not in self.prediction_tree_row_map:
            logger.error(f"Cannot find original data for selected tree item ID: {selected_item_id}")
            messagebox.showerror("Save Error", "Internal error: Could not map selected row to original data. Please reload predictions.")
            return

        original_df_index = self.prediction_tree_row_map[selected_item_id]
        logger.debug(f"Attempting to save updates for Tree Item ID {selected_item_id} mapped to DataFrame index {original_df_index}")

        try:
            # Get values from GUI
            real_lead_str = self.real_lead_entry.get().strip()
            real_cost_str = self.real_cost_entry.get().strip()
            real_stock_str = self.real_stock_var.get()

            real_lead = safe_float(real_lead_str, default=np.nan)
            real_cost = safe_float(real_cost_str, default=np.nan)
            # Convert '?' to None (which becomes NaN/NA in pandas)
            real_stock = None if real_stock_str == "?" else (real_stock_str == "True")

            # --- Load current predictions CSV into DataFrame ---
            try:
                # Specify dtypes more precisely, keep_default_na=False is crucial
                df = pd.read_csv(PREDICTION_FILE, dtype=str, keep_default_na=False)
                df = df.reset_index(drop=True) # Ensure consistent index
                # Validate header existence after load
                missing_cols = [c for c in self.pred_header if c not in df.columns]
                if missing_cols:
                     raise KeyError(f"Prediction CSV {PREDICTION_FILE.name} is missing columns: {missing_cols}. Please backup/delete and let the app recreate it.")
            except FileNotFoundError:
                 logger.error(f"Prediction file {PREDICTION_FILE} not found during save operation.")
                 messagebox.showerror("File Error", f"Prediction file not found:\n{PREDICTION_FILE.name}\nCannot save.")
                 return
            except Exception as e:
                logger.error(f"Failed to load {PREDICTION_FILE.name} for update: {e}", exc_info=True)
                messagebox.showerror("File Error", f"Could not load prediction file for saving.\n{e}")
                return

            # --- Validate index ---
            if original_df_index >= len(df):
                logger.error(f"Original DataFrame index {original_df_index} is out of bounds for the loaded CSV (length {len(df)}). File might have changed drastically.")
                messagebox.showerror("Save Error", "Could not find the corresponding row in the CSV file. It might have been modified externally or corrupted. Please reload.")
                return
            row_index = original_df_index # Use the mapped original index

            # --- Update DataFrame ---
            logger.info(f"Updating DataFrame index: {row_index}")

            # Store actuals (convert to string for consistency if needed, or handle types carefully)
            df.loc[row_index, 'Real_Lead'] = str(real_lead) if pd.notna(real_lead) else ''
            df.loc[row_index, 'Real_Cost'] = str(real_cost) if pd.notna(real_cost) else ''
            #df.loc[row_index, 'Real_Stock'] = str(real_stock) if real_stock is not None else ''
            if real_stock is True:
                real_stock_save_val = "True"
            elif real_stock is False:
                real_stock_save_val = "False"
            else: # Handles None and anything else
                real_stock_save_val = ""
            df.loc[row_index, 'Real_Stock'] = real_stock_save_val

            # Calculate and store accuracies *only if* both lead and cost are valid
            if pd.notna(real_lead) and pd.notna(real_cost):
                 # Get predicted values from the loaded DataFrame row
                 prophet_lead_pred = safe_float(df.loc[row_index, 'Prophet_Lead'], default=np.nan)
                 ai_lead_pred = safe_float(df.loc[row_index, 'AI_Lead'], default=np.nan)
                 prophet_cost_pred = safe_float(df.loc[row_index, 'Prophet_Cost'], default=np.nan)
                 ai_cost_pred = safe_float(df.loc[row_index, 'AI_Cost'], default=np.nan)

                 # Parse RAG Lead
                 rag_lead_range = df.loc[row_index, 'RAG_Lead']; rag_lead_mid = np.nan
                 if isinstance(rag_lead_range, str) and '-' in rag_lead_range:
                     try:
                         parts = [safe_float(p) for p in rag_lead_range.split('-')]
                         if len(parts) == 2 and not any(pd.isna(p) for p in parts): rag_lead_mid = (parts[0] + parts[1]) / 2
                     except: pass
                 # Parse RAG Cost
                 rag_cost_range = df.loc[row_index, 'RAG_Cost']; rag_cost_mid = np.nan
                 if isinstance(rag_cost_range, str) and '-' in rag_cost_range:
                     try:
                         parts = [safe_float(p) for p in rag_cost_range.split('-')]
                         if len(parts) == 2 and not any(pd.isna(p) for p in parts): rag_cost_mid = (parts[0] + parts[1]) / 2
                     except: pass

                 acc_results = self.calculate_prediction_accuracy(
                     real_lead, prophet_lead_pred, rag_lead_mid, ai_lead_pred,
                     real_cost, prophet_cost_pred, rag_cost_mid, ai_cost_pred
                 )

                 # Update DataFrame - Accuracies (store as string with fixed precision)
                 for model in ["Prophet", "RAG", "AI"]:
                     ld_acc = acc_results['Ld'].get(model, np.nan)
                     cost_acc = acc_results['Cost'].get(model, np.nan)
                     df.loc[row_index, f'{model}_Ld_Acc'] = f"{ld_acc:.1f}" if pd.notna(ld_acc) else ''
                     df.loc[row_index, f'{model}_Cost_Acc'] = f"{cost_acc:.1f}" if pd.notna(cost_acc) else ''
            else:
                 # Clear accuracy columns if actuals are not fully provided
                 for model in ["Prophet", "RAG", "AI"]:
                     df.loc[row_index, f'{model}_Ld_Acc'] = ''
                     df.loc[row_index, f'{model}_Cost_Acc'] = ''

            # --- Save DataFrame back to CSV ---
            try:
                 # Ensure columns are in the correct order before saving
                 df = df[self.pred_header]
                 df.to_csv(PREDICTION_FILE, index=False, quoting=csv.QUOTE_MINIMAL, encoding='utf-8')
                 component_name = df.loc[row_index, 'Component'] # Get name for status msg
                 date_str = df.loc[row_index, 'Date']
                 logger.info(f"Successfully updated prediction row for '{component_name}' on {date_str}.")
                 self.update_status_threadsafe("Prediction update saved.", "success")

                 # Refresh GUI table to show updated values and accuracies
                 self.load_predictions_to_gui()

            except IOError as e:
                logger.error(f"Failed to save updated predictions CSV: {e}", exc_info=True)
                messagebox.showerror("Save Error", f"Could not save updates to prediction file.\nCheck file permissions or if it's open elsewhere.\n{e}")
            except Exception as e:
                logger.error(f"Unexpected error saving predictions CSV: {e}", exc_info=True)
                messagebox.showerror("Save Error", f"An unexpected error occurred saving updates.\n{e}")

        except KeyError as e:
             logger.error(f"Save Error: Column mismatch or missing data for row index {original_df_index}. Error: {e}", exc_info=True)
             messagebox.showerror("Save Error", f"Could not find expected data for the selected row.\nTable might be malformed or file corrupted.\nError: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during prediction save: {e}", exc_info=True)
            messagebox.showerror("Error", f"An unexpected error occurred during save:\n{e}")
            

    def clear_prediction_actuals_inputs(self):
        """Clears the input fields for prediction actuals and disables save button."""
        if hasattr(self, 'real_lead_entry') and self.real_lead_entry.winfo_exists():
            self.real_lead_entry.delete(0, tk.END)
        if hasattr(self, 'real_cost_entry') and self.real_cost_entry.winfo_exists():
            self.real_cost_entry.delete(0, tk.END)
        if hasattr(self, 'real_stock_var'):
            self.real_stock_var.set("?")
        if hasattr(self, 'save_pred_update_btn') and self.save_pred_update_btn.winfo_exists():
            self.save_pred_update_btn.config(state="disabled")
        if hasattr(self, 'selected_pred_id_label') and self.selected_pred_id_label.winfo_exists():
            self.selected_pred_id_label.config(text=" ") # Clear selection label
            

    def calculate_prediction_accuracy(self, real_lead, prophet_lead, rag_lead_mid, ai_lead,
                                      real_cost, prophet_cost, rag_cost_mid, ai_cost):
        """Calculates lead time AND cost prediction accuracy percentage (0-100)."""
        results = {'Ld': {'Prophet': np.nan, 'RAG': np.nan, 'AI': np.nan},
                   'Cost': {'Prophet': np.nan, 'RAG': np.nan, 'AI': np.nan}}

        def calc_single_acc(predicted, actual):
            """Calculates accuracy for a single prediction vs actual."""
            if pd.isna(predicted) or pd.isna(actual): return np.nan
            # Handle zero or near-zero actual value carefully
            if abs(actual) < 1e-9:
                return 100.0 if abs(predicted) < 1e-9 else 0.0 # 100% if both zero, 0% otherwise

            error_margin = abs(predicted - actual)
            # Accuracy = 100 * (1 - |error| / |actual|)
            # Clamp result between 0 and 100
            accuracy = max(0.0, 100.0 * (1.0 - error_margin / abs(actual)))
            return accuracy

        # Calculate Lead Time Accuracies (ensure actual lead time is non-negative)
        if pd.notna(real_lead) and real_lead >= 0:
            # Clamp predicted lead times >= 0 before calculating accuracy
            preds_ld = {
                'Prophet': max(0, prophet_lead) if pd.notna(prophet_lead) else np.nan,
                'RAG': max(0, rag_lead_mid) if pd.notna(rag_lead_mid) else np.nan,
                'AI': max(0, ai_lead) if pd.notna(ai_lead) else np.nan
            }
            for model, pred_val in preds_ld.items():
                results['Ld'][model] = calc_single_acc(pred_val, real_lead)

        # Calculate Cost Accuracies (ensure actual cost is positive)
        if pd.notna(real_cost) and real_cost > 0:
            # Clamp predicted costs > 0 (e.g., minimum $0.001)
            preds_cost = {
                'Prophet': max(0.001, prophet_cost) if pd.notna(prophet_cost) else np.nan,
                'RAG': max(0.001, rag_cost_mid) if pd.notna(rag_cost_mid) else np.nan,
                'AI': max(0.001, ai_cost) if pd.notna(ai_cost) else np.nan
            }
            for model, pred_val in preds_cost.items():
                results['Cost'][model] = calc_single_acc(pred_val, real_cost)

        return results

    # --- Data File Handling ---
    def initialize_data_files(self):
        """Ensure historical and prediction CSV files exist and load initial data."""
        logger.info("Initializing data files...")
        init_csv_file(HISTORICAL_DATA_FILE, self.hist_header)
        init_csv_file(PREDICTION_FILE, self.pred_header)

        # --- Load Historical Data ---
        try:
            self.historical_data_df = pd.read_csv(HISTORICAL_DATA_FILE, dtype=str, keep_default_na=False)
            # Validate header
            missing_hist_cols = [col for col in self.hist_header if col not in self.historical_data_df.columns]
            if missing_hist_cols:
                 raise KeyError(f"Historical CSV missing columns: {missing_hist_cols}. Backup/delete the file to regenerate.")

            # Convert types after validation
            if 'Fetch_Timestamp' in self.historical_data_df.columns:
                self.historical_data_df['Fetch_Timestamp'] = pd.to_datetime(self.historical_data_df['Fetch_Timestamp'], errors='coerce', utc=True) # Assume UTC or make aware
            numeric_cols_hist = ['Lead_Time_Days', 'Cost', 'Inventory', 'Stock_Probability']
            for col in numeric_cols_hist:
                if col in self.historical_data_df.columns:
                    self.historical_data_df[col] = pd.to_numeric(self.historical_data_df[col], errors='coerce')
            logger.info(f"Loaded {len(self.historical_data_df)} historical records.")

        except FileNotFoundError:
             logger.warning(f"{HISTORICAL_DATA_FILE.name} not found. Will be created on first analysis.")
             self.historical_data_df = pd.DataFrame(columns=self.hist_header)
        except KeyError as e:
            logger.error(f"Failed to load historical data - {e}. Check CSV header. Initializing empty DataFrame.")
            messagebox.showerror("Data Load Error", f"Error loading historical data from {HISTORICAL_DATA_FILE.name}:\n\n{e}\n\nPlease check the file header or delete it to start fresh.")
            self.historical_data_df = pd.DataFrame(columns=self.hist_header)
        except Exception as e:
            logger.error(f"Unexpected error loading historical data: {e}", exc_info=True)
            messagebox.showerror("Data Load Error", f"Unexpected error loading historical data:\n\n{e}")
            self.historical_data_df = pd.DataFrame(columns=self.hist_header)

        # --- Load Prediction Data ---
        try:
            self.predictions_df = pd.read_csv(PREDICTION_FILE, dtype=str, keep_default_na=False)
            # Validate header
            missing_pred_cols = [col for col in self.pred_header if col not in self.predictions_df.columns]
            if missing_pred_cols:
                 logger.warning(f"Prediction file ({PREDICTION_FILE.name}) is missing columns: {missing_pred_cols}. Adding them.")
                 for col in missing_pred_cols: self.predictions_df[col] = ''
                 try: # Save the corrected file
                      self.predictions_df[self.pred_header].to_csv(PREDICTION_FILE, index=False, quoting=csv.QUOTE_MINIMAL, encoding='utf-8')
                      logger.info(f"Added missing columns to {PREDICTION_FILE.name} and saved.")
                 except IOError as save_err: logger.error(f"Could not save prediction file after adding columns: {save_err}")

            # Convert types after validation
            if 'Date' in self.predictions_df.columns:
                # Try multiple formats or inference
                self.predictions_df['Date'] = pd.to_datetime(self.predictions_df['Date'], errors='coerce')

            numeric_cols_pred = [ # Ensure all potentially numeric columns are listed
                'Prophet_Lead', 'Prophet_Cost', 'AI_Lead', 'AI_Cost', 'Stock_Probability',
                'Real_Lead', 'Real_Cost',
                'Prophet_Ld_Acc', 'Prophet_Cost_Acc',
                'RAG_Ld_Acc', 'RAG_Cost_Acc',
                'AI_Ld_Acc', 'AI_Cost_Acc'
            ]
            # RAG columns are ranges (strings), don't convert them here
            rag_cols = ['RAG_Lead', 'RAG_Cost']

            for col in numeric_cols_pred:
                if col in self.predictions_df.columns:
                    self.predictions_df[col] = pd.to_numeric(self.predictions_df[col], errors='coerce')

            if 'Real_Stock' in self.predictions_df.columns:
                # Use map for boolean conversion, handle various cases, keep NA as NA
                 self.predictions_df['Real_Stock'] = self.predictions_df['Real_Stock'].str.lower().map(
                     {'true': True, 't': True, '1': True, 'yes': True, 'false': False, 'f': False, '0': False, 'no': False,
                      '': pd.NA, '?': pd.NA, 'none': pd.NA, 'nan': pd.NA, 'na': pd.NA}
                 ).astype('boolean') # Use pandas nullable boolean type

            logger.info(f"Loaded {len(self.predictions_df)} prediction records.")

        except FileNotFoundError:
            logger.warning(f"{PREDICTION_FILE.name} not found. Will be created when predictions run.")
            self.predictions_df = pd.DataFrame(columns=self.pred_header)
        except KeyError as e:
             logger.error(f"Failed to load prediction data - {e}. Check CSV header. Initializing empty DataFrame.")
             messagebox.showerror("Data Load Error", f"Error loading prediction data from {PREDICTION_FILE.name}:\n\n{e}\n\nPlease check the file header or delete it to start fresh.")
             self.predictions_df = pd.DataFrame(columns=self.pred_header)
        except Exception as e:
            logger.error(f"Unexpected error loading prediction data: {e}", exc_info=True)
            messagebox.showerror("Data Load Error", f"Unexpected error loading prediction data:\n\n{e}")
            self.predictions_df = pd.DataFrame(columns=self.pred_header)


    # --- Input Validation ---
    def _is_config_valid(self):
         """ Internal check for config validity without updating GUI. """
         try:
             # Fetch values using safe_float
             total_units = safe_float(self.config_vars["total_units"].get(), default=-1)
             max_premium = safe_float(self.config_vars["max_premium"].get(), default=-1)
             target_lead_time = safe_float(self.config_vars["target_lead_time_days"].get(), default=-1)
             cost_weight = safe_float(self.config_vars["cost_weight"].get(), default=-1)
             lead_time_weight = safe_float(self.config_vars["lead_time_weight"].get(), default=-1)
             buy_up_threshold = safe_float(self.config_vars["buy_up_threshold"].get(), default=-1) # Added

             # Check core constraints
             if total_units <= 0: return False
             if not (0 <= max_premium <= 1000): return False # Allow up to 1000% premium? Adjust if needed
             if target_lead_time < 0: return False
             if not (0 <= cost_weight <= 1): return False
             if not (0 <= lead_time_weight <= 1): return False
             if not np.isclose(cost_weight + lead_time_weight, 1.0, atol=0.01): return False
             if not (0 <= buy_up_threshold <= 100): return False # Buy-up threshold between 0% and 100%

             # Check tariff rates
             for entry in self.tariff_entries.values():
                 val = entry.get()
                 if val: # Only validate if not blank
                     rate = safe_float(val, default=-999)
                     if not (0 <= rate <= 1000): # Allow high tariff %
                         return False
             return True # All checks passed
         except (AttributeError, KeyError, tk.TclError):
             # Widgets might not be ready yet during init
             return False
         except Exception as e:
             logger.error(f"Unexpected error during internal config validation: {e}", exc_info=True)
             return False

    def validate_inputs(self, event=None):
        """Validates configuration inputs, updates validation label, and button states."""
        errors = []
        try:
            # Fetch values using safe_float
            total_units = safe_float(self.config_vars["total_units"].get(), default=-1)
            max_premium = safe_float(self.config_vars["max_premium"].get(), default=-1)
            target_lead_time = safe_float(self.config_vars["target_lead_time_days"].get(), default=-1)
            cost_weight = safe_float(self.config_vars["cost_weight"].get(), default=-1)
            lead_time_weight = safe_float(self.config_vars["lead_time_weight"].get(), default=-1)
            buy_up_threshold = safe_float(self.config_vars["buy_up_threshold"].get(), default=-1) # Added

            # Check constraints and build error list
            if total_units <= 0: errors.append("Total Units must be > 0.")
            if not (0 <= max_premium <= 1000): errors.append("Max Premium must be 0-1000%.")
            if target_lead_time < 0: errors.append("Target Lead Time must be >= 0.")
            if not (0 <= cost_weight <= 1): errors.append("Cost Weight must be 0-1.")
            if not (0 <= lead_time_weight <= 1): errors.append("Lead Time Weight must be 0-1.")
            if not np.isclose(cost_weight + lead_time_weight, 1.0, atol=0.01): errors.append("Cost + Lead Time Weights must sum to 1.0.")
            if not (0 <= buy_up_threshold <= 100): errors.append("Buy-Up Threshold must be 0-100%.")

            # Check tariff rates
            for country, entry in self.tariff_entries.items():
                val = entry.get()
                if val: # Only validate if not blank
                    rate = safe_float(val, default=-999)
                    if not (0 <= rate <= 1000):
                        errors.append(f"Tariff for {country} invalid (must be % >= 0).")

            # Update validation label
            if errors:
                self.validation_label.config(text="Config Issues: " + " ".join(errors), foreground="red")
                is_valid = False
            else:
                self.validation_label.config(text="Configuration Valid", foreground="green")
                is_valid = True

        except (AttributeError, KeyError, tk.TclError):
             # Widgets might not be ready yet during init
             self.validation_label.config(text="Initializing...", foreground="orange")
             is_valid = False
        except Exception as e:
            self.validation_label.config(text=f"Validation Error: {e}", foreground="red")
            logger.error(f"Input validation error: {e}", exc_info=True)
            is_valid = False

        # Update button states regardless of validation outcome
        # Schedule this update to ensure it runs after the current event processing
        self.root.after(10, self.update_analysis_controls_state, self.running_analysis)

        return is_valid # Return validity status

    # --- BOM Loading ---
    def load_bom(self):
        """Loads BOM data from a CSV file, performs cleaning and validation."""
        if self.running_analysis:
            messagebox.showwarning("Busy", "Analysis is currently running. Please wait.")
            return

        filepath = filedialog.askopenfilename(
            title="Select BOM CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not filepath: return # User cancelled

        try:
            self.update_status_threadsafe("Loading BOM...", "info")
            df = pd.read_csv(filepath, dtype=str, keep_default_na=False)

            # --- Column Mapping (Case-insensitive, flexible names) ---
            col_map_needed = { # Target Name: [Possible Input Names]
                "Part Number": ["part number", "part #", "part_number", "pn", "mpn"],
                "Quantity": ["quantity", "qty", "bom qty", "quantity per", "qty per"],
                "Manufacturer": ["manufacturer", "mfg", "make"],
            }
            rename_dict = {}
            found_cols = {c.lower().strip(): c for c in df.columns} # Map lowercase input col to original case

            for target_col, possible_names in col_map_needed.items():
                found = False
                for name in possible_names:
                    if name in found_cols:
                         original_case_col = found_cols[name]
                         if target_col not in df.columns: # Avoid renaming if target already exists
                             rename_dict[original_case_col] = target_col
                         elif original_case_col != target_col:
                              # If target exists but we found a different source, log maybe?
                              pass
                         found = True
                         break
                if not found and target_col != "Manufacturer": # Manufacturer is optional
                     raise ValueError(f"Missing required BOM column: '{target_col}'. Please ensure your CSV has a column like: {possible_names}")

            if rename_dict:
                 logger.info(f"Renaming BOM columns: {rename_dict}")
                 df.rename(columns=rename_dict, inplace=True)

            # --- Data Cleaning and Validation ---
            # Part Number
            if "Part Number" not in df.columns: raise ValueError("Logic Error: 'Part Number' column missing after mapping.")
            df["Part Number"] = df["Part Number"].astype(str).str.strip()
            df.dropna(subset=["Part Number"], inplace=True) # Drop rows with truly NaN PN
            df = df[df["Part Number"] != ''] # Drop rows with empty string PN

            # Quantity
            if "Quantity" not in df.columns: raise ValueError("Logic Error: 'Quantity' column missing after mapping.")
            df["Quantity"] = pd.to_numeric(df["Quantity"], errors='coerce')
            df.dropna(subset=["Quantity"], inplace=True) # Drop rows where Quantity couldn't be converted
            df = df[df["Quantity"] > 0] # Keep only positive quantities
            df["Quantity"] = df["Quantity"].astype(int)

            # Manufacturer (Optional)
            if "Manufacturer" not in df.columns:
                df["Manufacturer"] = "" # Add empty Manufacturer if missing
                logger.warning("BOM missing optional 'Manufacturer' column. Added empty column.")
            else:
                df["Manufacturer"] = df["Manufacturer"].astype(str).str.strip()

            # Check for duplicates (optional, log warning)
            duplicates = df[df.duplicated(subset=["Part Number"], keep=False)]
            if not duplicates.empty:
                 logger.warning(f"Found {len(duplicates)} duplicate Part Numbers in BOM. Analyzing each instance.")
                 # Consider how to handle duplicates - aggregate qty? analyze separately? Current code analyzes separately.

            # Final Check
            if df.empty:
                 raise ValueError("No valid parts found in the BOM after cleaning (check Part Numbers and Quantities > 0).")

            original_count = len(pd.read_csv(filepath, dtype=str, keep_default_na=False)) # Read again to get original count
            removed_count = original_count - len(df)
            valid_parts_count = len(df)

            self.bom_df = df
            self.bom_filepath = Path(filepath)
            self.file_label.config(text=f"{self.bom_filepath.name} ({valid_parts_count} parts)")
            self.update_status_threadsafe(f"BOM loaded: {valid_parts_count} valid parts found ({removed_count} rows removed/invalid).", level="success")

            # Clear previous results and GUI tables
            self.analysis_results = {}
            self.strategies_for_export = {}
            self.clear_treeview(self.tree)
            self.clear_treeview(self.analysis_table)
            self.ai_summary_text.configure(state='normal')
            self.ai_summary_text.delete(1.0, tk.END)
            self.ai_summary_text.insert(tk.END, "Run analysis and then click 'AI Summary' (requires OpenAI key).")
            self.ai_summary_text.configure(state='disabled')

            self.validate_inputs() # Re-validate to potentially enable run button
            self.update_export_buttons_state() # Disable export buttons

        except ValueError as ve:
             self.bom_df = None; self.bom_filepath = None
             self.file_label.config(text="BOM Load Failed!")
             err_msg = f"BOM Load Error: {ve}"
             logger.error(err_msg)
             self.update_status_threadsafe(err_msg, level="error")
             messagebox.showerror("BOM Load Error", err_msg)
             self.validate_inputs() # Update button states
             self.update_export_buttons_state()
        except Exception as e:
            self.bom_df = None; self.bom_filepath = None
            self.file_label.config(text="BOM Load Failed!")
            err_msg = f"Failed to load/parse BOM: {e}"
            logger.error(err_msg, exc_info=True)
            self.update_status_threadsafe(err_msg, level="error")
            messagebox.showerror("BOM Load Error", f"Could not load or process the BOM file.\n\nError: {e}")
            self.validate_inputs() # Update button states
            self.update_export_buttons_state()

    # --- Alternates Popup ---
    def show_alternates_popup(self, event):
        """Displays a pop-up window with alternate parts for the selected row."""
        selected_items = self.tree.selection()
        if not selected_items: return

        selected_item_id = selected_items[0]

        # Get data stored for this item_id
        item_data = self.tree_item_data_map.get(selected_item_id)
        if not item_data:
            logger.warning(f"No backend data found for selected tree item {selected_item_id}")
            # Try getting values directly from tree as fallback
            try:
                item_values = self.tree.item(selected_item_id, 'values')
                cols = list(self.tree['columns'])
                bom_pn_idx = cols.index('PartNumber')
                mfg_pn_idx = cols.index('MfgPN')
                bom_pn = item_values[bom_pn_idx]
                mfg_pn = item_values[mfg_pn_idx]
                alternates_list = [] # Cannot get alternates from tree values alone
            except (ValueError, IndexError, tk.TclError):
                 messagebox.showerror("Error", "Could not retrieve part details for alternates.")
                 return
        else:
            bom_pn = item_data.get("PartNumber", "N/A")
            mfg_pn = item_data.get("MfgPN", "N/A")
            # Find the corresponding part_summary in analysis_results using bom_pn
            alternates_list = []
            if self.analysis_results and isinstance(self.analysis_results.get("part_summaries"), list):
                part_summary_found = next((ps for ps in self.analysis_results["part_summaries"] if ps.get("bom_pn") == bom_pn), None)
                if part_summary_found and 'alternates' in part_summary_found:
                    # Ensure it's a list of dicts
                    raw_alternates = part_summary_found['alternates']
                    if isinstance(raw_alternates, list):
                        alternates_list = [alt for alt in raw_alternates if isinstance(alt, dict)]
                    else:
                         logger.warning(f"Alternates data for {bom_pn} is not a list: {type(raw_alternates)}")

        logger.info(f"Showing alternates pop-up for BOM P/N: {bom_pn} (Mfg P/N: {mfg_pn})")

        # --- Create Pop-up --- (Standard Tkinter Toplevel)
        popup = tk.Toplevel(self.root)
        popup.title(f"Alternates for {mfg_pn}")
        popup.geometry("700x450") # Wider popup
        popup.transient(self.root)
        popup.grab_set()

        popup_frame = ttk.Frame(popup, padding=10)
        popup_frame.pack(fill=tk.BOTH, expand=True)
        popup_frame.rowconfigure(1, weight=1) # Allow text area to expand
        popup_frame.columnconfigure(0, weight=1)

        ttk.Label(popup_frame, text=f"Potential Substitutes/Alternates (from DigiKey):", font=("Segoe UI", 10, "bold")).grid(row=0, column=0, sticky='w', pady=(0, 2))
        ttk.Label(popup_frame, text=f"Mfg P/N: {mfg_pn}", style="Hint.TLabel").grid(row=1, column=0, sticky='w', pady=(0, 10))

        alt_text_area = scrolledtext.ScrolledText(popup_frame, wrap=tk.WORD, height=18, width=90, font=("Courier New", 9), relief="solid", borderwidth=1)
        alt_text_area.grid(row=2, column=0, sticky="nsew", pady=(5, 5))
        alt_text_area.configure(state='disabled')

        # Populate Content
        content = ""
        if not alternates_list:
            content = "No alternate parts found via DigiKey API for this Manufacturer Part Number."
        else:
            content = "Type                 | Manufacturer         | Alt Mfg Part Num        | Description\n"
            content += "---------------------+----------------------+-------------------------+--------------------------\n"
            # Expecting list of dicts from get_digikey_substitutions
            for alt in alternates_list:
                 sub_type = alt.get('SubstituteType', 'Unknown')[:20].ljust(20)
                 # Manufacturer name is nested inside 'Manufacturer' dict
                 mfg_name = alt.get('Manufacturer', {}).get('Name', 'N/A')[:20].ljust(20)
                 mfg_pn_alt = alt.get('ManufacturerProductNumber', 'N/A')[:23].ljust(23)
                 # Description is nested inside 'ProductDescription' dict
                 desc = alt.get('ProductDescription', {}).get('ProductDescription', 'N/A')[:40] # Limit description length
                 content += f"{sub_type} | {mfg_name} | {mfg_pn_alt} | {desc}\n"

        alt_text_area.configure(state='normal')
        alt_text_area.insert(tk.END, content)
        alt_text_area.configure(state='disabled')

        close_button = ttk.Button(popup_frame, text="Close", command=popup.destroy)
        close_button.grid(row=3, column=0, pady=(10, 0))

        # Center the popup
        popup.update_idletasks()
        self.center_window(popup)

    def center_window(self, window):
        """ Centers a Tkinter window (Toplevel or Root) on the screen. """
        window.update_idletasks()
        width = window.winfo_width()
        height = window.winfo_height()
        screen_width = window.winfo_screenwidth()
        screen_height = window.winfo_screenheight()
        x = (screen_width // 2) - (width // 2)
        y = (screen_height // 2) - (height // 2)
        window.geometry(f'{width}x{height}+{x}+{y}')


    # --- Authentication (DigiKey, Nexar) ---

    # --- DigiKey Authentication (HTTPS Method) ---
    def load_digikey_token_from_cache(self):
        """Loads DigiKey token from cache file if valid."""
        if not TOKEN_FILE.exists():
             logger.info("No DigiKey token cache file found.")
             self.digikey_token_data = None
             return False
        try:
            with open(TOKEN_FILE, 'r') as f:
                self.digikey_token_data = json.load(f)
            expires_at = self.digikey_token_data.get('expires_at', 0)
            now = time.time()

            if now < expires_at and self.digikey_token_data.get('access_token'):
                 logger.info("Valid DigiKey token loaded from cache.")
                 refresh_in = expires_at - now - 300 # Refresh 5 mins before expiry
                 if refresh_in > 0:
                      self._schedule_digikey_refresh(int(refresh_in * 1000))
                 return True
            elif self.digikey_token_data.get('refresh_token'):
                 logger.info("Cached DigiKey token expired, refresh needed.")
                 return True # Allow proceeding, refresh will be attempted
            else:
                 logger.warning("Cached DigiKey token invalid or missing refresh token. Re-authentication required.")
                 self.digikey_token_data = None
                 return False
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.error(f"Failed to load or validate cached DigiKey token: {e}", exc_info=True)
            self.digikey_token_data = None
            try: os.remove(TOKEN_FILE); logger.info(f"Removed potentially corrupted token file: {TOKEN_FILE.name}")
            except OSError: pass
            return False

    def _schedule_digikey_refresh(self, delay_ms):
        """Schedules or reschedules the DigiKey token refresh."""
        # Cancel previous timer if exists
        if hasattr(self, '_digikey_refresh_after_id') and self._digikey_refresh_after_id:
            try: self.root.after_cancel(self._digikey_refresh_after_id)
            except: pass
        self._digikey_refresh_after_id = self.root.after(delay_ms, self.refresh_digikey_token)
        logger.debug(f"Scheduled DigiKey token refresh in {delay_ms / 1000:.0f}s")

    def get_digikey_token(self, force_reauth=False):
        """Gets a valid DigiKey token, handling expiry, refresh, and OAuth flow via HTTPS."""
        logger.debug(f"get_digikey_token called (force_reauth={force_reauth})")
        if not API_KEYS["DigiKey"]:
            self.update_status_threadsafe("DigiKey API keys not set.", "error")
            return None

        # 1. Check cache / existing token data
        if not force_reauth and self.digikey_token_data:
            expires_at = self.digikey_token_data.get('expires_at', 0)
            if time.time() < expires_at:
                logger.debug("Using valid cached DigiKey token.")
                return self.digikey_token_data['access_token']
            else: # Token expired, try refresh
                logger.info("Cached DigiKey token expired. Attempting refresh...")
                if self.refresh_digikey_token(): # refresh_digikey_token handles its own logging/status
                    return self.digikey_token_data['access_token']
                else:
                    logger.warning("DigiKey token refresh failed. Forcing full re-authentication.")
                    # Fall through to full OAuth flow

        # 2. Full OAuth Flow (if no token, refresh failed, or forced)
        logger.info("Starting full DigiKey OAuth2 flow (HTTPS)...")
        self.update_status_threadsafe("DigiKey auth required: Check browser", "warning")

        # Ensure SSL certificate exists
        if not CERT_FILE.exists():
            err_msg = f"SSL Certificate file not found: {CERT_FILE.name}\n\nPlease generate it using OpenSSL: \nopenssl req -new -x509 -keyout {CERT_FILE.name} -out {CERT_FILE.name} -days 365 -nodes"
            logger.error(err_msg)
            self.update_status_threadsafe(f"SSL Error: {CERT_FILE.name} not found.", "error")
            # Can't use messagebox directly from non-main thread if called by background task
            self.root.after(0, messagebox.showerror, "SSL Error", err_msg)
            return None

        redirect_uri = "https://localhost:8000" # Use a common HTTPS port
        auth_port = 8000
        auth_url = f"https://api.digikey.com/v1/oauth2/authorize?response_type=code&client_id={DIGIKEY_CLIENT_ID}&redirect_uri={urllib.parse.quote(redirect_uri)}"

        logger.debug(f"Opening browser to: {auth_url}")
        try:
            webbrowser.open(auth_url)
        except Exception as e:
            logger.error(f"Failed to open browser for OAuth: {e}")
            self.update_status_threadsafe(f"Browser Error: {e}", "error")
            return None

        # Start HTTPS server to catch redirect
        server = None
        auth_code = None
        server_start_time = time.time()

        try:
            logger.info(f"Attempting to bind HTTPS server to localhost:{auth_port}")
            server_address = ('localhost', auth_port)
            httpd = HTTPServer(server_address, OAuthHandler)
            httpd.auth_code = None # Attribute to store the received code
            httpd.timeout = 300 # 5 minute timeout for user interaction
            httpd.app_instance = self # Pass app instance to handler for status updates

            # Create SSL context
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(certfile=str(CERT_FILE)) # Assumes key is in the same file
            httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
            server = httpd # Assign the wrapped server

            logger.info(f"Waiting for DigiKey OAuth callback on {redirect_uri}...")
            server.handle_request() # Wait for one HTTPS GET request
            auth_code = getattr(server, 'auth_code', None) # Retrieve code set by handler
            logger.info(f"Local HTTPS server finished after {time.time() - server_start_time:.1f}s.")

        except ssl.SSLError as e:
            err_msg = f"SSL Error creating/binding HTTPS server: {e}. Check certificate/permissions."
            logger.error(err_msg, exc_info=True)
            self.update_status_threadsafe(f"SSL Error: {e}", "error")
            self.root.after(0, messagebox.showerror, "SSL Error", err_msg)
            return None
        except OSError as e:
            err_msg = f"OAuth Error: Port {auth_port} likely in use. Close other apps using it. ({e})"
            logger.error(err_msg)
            self.update_status_threadsafe(f"Port Error: {e}", "error")
            self.root.after(0, messagebox.showerror, "OAuth Error", err_msg)
            return None
        except Exception as e:
            err_msg = f"OAuth callback server error: {e}"
            logger.error(err_msg, exc_info=True)
            self.update_status_threadsafe(f"OAuth Server Error: {e}", "error")
            return None
        finally:
            if server:
                 # Close server socket cleanly in a separate thread to avoid blocking
                 threading.Thread(target=server.server_close, daemon=True).start()
                 logger.debug("OAuth server close requested.")


        # 3. Exchange Code for Token
        if auth_code:
            logger.info("Attempting to exchange authorization code for token...")
            token_exchange_start_time = time.time()
            try:
                token_url = "https://api.digikey.com/v1/oauth2/token"
                payload = {
                    'client_id': DIGIKEY_CLIENT_ID,
                    'client_secret': DIGIKEY_CLIENT_SECRET,
                    'grant_type': 'authorization_code',
                    'code': auth_code,
                    'redirect_uri': redirect_uri # Must match exactly
                }
                response = requests.post(token_url, data=payload, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=API_TIMEOUT_SECONDS + 10)
                response.raise_for_status() # Check for HTTP errors
                token_data = response.json()
                logger.info(f"Token exchange successful ({time.time() - token_exchange_start_time:.1f}s).")

                expires_in = token_data.get('expires_in', 1800) # Default 30 mins
                token_data['expires_at'] = time.time() + expires_in
                self.digikey_token_data = token_data

                # Cache the new token data
                try:
                    with open(TOKEN_FILE, 'w') as f: json.dump(self.digikey_token_data, f, indent=2)
                    logger.info("DigiKey token successfully cached.")
                except IOError as e:
                    logger.error(f"Failed to cache DigiKey token: {e}")
                    self.update_status_threadsafe("Error caching token", "warning")

                if self.digikey_token_data:
                 # Try to get limits from the token exchange response headers
                 self.digikey_token_data["rate_limit_remaining"] = response.headers.get('X-RateLimit-Remaining', 'NA') # response is from requests.post
                 self.digikey_token_data["rate_limit"] = response.headers.get('X-RateLimit-Limit', 'NA')
                 # Schedule the GUI update
                 self.root.after(0, self.update_rate_limit_display)

                self.update_status_threadsafe("DigiKey authentication successful.", "success")
                self.root.after(0, self.update_rate_limit_display) # Update limits in GUI thread
                self._schedule_digikey_refresh(int((expires_in - 300) * 1000)) # Schedule refresh
                return self.digikey_token_data.get('access_token')

            except requests.RequestException as e:
                error_detail = f"Status: {e.response.status_code}" if e.response else str(e)
                try: error_detail += f" - {e.response.json().get('error_description', e.response.text)}" if e.response else ""
                except: pass # Ignore JSON parsing errors on error response
                logger.error(f"DigiKey token exchange failed: {error_detail}", exc_info=True)
                self.update_status_threadsafe(f"DigiKey Token Error: {error_detail}", "error")
                self.root.after(0, messagebox.showerror, "DigiKey Auth Error", f"Failed to exchange code for token:\n{error_detail}")
                self.digikey_token_data = None # Clear invalid data
                return None
            except Exception as e:
                 logger.error(f"Unexpected error during token exchange/caching: {e}", exc_info=True)
                 self.update_status_threadsafe(f"Unexpected Auth Error: {e}", "error")
                 return None
        else: # Auth code not received
            logger.error("Did not receive auth_code from local HTTPS server. Auth timeout or failure.")
            # Status was likely updated by the handler already
            # self.update_status_threadsafe("DigiKey auth timed out or failed.", "error")
            self.root.after(0, messagebox.showerror, "DigiKey Auth Error", "Did not receive authorization code.\nPlease ensure you completed the login/authorization in your browser and accepted any security warnings for the local callback server (localhost:8443).")
            return None

    def refresh_digikey_token(self):
        """Refreshes the DigiKey access token using the refresh token. Returns True on success."""
        logger.info("Attempting to refresh DigiKey token...")
        if not self.digikey_token_data or 'refresh_token' not in self.digikey_token_data:
            logger.warning("No DigiKey refresh token available. Manual re-authentication needed.")
            self.update_status_threadsafe("DigiKey re-authentication required.", "warning")
            self.digikey_token_data = None
            try: os.remove(TOKEN_FILE)
            except OSError: pass
            return False

        try:
            token_url = "https://api.digikey.com/v1/oauth2/token"
            payload = {
                'client_id': DIGIKEY_CLIENT_ID,
                'client_secret': DIGIKEY_CLIENT_SECRET,
                'grant_type': 'refresh_token',
                'refresh_token': self.digikey_token_data['refresh_token']
            }
            response = requests.post(token_url, data=payload, headers={'Content-Type': 'application/x-www-form-urlencoded'}, timeout=API_TIMEOUT_SECONDS)
            response.raise_for_status()
            new_token_data = response.json()
            logger.info("DigiKey token refresh successful.")

            expires_in = new_token_data.get('expires_in', 1800)
            new_token_data['expires_at'] = time.time() + expires_in

            # Keep the same refresh token unless a new one is provided
            if 'refresh_token' not in new_token_data:
                new_token_data['refresh_token'] = self.digikey_token_data['refresh_token']
            elif new_token_data['refresh_token'] != self.digikey_token_data['refresh_token']:
                 logger.info("Received a new DigiKey refresh token.")

            self.digikey_token_data = new_token_data
            
            if self.digikey_token_data:
                 self.digikey_token_data["rate_limit_remaining"] = response.headers.get('X-RateLimit-Remaining', 'NA')
                 self.digikey_token_data["rate_limit"] = response.headers.get('X-RateLimit-Limit', 'NA')
                 # Schedule the GUI update
                 self.root.after(0, self.update_rate_limit_display)

            self.update_status_threadsafe("DigiKey token refreshed.", "info")
            self.root.after(0, self.update_rate_limit_display)
            self._schedule_digikey_refresh(int((expires_in - 300) * 1000))
            return True

        except requests.RequestException as e:
            error_detail = f"Status: {e.response.status_code}" if e.response else str(e)
            try: error_detail += f" - {e.response.json().get('error', '')}: {e.response.json().get('error_description', e.response.text)}" if e.response else ""
            except: pass
            logger.error(f"DigiKey token refresh failed: {error_detail}", exc_info=True)
            self.update_status_threadsafe(f"Token refresh failed: {error_detail}", "error")
            self.digikey_token_data = None # Clear invalid token
            try: os.remove(TOKEN_FILE)
            except OSError: pass
            # Potentially trigger full re-auth? Or just let next API call fail?
            return False
        except Exception as e:
             logger.error(f"Unexpected error during token refresh: {e}", exc_info=True)
             self.update_status_threadsafe(f"Unexpected token refresh error: {e}", "error")
             return False

    # --- Nexar Authentication ---
    def load_nexar_token_from_cache(self):
        """Loads Nexar token from cache if valid."""
        if not NEXAR_TOKEN_FILE.exists():
            logger.info("No Nexar token cache file found.")
            self.nexar_token_data = None
            return False
        try:
            with open(NEXAR_TOKEN_FILE, 'r') as f:
                token_data = json.load(f)
            expires_at = token_data.get('expires_at', 0)
            if time.time() < expires_at and token_data.get('access_token'):
                 logger.info("Valid Nexar token loaded from cache.")
                 self.nexar_token_data = token_data
                 return True
            else:
                 logger.info("Nexar token expired or invalid in cache.")
                 self.nexar_token_data = None
                 return False
        except (FileNotFoundError, json.JSONDecodeError, KeyError, Exception) as e:
            logger.error(f"Failed loading/validating Nexar token: {e}", exc_info=True)
            self.nexar_token_data = None
            try: os.remove(NEXAR_TOKEN_FILE); logger.info("Removed potentially corrupt Nexar token file.")
            except OSError: pass
            return False

    def get_nexar_token(self):
        """Gets a valid Nexar access token using Client Credentials grant."""
        if not API_KEYS["Octopart (Nexar)"]:
            logger.error("Nexar API keys not set.")
            return None

        # Check cache first
        if self.nexar_token_data:
             if time.time() < self.nexar_token_data.get('expires_at', 0):
                  logger.debug("Using valid cached Nexar token.")
                  return self.nexar_token_data.get('access_token')
             else:
                 logger.info("Cached Nexar token expired.")

        logger.info("Attempting to get new Nexar token...")
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        payload = {
            'grant_type': 'client_credentials',
            'client_id': NEXAR_CLIENT_ID,
            'client_secret': NEXAR_CLIENT_SECRET,
            'scope': 'supply.domain' # Ensure correct scope
        }
        try:
            response = requests.post(NEXAR_TOKEN_URL, headers=headers, data=payload, timeout=API_TIMEOUT_SECONDS)
            response.raise_for_status()
            token_data = response.json()
            access_token = token_data.get('access_token')

            if access_token:
                logger.info("Nexar token received successfully.")
                expires_in = token_data.get('expires_in', 3600) # Default 1 hour
                token_data['expires_at'] = time.time() + expires_in - 60 # Add buffer
                self.nexar_token_data = token_data
                try:
                    with open(NEXAR_TOKEN_FILE, 'w') as f: json.dump(self.nexar_token_data, f, indent=2)
                    logger.info("Nexar token cached.")
                except IOError as e:
                    logger.error(f"Failed to cache Nexar token: {e}")
                return access_token
            else:
                logger.error(f"Nexar token request succeeded but no access_token found. Response: {token_data}")
                return None

        except requests.RequestException as e:
            error_text = f"Status: {e.response.status_code}, Response: {e.response.text[:200]}" if e.response else str(e)
            logger.error(f"Failed to get Nexar token: {error_text}", exc_info=True)
            self.root.after(0, messagebox.showerror, "Nexar Auth Error", f"Failed to get Nexar token.\nCheck credentials and connectivity.\nError: {error_text}")
            return None
        except Exception as e:
             logger.error(f"Unexpected error getting Nexar token: {e}", exc_info=True)
             return None

    # --- Mouser Rate Limiting ---
    def load_mouser_request_counter(self):
        """Loads the Mouser API request counter and last reset date."""
        today_utc = datetime.now(timezone.utc).date()
        if MOUSER_COUNTER_FILE.exists():
            try:
                with open(MOUSER_COUNTER_FILE, 'r') as f: data = json.load(f)
                last_reset_iso = data.get('last_reset_date')
                if last_reset_iso:
                    last_reset_date = datetime.fromisoformat(last_reset_iso).date()
                    if last_reset_date == today_utc:
                        self.mouser_requests_today = data.get('requests', 0)
                        self.mouser_last_reset_date = last_reset_date
                        logger.info(f"Mouser counter loaded: {self.mouser_requests_today} requests for {today_utc}.")
                        return # Success
                    else: logger.info(f"Mouser counter date ({last_reset_date}) is old. Resetting for {today_utc}.")
                else: logger.warning("Mouser counter file missing reset date. Resetting.")
            except (json.JSONDecodeError, KeyError, ValueError, Exception) as e:
                logger.error(f"Failed to load Mouser counter file: {e}. Resetting count.")
        else:
            logger.info(f"Mouser counter file not found. Initializing.")

        # Reset if loaded data was old, file missing, or error occurred
        self.mouser_requests_today = 0
        self.mouser_last_reset_date = today_utc
        self.save_mouser_request_counter() # Save the reset state

    def save_mouser_request_counter(self):
        """Saves the current Mouser API request count and reset date."""
        if self.mouser_last_reset_date is None: return # Don't save if not initialized
        try:
            os.makedirs(CACHE_DIR, exist_ok=True) # Ensure directory exists
            with open(MOUSER_COUNTER_FILE, 'w') as f:
                json.dump({
                    'requests': self.mouser_requests_today,
                    'last_reset_date': self.mouser_last_reset_date.isoformat()
                }, f)
        except IOError as e:
            logger.error(f"Failed to save Mouser request counter: {e}")

    def check_and_wait_mouser_rate_limit(self):
        """Checks Mouser rate limit, waits if necessary. Returns False if interrupted."""
        if not API_KEYS["Mouser"]: return True # Skip if no key

        # Ensure counter is up-to-date for today
        today_utc = datetime.now(timezone.utc).date()
        if self.mouser_last_reset_date != today_utc:
            logger.info(f"Mouser counter reset for new day: {today_utc} (in check_and_wait)")
            self.mouser_requests_today = 0
            self.mouser_last_reset_date = today_utc
            self.save_mouser_request_counter()
            self.root.after(0, self.update_rate_limit_display)

        if self.mouser_requests_today >= self.mouser_daily_limit:
            now_utc = datetime.now(timezone.utc)
            # Calculate time until next UTC midnight
            next_reset_dt = datetime.combine(today_utc + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
            wait_seconds = (next_reset_dt - now_utc).total_seconds()

            if wait_seconds > 0:
                wait_hours = wait_seconds / 3600
                msg = f"Mouser API limit ({self.mouser_daily_limit}) reached. Waiting {wait_hours:.1f}h..."
                logger.warning(msg)
                self.update_status_threadsafe(msg, "warning")

                # Sleep in intervals, checking for cancellation
                wait_interval = 5 # seconds
                end_time = time.time() + wait_seconds
                while time.time() < end_time:
                     if not self.running_analysis: # Check cancellation flag
                          logger.info("Mouser wait interrupted by analysis cancellation.")
                          self.update_status_threadsafe("Mouser wait cancelled.", "warning")
                          return False # Indicate interruption

                     remaining = end_time - time.time()
                     wait_msg = f"Mouser limit wait: {remaining // 60:.0f}m {remaining % 60:.0f}s..."
                     self.update_status_threadsafe(wait_msg, "warning")
                     time.sleep(min(wait_interval, remaining + 0.1))

                # After waiting (if not interrupted)
                self.mouser_requests_today = 0 # Reset counter
                self.mouser_last_reset_date = datetime.now(timezone.utc).date() # Update date
                self.save_mouser_request_counter()
                self.update_status_threadsafe("Mouser API limit reset. Resuming...", "info")
                logger.info("Mouser API limit reset after waiting.")
                self.root.after(0, self.update_rate_limit_display)
                return True
            else: # wait_seconds <= 0 (already past midnight) - Reset immediately
                 logger.info("Mouser counter reset (already past midnight in check_and_wait).")
                 self.mouser_requests_today = 0
                 self.mouser_last_reset_date = today_utc
                 self.save_mouser_request_counter()
                 self.root.after(0, self.update_rate_limit_display)
                 return True
        else: # Limit not reached
            return True

    # --- Supplier API Wrappers ---
    def _make_api_request(self, method, url, **kwargs):
        """ General API request wrapper with timeout, error handling, and retry on 429. """
        retries = 1 # Number of retries specifically for 429
        last_exception = None

        for attempt in range(retries + 1):
             try:
                 response = requests.request(method, url, timeout=API_TIMEOUT_SECONDS, **kwargs)

                 # Handle Rate Limiting (429)
                 if response.status_code == 429:
                     if attempt < retries:
                         retry_after = int(response.headers.get('Retry-After', '10')) # Default 10s
                         logger.warning(f"Rate limit hit ({url}). Waiting {retry_after}s (Attempt {attempt+1}/{retries+1}).")
                         self.update_status_threadsafe(f"API rate limit hit, waiting {retry_after}s...", "warning")
                         time.sleep(retry_after)
                         continue # Retry the request
                     else:
                         logger.error(f"Rate limit hit ({url}) - Max retries exceeded.")
                         response.raise_for_status() # Raise exception after max retries

                 response.raise_for_status() # Raise HTTPError for other bad responses (4xx or 5xx)
                 return response # Success

             except requests.Timeout as e:
                 logger.error(f"API request timed out: {method} {url}")
                 last_exception = TimeoutError(f"API request timed out ({API_TIMEOUT_SECONDS}s)")
                 break # Don't retry on timeout
             except requests.ConnectionError as e:
                  logger.error(f"API connection error: {method} {url} - {e}")
                  last_exception = ConnectionError(f"API connection error: {e}")
                  break # Don't retry on connection error
             except requests.HTTPError as e:
                 logger.error(f"API HTTP error: {method} {url} - Status {e.response.status_code} - Response: {e.response.text[:200]}")
                 last_exception = e # Store the original HTTPError
                 # Decide whether to retry based on status code? (e.g., retry 5xx?)
                 # For now, break on any HTTP error except 429 handled above.
                 break
             except requests.RequestException as e:
                 logger.error(f"API request failed: {method} {url} - {e}", exc_info=True)
                 last_exception = RuntimeError(f"API request failed: {e}")
                 break
             except Exception as e: # Catch any other unexpected errors
                  logger.error(f"Unexpected error during API request: {method} {url} - {e}", exc_info=True)
                  last_exception = e
                  break

        # If loop finished without returning, raise the last captured exception
        raise last_exception from None

    def search_digikey(self, part_number, manufacturer=""):
        """Searches DigiKey for a part number. Returns processed dict or None."""
        if not API_KEYS["DigiKey"]: return None
        access_token = self.get_digikey_token()
        if not access_token: return None

        url = "https://api.digikey.com/products/v4/search/keyword"
        headers = {
            'Authorization': f"Bearer {access_token}",
            'X-DIGIKEY-Client-Id': DIGIKEY_CLIENT_ID,
            'X-DIGIKEY-Locale-Site': 'US',
            'X-DIGIKEY-Locale-Language': 'en',
            'X-DIGIKEY-Locale-Currency': 'USD',
            'Content-Type': 'application/json'
        }
        keywords = f"{manufacturer} {part_number}".strip() if manufacturer else part_number
        payload = {"Keywords": keywords, "Limit": 5, "Offset": 0, "FilterOptions": {"Family": None}} # Broader search initially

        try:
            response = self._make_api_request("POST", url, headers=headers, json=payload)
            data = response.json()

            # Update rate limit display
            if self.digikey_token_data: # Check if token data exists
                 self.digikey_token_data["rate_limit_remaining"] = response.headers.get('X-RateLimit-Remaining', 'NA')
                 self.digikey_token_data["rate_limit"] = response.headers.get('X-RateLimit-Limit', 'NA')
                 # Schedule the GUI update from the worker thread
                 self.root.after(0, self.update_rate_limit_display)

            products = data.get("Products", [])
            if not products:
                logger.debug(f"DigiKey: No results for '{keywords}'.")
                return None

            # --- Find best match ---
            best_match = None
            exact_pn_upper = part_number.upper()
            # Prioritize exact MPN match
            for p in products:
                 mpn = p.get("ManufacturerProductNumber", "").upper()
                 if mpn == exact_pn_upper:
                      best_match = p; break
            # If no exact MPN, take the first result (API relevance assumed)
            if not best_match:
                 best_match = products[0]
                 logger.debug(f"DigiKey: No exact MPN match for '{keywords}', using first result: {best_match.get('ManufacturerProductNumber')}")

            # --- Extract Data ---
            mfg_name = best_match.get("Manufacturer", {}).get("Name", "N/A")
            mfg_pn = best_match.get("ManufacturerProductNumber", "N/A")

            # Get stock/MOQ/packaging from variations if possible, fallback to product level
            stock = 0; min_order_qty = 0; package_type = "N/A"; digikey_pn = best_match.get("DigiKeyProductNumber", "N/A")
            variations = best_match.get("ProductVariations", [])
            active_variation = None
            if variations:
                # Find the variation matching the top-level DKPN if possible, or just take first active one
                for v in variations:
                    if v.get("DigiKeyProductNumber") == digikey_pn and v.get("VariationStatus", "").lower() == 'active':
                        active_variation = v; break
                if not active_variation: # Fallback to first active variation
                     active_variation = next((v for v in variations if v.get("VariationStatus", "").lower() == 'active'), None)

            if active_variation:
                stock = int(safe_float(active_variation.get("QuantityAvailable", 0), default=0))
                min_order_qty = int(safe_float(active_variation.get("MinimumOrderQuantity", 0), default=0))
                package_type = active_variation.get("PackageType", {}).get("Name", "N/A")
                digikey_pn = active_variation.get("DigiKeyProductNumber", digikey_pn) # Update DKPN if variation specific
                logger.debug(f"DigiKey: Using data from product variation for {mfg_pn}")
            else: # Fallback to product level data
                stock = int(safe_float(best_match.get("QuantityAvailable", 0), default=0))
                min_order_qty = int(safe_float(best_match.get("MinimumOrderQuantity", 0), default=0))
                logger.debug(f"DigiKey: No active variation found, using product level data for {mfg_pn}")


            lead_time_weeks_str = best_match.get("ManufacturerLeadWeeks") # Seems to be at product level
            lead_time_days = convert_lead_time_to_days(lead_time_weeks_str)

            pricing_raw = best_match.get("StandardPricing", [])
            pricing = [{"qty": int(p["BreakQuantity"]), "price": safe_float(p["UnitPrice"])} for p in pricing_raw if safe_float(p.get("UnitPrice")) is not None and int(p.get("BreakQuantity", 0)) > 0]
            pricing.sort(key=lambda x: x['qty']) # Ensure sorted

            status_str = best_match.get("ProductStatus",{}).get("Value", "").lower()

            result = {
                "Source": "DigiKey", "SourcePartNumber": digikey_pn,
                "ManufacturerPartNumber": mfg_pn, "Manufacturer": mfg_name,
                "Description": best_match.get("Description", {}).get("Value", "N/A"),
                "Stock": stock, "LeadTimeDays": lead_time_days,
                "MinOrderQty": min_order_qty, "Packaging": package_type,
                "Pricing": pricing,
                "CountryOfOrigin": best_match.get("Classifications", {}).get("CountryOfOrigin", "N/A"),
                "TariffCode": best_match.get("Classifications", {}).get("HtsusCode", "N/A"),
                "NormallyStocking": best_match.get("Parameters", {}).get("IsNormallyStocking", False), # Check Parameters field
                "Discontinued": status_str == 'discontinued',
                "EndOfLife": status_str in ['obsolete', 'last time buy', 'not recommended for new designs', 'nrnd'],
                "DatasheetUrl": best_match.get("DatasheetUrl", "N/A"),
                "ApiTimestamp": datetime.now(timezone.utc).isoformat(timespec='seconds'),
            }
            return result

        except requests.HTTPError as e:
            # Handle specific errors like 401 for token issues
            if e.response is not None and e.response.status_code == 401:
                logger.error("DigiKey 401 Unauthorized. Token likely invalid. Re-authentication required.", exc_info=False)
                self.digikey_token_data = None # Clear bad token
                self.root.after(0, self.load_digikey_token_from_cache) # Try reload/prompt on next action
            elif e.response is not None and e.response.status_code == 404:
                 logger.debug(f"DigiKey 404 Not Found for {part_number}.") # Less alarming
            else:
                 logger.error(f"DigiKey API HTTP Error for {part_number}: {e}", exc_info=True)
            return None
        except (TimeoutError, ConnectionError, RuntimeError, Exception) as e:
            logger.error(f"DigiKey search failed for {part_number}: {e}", exc_info=True)
            return None

    def search_mouser(self, part_number, manufacturer=""):
        """Searches Mouser for a part number. Returns processed dict or None."""
        if not API_KEYS["Mouser"]: return None
        if not self.check_and_wait_mouser_rate_limit(): return None # Check limit, wait if needed

        url = "https://api.mouser.com/api/v1/search/partnumber" # Use Part Number Search API v1
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        params = {'apiKey': MOUSER_API_KEY}
        body = {'SearchByPartRequest': {'mouserPartNumber': part_number, 'partSearchOptions': 'RohsAndReach'}} # Search by MPN

        logger.debug(f"Mouser Searching (PartNumber API) for: {part_number}")

        try:
            response = self._make_api_request("POST", url, headers=headers, params=params, json=body)
            raw_response_text = response.text # Get text before potential json error

            # Increment count *after* successful request OR non-401/429 error
            self.mouser_requests_today += 1
            self.save_mouser_request_counter()
            self.root.after(0, self.update_rate_limit_display) # Schedule GUI update

            try: data = response.json()
            except json.JSONDecodeError as json_err:
                 logger.error(f"Mouser JSON Decode Error for {part_number}: {json_err}. Response: {raw_response_text[:500]}")
                 return None

            if 'Errors' in data and data['Errors']:
                err_msg = data['Errors'][0].get('Message', 'Unknown Mouser Error') if data['Errors'] else 'Unknown'
                if "not found" in err_msg.lower(): logger.debug(f"Mouser: Part {part_number} not found via PartNumber API.")
                else: logger.error(f"Mouser API Error for {part_number}: {err_msg}")
                return None

            parts = data.get('SearchResults', {}).get('Parts', [])
            if not parts:
                logger.debug(f"Mouser: No parts found for '{part_number}' via PartNumber API.")
                return None

            # API returns single best match usually for PartNumber search
            best_match = parts[0] if isinstance(parts, list) and parts else None
            if not isinstance(best_match, dict):
                 logger.warning(f"Mouser: Unexpected part data format for {part_number}: {best_match}")
                 return None

            # --- Extract Data ---
            lead_time_str = best_match.get('LeadTime') # E.g., "10 Weeks", "In Stock"
            lead_time_days = convert_lead_time_to_days(lead_time_str)

            pricing_raw = best_match.get('PriceBreaks', [])
            pricing = [{"qty": int(p["Quantity"]), "price": safe_float(p["Price"].replace('$',''))} for p in pricing_raw if isinstance(p, dict) and safe_float(p.get("Price")) is not None and int(p.get("Quantity", 0)) > 0]
            pricing.sort(key=lambda x: x['qty'])

            lifecycle_status = best_match.get('LifecycleStatus', '') or "" # Ensure string

            result = {
                "Source": "Mouser",
                "SourcePartNumber": best_match.get('MouserPartNumber', "N/A"),
                "ManufacturerPartNumber": best_match.get('ManufacturerPartNumber', "N/A"),
                "Manufacturer": best_match.get('Manufacturer', "N/A"),
                "Description": best_match.get('Description', "N/A"),
                "Stock": int(safe_float(best_match.get('AvailabilityInStock', 0), default=0)),
                "LeadTimeDays": lead_time_days,
                "MinOrderQty": int(safe_float(best_match.get('Min', 0), default=0)),
                "Packaging": best_match.get('Packaging', "N/A"),
                "Pricing": pricing,
                "CountryOfOrigin": best_match.get("CountryOfOrigin", "N/A"), # Often missing
                "TariffCode": "N/A", # Not provided by this API endpoint
                "NormallyStocking": True, # Assumption
                "Discontinued": "discontinued" in lifecycle_status.lower(),
                "EndOfLife": any(s in lifecycle_status.lower() for s in ["obsolete", "nrnd", "not recommended"]),
                "DatasheetUrl": best_match.get('DataSheetUrl', "N/A"),
                "ApiTimestamp": datetime.now(timezone.utc).isoformat(timespec='seconds'),
            }
            return result

        except requests.HTTPError as e:
             # Check specifically for 401 Unauthorized / 403 Forbidden
             if e.response is not None and e.response.status_code in [401, 403]:
                 logger.error(f"Mouser API Key Invalid or Unauthorized ({e.response.status_code}). Disabling.", exc_info=False)
                 API_KEYS["Mouser"] = False
                 self.root.after(0, lambda: self.api_status_labels["Mouser"].config(text="Mouser: Invalid Key", foreground="red"))
                 self.root.after(0, self.update_rate_limit_display)
             elif e.response is not None and e.response.status_code == 404:
                  logger.debug(f"Mouser 404 Not Found for {part_number}.")
             else:
                  logger.error(f"Mouser API HTTP Error for {part_number}: {e}", exc_info=True)
             # Decrement count if error wasn't rate limit related (429 handled in _make_api_request)
             if e.response is None or e.response.status_code != 429:
                  self.mouser_requests_today -= 1
                  self.save_mouser_request_counter()
                  self.root.after(0, self.update_rate_limit_display)
             return None
        except (TimeoutError, ConnectionError, RuntimeError, Exception) as e:
            logger.error(f"Mouser search failed for {part_number}: {e}", exc_info=True)
            # Decrement count as request likely failed before incrementing in try block
            self.mouser_requests_today -= 1
            self.save_mouser_request_counter()
            self.root.after(0, self.update_rate_limit_display)
            return None

    def search_octopart_nexar(self, part_number, manufacturer=""):
        """Searches Octopart/Nexar using GraphQL. Returns processed dict or None."""
        if not API_KEYS["Octopart (Nexar)"]: return None
        access_token = self.get_nexar_token()
        if not access_token: return None

        headers = { 'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
        # Search by MPN primarily
        search_term = part_number

        # Refined GraphQL Query - Requesting more potentially useful fields
        graphql_query = f"""
        query NexarPartSearch {{
          supSearchMpn(q: "{search_term}", limit: 1, country: "US", currency: "USD") {{
            hits
            results {{
              part {{
                mpn
                manufacturer {{ name }}
                shortDescription
                category {{ name path }}
                specs {{ attribute {{ shortname }} value }}
                bestDatasheet {{ url }}
                sellers(authorizedOnly: false, limit: 15) {{
                  company {{ name homepageUrl }}
                  isAuthorized
                  offers(limit: 5) {{
                    clickUrl
                    sku
                    inventoryLevel
                    moq
                    packaging
                    factoryLeadDays # Requested lead time field
                    updatedAt # See how fresh the data is
                    prices {{ quantity price currency }}
                  }}
                }}
              }}
            }}
          }}
        }}
        """
        logger.debug(f"Nexar GraphQL Query for MPN '{search_term}'")

        try:
            response = self._make_api_request("POST", NEXAR_API_URL, headers=headers, json={'query': graphql_query})
            data = response.json()

            if "errors" in data:
                logger.error(f"Nexar GraphQL Errors for '{search_term}': {data['errors']}")
                return None

            search_results = data.get("data", {}).get("supSearchMpn", {}).get("results", [])
            if not search_results:
                logger.info(f"Nexar: No results found via supSearchMpn for '{search_term}'.")
                return None

            part_data = search_results[0].get("part", {})
            if not part_data: return None

            # --- Select Best Offer ---
            # Iterate through sellers and their offers, prioritize stock, then authorized, then lowest MOQ
            potential_offers = []
            sellers = part_data.get("sellers", [])
            if sellers and isinstance(sellers, list):
                for seller in sellers:
                    seller_name = seller.get("company", {}).get("name", "Unknown Seller")
                    is_authorized = seller.get("isAuthorized", False)
                    offers = seller.get("offers", [])
                    if offers and isinstance(offers, list):
                        for offer in offers:
                            if isinstance(offer, dict):
                                offer['seller_name'] = seller_name
                                offer['is_authorized'] = is_authorized
                                potential_offers.append(offer)

            if not potential_offers:
                 logger.warning(f"Nexar: No valid offers found for {search_term} among sellers.")
                 best_offer_data = {} # Use empty dict
                 best_seller_name = "Nexar Aggregate"
            else:
                 # Sort: Stock (desc), Authorized (desc), MOQ (asc), Freshness (desc - optional)
                 potential_offers.sort(key=lambda x: (
                     -(x.get('inventoryLevel', 0) or 0), # Stock first (negative for desc)
                     -int(x.get('is_authorized', False)), # Authorized next (True=1, False=0, negative for desc)
                     x.get('moq', 0) or float('inf'), # MOQ third (asc)
                     # -(pd.to_datetime(x.get('updatedAt'), errors='coerce', utc=True).timestamp() or 0) # Optional: Sort by freshness
                 ))
                 best_offer_data = potential_offers[0]
                 best_seller_name = best_offer_data.get('seller_name', "Nexar Aggregate")
                 logger.debug(f"Nexar: Selected best offer for {search_term} from '{best_seller_name}' (Stock: {best_offer_data.get('inventoryLevel')}, Auth: {best_offer_data.get('is_authorized')}, MOQ: {best_offer_data.get('moq')})")

            # --- Extract Data from Best Offer & Part ---
            prices_raw = best_offer_data.get("prices", [])
            pricing = [{"qty": int(p["quantity"]), "price": safe_float(p["price"])} for p in prices_raw if isinstance(p, dict) and p.get("currency", "USD") == "USD" and safe_float(p.get("price")) is not None and int(p.get("quantity", 0)) > 0]
            pricing.sort(key=lambda x: x['qty'])

            # **Lead Time Handling**
            raw_lead_value = best_offer_data.get("factoryLeadDays")
            lead_time_days = safe_float(raw_lead_value, default=np.nan) # Treat number directly as days
            logger.debug(f"Nexar Raw 'factoryLeadDays': {raw_lead_value}, Converted: {lead_time_days}")

            mfg_name = part_data.get('manufacturer', {}).get('name', manufacturer or "N/A")
            mpn = part_data.get('mpn', part_number)

            result = {
                "Source": "Octopart (Nexar)",
                "SourcePartNumber": best_offer_data.get('sku', "N/A"),
                "ManufacturerPartNumber": mpn,
                "Manufacturer": mfg_name,
                "Description": part_data.get('shortDescription', "N/A"),
                "Stock": int(safe_float(best_offer_data.get('inventoryLevel', 0), default=0)),
                "LeadTimeDays": lead_time_days, # Use converted value
                "MinOrderQty": int(safe_float(best_offer_data.get('moq', 0), default=0)),
                "Packaging": best_offer_data.get('packaging', "N/A"),
                "Pricing": pricing,
                "CountryOfOrigin": "N/A", # Not directly available in this query structure easily
                "TariffCode": "N/A", # Not directly available
                "NormallyStocking": True, # Assumption
                "Discontinued": False, # Placeholder - Could infer from lifecycle spec if available
                "EndOfLife": False, # Placeholder - Could infer from lifecycle spec if available
                "DatasheetUrl": part_data.get('bestDatasheet', {}).get('url', 'N/A'),
                "ApiTimestamp": datetime.now(timezone.utc).isoformat(timespec='seconds'),
            }
            return result

        except requests.HTTPError as e:
             # Check specific status codes if needed (e.g., 401 for auth)
             logger.error(f"Nexar API HTTP Error for {part_number}: {e}", exc_info=True)
             return None
        except (TimeoutError, ConnectionError, RuntimeError, Exception) as e:
            logger.error(f"Nexar search failed for {part_number}: {e}", exc_info=True)
            return None


    def search_arrow(self, part_number, manufacturer=""):
        # This function assumes it's only called if API_KEYS["Arrow"] is True.
        # It should contain ONLY the logic for the REAL Arrow API call.

        # Safety check (should ideally not be needed if get_part_data_parallel works correctly)
        if not API_KEYS["Arrow"]:
             logger.error("search_arrow called unexpectedly without API key.")
             return None

        logger.warning(f"Real Arrow API search for {part_number} not implemented.")

        # <<< Placeholder for REAL Arrow API Logic >>>
        # TODO: Implement the actual API call using ARROW_API_KEY
        # Example structure:
        # try:
        #     # ... build URL, headers, payload for Arrow API ...
        #     # response = self._make_api_request(...)
        #     # data = response.json()
        #     # ... process data into standard result dict format ...
        #     # result = { ... fields ... }
        #     # return result
        # except Exception as e:
        #     logger.error(f"Real Arrow API search failed for {part_number}: {e}", exc_info=True)
        #     return None
        # <<< End Placeholder >>>

        return None # Return None until the real API call is implemented

    def search_avnet(self, part_number, manufacturer=""):
        # This function assumes it's only called if API_KEYS["Avnet"] is True.
        # It should contain ONLY the logic for the REAL Avnet API call.

        # Safety check (should ideally not be needed if get_part_data_parallel works correctly)
        if not API_KEYS["Avnet"]:
             logger.error("search_avnet called unexpectedly without API key.")
             return None

        logger.warning(f"Real Avnet API search for {part_number} not implemented.")

        # <<< Placeholder for REAL Avnet API Logic >>>
        # TODO: Implement the actual API call using AVNET_API_KEY
        # Example structure:
        # try:
        #     # ... build URL, headers, payload for Avnet API ...
        #     # response = self._make_api_request(...)
        #     # data = response.json()
        #     # ... process data into standard result dict format ...
        #     # result = { ... fields ... }
        #     # return result
        # except Exception as e:
        #     logger.error(f"Real Avnet API search failed for {part_number}: {e}", exc_info=True)
        #     return None
        # <<< End Placeholder >>>

        return None # Return None until the real API call is implemented

    # --- Core Analysis Logic ---
    def get_part_data_parallel(self, part_number, manufacturer):
        """Fetches data for a single part from all enabled/mocked suppliers in parallel."""
        part_number = str(part_number).strip()
        manufacturer = str(manufacturer).strip()
        if not part_number: return {}

        futures = {}
        results = {}
        # Ensure this maps to the actual search functions
        supplier_funcs = {
            "DigiKey": self.search_digikey,
            "Mouser": self.search_mouser,
            "Octopart (Nexar)": self.search_octopart_nexar,
            "Arrow": self.search_arrow, # Function exists but will only be called if key is TRUE
            "Avnet": self.search_avnet, # Function exists but will only be called if key is TRUE
        }

        # Use a ThreadPoolExecutor scoped to this function call
        with ThreadPoolExecutor(max_workers=MAX_API_WORKERS, thread_name_prefix="APIFetcher") as executor:
            # Submit tasks ONLY for suppliers where API keys are set to True
            for name, func in supplier_funcs.items():
                if API_KEYS.get(name, False): # <<<< This check is the gatekeeper
                    logger.debug(f"Submitting REAL API task for {name} - {part_number}")
                    futures[executor.submit(func, part_number, manufacturer)] = name
                else:
                    # If key is not set (False), simply skip submitting the task
                    logger.debug(f"Skipping API task for {name} (API key not set or False)")

            # Process results as they complete
            for future in as_completed(futures):
                supplier_name = futures[future]
                try:
                    result = future.result() # Get result from completed future
                    if result and isinstance(result, dict):
                        results[supplier_name] = result
                        logger.debug(f"Successfully processed REAL result from {supplier_name} for {part_number}")
                    elif result is None:
                         logger.debug(f"No REAL result returned from {supplier_name} for {part_number}")
                    else:
                         logger.warning(f"Unexpected REAL result type from {supplier_name} for {part_number}: {type(result)}")
                except Exception as e:
                    logger.error(f"Error fetching/processing REAL result from {supplier_name} for {part_number}: {e}", exc_info=False)

        # Return dictionary containing only results from suppliers with active keys
        return results

    def get_optimal_cost(self, qty_needed, pricing_breaks, min_order_qty=0, buy_up_threshold_pct=1.0):
        """
        Calculates the optimal UNIT and TOTAL cost for a given quantity, considering MOQ and price breaks.
        Includes logic to potentially buy more if the total cost is negligibly higher.

        Args:
            qty_needed (int): The exact number of units required for the build.
            pricing_breaks (list): List of dicts [{'qty': int, 'price': float}, ...].
            min_order_qty (int): Minimum order quantity from the supplier.
            buy_up_threshold_pct (float): Percentage (0-100) threshold for buying up.

        Returns:
            tuple: (optimal_unit_price, optimal_total_cost, actual_order_quantity, notes)
                   Returns (np.nan, np.nan, qty_needed, "Error") on failure.
        """
        notes = ""
        if not isinstance(qty_needed, (int, float)) or qty_needed <= 0: return np.nan, np.nan, qty_needed, "Invalid Qty Needed"
        if not isinstance(pricing_breaks, list): return np.nan, np.nan, qty_needed, "Invalid Pricing Data"

        # Clean and sort breaks
        try:
            valid_breaks = [{'qty': int(pb['qty']), 'price': safe_float(pb['price'])} for pb in pricing_breaks if isinstance(pb, dict) and 'qty' in pb and 'price' in pb and int(pb['qty']) > 0 and pd.notna(safe_float(pb['price'])) and safe_float(pb['price']) >= 0]
            if not valid_breaks: return np.nan, np.nan, qty_needed, "No Valid Price Breaks"
            pricing_breaks = sorted(valid_breaks, key=lambda x: x['qty'])
            min_order_qty = max(1, int(safe_float(min_order_qty, default=1))) # Ensure MOQ is at least 1
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error cleaning/sorting pricing breaks: {e}", exc_info=True)
            return np.nan, np.nan, qty_needed, "Pricing Data Error"

        # --- Core Logic ---
        # 1. Determine minimum order quantity considering both need and MOQ
        base_order_qty = max(int(qty_needed), min_order_qty)

        # 2. Find price for the base_order_qty
        base_unit_price = np.nan
        applicable_break = None
        for pb in pricing_breaks:
            if base_order_qty >= pb['qty']:
                applicable_break = pb
            else:
                break # Stop checking higher breaks
        if applicable_break:
             base_unit_price = applicable_break['price']
        elif pricing_breaks: # If base_order_qty is less than the first break qty
             applicable_break = pricing_breaks[0]
             base_unit_price = applicable_break['price']
             base_order_qty = max(base_order_qty, applicable_break['qty']) # Must order at least first break qty
             notes += f"MOQ adjusted to first break ({base_order_qty}). "
        else: # Should not happen if valid_breaks exist
             return np.nan, np.nan, qty_needed, "Cannot Determine Base Price"

        best_total_cost = base_unit_price * base_order_qty
        best_unit_price = base_unit_price
        actual_order_qty = base_order_qty
        logger.debug(f"get_optimal_cost({qty_needed=}): Base Qty={base_order_qty}, Base Unit Price={base_unit_price:.4f}, Base Total={best_total_cost:.4f}")

        # 3. Check higher price breaks for lower TOTAL cost
        for pb in pricing_breaks:
            break_qty = pb['qty']
            break_price = pb['price']
            # Only consider breaks with quantity >= base_order_qty needed
            if break_qty >= base_order_qty:
                total_cost_at_break = break_qty * break_price
                logger.debug(f"  Checking break: Qty={break_qty}, Price={break_price:.4f}, Total={total_cost_at_break:.4f}")

                # Condition 1: Is buying this break *significantly* cheaper?
                if total_cost_at_break < best_total_cost * (1.0 - (buy_up_threshold_pct / 100.0)):
                    logger.debug(f"    >> Lower total cost found. New Best: Total={total_cost_at_break:.4f}, Qty={break_qty}")
                    best_total_cost = total_cost_at_break
                    best_unit_price = break_price
                    actual_order_qty = break_qty
                    notes = f"Price break @ {break_qty} lower total cost. "

                # Condition 2: Is buying this break *negligibly* more expensive but gives more parts?
                # Check only if the current best quantity is *less* than this break's quantity
                elif actual_order_qty < break_qty and \
                     total_cost_at_break <= best_total_cost * (1.0 + (buy_up_threshold_pct / 100.0)):
                     # If cost is within threshold % OR exactly equal, prefer higher qty
                     logger.debug(f"    >> Negligible cost increase/same for more parts. New Best: Total={total_cost_at_break:.4f}, Qty={break_qty}")
                     best_total_cost = total_cost_at_break
                     best_unit_price = break_price
                     actual_order_qty = break_qty
                     notes = f"Bought up to {break_qty} for similar total cost. "


        logger.debug(f"Final Decision: Unit Price={best_unit_price}, Total Cost={best_total_cost}, Actual Order Qty={actual_order_qty}, Notes='{notes.strip()}'")
        return best_unit_price, best_total_cost, actual_order_qty, notes.strip()

    # --- Tariff Lookup ---
    def fetch_usitc_tariff_rate(self, hts_code):
        """Fetches tariff rate from USITC HTS Search. Caches results per run."""
        if not hts_code or pd.isna(hts_code): return None
        hts_clean = str(hts_code).strip().replace(".", "").replace(" ", "")
        if not hts_clean.isdigit(): return None # Basic validation

        # Use internal cache for this run
        if hts_clean in self._hts_cache:
             return self._hts_cache[hts_clean]

        rate = None
        try:
            # Using a known API endpoint structure (adjust if needed)
            search_url = f"https://hts.usitc.gov/rest/v1/search?query={hts_clean}"
            logger.debug(f"Fetching USITC tariff for HTS: {hts_clean} from {search_url}")
            # Use a library with better structure awareness if available, else parse JSON
            response = requests.get(search_url, timeout=10, headers={'Accept': 'application/json'})
            response.raise_for_status()
            data = response.json()

            # Parse the response structure (this might change based on USITC API updates)
            # Example: Assuming results are in a list, and we want 'general_rate'
            if isinstance(data, dict) and 'results' in data and isinstance(data['results'], list) and data['results']:
                # Find exact match or best result
                found_article = None
                for article in data['results']:
                    # Compare cleaned HTS numbers (removing dots/spaces)
                    api_hts = article.get('htsno', '').replace('.', '').replace(' ', '')
                    if api_hts == hts_clean:
                        found_article = article
                        break
                if not found_article: found_article = data['results'][0] # Fallback to first

                general_rate_str = found_article.get('general_rate')
                if general_rate_str and isinstance(general_rate_str, str):
                    # Extract numeric value, handling "%" and "Free"
                    if general_rate_str.lower() == 'free':
                         rate = 0.0
                    else:
                         rate_match = re.search(r'(\d+(\.\d+)?)', general_rate_str)
                         if rate_match:
                              rate_val = safe_float(rate_match.group(1))
                              if rate_val is not None: rate = rate_val / 100.0

            if rate is not None: logger.debug(f"USITC rate found for {hts_code}: {rate*100:.2f}%")
            else: logger.warning(f"No general tariff rate found for HTS {hts_code} in USITC response.")

        except requests.RequestException as e: logger.error(f"USITC request failed for HTS {hts_code}: {e}")
        except (json.JSONDecodeError, ValueError, KeyError, AttributeError) as e: logger.error(f"Error parsing USITC response for HTS {hts_code}: {e}")
        except Exception as e: logger.error(f"Unexpected error fetching USITC tariff for HTS {hts_code}: {e}", exc_info=True)

        self._hts_cache[hts_clean] = rate # Cache result (even if None)
        return rate

    def get_tariff_info(self, hts_code, country_of_origin, custom_tariff_rates):
        """Determines the applicable tariff rate (as fraction)."""
        base_tariff_rate = None
        source_info = "N/A"

        # 1. Custom Rates (Highest Priority)
        coo_clean = str(country_of_origin).strip() if country_of_origin else ""
        if coo_clean and coo_clean != "N/A":
             custom_rate = custom_tariff_rates.get(coo_clean) # Expects rate as fraction (e.g., 0.035)
             if custom_rate is not None:
                  base_tariff_rate = custom_rate
                  source_info = f"Custom ({coo_clean})"
                  return base_tariff_rate, source_info

        # 2. USITC Lookup (If HTS available)
        if base_tariff_rate is None and hts_code:
             fetched_rate = self.fetch_usitc_tariff_rate(hts_code)
             if fetched_rate is not None:
                  base_tariff_rate = fetched_rate
                  source_info = f"USITC ({hts_code})"
                  return base_tariff_rate, source_info

        # 3. Fallback Prediction based on COO (if COO known)
        if base_tariff_rate is None and coo_clean and coo_clean != "N/A":
             # Example predictive increases (adjust based on policy/data)
             predictive_increase = {
                 'China': 0.15, 'Mexico': 0.02, 'India': 0.08, 'Vietnam': 0.05,
                 'Taiwan': 0.03, 'Malaysia': 0.03, 'Japan': 0.01, 'Germany': 0.01,
                 'USA': 0.0, 'United States': 0.0,
             }
             # Default to global default rate if country not in prediction map
             country_rate = predictive_increase.get(coo_clean, DEFAULT_TARIFF_RATE)
             base_tariff_rate = country_rate
             source_info = f"Predicted ({coo_clean})"
             return base_tariff_rate, source_info

        # 4. Absolute Fallback (Default Rate)
        if base_tariff_rate is None:
             base_tariff_rate = DEFAULT_TARIFF_RATE
             source_info = "Default Rate"

        return base_tariff_rate, source_info

    # --- Strategy Export ---
    def export_strategy_gui(self, strategy_name):
        """Handles button click and exports the selected strategy to CSV."""
        logger.info(f"Exporting strategy: '{strategy_name}'")

        # Use the dedicated strategies_for_export dictionary
        if not self.strategies_for_export:
            messagebox.showerror("Export Error", "No strategy data available. Please run analysis first.")
            return
        strategy_dict = self.strategies_for_export.get(strategy_name)
        if not strategy_dict:
            messagebox.showerror("Export Error", f"Data for '{strategy_name}' strategy not found. Results might be incomplete.")
            return

        output_data = []
        # Define Header row
        output_header = [
            "BOM Part Number", "Manufacturer", "Manufacturer PN", "Qty Per Unit", "Total Qty Needed",
            "Chosen Source", "Source PN", "Unit Cost ($)", "Total Cost ($)", "Actual Qty Ordered", # Added Actual Qty Ordered
            "Lead Time (Days)", "Stock", "Notes/Score" # Combined Notes/Score
        ]
        output_data.append(output_header)

        parts_exported = 0
        for bom_pn, chosen_option in strategy_dict.items():
            if not isinstance(chosen_option, dict): continue # Skip invalid entries

            # Extract data using .get() for safety
            unit_cost = chosen_option.get('unit_cost', np.nan)
            total_cost = chosen_option.get('cost', np.nan)
            lead_time = chosen_option.get('lead_time', np.nan) # Use NaN instead of Inf for consistency

            # Format for CSV - handle NaN/None gracefully
            unit_cost_str = f"{unit_cost:.4f}" if pd.notna(unit_cost) else "N/A"
            total_cost_str = f"{total_cost:.2f}" if pd.notna(total_cost) else "N/A"
            lead_time_str = f"{lead_time:.0f}" if pd.notna(lead_time) and lead_time != np.inf else "N/A" # Handle Inf too
            actual_qty_str = str(chosen_option.get("actual_order_qty", 'N/A'))
            notes = str(chosen_option.get('notes', ''))
            score = str(chosen_option.get('optimized_strategy_score', ''))
            notes_score = f"{notes} {('Score: '+score) if score else ''}".strip()

            output_data.append([
                chosen_option.get("bom_pn", "N/A"),
                chosen_option.get("Manufacturer", "N/A"),
                chosen_option.get("ManufacturerPartNumber", "N/A"),
                chosen_option.get("original_qty_per_unit", "N/A"),
                chosen_option.get("total_qty_needed", "N/A"),
                chosen_option.get("source", "N/A"),
                chosen_option.get("SourcePartNumber", "N/A"),
                unit_cost_str, total_cost_str, actual_qty_str, # Added actual_qty_str
                lead_time_str,
                chosen_option.get("stock", 0),
                notes_score # Combined Notes/Score
            ])
            parts_exported += 1

        if parts_exported == 0:
            messagebox.showinfo("Export Info", f"No valid part data found to export for the '{strategy_name}' strategy.")
            return

        # Ask user for filename
        default_filename = f"BOM_Strategy_{strategy_name.replace(' ', '_')}_{datetime.now():%Y%m%d_%H%M}.csv"
        filepath = filedialog.asksaveasfilename(
            title=f"Save {strategy_name} Strategy As", defaultextension=".csv",
            initialfile=default_filename, filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not filepath: return # User cancelled

        # Write to CSV
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerows(output_data)
            self.update_status_threadsafe(f"Exported '{strategy_name}' ({parts_exported} parts) to {Path(filepath).name}", "success")
            messagebox.showinfo("Export Successful", f"Successfully exported {parts_exported} parts for strategy '{strategy_name}' to:\n{filepath}")
        except IOError as e:
            logger.error(f"Failed to export strategy CSV: {e}", exc_info=True)
            messagebox.showerror("Export Error", f"Failed to write CSV file:\n{filepath}\n\nError: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during strategy export: {e}", exc_info=True)
            messagebox.showerror("Export Error", f"An unexpected error occurred during export:\n\n{e}")

    # --- Main Analysis Function (Per Part) ---
    def analyze_single_part(self, bom_part_number, bom_manufacturer, bom_qty_per_unit, config):
        """ Analyzes a single BOM line item. Returns GUI row(s), historical entries, and summary data. """
        # Get config values
        total_units = config['total_units']
        buy_up_threshold_pct = config['buy_up_threshold'] # Get threshold
        total_qty_needed = int(bom_qty_per_unit * total_units)

        # Fetch data from APIs
        part_results_by_supplier = self.get_part_data_parallel(bom_part_number, bom_manufacturer)

        gui_entry = {} # For the main treeview row
        historical_entries = [] # For historical CSV
        part_summary = { # For strategy calculation & export later
            "bom_pn": bom_part_number, "bom_mfg": bom_manufacturer,
            "original_qty_per_unit": bom_qty_per_unit,
            "total_qty_needed": total_qty_needed,
            "options": [], # Detailed options list
            "alternates": []
        }
        temp_supplier_data = {} # Intermediate storage if needed

        # --- Handle Case: No Suppliers Found ---
        if not part_results_by_supplier:
             gui_entry = {
                 "PartNumber": bom_part_number, "Manufacturer": bom_manufacturer or "N/A", "MfgPN": "NOT FOUND",
                 "QtyNeed": total_qty_needed, "Status": "Unknown", "Sources": 0, "StockAvail": "0",
                 "COO": "N/A", "RiskScore": "10.0", "TariffPct": "N/A", # Max risk if not found
                 "BestCostPer": "N/A", "BestTotalCost": "N/A", "ActualBuyQty": "N/A", "BestCostLT": "N/A", "BestCostSrc": "N/A",
                 "FastestLT": "N/A", "FastestCost": "N/A", "FastestLTSrc": "N/A",
                 "Alternates": "No", "Notes": "No suppliers found",
                 "RiskFactors": {'Sourcing': 10, 'Stock': 10, 'LeadTime': 10, 'Lifecycle': 5, 'Geographic': 3} # High risk
             }
             part_summary["options"] = []
             return [gui_entry], historical_entries, part_summary # Return list containing one entry

        # --- Process Supplier Data & Create Options List ---
        all_options_data = [] # Temp list for part_summary['options']
        consolidated_mfg = bom_manufacturer or "N/A"; mfg_source = "BOM"
        consolidated_mpn = bom_part_number; mpn_source = "BOM"

        for source, data in part_results_by_supplier.items():
            if not isinstance(data, dict): continue

            # Consolidate Mfg/MPN - Prefer non-N/A API result over BOM input
            api_mfg = data.get('Manufacturer')
            api_mpn = data.get('ManufacturerPartNumber')
            if api_mfg and api_mfg != "N/A" and consolidated_mfg == "N/A":
                 consolidated_mfg = api_mfg; mfg_source = source
            if api_mpn and api_mpn != "N/A" and consolidated_mpn == "N/A":
                 consolidated_mpn = api_mpn; mpn_source = source
            # If BOM had a value, but API differs, log it? For now, first valid API wins if BOM was N/A
            # Or implement logic to find *most common* non-N/A Mfg/MPN across sources

            # Calculate optimal cost for THIS supplier's pricing
            unit_cost, total_cost, actual_order_qty, cost_notes = self.get_optimal_cost(
                total_qty_needed, data.get('Pricing', []), data.get('MinOrderQty', 0), buy_up_threshold_pct
            )

            # Create option dict for this supplier
            option_dict = {
                "source": source,
                "cost": total_cost if pd.notna(total_cost) else np.inf, # Use inf for invalid cost
                "lead_time": data.get('LeadTimeDays', np.inf), # Use inf for invalid LT
                "stock": data.get('Stock', 0),
                "unit_cost": unit_cost,
                "actual_order_qty": actual_order_qty,
                "moq": data.get('MinOrderQty', 0),
                "discontinued": data.get('Discontinued', False),
                "eol": data.get('EndOfLife', False),
                'bom_pn': bom_part_number, # Keep original BOM PN link
                'original_qty_per_unit': bom_qty_per_unit,
                'total_qty_needed': total_qty_needed,
                'Manufacturer': data.get('Manufacturer', 'N/A'), # Store source-specific Mfg/MPN
                'ManufacturerPartNumber': data.get('ManufacturerPartNumber', 'N/A'),
                'SourcePartNumber': data.get('SourcePartNumber', 'N/A'),
                'TariffCode': data.get('TariffCode'),
                'CountryOfOrigin': data.get('CountryOfOrigin'),
                'ApiTimestamp': data.get('ApiTimestamp'),
                'tariff_rate': None, # Calculated later based on consolidated COO/HTS
                'stock_prob': 0.0, # Calculated later
                'notes': cost_notes, # Add notes from cost calculation (e.g., buy-up reason)
            }
            all_options_data.append(option_dict)

        part_summary["options"] = all_options_data

        # --- Consolidate Part Info (Mfg, MPN, COO, HTS, Alternates) ---
        # Use the potentially updated consolidated Mfg/MPN
        logger.debug(f"Consolidated Mfg for {bom_part_number}: '{consolidated_mfg}' (Source: {mfg_source})")
        logger.debug(f"Consolidated MPN for {bom_part_number}: '{consolidated_mpn}' (Source: {mpn_source})")
        final_component_name = f"{consolidated_mfg} {consolidated_mpn}".strip()

        # Get alternates based on consolidated MPN
        substitutes = self.get_digikey_substitutions(consolidated_mpn) if API_KEYS["DigiKey"] else []
        part_summary['alternates'] = substitutes

        # Consolidate COO/HTS (find first valid, prefer non-N/A, non-aggregate)
        consolidated_coo = "N/A"; coo_source_log = "None Found"
        consolidated_hts = "N/A"; hts_source_log = "None Found"

        for option in all_options_data:
            api_coo = option.get('CountryOfOrigin')
            if api_coo and isinstance(api_coo, str) and api_coo.strip().upper() not in ["N/A", "", "UNKNOWN", "AGGREGATE"]:
                consolidated_coo = api_coo.strip(); coo_source_log = f"API ({option['source']})"; break
        if consolidated_coo == "N/A": # If no COO, try inferring from first valid HTS
            for option in all_options_data:
                 api_hts = option.get('TariffCode')
                 if api_hts and isinstance(api_hts, str) and api_hts.strip().lower() not in ['n/a', '']:
                      consolidated_hts = api_hts.strip(); hts_source_log = f"API ({option['source']})"
                      inferred_coo = self.infer_coo_from_hts(consolidated_hts)
                      if inferred_coo != "Unknown":
                           consolidated_coo = inferred_coo; coo_source_log = f"Inferred from HTS ({consolidated_hts} via {option['source']})"
                      break # Stop after finding first valid HTS
        elif consolidated_hts == "N/A": # If COO was found, still try to find *any* HTS for display
             for option in all_options_data:
                  api_hts = option.get('TariffCode')
                  if api_hts and isinstance(api_hts, str) and api_hts.strip().lower() not in ['n/a', '']:
                       consolidated_hts = api_hts.strip(); hts_source_log = f"API ({option['source']})"; break

        logger.debug(f"Consolidated COO for {bom_part_number}: {consolidated_coo} (Source: {coo_source_log})")
        logger.debug(f"Consolidated HTS for {bom_part_number}: {consolidated_hts} (Source: {hts_source_log})")

        # --- Calculate Consolidated Metrics & Final GUI Row ---
        valid_options = [opt for opt in all_options_data if opt['cost'] != np.inf and opt['lead_time'] != np.inf]
        if not valid_options: valid_options = all_options_data # Fallback if no options have both cost & LT

        # Calculate tariff ONCE using consolidated info
        consolidated_tariff_rate, tariff_source_info = self.get_tariff_info(consolidated_hts, consolidated_coo, config.get('custom_tariff_rates', {}))
        stock_prob = self.calculate_stock_probability_simple(all_options_data, total_qty_needed)

        total_stock_available = 0
        min_lead_no_stock = np.inf
        lifecycle_notes = set()
        has_stock_gap = False # Assume no gap initially

        for option in all_options_data: # Iterate original options to build historical/check lifecycle
             option['tariff_rate'] = consolidated_tariff_rate # Store consolidated rate in each option
             option['stock_prob'] = stock_prob # Store calculated probability

             total_stock_available += option.get('stock', 0)
             if option.get('discontinued'): lifecycle_notes.add("DISC")
             if option.get('eol'): lifecycle_notes.add("EOL")
             if option.get('stock', 0) < total_qty_needed and option['lead_time'] != np.inf:
                  min_lead_no_stock = min(min_lead_no_stock, option['lead_time'])

             # --- Historical Logging ---
             historical_entries.append([
                 final_component_name, # Use consolidated name
                 option.get('Manufacturer', 'N/A'), option.get('ManufacturerPartNumber', 'N/A'),
                 option.get('source'),
                 option.get('lead_time', np.nan), # Use NaN for CSV if Inf
                 option.get('unit_cost', np.nan), # Use the unit cost associated with the optimal total cost
                 option.get('stock', 0),
                 stock_prob,
                 option.get('ApiTimestamp', datetime.now(timezone.utc).isoformat(timespec='seconds'))
             ])

        # Determine if there's a stock gap across ALL suppliers
        has_stock_gap = not any(opt.get('stock', 0) >= total_qty_needed for opt in all_options_data)

        # --- Calculate Risk Factors & Score ---
        risk_factors = {'Sourcing': 0, 'Stock': 0, 'LeadTime': 0, 'Lifecycle': 0, 'Geographic': 0}
        num_valid_sources = len(valid_options) # Count options with valid cost/LT primarily
        if num_valid_sources <= 1: risk_factors['Sourcing'] = 10
        elif num_valid_sources == 2: risk_factors['Sourcing'] = 5
        else: risk_factors['Sourcing'] = 1

        if has_stock_gap: risk_factors['Stock'] = 8
        elif total_stock_available < 1.5 * total_qty_needed: risk_factors['Stock'] = 4
        else: risk_factors['Stock'] = 0

        if has_stock_gap: # Only consider lead time risk if stock is insufficient
            if min_lead_no_stock == np.inf: risk_factors['LeadTime'] = 9 # No stock and no lead time info = high risk
            elif min_lead_no_stock > 90: risk_factors['LeadTime'] = 7
            elif min_lead_no_stock > 45: risk_factors['LeadTime'] = 4
            else: risk_factors['LeadTime'] = 1
        else: risk_factors['LeadTime'] = 0 # No LT risk if stock available

        if "EOL" in lifecycle_notes or "DISC" in lifecycle_notes: risk_factors['Lifecycle'] = 10
        else: risk_factors['Lifecycle'] = 0 # Active

        risk_factors['Geographic'] = self.GEO_RISK_TIERS.get(consolidated_coo, self.GEO_RISK_TIERS["_DEFAULT_"])

        overall_risk_score = sum(risk_factors[factor] * self.RISK_WEIGHTS[factor] for factor in self.RISK_WEIGHTS)
        overall_risk_score = round(max(0, min(10, overall_risk_score)), 1)

        # --- Find Best Cost & Fastest Options from *valid* options ---
        best_cost_option = min(valid_options, key=lambda x: (x.get('cost', np.inf), x.get('lead_time', np.inf))) if valid_options else None
        fastest_lt_option = min(valid_options, key=lambda x: (x.get('lead_time', np.inf), x.get('cost', np.inf))) if valid_options else None

        # --- Determine Status and Notes for GUI ---
        status = "Active"; notes_str = ""
        if "EOL" in lifecycle_notes: status = "EOL"
        elif "DISC" in lifecycle_notes: status = "Discontinued"
        # Add lifecycle notes even if status is active (e.g., NRND) - not currently captured, add if needed
        if has_stock_gap: notes_str = "Stock Gap"
        # Add buy-up notes from the best cost option
        best_cost_notes = best_cost_option.get('notes', '') if best_cost_option else ''
        if best_cost_notes: notes_str = f"{notes_str}; {best_cost_notes}".strip('; ')

        # --- Create Final GUI Row Data ---
        gui_entry = {
            "PartNumber": bom_part_number,
            "Manufacturer": consolidated_mfg,
            "MfgPN": consolidated_mpn,
            "QtyNeed": total_qty_needed,
            "Status": status,
            "Sources": f"{len(all_options_data)}", # Show total sources found initially
            "StockAvail": f"{total_stock_available}",
            "COO": consolidated_coo,
            "RiskScore": f"{overall_risk_score:.1f}" if pd.notna(overall_risk_score) else "N/A",
            "TariffPct": f"{consolidated_tariff_rate * 100:.1f}%" if pd.notna(consolidated_tariff_rate) else "N/A",
            # Best Cost Info
            "BestCostPer": f"{best_cost_option['unit_cost']:.4f}" if best_cost_option and pd.notna(best_cost_option.get('unit_cost')) else "N/A",
            "BestTotalCost": f"{best_cost_option['cost']:.2f}" if best_cost_option and best_cost_option.get('cost') != np.inf else "N/A",
            "ActualBuyQty": f"{best_cost_option['actual_order_qty']}" if best_cost_option else "N/A",
            "BestCostLT": f"{best_cost_option['lead_time']:.0f}" if best_cost_option and best_cost_option.get('lead_time') != np.inf else "N/A",
            "BestCostSrc": best_cost_option['source'] if best_cost_option else "N/A",
            # Fastest Lead Time Info
            "FastestLT": f"{fastest_lt_option['lead_time']:.0f}" if fastest_lt_option and fastest_lt_option.get('lead_time') != np.inf else "N/A",
            "FastestCost": f"{fastest_lt_option['cost']:.2f}" if fastest_lt_option and fastest_lt_option.get('cost') != np.inf else "N/A",
            "FastestLTSrc": fastest_lt_option['source'] if fastest_lt_option else "N/A",
            "Alternates": "Yes" if substitutes else "No",
            "Notes": notes_str,
            "RiskFactors": risk_factors # Store for potential detailed view/tooltip later
        }

        return [gui_entry], historical_entries, part_summary


    # --- Main Analysis Execution Flow ---
    def validate_and_run_analysis(self):
        """Validates inputs and starts the analysis thread."""
        if self.running_analysis:
            messagebox.showwarning("Busy", "Analysis is already in progress.")
            return
        if not self.validate_inputs(): # Validation updates GUI label
            messagebox.showerror("Invalid Config", "Please fix the configuration errors (marked red) before running analysis.")
            return
        if self.bom_df is None or self.bom_df.empty:
            messagebox.showerror("No BOM", "Please load a valid BOM file first.")
            return

        # Get config values safely
        try:
             config = {
                 "total_units": int(safe_float(self.config_vars["total_units"].get())),
                 "max_premium": safe_float(self.config_vars["max_premium"].get(), default=15.0), # Store as %, use directly
                 "target_lead_time_days": int(safe_float(self.config_vars["target_lead_time_days"].get(), default=56)), # Finish this line
                 "cost_weight": safe_float(self.config_vars["cost_weight"].get(), default=0.5),
                 "lead_time_weight": safe_float(self.config_vars["lead_time_weight"].get(), default=0.5),
                 "buy_up_threshold": safe_float(self.config_vars["buy_up_threshold"].get(), default=1.0), # Store as %
                 "custom_tariff_rates": {} # Initialize empty
             }
             # Process tariffs safely (store as fraction 0-1)
             for country, entry in self.tariff_entries.items():
                 rate_str = entry.get()
                 if rate_str:
                      rate = safe_float(rate_str)
                      if rate is not None and rate >= 0:
                           config["custom_tariff_rates"][country] = rate / 100.0
                      else: # Invalid rate entered
                           raise ValueError(f"Invalid tariff rate for {country}: '{rate_str}'")

        except (ValueError, KeyError, AttributeError) as e: # Catch specific errors during config parsing
            messagebox.showerror("Config Error", f"Invalid value in configuration: {e}")
            logger.error(f"Configuration error reading values: {e}", exc_info=True)
            return
        except Exception as e:
             messagebox.showerror("Config Error", f"Unexpected error reading configuration: {e}")
             logger.error("Unexpected config error", exc_info=True)
             return

        # Reset results before starting
        self.analysis_results = {}
        self.strategies_for_export = {}
        self.clear_treeview(self.tree)
        self.clear_treeview(self.analysis_table)
        self.ai_summary_text.configure(state='normal'); self.ai_summary_text.delete(1.0, tk.END); self.ai_summary_text.insert(tk.END, "Analysis running..."); self.ai_summary_text.configure(state='disabled')
        self.update_export_buttons_state() # Ensure disabled

        # Start the analysis in a separate thread
        self.running_analysis = True
        self.update_analysis_controls_state(True) # Disable buttons
        self.update_status_threadsafe("Starting analysis...", "info")
        logger.info("Submitting analysis task to thread pool.")
        self.thread_pool.submit(self.run_analysis_thread, config) # Pass validated config dict

    def run_analysis_thread(self, config):
        """ The function that runs in a separate thread to perform analysis. """
        start_time = time.time()
        try:
            # Initial check if BOM is still valid (unlikely to change, but good practice)
            if self.bom_df is None or self.bom_df.empty:
                 self.update_status_threadsafe("Error: BOM became invalid before analysis start.", "error")
                 return

            # Re-check DigiKey token right before heavy API usage
            if API_KEYS["DigiKey"]:
                 self.update_status_threadsafe("Checking DigiKey token...", "info")
                 access_token = self.get_digikey_token() # This might block/trigger OAuth
                 if not access_token:
                      logger.error("Analysis cancelled: Failed to obtain DigiKey token.")
                      self.update_status_threadsafe("Analysis Cancelled: DigiKey Auth Failed.", "error")
                      self.root.after(0, messagebox.showerror, "Auth Error", "Could not get/refresh DigiKey token. Analysis cancelled.")
                      return # Exit thread gracefully
                 else:
                      self.update_status_threadsafe("API tokens check OK.", "info")

            # Clear temporary caches
            self._hts_cache = {} # Clear HTS cache for this run

            # Prepare lists for results
            all_analysis_entries = []
            all_historical_entries = []
            all_part_summaries = []
            total_parts_in_bom = len(self.bom_df)
            self.update_progress_threadsafe(0, total_parts_in_bom, "Initializing...")

            # --- Process each BOM line ---
            for i, row in self.bom_df.iterrows():
                if not self.running_analysis: # Check for cancellation signal
                     logger.info("Analysis run cancelled by user request.")
                     self.update_status_threadsafe("Analysis Cancelled.", "warning")
                     return # Exit thread

                bom_pn = row['Part Number']
                bom_mfg = row.get('Manufacturer', '')
                bom_qty_per = row['Quantity'] # Already validated during load

                # Update progress in GUI thread
                self.update_progress_threadsafe(i, total_parts_in_bom, f"Analyzing {bom_pn[:25]}...")

                # --- Call the core analysis function for the part ---
                part_gui_rows, part_hist_rows, part_summary = self.analyze_single_part(
                    bom_pn, bom_mfg, bom_qty_per, config # Pass qty_per_unit
                )

                # Append results
                all_analysis_entries.extend(part_gui_rows)
                all_historical_entries.extend(part_hist_rows)
                if part_summary and part_summary.get('options'): # Only add summaries if options were found
                    all_part_summaries.append(part_summary)

                # Brief sleep to allow GUI updates and prevent overwhelming API (optional)
                # time.sleep(0.01)

            # --- Post-Processing After Loop ---
            if not self.running_analysis: return # Final check before aggregation

            self.update_progress_threadsafe(total_parts_in_bom, total_parts_in_bom, "Aggregating results...")

            # Store results in the main dictionary
            self.analysis_results["config"] = config
            self.analysis_results["part_summaries"] = all_part_summaries
            self.analysis_results["gui_entries"] = all_analysis_entries # Store GUI rows

            # 1. Populate Main Treeview (schedule on GUI thread)
            self.root.after(0, self.populate_treeview, self.tree, all_analysis_entries)

            # 2. Save Historical Data (submit to thread pool to avoid blocking)
            if all_historical_entries:
                logger.info(f"Submitting append of {len(all_historical_entries)} rows to {HISTORICAL_DATA_FILE.name}")
                self.thread_pool.submit(append_to_csv, HISTORICAL_DATA_FILE, all_historical_entries)
                # Consider reloading historical data in background if needed immediately for predictions
                # self.historical_data_df = pd.concat([self.historical_data_df, pd.DataFrame(all_historical_entries, columns=self.hist_header)], ignore_index=True)

            # 3. Calculate Summary Metrics (this populates self.strategies_for_export)
            summary_metrics = self.calculate_summary_metrics(all_part_summaries, config)
            self.analysis_results["summary_metrics"] = summary_metrics # Store list of tuples

            # 4. Populate Analysis Summary Table (schedule on GUI thread)
            self.root.after(0, self.populate_treeview, self.analysis_table, summary_metrics)

            # 5. Store calculated strategies (already populated in self.strategies_for_export by calculate_summary_metrics)
            self.analysis_results["strategies"] = self.strategies_for_export
            logger.debug(f"Strategies available for export after run: {list(self.analysis_results.get('strategies', {}).keys())}")

            # 6. Enable Export Buttons (schedule on GUI thread after a delay)
            self.root.after(100, self.update_export_buttons_state)

            elapsed_time = time.time() - start_time
            self.update_status_threadsafe(f"Analysis complete ({total_parts_in_bom} parts in {elapsed_time:.1f}s).", "success")
            self.update_progress_threadsafe(total_parts_in_bom, total_parts_in_bom, "Done")

        except Exception as e:
             logger.error(f"Analysis thread encountered an error: {e}", exc_info=True)
             self.update_status_threadsafe(f"Analysis Error: {e}", "error")
             # Show error popup in GUI thread
             self.root.after(0, messagebox.showerror, "Analysis Error", f"An error occurred during analysis:\n\n{e}\n\nCheck logs for details.")
             # Clear potentially partial results
             self.analysis_results = {}
             self.strategies_for_export = {}
             self.root.after(0, self.clear_treeview, self.tree)
             self.root.after(0, self.clear_treeview, self.analysis_table)
        finally:
             self.running_analysis = False
             # Re-enable controls in GUI thread
             self.root.after(50, self.update_analysis_controls_state, False)


    def calculate_summary_metrics(self, part_summaries, config):
        """
        Calculates aggregate BOM metrics, stores detailed strategy options for export,
        and calculates clear-to-build time using refined logic.
        """
        logger.info(f"Calculating summary metrics for {len(part_summaries)} parts...")

        # --- Initialize Aggregates & Flags ---
        total_bom_cost_min_strategy = 0.0
        total_bom_cost_max_potential = 0.0 # Max cost based on most expensive option per part
        total_bom_cost_fastest_strategy = 0.0
        total_bom_cost_optimized_strategy = 0.0

        max_lead_time_min_cost_strategy = 0
        max_lead_time_fastest_strategy = 0
        max_lead_time_optimized_strategy = 0

        # Flags to track if any part made a strategy invalid
        invalid_min_cost = False
        invalid_max_cost = False
        invalid_fastest_cost = False
        invalid_optimized_cost = False

        clear_to_build_stock_only = True # Assume true initially
        time_to_acquire_all_parts = 0    # Track max time needed
        stock_gap_parts = []             # List parts with stock gaps
        parts_with_stock_avail = 0       # Count parts fully stocked
        total_parts_analyzed = len(part_summaries) if part_summaries else 0

        # --- Initialize Strategy Storage Dictionaries ---
        lowest_cost_strategy_details = {}
        fastest_strategy_details = {}
        optimized_strategy_details = {}
        near_miss_info = {}

        # --- Iterate Through Each Part Summary ---
        for i, summary in enumerate(part_summaries):
            bom_pn = summary.get('bom_pn', f'UnknownPart_{i}')
            qty_needed = summary.get('total_qty_needed', 0)
            options = summary.get('options', []) # Original list of dicts

            # Basic check for valid options list structure
            valid_options = [opt for opt in options if isinstance(opt, dict)]

            if not valid_options:
                 # --- Handling for no options (Keep this block exactly as it was) ---
                 logger.warning(f"No valid supplier options found for {bom_pn}. Marking as unavailable.")
                 clear_to_build_stock_only = False
                 stock_gap_parts.append(f"{bom_pn}: No suppliers/options found.")
                 invalid_min_cost = invalid_max_cost = invalid_fastest_cost = invalid_optimized_cost = True
                 time_to_acquire_all_parts = np.inf
                 placeholder = self.create_strategy_entry({'notes': 'No Suppliers Found'})
                 lowest_cost_strategy_details[bom_pn] = placeholder
                 fastest_strategy_details[bom_pn] = placeholder
                 optimized_strategy_details[bom_pn] = placeholder
                 continue # Skip calculations for this part

            # --- Per-Part Calculations & Strategy Determination ---

            # Initialize part-specific best options and costs
            best_cost_option = None
            part_min_cost = np.inf
            fastest_option = None
            part_fastest_cost = np.inf
            part_min_lead = np.inf

            # --- Find LOWEST COST Option ---
            # Use min() with key: (cost, lead_time)
            if valid_options:
                 try:
                     best_cost_option = min(valid_options, key=lambda x: (
                                             x.get('cost', np.inf),
                                             x.get('lead_time', np.inf)
                                             ))
                     part_min_cost = best_cost_option.get('cost', np.inf)
                 except ValueError: # Handles empty valid_options case implicitly
                      best_cost_option = None
                      part_min_cost = np.inf

            # Store LOWEST COST strategy details and update totals
            if best_cost_option and part_min_cost != np.inf:
                lowest_cost_strategy_details[bom_pn] = self.create_strategy_entry(best_cost_option)
                if not invalid_min_cost: # Check overall strategy validity flag
                    total_bom_cost_min_strategy += part_min_cost
                    max_lead_time_min_cost_strategy = max(max_lead_time_min_cost_strategy, best_cost_option.get('lead_time', np.inf))
            else:
                invalid_min_cost = True
                lowest_cost_strategy_details[bom_pn] = self.create_strategy_entry({'notes': 'No Valid Cost Option'})
                best_cost_option = None # Ensure None for optimized fallback logic


            # --- Find FASTEST Option (Explicit Two-Step Logic) ---
            fastest_option = None
            part_fastest_cost = np.inf
            part_min_lead = np.inf

            if valid_options:
                # 1. Find all options with sufficient stock (stock >= qty_needed)
                options_with_stock = [
                    opt for opt in valid_options
                    if opt.get('stock', 0) >= qty_needed
                ]

                if options_with_stock:
                    # 2. From options with sufficient stock, select the CHEAPEST (breaking ties by lead time)
                    logger.debug(f"Part {bom_pn}: Found {len(options_with_stock)} options with sufficient stock. Selecting cheapest (then shortest lead time).")
                    try:
                        fastest_option = min(options_with_stock, key=lambda x: (
                            x.get('cost', np.inf),      # Prioritize lowest cost
                            x.get('lead_time', np.inf)  # Break ties by lead time
                        ))
                        part_fastest_cost = fastest_option.get('cost', np.inf)
                        part_min_lead = fastest_option.get('lead_time', np.inf)
                    except ValueError:
                        fastest_option = None
                        part_fastest_cost = np.inf
                        part_min_lead = np.inf
                else:
                    # 3. If NO options have sufficient stock, find the shortest lead time overall
                    logger.debug(f"Part {bom_pn}: No options with sufficient stock found. Selecting shortest lead time.")
                    try:
                        fastest_option = min(valid_options, key=lambda x: (
                            x.get('lead_time', np.inf),
                            x.get('cost', np.inf)
                        ))
                        part_fastest_cost = fastest_option.get('cost', np.inf)
                        part_min_lead = fastest_option.get('lead_time', np.inf)
                    except ValueError:
                        fastest_option = None
                        part_fastest_cost = np.inf
                        part_min_lead = np.inf

            # Store FASTEST strategy details and update totals
            if fastest_option and part_min_lead != np.inf:
                fastest_strategy_details[bom_pn] = self.create_strategy_entry(fastest_option)
                if not invalid_fastest_cost:
                    if part_fastest_cost == np.inf:
                        invalid_fastest_cost = True
                    else:
                        total_bom_cost_fastest_strategy += part_fastest_cost
                    max_lead_time_fastest_strategy = max(max_lead_time_fastest_strategy, part_min_lead)
            else:
                invalid_fastest_cost = True
                fastest_strategy_details[bom_pn] = self.create_strategy_entry({'notes': 'No Valid Lead Time Option'})
                fastest_option = None


            # Calculate Max Potential Cost for this part
            part_max_cost = 0.0
            # Use the initial valid_options list
            valid_costs_part = [opt.get('cost', 0) for opt in valid_options if opt.get('cost', np.inf) != np.inf]
            if valid_costs_part:
                part_max_cost = max(valid_costs_part)
                if not invalid_max_cost: # Check overall strategy validity flag
                    total_bom_cost_max_potential += part_max_cost
            else:
                 invalid_max_cost = True # No valid costs found for this part, invalidate max potential

            # *** Corrected Logic for Lowest/Fastest/Max Ends Here ***

            # 4. Clear to Build & Time to Acquire Logic (uses valid_options)
            part_has_stock = any(opt.get('stock', 0) >= qty_needed for opt in valid_options)
            if part_has_stock:
                parts_with_stock_avail += 1
                part_time_to_acquire = 0 # Can get immediately from stock
            else:
                clear_to_build_stock_only = False # If any part lacks stock, flag is false
                # Find minimum lead time among options WITHOUT enough stock
                min_lead_for_part = min((opt.get('lead_time', np.inf) for opt in valid_options if opt.get('stock', 0) < qty_needed), default=np.inf)
                # If all options have stock OR no lead time info available, use overall fastest LT found earlier
                if min_lead_for_part == np.inf:
                     min_lead_for_part = part_min_lead # Use the lead time of the fastest option overall

                part_time_to_acquire = min_lead_for_part
                issue = f"{bom_pn}: Stock Gap ({qty_needed} needed)"
                if min_lead_for_part != np.inf: issue += f", Min LT {min_lead_for_part:.0f}d."
                else: issue += ", No Lead Time info."
                stock_gap_parts.append(issue)

            if time_to_acquire_all_parts != np.inf: # Only update if not already infinite
                time_to_acquire_all_parts = max(time_to_acquire_all_parts, part_time_to_acquire)


            # 5. Optimized Strategy Calculation (uses valid_options, part_min_cost, best_cost_option, fastest_option)
            # This section should use the part_min_cost, best_cost_option, and fastest_option determined above
            target_lt_days = config['target_lead_time_days']
            max_prem_pct = config['max_premium'] # Stored as %
            cost_weight = config['cost_weight']
            lead_weight = config['lead_time_weight']
            chosen_option_opt = None # Clear choice for this part
            opt_notes = ""           # Clear notes for this part
            best_score = np.inf

            # Check if we have a valid base cost to compare against
            if part_min_cost == np.inf:
                 logger.warning(f"Optimized Strategy: Skipping {bom_pn} calculation as base min cost is invalid (infinity).")
                 # Use the placeholder already created for lowest cost
                 optimized_option = lowest_cost_strategy_details.get(bom_pn, self.create_strategy_entry({'notes': 'Invalid Base Cost'}))
                 opt_notes = "N/A (Invalid Base Cost)"
                 invalid_optimized_cost = True # Mark optimized as invalid
            else:
                # Filter options meeting basic constraints (valid cost/LT, LT <= Target)
                constrained_options = []
                for opt in valid_options:
                    cost = opt.get('cost', np.inf)
                    lead = opt.get('lead_time', np.inf)
                    if cost == np.inf or lead == np.inf: continue # Skip invalid
                    if lead > target_lt_days: continue          # Check Lead Time

                    # Check Cost Premium (handle division by zero)
                    cost_premium_pct_calc = ((cost - part_min_cost) / part_min_cost * 100.0) if part_min_cost > 1e-9 else 0.0
                    if cost_premium_pct_calc > max_prem_pct : continue # Check Premium

                    constrained_options.append(opt) # Passed all constraints


                # --- Select from Constrained Options or Fallback ---
                if not constrained_options:
                     logger.warning(f"Optimized Strategy: No option met constraints for {bom_pn}.")
                     # Fallback Logic: Compare lowest_cost_option vs fastest_option found earlier
                     lowest_cost_lt = best_cost_option.get('lead_time', np.inf) if best_cost_option else np.inf
                     # Check if lowest cost LT badly violates target (e.g., > 50% over)
                     # Use the actual determined best_cost_option and fastest_option here
                     if lowest_cost_lt > target_lt_days * 1.5 and target_lt_days > 0 and fastest_option:
                          logger.warning(f"  -> Lowest cost LT ({lowest_cost_lt}d) significantly exceeds target ({target_lt_days}d). Falling back to fastest option.")
                          chosen_option_opt = fastest_option
                          opt_notes = f"Constraints Failed. Fallback to Fastest (Low Cost LT: {lowest_cost_lt}d)."
                     elif best_cost_option: # Check if best_cost_option exists
                          chosen_option_opt = best_cost_option # Default fallback to lowest cost
                          opt_notes = f"Constraints Failed. Fallback to Lowest Cost."
                     else: # Cannot fallback if lowest_cost is also invalid
                          chosen_option_opt = None
                          opt_notes = f"Constraints Failed. No valid fallback."
                          invalid_optimized_cost = True

                     best_score = np.nan # Indicate fallback score

                else: # We have constrained options, calculate scores
                     # Calculate score ranges *only* based on constrained options
                     min_viable_cost = min(opt['cost'] for opt in constrained_options)
                     max_viable_cost = max(opt['cost'] for opt in constrained_options)
                     min_viable_lt = min(opt['lead_time'] for opt in constrained_options)
                     max_viable_lt = max(opt['lead_time'] for opt in constrained_options)

                     cost_range = (max_viable_cost - min_viable_cost) if max_viable_cost > min_viable_cost else 1.0 # Avoid div by zero
                     lead_range = (max_viable_lt - min_viable_lt) if max_viable_lt > min_viable_lt else 1.0 # Avoid div by zero

                     for option in constrained_options:
                         cost = option['cost']
                         lead_time = option['lead_time']
                         stock = option.get('stock', 0)
                         # Normalize cost and lead time within the viable range (0-1)
                         norm_cost = (cost - min_viable_cost) / cost_range if cost_range > 1e-9 else 0
                         norm_lead = (lead_time - min_viable_lt) / lead_range if lead_range > 1e-9 else 0

                         # Calculate Score = Weighted Norm Cost + Weighted Norm LT + Penalties
                         score = (cost_weight * norm_cost) + (lead_weight * norm_lead)
                         if option.get('discontinued') or option.get('eol'): score += 0.5
                         if stock < qty_needed: score += 0.1

                         if score < best_score:
                             best_score = score
                             chosen_option_opt = option

                     if chosen_option_opt:
                         # Add score to notes
                         opt_notes = f"Score: {best_score:.3f}"
                         # Prepend any existing notes (like buy-up reason) from the chosen option
                         existing_notes = chosen_option_opt.get('notes', '')
                         if existing_notes: opt_notes = f"{existing_notes}; {opt_notes}"
                     else: # Should not happen if constrained_options was not empty
                         logger.error(f"Optimized Strategy: Logic error - Failed to select option for {bom_pn} from constrained list.")
                         chosen_option_opt = best_cost_option # Fallback again
                         opt_notes = "N/A (Selection Error)"
                         best_score = np.nan
                         if not chosen_option_opt: invalid_optimized_cost = True # Cannot fallback

            # --- Store Optimized Strategy Choice ---
            # Make sure chosen_option_opt is valid before proceeding
            if chosen_option_opt:
                # Add the calculated score/notes back into the chosen option dict before creating the entry
                chosen_option_opt['optimized_strategy_score'] = f"{best_score:.3f}" if pd.notna(best_score) else "N/A"
                chosen_option_opt['notes'] = opt_notes # Ensure notes reflect fallback or score
                optimized_strategy_details[bom_pn] = self.create_strategy_entry(chosen_option_opt)

                # Update total optimized cost (only if overall strategy still valid)
                cost_to_add = chosen_option_opt.get('cost', np.inf)
                if not invalid_optimized_cost:
                    if cost_to_add == np.inf: invalid_optimized_cost = True
                    else: total_bom_cost_optimized_strategy += cost_to_add
                # Update max lead time for optimized strategy (only if overall strategy still valid)
                if not invalid_optimized_cost:
                    max_lead_time_optimized_strategy = max(max_lead_time_optimized_strategy, chosen_option_opt.get('lead_time', np.inf))

            else: # Handle case where optimized_option remained None
                  if bom_pn not in optimized_strategy_details: # Ensure placeholder exists
                       placeholder = lowest_cost_strategy_details.get(bom_pn, self.create_strategy_entry({}))
                       # Use notes determined during fallback logic
                       placeholder['notes'] = opt_notes if opt_notes else "N/A (Processing Error)"
                       optimized_strategy_details[bom_pn] = placeholder
                  if not invalid_optimized_cost: # Mark as invalid if we reach here without a choice
                       invalid_optimized_cost = True

            part_near_misses = {}
            if chosen_option_opt and best_score != np.inf: # Only look for near misses if constraints *were* met by chosen_option_opt or fallback logic ran
                 options_that_failed = [
                    opt for opt in valid_options
                    if opt not in constrained_options # Focus on those that failed constraints
                    and opt.get('cost', np.inf) != np.inf # Must have valid cost
                    and opt.get('lead_time', np.inf) != np.inf # Must have valid LT
                 ]

                 if options_that_failed:
                     # Find best option slightly over LT constraint
                     over_lt_candidates = [
                        opt for opt in options_that_failed
                        if opt.get('lead_time', np.inf) > target_lt_days
                     ]
                     if over_lt_candidates:
                          # Sort by how *little* they exceed LT, then by cost
                          over_lt_candidates.sort(key=lambda x: (x['lead_time'] - target_lt_days, x['cost']))
                          best_over_lt = over_lt_candidates[0]
                          # Check if it's only *slightly* over (e.g., within 14 days or 25% - configurable?)
                          if best_over_lt['lead_time'] <= target_lt_days + 14:
                               part_near_misses['slightly_over_lt'] = {
                                   'option': self.create_strategy_entry(best_over_lt),
                                   'over_by_days': best_over_lt['lead_time'] - target_lt_days
                               }

                     # Find best option slightly over Cost Premium constraint
                     over_cost_candidates = [
                        opt for opt in options_that_failed
                        if opt.get('lead_time', np.inf) <= target_lt_days # Must meet LT constraint
                     ]
                     if over_cost_candidates:
                          # Sort by how *little* they exceed cost premium, then by lead time
                          over_cost_candidates.sort(key=lambda x: (
                              ((x['cost'] - part_min_cost) / part_min_cost * 100.0) - max_prem_pct if part_min_cost > 1e-9 else x['cost'], # Amount over premium
                              x['lead_time']
                          ))
                          best_over_cost = over_cost_candidates[0]
                          # Check if it's only *slightly* over (e.g., within 5% points - configurable?)
                          actual_premium = ((best_over_cost['cost'] - part_min_cost) / part_min_cost * 100.0) if part_min_cost > 1e-9 else 0.0
                          if actual_premium <= max_prem_pct + 5.0:
                               part_near_misses['slightly_over_cost'] = {
                                   'option': self.create_strategy_entry(best_over_cost),
                                   'over_by_pct': actual_premium - max_prem_pct
                               }

            if part_near_misses:
                 near_miss_info[bom_pn] = part_near_misses


        # --- End of Part Loop ---

        # Final check on aggregated costs/lead times - if any part was invalid, mark whole strategy as N/A
        if invalid_min_cost: total_bom_cost_min_strategy = np.nan; max_lead_time_min_cost_strategy = np.inf
        # Max potential cost check (already handled per part)
        if invalid_max_cost: total_bom_cost_max_potential = np.nan
        if invalid_fastest_cost: total_bom_cost_fastest_strategy = np.nan; max_lead_time_fastest_strategy = np.inf
        if invalid_optimized_cost: total_bom_cost_optimized_strategy = np.nan; max_lead_time_optimized_strategy = np.inf

        # If time_to_acquire is infinite, clear_to_build must be False
        if time_to_acquire_all_parts == np.inf: clear_to_build_stock_only = False

        # --- Store Strategies for Export ---
        # Ensure all parts have an entry, even if invalid
        all_boms = [s.get('bom_pn') for s in part_summaries if s.get('bom_pn')]
        for strategy_dict in [lowest_cost_strategy_details, fastest_strategy_details, optimized_strategy_details]:
            for bom_pn in all_boms:
                if bom_pn not in strategy_dict:
                     strategy_dict[bom_pn] = self.create_strategy_entry({'notes': 'Processing Error or Unavailable'})

        self.strategies_for_export = {
             "Lowest Cost": lowest_cost_strategy_details,
             "Fastest": fastest_strategy_details,
             "Optimized Strategy": optimized_strategy_details,
        }
        self.analysis_results['near_miss_info'] = near_miss_info
        logger.debug(f"Near miss info calculated for {len(near_miss_info)} parts.")
        
        logger.debug(f"Strategies calculated and stored for export. Keys: {list(self.strategies_for_export.keys())}")

        # --- Format Summary Data for GUI Table ---
        summary_list = [] # Use list of tuples to maintain order
        summary_list.append(("Total Parts Analyzed", f"{total_parts_analyzed}"))

        ctb_stock_value = f"{clear_to_build_stock_only} ({parts_with_stock_avail}/{total_parts_analyzed} parts fully stocked)"
        summary_list.append(("Immediate Stock Availability", ctb_stock_value))

        ctb_lt_value = f"{time_to_acquire_all_parts:.0f} days" if time_to_acquire_all_parts != np.inf else "N/A (Parts Unavailable)"
        summary_list.append(("Est. Time to Full Kit (Days)", ctb_lt_value))

        if stock_gap_parts:
             issues_str = "; ".join(stock_gap_parts)
             summary_list.append(("Parts with Stock Gaps", issues_str[:500] + ('...' if len(issues_str) > 500 else '')))
        else:
             summary_list.append(("Parts with Stock Gaps", "None"))


        min_max_cost_str = "N/A"
        if pd.notna(total_bom_cost_min_strategy) and pd.notna(total_bom_cost_max_potential): min_max_cost_str = f"${total_bom_cost_min_strategy:.2f} / ${total_bom_cost_max_potential:.2f}"
        elif pd.notna(total_bom_cost_min_strategy): min_max_cost_str = f"${total_bom_cost_min_strategy:.2f} / N/A"
        summary_list.append(("Potential Cost Range ($)", min_max_cost_str))

        lowest_cost_str = "N/A"
        if pd.notna(total_bom_cost_min_strategy):
             lt_str = f"{max_lead_time_min_cost_strategy:.0f} days" if max_lead_time_min_cost_strategy != np.inf else "N/A"
             lowest_cost_str = f"${total_bom_cost_min_strategy:.2f} / {lt_str}"
        summary_list.append(("Lowest Cost Strategy / LT ($ / Days)", lowest_cost_str))

        fastest_str = "N/A"
        if pd.notna(total_bom_cost_fastest_strategy):
             lt_str = f"{max_lead_time_fastest_strategy:.0f} days" if max_lead_time_fastest_strategy != np.inf else "N/A"
             fastest_str = f"${total_bom_cost_fastest_strategy:.2f} / {lt_str}"
        summary_list.append(("Fastest Strategy Cost / LT ($ / Days)", fastest_str))

        # Use consistent naming for the optimized strategy key
        optimized_summary_key = "Balanced (Optimized Strategy) Cost / LT ($ / Days)"
        optimized_str = "N/A"
        if pd.notna(total_bom_cost_optimized_strategy):
             lt_str = f"{max_lead_time_optimized_strategy:.0f} days" if max_lead_time_optimized_strategy != np.inf else "N/A"
             optimized_str = f"${total_bom_cost_optimized_strategy:.2f} / {lt_str}"
        # Provide reason if N/A - check the flag
        elif invalid_optimized_cost:
             optimized_str = "N/A (Constraints Failed / Parts Unavailable)"
        summary_list.append((optimized_summary_key, optimized_str))

        # --- Tariff Calculation (based on chosen strategy) ---
        total_tariff_cost = 0.0
        calculated_bom_cost_for_tariff = 0.0
        chosen_strategy_for_tariff_calc = {}
        tariff_basis_name = "N/A"

        # Prefer Optimized Strategy if valid, otherwise fallback to Lowest Cost if valid
        if not invalid_optimized_cost: # Check flag instead of just NaN cost
             chosen_strategy_for_tariff_calc = optimized_strategy_details
             tariff_basis_name = "Optimized Strategy"
        elif not invalid_min_cost: # Check flag
             chosen_strategy_for_tariff_calc = lowest_cost_strategy_details
             tariff_basis_name = "Lowest Cost"

        if tariff_basis_name != "N/A":
            for bom_pn, chosen_option in chosen_strategy_for_tariff_calc.items():
                # Ensure option is valid before calculating
                if isinstance(chosen_option, dict) and chosen_option.get('source', 'N/A') != 'N/A':
                    part_cost_basis = chosen_option.get('cost', np.inf)
                    if part_cost_basis != np.inf:
                        calculated_bom_cost_for_tariff += part_cost_basis
                        # Tariff rate was stored in the option dict during analyze_single_part
                        tariff_rate = chosen_option.get('tariff_rate') # Expects fraction
                        if tariff_rate is not None and pd.notna(tariff_rate):
                             total_tariff_cost += part_cost_basis * tariff_rate

            total_tariff_pct = (total_tariff_cost / calculated_bom_cost_for_tariff * 100) if calculated_bom_cost_for_tariff > 1e-6 else 0.0
            summary_list.append((f"Est. Total Tariff Cost ({tariff_basis_name})", f"${total_tariff_cost:.2f}"))
            summary_list.append((f"Est. Total Tariff % ({tariff_basis_name})", f"{total_tariff_pct:.2f}%"))
        else:
             summary_list.append(("Est. Total Tariff Cost (N/A)", "N/A"))
             summary_list.append(("Est. Total Tariff % (N/A)", "N/A"))

        logger.info("Summary metrics calculation complete.")
        return summary_list # Return list of tuples for the GUI table


    # --- Predictive Analysis (Prophet, RAG Mock) ---

    def run_prophet(self, component_historical_data, metric='Lead_Time_Days', periods=90, min_data_points=5):
        """ Runs Prophet forecasting with outlier filtering. Returns prediction or None. """
        if component_historical_data is None or component_historical_data.empty: return None
        if metric not in component_historical_data.columns: return None

        df_prophet = component_historical_data[['Fetch_Timestamp', metric]].copy()
        df_prophet = df_prophet.dropna(subset=[metric])
        df_prophet.rename(columns={'Fetch_Timestamp': 'ds', metric: 'y'}, inplace=True)

        if len(df_prophet) < min_data_points:
            logger.debug(f"Prophet ({metric}): Insufficient data ({len(df_prophet)} < {min_data_points})")
            return None

        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], errors='coerce').dt.tz_localize(None) # Ensure timezone naive
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet = df_prophet.dropna(subset=['ds', 'y'])

        # IQR Outlier Filtering
        q1 = df_prophet['y'].quantile(0.25); q3 = df_prophet['y'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr; upper_bound = q3 + 1.5 * iqr
        initial_rows = len(df_prophet)
        df_prophet = df_prophet[(df_prophet['y'] >= lower_bound) & (df_prophet['y'] <= upper_bound)]
        if initial_rows - len(df_prophet) > 0: logger.debug(f"Prophet ({metric}): Removed {initial_rows - len(df_prophet)} outliers.")

        if len(df_prophet) < min_data_points: # Check again after filtering
             logger.debug(f"Prophet ({metric}): Insufficient data after filtering ({len(df_prophet)} < {min_data_points})")
             return None

        try:
            # Suppress Prophet logs further
            with open(os.devnull, 'w') as stderr, contextlib.redirect_stderr(stderr):
                model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False, changepoint_prior_scale=0.05)
                model.fit(df_prophet)

            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)
            predicted_value = forecast.iloc[-1]['yhat'] # Get last predicted value

            # Apply bounds
            if metric == 'Lead_Time_Days': predicted_value = max(0, predicted_value)
            elif metric == 'Cost': predicted_value = max(0.001, predicted_value)

            logger.debug(f"Prophet forecast for {metric}: {predicted_value:.2f}")
            return predicted_value

        except Exception as e:
            # Log only if error is unexpected (Prophet can fail on flat data etc.)
            if "values must be unique" not in str(e).lower() and "less than 2 non-NaN values" not in str(e).lower():
                 logger.error(f"Prophet forecasting failed for {metric}: {e}", exc_info=False)
            else:
                 logger.debug(f"Prophet skipping for {metric} due to data issues (e.g., constant values).")
            return None

    def run_rag_mock(self, prophet_lead, prophet_cost, stock_prob, context=""):
        """ Mock RAG function adding variability. """
        rag_lead_range, rag_cost_range = "N/A", "N/A"
        adj_stock_prob = stock_prob if pd.notna(stock_prob) else 50.0 # Default if missing

        context_lower = context.lower()
        has_issues = any(term in context_lower for term in ["shortage", "delay", "constrained", "tariff", "increase"])

        if pd.notna(prophet_lead):
            base = prophet_lead
            var = max(7, base * 0.15) # Variability: +/- 1 week or 15%
            lead_min = base - var * np.random.uniform(0.5, 1.0)
            lead_max = base + var * np.random.uniform(1.0, 1.5)
            if has_issues: lead_min += 7; lead_max += 14; adj_stock_prob *= 0.8
            rag_lead_range = f"{max(0, lead_min):.0f}-{max(0, lead_max):.0f}"

        if pd.notna(prophet_cost):
            base = prophet_cost
            var = max(0.01, base * 0.05) # Variability: +/- 1 cent or 5%
            cost_min = base - var * np.random.uniform(0.5, 1.0)
            cost_max = base + var * np.random.uniform(1.0, 1.2)
            if has_issues: cost_min = max(cost_min, base * 0.98); cost_max += base * 0.1 # Increase range if issues
            rag_cost_range = f"{max(0.001, cost_min):.3f}-{max(0.001, cost_max):.3f}"

        return rag_lead_range, rag_cost_range, round(max(0.0, min(100.0, adj_stock_prob)), 1)


    def run_ai_comparison(self, prophet_lead, prophet_cost, rag_lead_range, rag_cost_range, stock_prob):
        """ Mock AI comparison - simple averaging for now. """
        ai_lead, ai_cost = np.nan, np.nan # Start with NaN

        # Average Lead
        rag_mid_lead = np.nan
        if isinstance(rag_lead_range, str) and '-' in rag_lead_range:
            try: parts = [safe_float(p) for p in rag_lead_range.split('-')]; rag_mid_lead = np.mean(parts) if len(parts)==2 else np.nan
            except: pass
        valid_leads = [p for p in [prophet_lead, rag_mid_lead] if pd.notna(p)]
        if valid_leads: ai_lead = max(0, np.mean(valid_leads)) # Average valid predictions, floor at 0

        # Average Cost
        rag_mid_cost = np.nan
        if isinstance(rag_cost_range, str) and '-' in rag_cost_range:
            try: parts = [safe_float(p) for p in rag_cost_range.split('-')]; rag_mid_cost = np.mean(parts) if len(parts)==2 else np.nan
            except: pass
        valid_costs = [c for c in [prophet_cost, rag_mid_cost] if pd.notna(c)]
        if valid_costs: ai_cost = max(0.001, np.mean(valid_costs)) # Average valid predictions, floor at 0.001

        # Stock prob just passed through from RAG mock for now
        ai_stock_prob = stock_prob if pd.notna(stock_prob) else 50.0

        return ai_lead, ai_cost, ai_stock_prob


    def run_predictive_analysis_gui(self):
        """ Handles 'Run Predictions' button click. """
        if self.running_analysis:
            messagebox.showwarning("Busy", "Analysis/Prediction is already in progress.")
            return
        if self.historical_data_df is None or self.historical_data_df.empty or 'Component' not in self.historical_data_df.columns:
            messagebox.showerror("No Data", "No historical data available. Run main BOM analysis first to generate data.")
            return

        # Optional: Simple dialog for context
        # context = simpledialog.askstring("Prediction Context", "Enter brief market context (optional):", parent=self.root)
        # context = context if context else "General market conditions"
        context = "General market conditions" # Placeholder

        self.running_analysis = True
        self.update_analysis_controls_state(True)
        self.update_status_threadsafe("Generating predictions...", "info")
        logger.info("Submitting prediction task to thread pool.")
        self.thread_pool.submit(self.run_predictive_analysis_thread, context)


    def run_predictive_analysis_thread(self, context):
        """ Thread function to generate predictions for all unique components. """
        start_time = time.time()
        try:
            if self.historical_data_df is None or self.historical_data_df.empty or 'Component' not in self.historical_data_df.columns:
                self.update_status_threadsafe("Error: Historical data missing.", "error")
                return

            unique_components = self.historical_data_df['Component'].dropna().unique()
            total_comps = len(unique_components)
            if total_comps == 0:
                self.update_status_threadsafe("No components in historical data.", "warning")
                return

            self.update_progress_threadsafe(0, total_comps, "Predicting...")
            new_predictions = []
            today_str = datetime.now().strftime('%Y-%m-%d')

            for i, component in enumerate(unique_components):
                 if not self.running_analysis: logger.info("Prediction run cancelled."); return

                 self.update_progress_threadsafe(i, total_comps, f"Predicting {component[:30]}...")
                 component_data = self.historical_data_df[self.historical_data_df['Component'] == component].copy()
                 if component_data.empty: continue

                 # Sort by timestamp to get latest stock probability reliably
                 component_data.sort_values('Fetch_Timestamp', ascending=False, inplace=True)
                 latest_stock_prob = component_data['Stock_Probability'].iloc[0] if not component_data.empty and not pd.isna(component_data['Stock_Probability'].iloc[0]) else 50.0 # Default 50%

                 # Run predictions (Prophet runs first)
                 prophet_lead = self.run_prophet(component_data, 'Lead_Time_Days')
                 prophet_cost = self.run_prophet(component_data, 'Cost')

                 # Run mocks/comparisons based on Prophet results
                 rag_lead, rag_cost, rag_stock_prob = self.run_rag_mock(prophet_lead, prophet_cost, latest_stock_prob, context)
                 ai_lead, ai_cost, ai_stock_prob = self.run_ai_comparison(prophet_lead, prophet_cost, rag_lead, rag_cost, rag_stock_prob)

                 # Format prediction row (ensure order matches self.pred_header)
                 pred_row = {
                     'Component': str(component).replace('\n', ' ').replace('\r', ''),
                     'Date': today_str,
                     'Prophet_Lead': f"{prophet_lead:.1f}" if pd.notna(prophet_lead) else '',
                     'Prophet_Cost': f"{prophet_cost:.3f}" if pd.notna(prophet_cost) else '',
                     'RAG_Lead': rag_lead if rag_lead != "N/A" else '',
                     'RAG_Cost': rag_cost if rag_cost != "N/A" else '',
                     'AI_Lead': f"{ai_lead:.1f}" if pd.notna(ai_lead) else '',
                     'AI_Cost': f"{ai_cost:.3f}" if pd.notna(ai_cost) else '',
                     'Stock_Probability': f"{ai_stock_prob:.1f}" if pd.notna(ai_stock_prob) else '',
                     'Real_Lead': '', 'Real_Cost': '', 'Real_Stock': '',
                     'Prophet_Ld_Acc': '', 'Prophet_Cost_Acc': '',
                     'RAG_Ld_Acc': '', 'RAG_Cost_Acc': '',
                     'AI_Ld_Acc': '', 'AI_Cost_Acc': '',
                 }
                 # Convert dict row to list in the correct order
                 new_predictions.append([pred_row.get(h, '') for h in self.pred_header])

            if not self.running_analysis: return # Check again before writing

            # Append new predictions to file
            if new_predictions:
                 append_to_csv(PREDICTION_FILE, new_predictions)
                 logger.info(f"Appended {len(new_predictions)} predictions to {PREDICTION_FILE.name}")
                 # Schedule reload and GUI update on main thread
                 self.root.after(0, self.load_predictions_to_gui)
            else:
                 logger.info("No new predictions generated.")

            elapsed_time = time.time() - start_time
            self.update_status_threadsafe(f"Predictive analysis complete ({elapsed_time:.1f}s).", "success")
            self.update_progress_threadsafe(total_comps, total_comps, "Done")

        except Exception as e:
            logger.error(f"Predictive analysis thread failed: {e}", exc_info=True)
            self.update_status_threadsafe(f"Prediction Error: {e}", "error")
            self.root.after(0, messagebox.showerror, "Prediction Error", f"An error occurred during prediction:\n\n{e}")
        finally:
            self.running_analysis = False
            self.root.after(50, self.update_analysis_controls_state, False)

    # --- Load Predictions to GUI ---
    def load_predictions_to_gui(self):
        """
        Loads data from prediction CSV into the GUI table, formats for display,
        maps rows for saving, and calculates accuracy. Ensures execution on main thread.
        """
        if not is_main_thread():
            # If not on the main thread, reschedule the call on the main thread
            self.root.after(0, self.load_predictions_to_gui)
            return

        logger.debug("Loading predictions into GUI...")
        self.clear_treeview(self.predictions_tree)
        self.prediction_tree_row_map = {} # Reset mapping on each load

        try:
            # 1. Ensure Data is Loaded and Validated
            # This call loads/reloads self.predictions_df from the CSV
            # and ensures required columns exist.
            self.initialize_data_files()

            # 2. Check if DataFrame is valid after initialization
            if self.predictions_df is None or self.predictions_df.empty:
                 logger.info("Prediction data is empty or failed to load. Nothing to display.")
                 # Update accuracies to show N/A or 0 if data is missing
                 self.calculate_and_display_average_accuracies()
                 return

            # 3. Prepare DataFrame for Display (Work on a copy)
            # Ensure a consistent 0-based index AFTER loading/initialization
            # This index will be used for mapping Treeview items back for saving.
            df_for_mapping = self.predictions_df.reset_index(drop=True)
            df_display = df_for_mapping.copy() # Copy for display formatting/sorting

            # 4. Format 'Real_Stock' for Display
            # The underlying data in df_for_mapping['Real_Stock'] should be pandas boolean type
            if 'Real_Stock' in df_display.columns:
                # Map True -> "True", False -> "False", pd.NA/None -> "?"
                df_display['Real_Stock_Display'] = df_display['Real_Stock'].map(
                    {True: 'True', False: 'False', pd.NA: ''}
                ).fillna('?') # Handle any other potential missing values
            else:
                # If column somehow missing after init_files, add placeholder
                df_display['Real_Stock_Display'] = ''
                logger.warning("'Real_Stock' column unexpectedly missing during GUI load prep.")

            # 5. Optional: Sort DataFrame for Display (Removed for index stability, add back if mapping is robust)
            # Sorting here makes mapping the Treeview ID back to the *original* DataFrame index difficult.
            # If sorting is desired, the mapping in step 7 needs a more robust method (e.g., unique ID).
            # if 'Date' in df_display.columns:
            #      df_display['Date_dt'] = pd.to_datetime(df_display['Date'], errors='coerce')
            #      df_display.sort_values(by=['Date_dt', 'Component'], ascending=[False, True], inplace=True, na_position='last')
            #      df_display.drop(columns=['Date_dt'], inplace=True)

            # 6. Format Other Columns (Optional, but can improve appearance)
            # Ensure all header columns exist in the display DataFrame
            for col in self.pred_header:
                if col not in df_display.columns:
                    df_display[col] = ''

            # Apply specific formatting to numeric columns for display consistency
            # This modifies the df_display DataFrame directly
            for col in self.pred_header:
                # Skip non-numeric or specially handled columns
                if col in ['Component', 'Date', 'RAG_Lead', 'RAG_Cost', 'Real_Stock', 'Real_Stock_Display']:
                    continue

                try:
                    # Convert to numeric first to allow formatting
                    numeric_col = pd.to_numeric(df_display[col], errors='coerce')

                    # Define format strings based on column name patterns
                    if 'Acc' in col or 'Probability' in col: fmt = "{:.1f}" # Accuracy/Probability: 1 decimal place
                    elif 'Cost' in col: fmt = "{:.3f}"                     # Cost: 3 decimal places
                    elif 'Lead' in col: fmt = "{:.1f}"                     # Lead Time: 1 decimal place (adjust if integer preferred)
                    else: fmt = "{}"                                       # Default: No specific format

                    # Apply formatting, replacing NaN with empty string
                    df_display[col] = numeric_col.apply(lambda x: fmt.format(x) if pd.notna(x) else '')
                except Exception as fmt_err:
                    logger.warning(f"Could not format column '{col}' for display: {fmt_err}")
                    # Ensure column is string type, handling potential errors
                    df_display[col] = df_display[col].astype(str).fillna('')


            # 7. Populate Treeview and Create Mapping
            # Iterate through the potentially formatted df_display
            # Use the index from the NON-SORTED df_for_mapping for saving later
            for index, row in df_display.iterrows():
                 values = []
                 for col in self.pred_header:
                     if col == 'Real_Stock':
                         # Use the specially formatted display value
                         values.append(str(row.get('Real_Stock_Display', '?')))
                     else:
                         # Get value from the potentially formatted row, default to empty string
                         values.append(str(row.get(col, '')))

                 # Insert row into Treeview
                 tree_item_id = self.predictions_tree.insert("", "end", values=values)

                 # Map the Treeview item ID back to the DataFrame index (from df_for_mapping)
                 # This index corresponds to the row in self.predictions_df that needs saving.
                 if index < len(df_for_mapping): # Basic sanity check
                      self.prediction_tree_row_map[tree_item_id] = index
                 else:
                      # This case should ideally not happen if indices are managed correctly
                      logger.error(f"Index mismatch during Treeview population. Row index {index} out of bounds for mapping DataFrame (len {len(df_for_mapping)}). Mapping failed for this row.")


            logger.info(f"Loaded {len(df_display)} predictions into GUI.")

            # 8. Update Average Accuracy Display
            self.calculate_and_display_average_accuracies()

        except FileNotFoundError:
              # This case should be handled by initialize_data_files, but keep for safety
              logger.info(f"{PREDICTION_FILE.name} not found. No predictions loaded.")
              self.calculate_and_display_average_accuracies() # Reset accuracies
        except Exception as e:
              logger.error(f"Failed to load or display predictions: {e}", exc_info=True)
              messagebox.showerror("Load Error", f"Could not load or display predictions from {PREDICTION_FILE.name}:\n\n{e}")
              self.calculate_and_display_average_accuracies() # Reset accuracies


    def calculate_and_display_average_accuracies(self):
        """ Calculates and displays average accuracies in the GUI. """
        if not is_main_thread():
             self.root.after(0, self.calculate_and_display_average_accuracies)
             return

        logger.debug("Calculating average prediction accuracies...")
        # Use the main predictions_df which should have correct types from initialize_data_files
        if self.predictions_df is None or self.predictions_df.empty:
            logger.warning("No prediction data loaded to calculate averages.")
            # Reset all labels to N/A or 0
            for key, label in self.avg_acc_labels.items():
                 default_text = "0" if "Count" in key else "N/A"
                 try: label.config(text=default_text)
                 except: pass # Ignore errors if label doesn't exist
            return

        acc_types = ["Prophet", "RAG", "AI"]
        try:
            for acc_type in acc_types:
                # Lead Time
                ld_col = f"{acc_type}_Ld_Acc"
                ld_key_acc = f"{acc_type}_Ld"
                ld_key_count = f"{acc_type}_Ld_Count"
                if ld_col in self.predictions_df.columns:
                    # Ensure column is numeric for calculations
                    acc_series_ld = pd.to_numeric(self.predictions_df[ld_col], errors='coerce')
                    avg_ld = acc_series_ld.mean(skipna=True)
                    count_ld = acc_series_ld.count() # Counts non-NaN values
                    # Update GUI labels safely
                    if ld_key_acc in self.avg_acc_labels:
                        self.avg_acc_labels[ld_key_acc].config(text=f"{avg_ld:.1f}%" if pd.notna(avg_ld) else "N/A")
                    if ld_key_count in self.avg_acc_labels:
                         self.avg_acc_labels[ld_key_count].config(text=f"{count_ld}")
                else: # Column doesn't exist
                    if ld_key_acc in self.avg_acc_labels: self.avg_acc_labels[ld_key_acc].config(text="N/A")
                    if ld_key_count in self.avg_acc_labels: self.avg_acc_labels[ld_key_count].config(text="0")

                # Cost
                cost_col = f"{acc_type}_Cost_Acc"
                cost_key_acc = f"{acc_type}_Cost"
                cost_key_count = f"{acc_type}_Cost_Count"
                if cost_col in self.predictions_df.columns:
                    acc_series_cost = pd.to_numeric(self.predictions_df[cost_col], errors='coerce')
                    avg_cost = acc_series_cost.mean(skipna=True)
                    count_cost = acc_series_cost.count()
                    if cost_key_acc in self.avg_acc_labels:
                        self.avg_acc_labels[cost_key_acc].config(text=f"{avg_cost:.1f}%" if pd.notna(avg_cost) else "N/A")
                    if cost_key_count in self.avg_acc_labels:
                         self.avg_acc_labels[cost_key_count].config(text=f"{count_cost}")
                else: # Column doesn't exist
                    if cost_key_acc in self.avg_acc_labels: self.avg_acc_labels[cost_key_acc].config(text="N/A")
                    if cost_key_count in self.avg_acc_labels: self.avg_acc_labels[cost_key_count].config(text="0")

            logger.debug("Average accuracies updated.")

        except Exception as e:
            logger.error(f"Error calculating/displaying average accuracies: {e}", exc_info=True)
            # Reset labels on error
            for key, label in self.avg_acc_labels.items():
                 default_text = "0" if "Count" in key else "Error"
                 try: label.config(text=default_text)
                 except: pass


    # --- AI Summary ---
    def generate_ai_summary_gui(self):
        """ Handles 'AI Summary' button click. """
        if self.running_analysis:
            messagebox.showwarning("Busy", "Analysis/Prediction is already in progress.")
            return
        if not self.analysis_results or not self.analysis_results.get("summary_metrics"):
             messagebox.showinfo("No Data", "Please run the main BOM analysis first to generate results.")
             return
        if not openai_client: # Check if client was initialized
             messagebox.showwarning("No API Key", "OpenAI API key is not configured or failed to initialize. Cannot generate summary.")
             return

        self.running_analysis = True
        self.update_analysis_controls_state(True) # Disable buttons
        self.update_status_threadsafe("Generating AI summary...", "info")
        self.ai_summary_text.configure(state='normal'); self.ai_summary_text.delete(1.0, tk.END); self.ai_summary_text.insert(tk.END, "Calling OpenAI...\nThis may take a moment."); self.ai_summary_text.configure(state='disabled')
        logger.info("Submitting AI summary task to thread pool.")
        self.thread_pool.submit(self.generate_ai_summary_thread)


    def generate_ai_summary_thread(self):
        """Thread function to call OpenAI and update the GUI."""
        logger.info("AI Summary Thread: Starting...")
        try:
            # --- Extract data from analysis_results ---
            if not self.analysis_results:
                 logger.error("AI Summary Thread: self.analysis_results is empty.")
                 self.root.after(0, self.update_status_threadsafe, "Error: No analysis results found for AI Summary.", "error")
                 return

            config = self.analysis_results.get("config", {})
            gui_entries = self.analysis_results.get("gui_entries", [])
            summary_metrics_list = self.analysis_results.get("summary_metrics", [])
            strategies = self.analysis_results.get("strategies", {})
            near_miss_data = self.analysis_results.get("near_miss_info", {}) # Get near miss info

            if not summary_metrics_list or not isinstance(summary_metrics_list, list):
                 logger.error("AI Summary Thread: Missing or invalid summary metrics in analysis_results.")
                 self.root.after(0, self.update_status_threadsafe, "Error: Missing summary data for AI Summary.", "error")
                 return

            summary_dict = dict(summary_metrics_list) # Create dict for easier lookup

            # --- Build ADVANCED Prompt ---
            prompt = f"You are a strategic supply chain advisor reporting to the C-suite for a build of {config.get('total_units', 'N/A')} units. Key constraints: Target Lead Time <= {config.get('target_lead_time_days', 'N/A')} days, Max Cost Premium <= {config.get('max_premium', 'N/A')}%. Analyze the following BOM data:\n\n"

            # === Key Performance Indicators ===
            prompt += "=== Key Performance Indicators ===\n"
            metric_map_display = { # Mapping for cleaner display names
                "Total Parts Analyzed": "BOM Lines Analyzed",
                "Immediate Stock Availability": "Immediate Stock Availability",
                "Est. Time to Full Kit (Days)": "Est. Time to Full Kit (Days)",
                "Parts with Stock Gaps": "Parts with Stock Gaps",
                "Potential Cost Range ($)": "Potential BOM Cost Range ($)",
                "Lowest Cost Strategy / LT ($ / Days)": "Lowest Cost Strategy / Max LT",
                "Fastest Strategy Cost / LT ($ / Days)": "Fastest Strategy Cost / Max LT",
                "Balanced (Optimized Strategy) Cost / LT ($ / Days)": "Optimized Strategy Cost / Max LT",
                # Tariff metrics will be handled dynamically below
            }
            # Add formatted metrics from the list
            for metric, value in summary_metrics_list:
                 display_name = metric_map_display.get(metric, metric) # Use mapped name or original
                 # Handle dynamic tariff naming
                 if "Est. Total Tariff Cost" in metric:
                     strategy_ctx = "(Optimized Strategy)" if "Optimized Strategy" in metric else "(Lowest Cost)" if "Lowest Cost" in metric else "(N/A)"
                     display_name = f"Est. Total Tariff Cost {strategy_ctx}"
                 elif "Est. Total Tariff %" in metric:
                      strategy_ctx = "(Optimized Strategy)" if "Optimized Strategy" in metric else "(Lowest Cost)" if "Lowest Cost" in metric else "(N/A)"
                      display_name = f"Est. Total Tariff % {strategy_ctx}"

                 prompt += f"- {display_name}: {value}\n"
            prompt += "\n"


            # === Strategic Risk Assessment ===
            prompt += "=== Strategic Risk Assessment ===\n"
            high_risk_parts = []
            moderate_risk_parts = []
            risks = [] # General risk statements

            if gui_entries: # Check if gui_entries exist
                 # Sort by RiskScore (descending) - handle potential 'N/A' scores
                 def get_risk_sort_key(entry):
                     score = entry.get('RiskScore', '0.0')
                     return safe_float(score, default=-1.0) # Sort N/A as low risk

                 sorted_entries = sorted(gui_entries, key=get_risk_sort_key, reverse=True)

                 for entry in sorted_entries:
                      score_val = safe_float(entry.get('RiskScore'), default=0.0)
                      # Use MfgPN if available, else BOM PN
                      part_id = entry.get('MfgPN', entry.get('PartNumber', 'Unknown Part'))
                      if part_id != 'NOT FOUND': # Avoid listing "NOT FOUND" parts as high risk
                          if score_val >= self.RISK_CATEGORIES['high'][0]:
                               high_risk_parts.append(f"{part_id} ({score_val:.1f})")
                          elif score_val >= self.RISK_CATEGORIES['moderate'][0]:
                               moderate_risk_parts.append(f"{part_id} ({score_val:.1f})")

            if high_risk_parts:
                 prompt += f"- High Risk Parts ({len(high_risk_parts)}): {', '.join(high_risk_parts[:7])}{'...' if len(high_risk_parts)>7 else ''}\n" # Show a few more
            if moderate_risk_parts:
                 prompt += f"- Moderate Risk Parts ({len(moderate_risk_parts)}): {', '.join(moderate_risk_parts[:5])}{'...' if len(moderate_risk_parts)>5 else ''}\n"

            # Check specific metrics for general risks
            clear_to_build_stock_only_val = summary_dict.get("Immediate Stock Availability", "True")
            time_to_acquire_val = summary_dict.get("Est. Time to Full Kit (Days)", "0 days")
            target_lt = config.get('target_lead_time_days', 90) # Default target if not in config

            if isinstance(clear_to_build_stock_only_val, str) and clear_to_build_stock_only_val.lower().startswith("false"):
                 risks.append("Immediate build hindered by stock gaps requiring lead time buys.")

            try: # Safely parse acquisition time
                current_acquire_time_str = str(time_to_acquire_val).split()[0]
                if current_acquire_time_str.lower() not in ['n/a', 'inf']:
                     current_acquire_time = float(current_acquire_time_str)
                     if current_acquire_time > target_lt: risks.append(f"Est. time to acquire parts ({current_acquire_time:.0f}d) exceeds target ({target_lt}d).")
            except (ValueError, IndexError): pass # Ignore parsing fails

            eol_parts = [entry.get('MfgPN', entry.get('PartNumber')) for entry in gui_entries if entry.get('Status') in ['EOL', 'Discontinued']]
            if eol_parts:
                prompt += f"- EOL/Discontinued parts identified: {', '.join(eol_parts[:5])}{'...' if len(eol_parts)>5 else ''}\n"
                risks.append("EOL/Discontinued parts require immediate action (last time buy, redesign).")

            # Add general risk statements if any were found
            if risks:
                 prompt += "- Other Key Risks: " + "; ".join(risks) + "\n"

            if not high_risk_parts and not moderate_risk_parts and not eol_parts and not risks:
                prompt += "- No major specific risks identified based on current data and thresholds.\n"
            prompt += "\n"


            # === Potential Trade-offs (Near Miss Analysis) ===
            prompt += "=== Potential Trade-offs (Near Miss Analysis) ===\n"
            if not near_miss_data:
                 prompt += "No significant near-miss options identified based on current thresholds (+14 days LT, +5% cost premium).\n"
            else:
                 prompt += "Consider these options if constraints can be slightly adjusted:\n"
                 # Limit number of near misses shown to keep prompt reasonable
                 near_miss_count = 0
                 max_near_miss_display = 5
                 for bom_pn, misses in near_miss_data.items():
                      if near_miss_count >= max_near_miss_display:
                           prompt += "- ... (further near-miss options truncated)\n"
                           break
                      # Try to get MfgPN for better identification
                      mfg_pn = "Unknown MPN"
                      for strategy_name in ["Optimized Strategy", "Lowest Cost", "Fastest"]:
                           part_detail = strategies.get(strategy_name, {}).get(bom_pn)
                           if part_detail and part_detail.get("ManufacturerPartNumber", "N/A") != "N/A":
                                mfg_pn = part_detail["ManufacturerPartNumber"]; break
                      if mfg_pn == "Unknown MPN": mfg_pn = bom_pn # Fallback to BOM PN

                      details = []
                      if 'slightly_over_lt' in misses:
                           opt = misses['slightly_over_lt'].get('option', {})
                           over_by = misses['slightly_over_lt'].get('over_by_days', 0)
                           details.append(f"Slightly over LT (+{over_by:.0f}d for ${opt.get('cost', 0):.2f} via {opt.get('source', 'N/A')})")
                      if 'slightly_over_cost' in misses:
                           opt = misses['slightly_over_cost'].get('option', {})
                           over_by = misses['slightly_over_cost'].get('over_by_pct', 0)
                           details.append(f"Slightly over Cost Premium (+{over_by:.1f}% for {opt.get('lead_time', 0):.0f}d LT via {opt.get('source', 'N/A')})")

                      if details:
                           prompt += f"- {mfg_pn}: {'; '.join(details)}\n"
                           near_miss_count += 1

                 if near_miss_count == 0: # Case where near_miss_data existed but no qualifying misses
                      prompt += "No options identified within the 'near miss' thresholds (+14 days LT, +5% cost premium).\n"
            prompt += "\n"


            # === Actionable Recommendations ===
            prompt += "=== Actionable Recommendations ===\n"
            prompt += "Provide concise, strategic recommendations for executive review, focusing on:\n"
            prompt += "1.  **Optimal Sourcing Strategy:** Clearly recommend ONE strategy (Lowest Cost, Fastest, or Optimized) as the primary plan. Justify this based on the KPIs (cost, lead time), risk profile, and whether the Optimized strategy met constraints or used fallbacks. Quantify the choice where possible (e.g., 'Optimized saves $X but adds Y days vs. Fastest').\n"
            prompt += "2.  **High/Moderate Risk Part Actions:** For the specific High and Moderate risk parts listed, suggest concrete next steps. Examples: Prioritize ordering, Seek second source, Qualify alternate, Increase buffer stock, Expedite options, Initiate redesign investigation (especially for EOL/sole source high risk).\n"
            prompt += "3.  **Potential Plan B / Trade-offs:** Analyze the 'Near Miss' data. If applicable, identify specific parts where slightly relaxing the LT target OR the cost premium could yield significant benefits (e.g., large cost savings for a few extra days LT, or meeting LT by slightly exceeding cost premium). Advise if these specific trade-offs seem strategically advantageous and worth proposing as alternatives.\n"
            prompt += "4.  **Overall Risk Mitigation:** Suggest 1-2 high-level actions based on the overall risk assessment (e.g., 'Develop secondary suppliers for sole-sourced high-risk items', 'Review buffer stock levels based on calculated lead times and stock gaps', 'Initiate value engineering for high-cost drivers').\n"
            prompt += "5.  **Buy-Up Decisions:** Comment on any parts where the chosen strategy involves buying significantly more than needed ('Actual Qty Ordered' >> 'Total Qty Needed'). Briefly assess the cost saving vs. the inventory risk for these specific parts.\n"

            # Find parts with significant overbuy in the *chosen* strategy (assume Optimized if valid, else Lowest Cost)
            overbuy_parts = []
            chosen_strategy_for_buyup = {}
            if pd.notna(summary_dict.get("Balanced (Optimized Strategy) Cost / LT ($ / Days)", np.nan)):
                 chosen_strategy_for_buyup = strategies.get("Optimized Strategy", {})
            elif pd.notna(summary_dict.get("Lowest Cost Strategy / LT ($ / Days)", np.nan)):
                 chosen_strategy_for_buyup = strategies.get("Lowest Cost", {})

            if chosen_strategy_for_buyup:
                 for bom_pn, option in chosen_strategy_for_buyup.items():
                     needed_val = option.get("total_qty_needed")
                     ordered_val = option.get("actual_order_qty")
                     # Check if types are numeric and ordered > needed * 1.5 (and more than a few units difference)
                     if isinstance(needed_val, (int, float)) and isinstance(ordered_val, (int, float)) and \
                        needed_val > 0 and ordered_val > needed_val * 1.5 and (ordered_val - needed_val) > 10:
                         mfg_pn = option.get("ManufacturerPartNumber", bom_pn)
                         overbuy_parts.append(f"{mfg_pn} (Need {needed_val}, Buy {ordered_val})")

            if overbuy_parts:
                 prompt += f"    *Note on Buy-Ups in Recommended Strategy: Significant over-buys for cost savings identified for: {'; '.join(overbuy_parts[:3])}{'...' if len(overbuy_parts)>3 else ''}.*\n"

            prompt += "6.  **Financial Summary:** Briefly summarize the total estimated cost of the recommended strategy and compare it to alternatives if relevant (e.g., '$X premium for Y days faster delivery compared to Lowest Cost'). Include estimated tariff impact.\n"
            prompt += "Keep language clear, direct, and action-oriented. Use bullet points for recommendations.\n"
            # --- End Prompt ---

            logger.info("AI Summary Thread: Prompt built (with near-miss section), preparing API call.")

            # --- Call OpenAI ---
            ai_response = call_chatgpt(prompt, model="gpt-4o") # Use appropriate model

            if ai_response and "OpenAI" not in ai_response: # Basic check for error messages from call_chatgpt
                 logger.info(f"AI Summary Thread: Received response from OpenAI (Length: {len(ai_response)} chars).")
            else:
                 logger.error(f"AI Summary Thread: Received error or no response from OpenAI. Response: {ai_response}")
                 ai_response = f"Error: Failed to get summary from AI.\n\nDetails: {ai_response}" # Provide error in GUI

            # --- Schedule GUI Update ---
            def update_gui():
                logger.debug("AI Summary Thread: Executing update_gui (SIMPLIFIED VERSION).")
                try:
                    # Basic check - still good practice
                    if not hasattr(self, 'ai_summary_text') or not self.ai_summary_text.winfo_exists():
                        logger.error("AI Summary Text widget missing.")
                        return

                    # Core update logic - similar to original concept
                    self.ai_summary_text.configure(state='normal')
                    self.ai_summary_text.delete(1.0, tk.END)
                    self.ai_summary_text.insert(tk.END, ai_response) # ai_response is from the outer scope
                    self.ai_summary_text.configure(state='disabled')
                    logger.info("AI Summary text updated.")

                    # Tab Switching - use the CORRECT variable name for the current code
                    if hasattr(self, 'results_notebook') and self.results_notebook.winfo_exists() and \
                       hasattr(self, 'predictive_tab') and self.predictive_tab.winfo_exists():
                         try:
                              self.results_notebook.select(self.predictive_tab)
                              logger.debug("Switched to Predictive Analysis tab.")
                         except tk.TclError as tab_err:
                              logger.error(f"Failed to switch to Predictive Analysis tab: {tab_err}")

                    # Update status using the threadsafe method
                    if "Error:" not in ai_response:
                         self.update_status_threadsafe("AI summary generated.", "success")
                    else:
                         self.update_status_threadsafe("Error generating AI summary.", "error")


                except tk.TclError as tk_err: # Catch Tkinter errors
                     logger.error(f"Tkinter Error updating AI summary: {tk_err}", exc_info=True)
                     self.update_status_threadsafe(f"GUI Error updating AI Summary: {tk_err}", "error")
                except Exception as e:
                     logger.error(f"Unexpected error updating AI summary: {e}", exc_info=True)
                     self.update_status_threadsafe(f"Error displaying AI Summary: {e}", "error")


            self.root.after(0, update_gui) # Schedule the update
            logger.info("AI Summary Thread: GUI update scheduled.")

        except Exception as e:
            # Catch errors during data extraction or prompt building
            logger.error(f"AI summary generation thread failed before API call: {e}", exc_info=True)
            self.root.after(0, self.update_status_threadsafe, f"AI Summary Prep Error: {e}", "error")
            # Schedule messagebox on main thread
            self.root.after(0, messagebox.showerror, "AI Summary Error", f"Failed preparing data for AI summary:\n\n{e}")
        finally:
            logger.info("AI Summary Thread: Finishing.")
            self.running_analysis = False
            # Schedule button state update ensures it runs on the main thread
            self.root.after(0, self.validate_inputs)


    # --- GUI Table Helpers ---
    def clear_treeview(self, tree):
        """ Clears all items from a ttk.Treeview. Must run on main thread. """
        if not is_main_thread():
            self.root.after(0, self.clear_treeview, tree); return
        if not hasattr(tree, 'winfo_exists') or not tree.winfo_exists(): return
        try:
             children = tree.get_children()
             if children: tree.delete(*children)
        except tk.TclError: pass # Ignore during shutdown

    def populate_treeview(self, tree, data):
        """ Populates a ttk.Treeview, applying risk tags to main tree. MUST run on main thread. """
        if not is_main_thread():
            self.root.after(0, self.populate_treeview, tree, data); return
        if not hasattr(tree, 'winfo_exists') or not tree.winfo_exists(): return

        self.clear_treeview(tree)
        if not data: return

        try:
             cols = tree['columns']
             if not cols: logger.error("Cannot populate treeview: Columns not defined."); return

             # Special handling for the main parts list tree (self.tree)
             is_main_tree = (tree == self.tree)
             if is_main_tree:
                 self.tree_item_data_map.clear() # Clear previous mapping

             for i, item_data in enumerate(data):
                 values = []
                 tags = () # Default empty tags tuple

                 if isinstance(item_data, dict):
                     # Extract values in the order of tree columns
                     values = [str(item_data.get(col, '')).replace('nan', 'N/A') for col in cols]

                     # Apply Risk Tag only to the main parts list tree
                     if is_main_tree and 'RiskScore' in item_data:
                         risk_score = safe_float(item_data.get('RiskScore'), default=np.nan)
                         if pd.notna(risk_score):
                             if risk_score >= self.RISK_CATEGORIES['high'][0]: tags = ('high_risk',)
                             elif risk_score >= self.RISK_CATEGORIES['moderate'][0]: tags = ('moderate_risk',)
                             else: tags = ('low_risk',)
                         else:
                             tags = ('na_risk',) # Tag for N/A risk scores

                     # Insert row and store mapping if it's the main tree
                     item_id = tree.insert("", "end", values=values, tags=tags)
                     if is_main_tree:
                         self.tree_item_data_map[item_id] = item_data # Store full dict for this row

                 elif isinstance(item_data, (list, tuple)):
                     # Ensure values are strings and handle length mismatch
                     if len(item_data) == len(cols):
                         values = [str(v).replace('nan', 'N/A') if v is not None else '' for v in item_data]
                         tree.insert("", "end", values=values) # No tags or mapping for tuple data
                     else:
                         logger.warning(f"Row length mismatch: Expected {len(cols)}, Got {len(item_data)}. Row: {item_data}")
                 else:
                     logger.warning(f"populate_treeview: Skipping unexpected data type: {type(item_data)}")

        except tk.TclError as e: logger.warning(f"Ignoring TclError populating treeview: {e}")
        except Exception as e:
             logger.error(f"Failed to populate treeview: {e}", exc_info=True)
             self.update_status_threadsafe(f"Error displaying results: {e}", "error")


    # --- Export Main Parts List ---
    def export_treeview_data(self):
        """ Exports the data currently displayed in the main parts list Treeview. """
        logger.info("Exporting main parts list treeview data.")
        if not hasattr(self, 'tree') or not self.tree.winfo_exists():
            messagebox.showerror("Export Error", "Parts list table not available.")
            return

        children = self.tree.get_children()
        if not children:
            messagebox.showinfo("Export Info", "Parts list table is empty. No data to export.")
            return

        # Prepare data for export
        output_data = []
        # Get header from the treeview headings
        header_display = [self.tree.heading(col, 'text') for col in self.tree['columns']]
        output_data.append(header_display)

        # Get data rows
        for item_id in children:
            values = self.tree.item(item_id, 'values')
            output_data.append(list(values))

        # Ask user for filename
        default_filename = f"BOM_Analysis_Summary_{datetime.now():%Y%m%d_%H%M}.csv"
        filepath = filedialog.asksaveasfilename(
            title="Save BOM Analysis Summary As", defaultextension=".csv",
            initialfile=default_filename, filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not filepath: return # User cancelled

        # Write to CSV
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerows(output_data)
            self.update_status_threadsafe(f"Exported Parts List Summary to {Path(filepath).name}", "success")
            messagebox.showinfo("Export Successful", f"Successfully exported parts list summary to:\n{filepath}")
        except IOError as e:
            logger.error(f"Failed to export parts list CSV: {e}", exc_info=True)
            messagebox.showerror("Export Error", f"Failed to write CSV file:\n{filepath}\n\nError: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during parts list export: {e}", exc_info=True)
            messagebox.showerror("Export Error", f"An unexpected error occurred during export:\n\n{e}")


    # --- Application Closing ---
    def on_closing(self):
        """ Handles application close event cleanly. """
        logger.info("Close requested. Initiating shutdown...")
        if messagebox.askokcancel("Quit", f"Are you sure you want to quit {APP_NAME}?"):
            self.update_status_threadsafe("Shutting down...", "info")
            self.running_analysis = False # Signal running threads to stop gracefully if possible

            # Shutdown thread pool - wait briefly for tasks to finish/cancel
            logger.info("Shutting down thread pool...")
            try:
                # Give threads a chance to finish current API call, but cancel pending ones
                self.thread_pool.shutdown(wait=False, cancel_futures=True) # Python 3.9+
                logger.info("Thread pool shutdown requested (cancel_futures=True).")
                # Optional: Add a short delay to allow cancellation to propagate if needed
                # time.sleep(0.5)
            except TypeError: # Fallback for older Python versions without cancel_futures
                self.thread_pool.shutdown(wait=False)
                logger.info("Thread pool shutdown requested (wait=False).")
            except Exception as e:
                 logger.error(f"Error during thread pool shutdown: {e}")

            # Cancel any pending Tkinter 'after' jobs (like token refresh)
            if hasattr(self, '_digikey_refresh_after_id') and self._digikey_refresh_after_id:
                 try: self.root.after_cancel(self._digikey_refresh_after_id)
                 except: pass

            logger.info("Destroying main window...")
            try:
                 self.root.destroy()
            except tk.TclError: pass # Ignore if already destroyed

            logger.info(f"{APP_NAME} closed.")
        else:
            logger.info("Quit cancelled by user.")


# --- Main Execution ---
if __name__ == "__main__":
    root = None # Initialize root to None
    try:
        # --- Matplotlib/Seaborn backend selection (Optional, but good practice for Tkinter embedding) ---
        # import matplotlib
        # matplotlib.use('TkAgg') # Set backend *before* importing pyplot
        # import matplotlib.pyplot as plt
        # import seaborn as sns
        # sns.set_theme(style="whitegrid") # Example Seaborn theme
        # ---
        import contextlib # For suppressing Prophet logs if needed

        root = tk.Tk()
        app = BOMAnalyzerApp(root)
        logger.info(f"Starting {APP_NAME} v{APP_VERSION} main loop...")
        root.mainloop()
        logger.info(f"{APP_NAME} main loop finished.")
    except Exception as main_err:
        logger.critical(f"Critical error during application execution: {main_err}", exc_info=True)
        try:
            # Use a basic Tkinter window for the final error if possible
            err_root = tk.Tk()
            err_root.withdraw()
            messagebox.showerror("Fatal Error", f"A critical error occurred:\n\n{main_err}\n\nSee logs for details. Application will exit.")
            err_root.destroy()
        except Exception as popup_err:
            print("\n" + "="*60)
            print(f"FATAL APPLICATION ERROR: {main_err}")
            print(f"(Also failed to show error popup: {popup_err})")
            print("Please check the application logs for details.")
            print("="*60 + "\n")
        finally:
            if root and root.winfo_exists():
                try: root.destroy()
                except: pass
            sys.exit(1) # Exit with error code 
    
