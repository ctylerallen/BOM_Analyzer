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
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import seaborn as sns 
import subprocess

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

# Log messages at different levels
logging.debug("This is a debug message")
logging.info("This is an info message")
logging.warning("This is a warning message")
logging.error("This is an error message")
logging.critical("This is a critical message")

# --- Configuration & Constants ---
try:
    # Try using __file__ first, as it's the most reliable when running as a script
    SCRIPT_DIR = Path(__file__).parent.resolve()
    logger.info(f"Script directory detected as: {SCRIPT_DIR} (using __file__)")
except NameError:
    # Fallback to current working directory if __file__ is not defined
    SCRIPT_DIR = Path.cwd().resolve()
    logger.warning(f"__file__ not defined, using current working directory as script directory: {SCRIPT_DIR}")
    logger.warning("Ensure keys.env and data files are relative to this directory or provide full paths.")
CACHE_DIR = SCRIPT_DIR / 'cache'
os.makedirs(CACHE_DIR, exist_ok=True)
CERT_FILE = SCRIPT_DIR / "localhost.pem" # Expected cert file location
APP_CONFIG_FILE = CACHE_DIR / 'app_config.json'
DEFAULT_APP_CONFIG = {
    "show_startup_guide": True
}

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
APP_VERSION = "1.0.0" # Updated Version

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

def load_app_config():
    """Loads application configuration from JSON file."""
    if not APP_CONFIG_FILE.exists():
        return DEFAULT_APP_CONFIG.copy() # Return default if file not found
    try:
        with open(APP_CONFIG_FILE, 'r') as f:
            config = json.load(f)
            # Ensure all default keys exist
            for key, value in DEFAULT_APP_CONFIG.items():
                config.setdefault(key, value)
            return config
    except (json.JSONDecodeError, IOError, Exception) as e:
        logger.error(f"Failed to load app config from {APP_CONFIG_FILE}: {e}. Using defaults.")
        return DEFAULT_APP_CONFIG.copy()
        

def save_app_config(config_data):
    """Saves application configuration to JSON file."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(APP_CONFIG_FILE, 'w') as f:
            json.dump(config_data, f, indent=2)
    except IOError as e:
        logger.error(f"Failed to save app config to {APP_CONFIG_FILE}: {e}")
        

# --- OAuth Handler (for DigiKey) ---
class OAuthHandler(BaseHTTPRequestHandler):
    """Handles the OAuth callback from DigiKey."""
    def do_GET(self):
        try:
            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)
            code = params.get('code', [None])[0]
            state = params.get('state', [None])[0] # Optional: For security

            if code:
                self.server.auth_code = code
                self.send_response(200)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(b"<html><body><h1>Authentication Successful!</h1>"
                                 b"<p>Authorization code received. You can close this window and return to the BOM Analyzer.</p>"
                                 b"</body></html>")
                logger.info("OAuth code received successfully.")
            else:
                error = params.get('error', ['Unknown error'])[0]
                error_desc = params.get('error_description', ['No description'])[0]
                self.send_response(400)
                self.send_header('Content-type', 'text/html')
                self.end_headers()
                self.wfile.write(f"<html><body><h1>Authentication Failed</h1>"
                                 f"<p>Error: {error}</p><p>Description: {error_desc}</p>"
                                 f"<p>Please close this window and try again.</p>"
                                 f"</body></html>".encode('utf-8'))
                logger.error(f"OAuth failed. Error: {error}, Description: {error_desc}")
                self.server.auth_code = None # Signal failure

        except Exception as e:
            self.send_response(500)
            self.end_headers()
            self.wfile.write(b"Internal Server Error during OAuth callback.")
            logger.error(f"Error in OAuthHandler: {e}", exc_info=True)
            self.server.auth_code = None

    def log_message(self, format, *args):
        # Quieten down the server logging unless debugging
        # logger.debug(f"OAuthServer: {format % args}")
        pass
        

# --- Main Application Class ---
class BOMAnalyzerApp:
 

    # Define color palette as class attributes for consistency
    COLOR_BACKGROUND = "#e1e1e1" # Medium-Light Gray (Warmer than default 'clam')
    COLOR_FRAME_BG = "#f0f0f0"   # Lighter Gray for content frames (subtle contrast)
    COLOR_TEXT = "#333333"      # Darker Gray Text
    COLOR_ACCENT = "#0078d4"    # A slightly different blue (Windows blue)
    COLOR_SUCCESS = "#107c10"   # Darker Green
    COLOR_WARN = "#ca5010"      # Darker Orange
    COLOR_ERROR = "#d13438"     # Darker Red
    COLOR_DISABLED = "#a19f9d"  # Gray for disabled elements
    COLOR_BORDER = "#c1c1c1"    # Slightly darker border
    COLOR_PLOT_BG = "#ffffff"    # Keep plots white
    COLOR_TREE_HEADING = "#d1d1d1" # Gray for tree headings
    COLOR_TREE_ROW_ALT = "#f5f5f5"  # Alternate row color
    COLOR_STATUS_BAR_BG = "#b1b1b1" # Darker status bar
    COLOR_STATUS_BAR_TEXT = "#111111" # Very dark text for status bar

    # --- Define Font Sizes as Class Attributes ---
    FONT_FAMILY = "Segoe UI" # Or choose another suitable font
    FONT_SIZE_SMALL = 8
    FONT_SIZE_NORMAL = 10     # Increased from 9
    FONT_SIZE_LARGE = 11     # For headings/tabs
    FONT_SIZE_XLARGE = 15    # For main titles
    # --- End Font Sizes ---
 

    # --- Risk Configuration Constants ---
    RISK_WEIGHTS = {'Sourcing': 0.30, 'Stock': 0.15, 'LeadTime': 0.15, 'Lifecycle': 0.30, 'Geographic': 0.10}
    GEO_RISK_TIERS = {
        "China": 7, "Russia": 9, "Taiwan": 5, "Malaysia": 4, "Vietnam": 4, "India": 5, "Philippines": 4,
        "Thailand": 4, "South Korea": 3, "USA": 1, "United States": 1, "Mexico": 2, "Canada": 1, "Japan": 1,
        "Germany": 1, "France": 1, "UK": 1, "Ireland": 1, "Switzerland": 1, "EU": 1,
        "Unknown": 4, "N/A": 4, "_DEFAULT_": 4
    }
    RISK_CATEGORIES = {'high': (6.6, 10.0), 'moderate': (3.6, 6.5), 'low': (0.0, 3.5)}
    # --- End Risk Configuration ---

    def __init__(self, root):
        self.root = root
        self.FONT_FAMILY = BOMAnalyzerApp.FONT_FAMILY
        self.FONT_SIZE_SMALL = BOMAnalyzerApp.FONT_SIZE_SMALL
        self.FONT_SIZE_NORMAL = BOMAnalyzerApp.FONT_SIZE_NORMAL
        self.FONT_SIZE_LARGE = BOMAnalyzerApp.FONT_SIZE_LARGE
        self.FONT_SIZE_XLARGE = BOMAnalyzerApp.FONT_SIZE_XLARGE

        # Initialize widget variables
        self.load_button = self.file_label = self.run_button = self.predict_button = None
        self.ai_summary_button = self.validation_label = self.tree = self.analysis_table = None
        self.predictions_tree = self.ai_summary_text = self.status_label = self.progress = None
        self.progress_label = self.rate_label = self.universal_status_bar = None
        self.universal_tooltip_label = self.plot_combo = self.plot_frame = None
        self.fig_canvas = self.toolbar = self.export_parts_list_btn = None
        self.lowest_cost_btn = self.fastest_btn = self.optimized_strategy_btn = None
        self.lowest_cost_strict_btn = self.in_stock_btn = self.with_lt_btn = None
        self.alt_popup = None  # Track the alternates popup
        self.summary_popup = None  # Track the summary details popup

        self.export_recommended_btn = None
        self.ai_recommended_strategy_key = None
        self.root.title(f"{APP_NAME} - v{APP_VERSION}")
        self.root.geometry("1500x900")
        self.root.minsize(1500, 900)  # Lock window size
        self.root.maxsize(1500, 900)
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=0)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.configure(bg=self.COLOR_BACKGROUND)

        # --- Theme and Style Configuration ---
        self.style = ttk.Style()
        available_themes = self.style.theme_names()
        logger.debug(f"Available themes: {available_themes}")
        preferred_themes = ['clam', 'alt', 'vista', 'xpnative', 'default']
        chosen_theme = None
        for theme in preferred_themes:
            if theme in available_themes:
                try: self.style.theme_use(theme); logger.info(f"Using theme: {theme}"); chosen_theme = theme; break
                except tk.TclError: logger.warning(f"Could not use theme '{theme}'.")
        if not chosen_theme: logger.warning("Using system default theme.")

        self.style.configure(".", background=self.COLOR_BACKGROUND, foreground=self.COLOR_TEXT, bordercolor=self.COLOR_BORDER, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL))
        self.style.configure("TFrame", background=self.COLOR_BACKGROUND)
        self.style.configure("Card.TFrame", background=self.COLOR_FRAME_BG, relief='raised', borderwidth=1)
        self.style.configure("InnerCard.TFrame", background=self.COLOR_FRAME_BG)
        self.style.configure("TLabelframe", background=self.COLOR_BACKGROUND, bordercolor=self.COLOR_BORDER, relief="groove", borderwidth=1, padding=10)
        self.style.configure("TLabelframe.Label", background=self.COLOR_BACKGROUND, foreground=self.COLOR_TEXT, font=(self.FONT_FAMILY, self.FONT_SIZE_LARGE, "bold"))
        self.style.configure("TLabel", background=self.COLOR_BACKGROUND, foreground=self.COLOR_TEXT, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL), padding=(0, 2))
        self.style.configure("Title.TLabel", font=(self.FONT_FAMILY, self.FONT_SIZE_XLARGE, "bold"), foreground=self.COLOR_ACCENT, background=self.COLOR_FRAME_BG)
        self.style.configure("Hint.TLabel", foreground="#555555", font=(self.FONT_FAMILY, self.FONT_SIZE_SMALL), background=self.COLOR_FRAME_BG)
        self.style.configure("Status.TLabel", font=(self.FONT_FAMILY, self.FONT_SIZE_SMALL), padding=[0, 10])
        self.style.configure("TButton", font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL), padding=(10, 5), relief="raised", borderwidth=1)
        self.style.map("TButton",
            background=[('active', '#c0c0c0'), ('!disabled', '#dcdcdc')],
            bordercolor=[('focus', self.COLOR_ACCENT)],
            relief=[('pressed', 'sunken')]
        )
        self.style.configure("Accent.TButton", font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL, "bold"), foreground="white", background=self.COLOR_ACCENT, borderwidth=1)
        self.style.map("Accent.TButton", background=[('active', '#005a9e'), ('!disabled', self.COLOR_ACCENT)])

        treeview_rowheight = 28
        self.style.configure("Treeview", background=self.COLOR_FRAME_BG, fieldbackground=self.COLOR_FRAME_BG, foreground=self.COLOR_TEXT, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL), rowheight=treeview_rowheight)
        self.style.map('Treeview', background=[('selected', self.COLOR_ACCENT)], foreground=[('selected', 'white')])
        self.style.configure("Treeview.Heading", font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL, "bold"), background=self.COLOR_TREE_HEADING, relief="raised", borderwidth=1, padding=(4, 4))
        self.style.map("Treeview.Heading", relief=[('active','raised'),('pressed','raised')])
        self.style.configure("high_risk.Treeview", background='#fee2e2')
        self.style.configure("moderate_risk.Treeview", background='#fef3c7')
        self.style.configure("low_risk.Treeview", background='#dcfce7')
        self.style.configure("na_risk.Treeview", background='#f3f4f6')
        self.style.configure("warn_metric.Treeview", background='#fffbeb')
        self.style.configure("error_metric.Treeview", background='#fff1f2')

        self.style.configure("TNotebook", background=self.COLOR_BACKGROUND, borderwidth=0, tabmargins=[5, 5, 5, 0])
        self.style.configure("TNotebook.Tab", font=(self.FONT_FAMILY, self.FONT_SIZE_LARGE, "bold"), padding=[12, 6], background="#d1d5db", foreground="#4b5563")
        self.style.map("TNotebook.Tab", background=[("selected", self.COLOR_FRAME_BG), ('!selected', '#e5e7eb')], foreground=[("selected", self.COLOR_ACCENT), ('!selected', '#4b5563')], expand=[("selected", [1, 1, 1, 0])])
        self.style.configure("TScrollbar", background=self.COLOR_BACKGROUND, troughcolor='#e5e7eb', bordercolor="#cccccc", arrowcolor='#333333')
        self.style.configure("TProgressbar", troughcolor='#e5e7eb', background=self.COLOR_ACCENT, thickness=15)
        self.style.configure("StatusBar.TFrame", background=self.COLOR_STATUS_BAR_BG, borderwidth=1, relief='flat')
        self.style.configure("Status.TLabel", background=self.COLOR_STATUS_BAR_BG, foreground=self.COLOR_STATUS_BAR_TEXT)

        logger.info("Initializing GUI...")

        # Configure tags for emphasis in the AI summary ScrolledText widget
        # Note: These are applied directly to the Text widget, not via ttk styles
        # Ensure ai_summary_text widget exists before configuring tags if called later,
        # but it's generally safe to configure tags after widget creation.
        # We will apply them *during* text insertion later.
        # No direct code needed here IF we apply during insertion, but good to be aware.
        # Alternatively, configure them after the widget is created:
        # self.ai_summary_text.tag_configure("critical", foreground="red", font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL, "bold"))
        # self.ai_summary_text.tag_configure("warning", foreground=self.COLOR_WARN, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL, "bold"))
        # self.ai_summary_text.tag_configure("highlight", background="yellow")
        self.ai_summary_text_widget = None
        
        self.tooltip_texts = {}
        self.hist_header = ['Component', 'Manufacturer', 'Part_Number', 'Distributor', 'Lead_Time_Days', 'Cost', 'Inventory', 'Stock_Probability', 'Fetch_Timestamp']
        self.pred_header = ['Component', 'Date', 'Prophet_Lead', 'Prophet_Cost', 'RAG_Lead', 'RAG_Cost', 'AI_Lead', 'AI_Cost', 'Stock_Probability', 'Real_Lead', 'Real_Cost', 'Real_Stock', 'Prophet_Ld_Acc', 'Prophet_Cost_Acc', 'RAG_Ld_Acc', 'RAG_Cost_Acc']

        # --- Main Layout ---
        self.main_paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned_window.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 5))
        logger.debug("Main paned window gridded onto root window.")

        # --- Universal Status Bar ---
        STATUS_BAR_HEIGHT = 50
        self.universal_status_bar = ttk.Frame(self.root, style="StatusBar.TFrame", height=STATUS_BAR_HEIGHT)
        # ... (Label setup and packing inside status bar frame) ...
        self.universal_tooltip_label = ttk.Label(self.universal_status_bar, text=" ", anchor='nw', wraplength=0, style="Status.TLabel", justify='left', font=("Segoe UI",12))
        # --- End Change ---
        self.universal_tooltip_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=(1, 1))
        # Grid status bar in root window, row 1
        self.universal_status_bar.grid(row=1, column=0, sticky="ew", padx=0, pady=(0, 0)) # columnspan=1 or remove
        self.universal_status_bar.grid_propagate(False) 

        # --- Left Pane: Configuration ---
        self.config_frame_outer = ttk.Frame(self.main_paned_window, padding=0, width=450, style="Card.TFrame")
        self.config_frame_outer.grid(row=0, column=0, sticky="nsew", padx=(10, 5), pady=(10, 5))
        self.config_frame = ttk.Frame(self.config_frame_outer, padding=(15, 15), style="InnerCard.TFrame") # Parent is now config_frame_outer
        self.config_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        #self.config_scroll_canvas = tk.Canvas(self.config_frame_outer, borderwidth=0, background=self.COLOR_FRAME_BG, highlightthickness=0)
        #self.config_scrollbar = ttk.Scrollbar(self.config_frame_outer, orient="vertical", command=self.config_scroll_canvas.yview)
        #self.config_frame = ttk.Frame(self.config_scroll_canvas, padding=(15, 15), style="InnerCard.TFrame")
        #self.config_frame.bind("<Configure>", lambda e: self.config_scroll_canvas.configure(scrollregion=self.config_scroll_canvas.bbox("all")))
        #self.config_scroll_canvas.create_window((0, 0), window=self.config_frame, anchor="nw")
        #self.config_scroll_canvas.configure(yscrollcommand=self.config_scrollbar.set)
        #self.config_scroll_canvas.pack(side="left", fill="both", expand=True); self.config_scrollbar.pack(side="right", fill="y")
        
        """# Mouse wheel binding
        def _on_mousewheel_config(event):
            delta = 0
            if event.num == 4: delta = -1; # Linux scroll up
            elif event.num == 5: delta = 1; # Linux scroll down
            elif event.delta > 0: delta = -1; # Windows/Mac scroll up
            elif event.delta < 0: delta = 1; # Windows/Mac scroll down
            if delta != 0: self.config_scroll_canvas.yview_scroll(delta, "units")
        self.config_frame.bind_all("<MouseWheel>", _on_mousewheel_config)
        self.config_frame.bind_all("<Button-4>", _on_mousewheel_config)
        self.config_frame.bind_all("<Button-5>", _on_mousewheel_config)
        """
        
        # --- Configuration Widgets ---
        ttk.Label(self.config_frame, text="BOM Analyzer v1.0.0", style="Title.TLabel").pack(fill="x", pady=(0, 15), anchor='w')

        instruction_label = ttk.Label(self.config_frame, text="Load BOM to Start BOM Analysis", style="Hint.TLabel", font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL,"bold", "italic")) 
        instruction_label.pack(fill='x', anchor='w', padx=5, pady=(5, 2)) #

        # Load BOM Section Frame (Container for the grid)
        load_bom_frame = ttk.Frame(self.config_frame, style="InnerCard.TFrame")
        load_bom_frame.pack(fill="x", pady=(0, 10))

        # Configure Grid Columns within load_bom_frame
        load_bom_frame.columnconfigure(0, weight=0)  # Load BOM button column
        load_bom_frame.columnconfigure(1, weight=0)  # Stacked buttons column
        load_bom_frame.columnconfigure(2, weight=1)  # File label column (expands)

        # 1. Load BOM Button (Primary Action)
        self.load_button = ttk.Button(load_bom_frame, text="Load BOM...", command=self.load_bom, style="Accent.TButton") # Apply Accent style
        self.load_button.grid(row=0, column=0, rowspan=2, padx=(0, 85), pady=3, ipady=5, sticky='nsew')
        self.create_tooltip(self.load_button, "Load a Bill of Materials (BOM) in CSV format. Requires columns like 'Part Number' and 'Quantity'.")

        # 2. Frame for Stacked Buttons (Edit Keys, Show Guide)
        stacked_buttons_frame = ttk.Frame(load_bom_frame, style="InnerCard.TFrame") # Use same style for consistency
        stacked_buttons_frame.grid(row=0, column=1, rowspan=2, sticky='ns', padx=5) # Place frame in grid column 1

        # Edit API Keys Button (inside stacked frame)
        self.edit_keys_button = ttk.Button(stacked_buttons_frame, text="Edit API Keys", command=self.edit_keys_file, width=12)
        self.edit_keys_button.pack(side=tk.TOP, pady=(2, 2), fill='x', expand=False) # Pack vertically inside stacked frame
        self.create_tooltip(self.edit_keys_button, f"Open the '{env_path.name}' file in your default text editor.\nRestart required after saving changes.")

        # Show Guide Button (inside stacked frame)
        self.show_guide_button = ttk.Button(stacked_buttons_frame, text="Getting Started", command=self.show_startup_guide_popup, width=12)
        self.show_guide_button.pack(side=tk.TOP, pady=(2, 2), fill='x', expand=False) # Pack vertically below edit button
        self.create_tooltip(self.show_guide_button, "Display the Getting Started guide popup again.")

        # 3. File Label
        self.file_label = ttk.Label(load_bom_frame, text="No BOM loaded.", style="Hint.TLabel", wraplength=200, background=self.COLOR_FRAME_BG, anchor='w')
        self.file_label.grid(row=0, column=2, rowspan=2, sticky='nsew', padx=(15, 0)) # Place in grid column 2

        # Analysis Controls Section
        run_frame = ttk.LabelFrame(self.config_frame, text="Analysis Controls", padding=10)
        run_frame.pack(fill="x", pady=(10, 10))
        self.run_button = ttk.Button(run_frame, text="1. Run Analysis", command=self.validate_and_run_analysis, state="disabled", style="Accent.TButton")
        self.run_button.pack(side=tk.LEFT, padx=(0,5), ipady=2) # Keep pack for this frame's internal layout
        self.create_tooltip(self.run_button, "Run the full analysis using current BOM and configuration.\nFetches data from suppliers, calculates risk, and determines strategies.")
        self.predict_button = ttk.Button(run_frame, text="2. Run Predictions", command=self.run_predictive_analysis_gui, state="disabled")
        self.predict_button.pack(side=tk.LEFT, padx=5, ipady=2) # Keep pack for this frame's internal layout
        self.create_tooltip(self.predict_button, "Generate future cost/lead time predictions based on historical data.\nRequires historical data from previous analysis runs.")
        self.ai_summary_button = ttk.Button(run_frame, text="3. AI Summary", command=self.generate_ai_summary_gui, state="disabled")
        self.ai_summary_button.pack(side=tk.LEFT, ipady=2) # Keep pack for this frame's internal layout
        self.create_tooltip(self.ai_summary_button, "Generate an executive summary and recommendations using OpenAI.\nRequires analysis results and an OpenAI API key.")
        ttk.Separator(self.config_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(15, 10), padx=5)
        
        # Optimized Strategy Config Section
        optimized_strategy_frame = ttk.LabelFrame(self.config_frame, text="Optimized Strategy Configuration", padding=10)
        optimized_strategy_frame.pack(fill="x", pady=(0, 5))
        config_entries = [ ("Total Units to Build:", "total_units", "100", "Number of finished units to build (calculates total quantity needed per part)."), ("Max Cost Premium (%):", "max_premium", "15", "Maximum percentage increase over the absolute lowest total cost allowed for a part in the Optimized Strategy."), ("Target Lead Time (days):", "target_lead_time_days", "56", "Maximum acceptable lead time (days) for any part chosen in the Optimized Strategy."), ("Cost Weight (0-1):", "cost_weight", "0.5", "Priority for minimizing cost (0=ignore, 1=only cost). Must sum to 1 with Lead Time Weight."), ("Lead Time Weight (0-1):", "lead_time_weight", "0.5", "Priority for minimizing lead time (0=ignore, 1=only LT). Must sum to 1 with Cost Weight."), ("Buy-Up Threshold (%):", "buy_up_threshold", "1", "Allow buying more parts (e.g., next price break) if total cost increases by no more than this percentage compared to buying the exact needed amount (or MOQ). Set to 0 to disable."), ]
        self.config_vars = {}; optimized_strategy_frame.columnconfigure(1, weight=1)
        for i, (label, attr, default, hint) in enumerate(config_entries):
            lbl = ttk.Label(optimized_strategy_frame, text=label)
            lbl.grid(row=i, column=0, sticky="w", padx=(0, 5), pady=3)
            entry = ttk.Entry(optimized_strategy_frame, width=8, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL))
            entry.grid(row=i, column=1, sticky="w", pady=3)
            entry.insert(0, default); self.config_vars[attr] = entry; entry.bind("<KeyRelease>", self.validate_inputs)
            self.create_tooltip(lbl, hint); self.create_tooltip(entry, hint)
            
        # Tariff Config Section
        self.tariff_frame = ttk.LabelFrame(self.config_frame, text="Custom Tariff Rates (%)", padding=10)
        self.tariff_frame.pack(fill="x", pady=(10, 5))
        self.tariff_entries = {}
        top_countries = sorted(["China", "Mexico", "India", "Vietnam", "Taiwan", "Japan", "Malaysia", "Germany", "USA", "Philippines", "Thailand", "South Korea"])
        num_cols_tariff = 3; self.tariff_frame.columnconfigure((1, 3, 5), weight=1)
        for i, country in enumerate(top_countries):
            row, col_idx = divmod(i, num_cols_tariff)
            frame = ttk.Frame(self.tariff_frame, style="InnerCard.TFrame")
            frame.grid(row=row, column=col_idx*2, columnspan=2, sticky="ew", padx=5, pady=2); frame.columnconfigure(1, weight=1)
            lbl = ttk.Label(frame, text=f"{country}:", width=12, anchor='w', background=self.COLOR_FRAME_BG)
            lbl.pack(side=tk.LEFT, padx=(0,2))
            entry = ttk.Entry(frame, width=5, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL))
            entry.insert(0, ""); entry.pack(side=tk.LEFT, fill=tk.X, expand=True); self.tariff_entries[country] = entry
            hint_tariff = f"Custom tariff rate (%) for parts from '{country}'.\nLeave blank to use USITC lookup or default/predicted rate."
            self.create_tooltip(lbl, hint_tariff); self.create_tooltip(entry, hint_tariff); entry.bind("<KeyRelease>", self.validate_inputs)
        ttk.Label(self.tariff_frame, text="(Blank uses default/predicted)", style="Hint.TLabel", background=self.COLOR_FRAME_BG).grid(row=(len(top_countries) + num_cols_tariff -1)//num_cols_tariff, column=0, columnspan=num_cols_tariff*2, pady=(8,0), sticky='w')
        
        # Validation Label
        self.validation_label = ttk.Label(self.config_frame, text="", foreground=self.COLOR_ERROR, wraplength=350, font=(self.FONT_FAMILY, self.FONT_SIZE_SMALL))
        self.validation_label.pack(fill="x", pady=(10, 10), anchor='w')
        
        # API Status Section
        api_status_frame = ttk.LabelFrame(self.config_frame, text="API Status", padding=10)
        api_status_frame.pack(fill="x", pady=(10, 5), anchor='w')
        self.api_status_labels = {}
        api_status_frame.columnconfigure(0, weight=0) # Label column 1 (fixed width)
        api_status_frame.columnconfigure(1, weight=1) # Status column 1 (expands)
        api_status_frame.columnconfigure(2, weight=0, pad=15) # Label column 2 (fixed width, add padding before)
        api_status_frame.columnconfigure(3, weight=1) # Status column 2 (expands)
        
        api_items = list(API_KEYS.items()) # Get items to iterate over
        num_items = len(api_items)
        num_rows = (num_items + 1) // 2 # Calculate needed rows for 2 columns
        
        for i, (api_name, is_set) in enumerate(api_items):
             # Determine status text and color
             if is_set: status_text = "OK"; color = self.COLOR_SUCCESS
             elif api_name == "OpenAI": status_text = "Not Set (Optional)"; color = self.COLOR_WARN
             else: status_text = "Not Set"; color = self.COLOR_ERROR

             # Calculate grid position (row, column_index)
             row_num = i % num_rows
             col_idx = (i // num_rows) * 2 # 0 for first column, 2 for second column

             # Create and grid the widgets
             lbl_name = ttk.Label(api_status_frame, text=f"{api_name}:", width=15)
             lbl_name.grid(row=row_num, column=col_idx, sticky='w', padx=(0, 5), pady=1)

             lbl_status = ttk.Label(api_status_frame, text=status_text, foreground=color, anchor='w')
             lbl_status.grid(row=row_num, column=col_idx + 1, sticky='ew', pady=1)
             self.api_status_labels[api_name] = lbl_status

        # --- Right Pane: Results ---
        self.results_frame = ttk.Frame(self.main_paned_window, padding=(5, 0, 10, 0))
        self.results_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=(10, 5))
        self.main_paned_window.add(self.config_frame_outer, weight=1) # Adjust weight as needed
        self.main_paned_window.add(self.results_frame, weight=3)
        self.results_frame.grid_rowconfigure(1, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)

        # Status Bar Area within Results Pane
        status_progress_frame = ttk.Frame(self.results_frame, padding=(0, 5))
        status_progress_frame.grid(row=0, column=0, sticky="ew")
        status_progress_frame.grid_columnconfigure(0, weight=3); 
        status_progress_frame.grid_columnconfigure(1, weight=1); 
        status_progress_frame.grid_columnconfigure(2, weight=0); 
        status_progress_frame.grid_columnconfigure(3, weight=2)

        self.status_label = ttk.Label(status_progress_frame, text="Ready", anchor="w"); self.status_label.grid(row=0, column=0, padx=(0, 5), sticky="ew")
        self.progress = ttk.Progressbar(status_progress_frame, orient="horizontal", length=150, mode="determinate"); self.progress.grid(row=0, column=1, padx=5, sticky="ew")
        self.progress_label = ttk.Label(status_progress_frame, text="0%", width=5); self.progress_label.grid(row=0, column=2, padx=(0, 5), sticky="w")
        self.rate_label = ttk.Label(status_progress_frame, text="API Rates: -", anchor="e", style="Hint.TLabel"); self.rate_label.grid(row=0, column=3, padx=(10, 0), sticky="ew")
        
        # --- Results Notebook ---
        self.results_notebook = ttk.Notebook(self.results_frame); self.results_notebook.grid(row=1, column=0, sticky="nsew", pady=(5,0))
        # --- Tab 1: BOM Analysis Summary ---
        self.analysis_tab = ttk.Frame(self.results_notebook, padding=(0, 5, 0, 5))
        self.results_notebook.add(self.analysis_tab, text=" BOM Analysis ")
        self.analysis_tab.grid_columnconfigure(0, weight=1); 
        self.analysis_tab.grid_rowconfigure(0, weight=1); 
        self.analysis_tab.grid_rowconfigure(1, weight=0)
        
        # --- Vertical PanedWindow ---
        self.analysis_pane = ttk.PanedWindow(self.analysis_tab, orient=tk.VERTICAL)
        self.analysis_pane.grid(row=0, column=0, sticky="nsew")
        
        # -- Top Pane: Parts Treeview --
        tree_frame_outer = ttk.Frame(self.analysis_pane, style="Card.TFrame")
        tree_frame_outer.grid_rowconfigure(0, weight=1); tree_frame_outer.grid_columnconfigure(0, weight=1)
        columns = [
    "PartNumber", "Manufacturer", "MfgPN", "QtyNeed", "Status", "Sources", "StockAvail", "COO", "RiskScore", "TariffPct",
    "BestCostPer", "BestTotalCost", "ActualBuyQty", "BestCostLT", "BestCostSrc", "Alternates", "Notes"
]
        headings = [
    "BOM P/N", "Manufacturer", "Mfg P/N", "Need", "Lifecycle", "Aval Sources", "Stock", "COO", "Risk", "Tariff",
    "Unit Cost", "Total Cost", "Buy Qty", "LT", "Source", "Alts?", "Notes/Flags"
]
        col_widths = {
    "PartNumber": 140, "Manufacturer": 110, "MfgPN": 90, "QtyNeed": 50, "Status": 65, "Sources": 80, "StockAvail": 70, "COO": 45, "RiskScore": 45, "TariffPct": 50,
    "BestCostPer": 70, "BestTotalCost": 75, "ActualBuyQty": 55, "BestCostLT": 35, "BestCostSrc": 50, "Alternates": 40, "Notes": 150
}
        col_align = {
    "PartNumber": 'w', "Manufacturer": 'w', "MfgPN": 'w', "QtyNeed": 'center', "Status": 'center', "Sources": 'center', "StockAvail": 'e', "COO": 'center', "RiskScore": 'center', "TariffPct": 'e',
    "BestCostPer": 'e', "BestTotalCost": 'e', "ActualBuyQty": 'center', "BestCostLT": 'center', "BestCostSrc": 'center', "Alternates": 'center', "Notes": 'w'
}
        col_tooltips = {
    "PartNumber": "Part number from the input BOM.", 
    "Manufacturer": "Consolidated Manufacturer Name.", 
    "MfgPN": "Consolidated Manufacturer Part Number.", 
    "QtyNeed": "Total quantity needed (BOM Qty/Unit * Total Units).", 
    "Status": "Lifecycle status (Active, EOL, Discontinued, NRND).", 
    "Aval Sources": "Number of suppliers found with data.", 
    "StockAvail": "Total stock across all valid sources.", 
    "COO": "Country of Origin.", 
    "RiskScore": "Overall Risk Score (0-10). Higher=More Risk.\nRed(>6.5), Yellow(3.6-6.5), Green(<=3.5).\nFactors: Sourcing, Stock, LeadTime, Lifecycle, Geo.", 
    "TariffPct": "Estimated Tariff Rate (%) based on COO/HTS.", 
    "BestCostPer": "Lowest Unit Cost ($) found for the chosen 'Actual Buy Qty'.", 
    "BestTotalCost": "Lowest Total Cost ($) for the 'Actual Buy Qty' (may include price break optimization).", 
    "ActualBuyQty": "Quantity chosen for the 'Best Total Cost' calculation (may be > QtyNeed due to MOQ or price breaks).", 
    "BestCostLT": "Lead Time (days) for the Best Total Cost option.", 
    "BestCostSrc": "Supplier for the Best Total Cost option.", 
    "Alternates": "Indicates if potential alternates were found. Double-click row to view.", 
    "Notes": "Additional notes: Stock Gap, EOL/Discontinued flags, Buy-up reasons."
}

        self.tree_hsb = ttk.Scrollbar(tree_frame_outer, orient="horizontal") 
        self.tree_vsb = ttk.Scrollbar(tree_frame_outer, orient="vertical")   
        self.tree_column_tooltips = {}; 
        self.tree = ttk.Treeview(tree_frame_outer, columns=columns, show="headings", height=18, selectmode="browse",
                                  yscrollcommand=self.tree_vsb.set, xscrollcommand=self.tree_hsb.set)
       
        self.tree_hsb.config(command=self.tree.xview)
        self.tree_vsb.config(command=self.tree.yview)
        self.tree_hsb.pack(side=tk.BOTTOM, fill=tk.X)     
        self.tree_vsb.pack(side=tk.RIGHT, fill=tk.Y)      
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True) 
        
        for col, heading in zip(columns, headings):
            width = col_widths.get(col, 90); align = col_align.get(col, 'w'); tooltip_text = col_tooltips.get(col, heading)
            self.tree.heading(col, text=heading, command=lambda c=col: self.sort_treeview(self.tree, c, False), anchor='center')
            self.tree.column(col, width=width, minwidth=10, stretch=True, anchor=align)
            self.tree_column_tooltips[col] = tooltip_text
        
        self.tree.tag_configure('high_risk', background='#fee2e2');
        self.tree.tag_configure('moderate_risk', background='#fef3c7'); self.tree.tag_configure('low_risk', background='#dcfce7'); self.tree.tag_configure('na_risk', background='#f3f4f6')
        self.tree.bind("<Motion>", self._on_treeview_motion);
        self.tree.bind("<Leave>", self._on_treeview_leave); 
        self.tree.bind("<Double-Button-1>", self.show_alternates_popup)
        self.analysis_pane.add(tree_frame_outer, weight=3)
        
        # -- Bottom Pane: Analysis Summary Table --
        self.analysis_table_frame = ttk.LabelFrame(self.analysis_pane, text="BOM Summary Metrics", padding=(10, 5))
        self.analysis_table_frame.grid_columnconfigure(0, weight=1); self.analysis_table_frame.grid_rowconfigure(0, weight=1)
        self.analysis_table = ttk.Treeview(self.analysis_table_frame, columns=["Metric", "Value"], show="headings", height=10, selectmode="browse")
        self.analysis_table.heading("Metric", text="Metric", anchor='w'); self.analysis_table.heading("Value", text="Value", anchor='w')
        self.analysis_table.column("Metric", width=280, stretch=False, anchor='w'); self.analysis_table.column("Value", width=450, stretch=True, anchor='w')
        self.analysis_table_scrollbar = ttk.Scrollbar(self.analysis_table_frame, orient="vertical", command=self.analysis_table.yview)
        self.analysis_table.configure(yscrollcommand=self.analysis_table_scrollbar.set); 
        self.analysis_table.grid(row=0, column=0, sticky="nsew"); 
        self.analysis_table_scrollbar.grid(row=0, column=1, sticky="ns")
        self.analysis_table.bind("<Enter>", self._on_widget_enter, add='+'); 
        self.analysis_table.bind("<Leave>", self._on_widget_leave, add='+'); 
        self.analysis_table.bind("<Motion>", self._on_summary_table_motion, add='+')
        self.analysis_pane.add(self.analysis_table_frame, weight=2)
        
        # -- Export Buttons (Below the PanedWindow) --
        export_buttons_main_frame = ttk.LabelFrame(self.analysis_tab, text="Export Options", padding=(10,5))
        export_buttons_main_frame.grid(row=1, column=0, sticky="ew", pady=(10, 5), padx=0)
        export_strategy_frame = ttk.Frame(export_buttons_main_frame); export_strategy_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Label(export_strategy_frame, text="Strategy:", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        self.lowest_cost_strict_btn = ttk.Button(export_strategy_frame, text="Strict Cost", command=lambda key="Strict Lowest Cost": self.export_strategy_gui(key), state="disabled"); self.lowest_cost_strict_btn.pack(side=tk.LEFT, padx=2); self.create_tooltip(self.lowest_cost_strict_btn, "Export CSV for 'Strict Lowest Cost' strategy regardless of lead time.")
        self.in_stock_btn = ttk.Button(export_strategy_frame, text="In Stock", command=lambda key="Lowest Cost In Stock": self.export_strategy_gui(key), state="disabled"); self.in_stock_btn.pack(side=tk.LEFT, padx=2); self.create_tooltip(self.in_stock_btn, "Export CSV for 'Lowest Cost In Stock' strategy. Shows ONLY parts in stock")
        self.with_lt_btn = ttk.Button(export_strategy_frame, text="w/ LT", command=lambda key="Lowest Cost with Lead Time": self.export_strategy_gui(key), state="disabled"); self.with_lt_btn.pack(side=tk.LEFT, padx=2); self.create_tooltip(self.with_lt_btn, "Export CSV for 'Lowest Cost with Lead Time' strategy.")
        self.fastest_btn = ttk.Button(export_strategy_frame, text="Fastest", command=lambda key="Fastest": self.export_strategy_gui(key), state="disabled"); self.fastest_btn.pack(side=tk.LEFT, padx=2); self.create_tooltip(self.fastest_btn, "Export CSV for 'Fastest' strategy regardless of cost.")
        self.optimized_strategy_btn = ttk.Button(export_strategy_frame, text="Optimized", command=lambda key="Optimized Strategy": self.export_strategy_gui(key), state="disabled"); self.optimized_strategy_btn.pack(side=tk.LEFT, padx=2); self.create_tooltip(self.optimized_strategy_btn, "Export CSV for 'Optimized Strategy'.")
        ttk.Separator(export_buttons_main_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=5)
        export_parts_frame = ttk.Frame(export_buttons_main_frame); export_parts_frame.pack(side=tk.RIGHT, padx=(5, 0))
        self.export_parts_list_btn = ttk.Button(export_parts_frame, text="Export View", command=self.export_treeview_data, state="disabled"); self.export_parts_list_btn.pack(side=tk.LEFT); self.create_tooltip(self.export_parts_list_btn, "Export the current data shown in the main BOM Analysis parts list table to a CSV file.")
        
        # --- Set initial sash position ---
        initial_analysis_sash_pos = 400
        self.root.after(150, lambda: self.set_sash_pos(self.analysis_pane, 0, initial_analysis_sash_pos))

        # --- Tab 2: AI & Predictive Analysis ---
        self.predictive_tab = ttk.Frame(self.results_notebook, padding=10)
        self.results_notebook.add(self.predictive_tab, text=" AI & Predictions ")

        # --- Grid Configuration for Predictive Tab ---
        # Row 0: Recommendation Box (fixed height)
        # Row 1: Separator
        # Row 2: Main AI Summary Text (expands vertically)
        # Row 3: Separator
        # Row 4: Predictions vs Actuals (expands vertically)
        # Row 5: Accuracy Display Frame (fixed height)
        self.predictive_tab.grid_rowconfigure(0, weight=0)
        self.predictive_tab.grid_rowconfigure(1, weight=0)
        self.predictive_tab.grid_rowconfigure(2, weight=1) # Main text area expands
        self.predictive_tab.grid_rowconfigure(3, weight=0)
        self.predictive_tab.grid_rowconfigure(4, weight=2) # Predictions frame expands more
        self.predictive_tab.grid_rowconfigure(5, weight=0)
        self.predictive_tab.grid_columnconfigure(0, weight=1) # Content expands horizontally

        # --- 1. Recommendation Frame ---
        recommend_frame = ttk.LabelFrame(self.predictive_tab, text="AI Recommended Strategy", padding=(10, 5))
        recommend_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        recommend_frame.grid_columnconfigure(0, weight=1) # Label expands horizontally
        recommend_frame.grid_columnconfigure(1, weight=0) # Button fixed size

        self.ai_recommendation_label = tk.Label(recommend_frame, text="Run AI Summary to get recommendation.",
                                                 anchor='nw', justify='left', wraplength=700,
                                                 font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL),
                                                 bg=self.COLOR_FRAME_BG, # Start with default background
                                                 fg=self.COLOR_TEXT)
        self.ai_recommendation_label.grid(row=0, column=0, sticky='nw', padx=(0,10), pady=5)

        self.export_recommended_btn = ttk.Button(recommend_frame, text="Export Recommended", command=self.export_ai_recommended_strategy, state="disabled")
        self.export_recommended_btn.grid(row=0, column=1, sticky='ne', padx=5, pady=5)
        self.create_tooltip(self.export_recommended_btn, "Export the specific purchasing strategy recommended by the AI analysis.")

        # --- 2. Separator ---
        ttk.Separator(self.predictive_tab, orient=tk.HORIZONTAL).grid(row=1, column=0, sticky="ew", pady=(5, 5), padx=5)

        # --- 3. Main AI Analysis Frame (ScrolledText) ---
        ai_frame = ttk.LabelFrame(self.predictive_tab, text="Full AI Analysis & Details", padding=5)
        ai_frame.grid(row=2, column=0, sticky="nsew", pady=(0, 10)) # Grid to row 2
        ai_frame.grid_rowconfigure(0, weight=1); ai_frame.grid_columnconfigure(0, weight=1)

        self.ai_summary_text = scrolledtext.ScrolledText(ai_frame, wrap=tk.WORD, height=15, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL), relief="solid", borderwidth=1, state='disabled', background="#fdfdfd", foreground="#111827")
        self.ai_summary_text.grid(row=0, column=0, sticky="nsew")

        # Configure tags for this main text area
        self.ai_summary_text.tag_configure("critical", foreground=self.COLOR_ERROR, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL, "bold"))
        self.ai_summary_text.tag_configure("warning", foreground=self.COLOR_WARN, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL, "bold"))
        self.ai_summary_text.tag_configure("bold", font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL, "bold"))

        self.ai_summary_text.insert(tk.END, "Run AI Summary to view full analysis details.")

        # --- 4. Separator ---
        ttk.Separator(self.predictive_tab, orient=tk.HORIZONTAL).grid(row=3, column=0, sticky="ew", pady=(10, 10), padx=5) # Grid to row 3

        # --- 5. Predictions vs Actuals Frame ---
        pred_update_frame = ttk.LabelFrame(self.predictive_tab, text="Predictions vs Actuals", padding=5)
        pred_update_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 10)); # Grid to row 4
        # Configure inner grid/widgets for predictions... (rest of this section unchanged)
        pred_update_frame.grid_columnconfigure(0, weight=1); pred_update_frame.grid_rowconfigure(1, weight=1)
        pred_tree_frame = ttk.Frame(pred_update_frame); pred_tree_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", pady=(5,5));
        pred_tree_frame.grid_rowconfigure(0, weight=1);
        pred_tree_frame.grid_columnconfigure(0, weight=1)
        pred_hsb = ttk.Scrollbar(pred_tree_frame, orient="horizontal")
        pred_vsb = ttk.Scrollbar(pred_tree_frame, orient="vertical")
        pred_col_widths = {c: 75 for c in self.pred_header}; pred_col_widths.update({'Component': 180, 'Date': 80, 'Stock_Probability': 65, 'Real_Lead': 60, 'Real_Cost': 70, 'Real_Stock': 60, 'Prophet_Ld_Acc': 60, 'Prophet_Cost_Acc': 60, 'RAG_Ld_Acc': 60, 'RAG_Cost_Acc': 60, 'AI_Ld_Acc': 60, 'AI_Cost_Acc': 60})
        pred_col_align = {c: 'center' for c in self.pred_header}; pred_col_align.update({'Component': 'w'})
        pred_col_tooltips = {'Component': 'Consolidated Component Name (Mfg + MPN)', 'Date': 'Date prediction generated.', 'Prophet_Lead': 'Prophet Lead Time (d)', 'Prophet_Cost': 'Prophet Unit Cost ($)', 'RAG_Lead': 'RAG Lead Time Range (d)', 'RAG_Cost': 'RAG Unit Cost Range ($)', 'AI_Lead': 'AI Combined Lead Time (d)', 'AI_Cost': 'AI Combined Unit Cost ($)', 'Stock_Probability': 'Predicted Stock Probability (%)', 'Real_Lead': 'ACTUAL Lead Time (d)', 'Real_Cost': 'ACTUAL Unit Cost ($)', 'Real_Stock': 'ACTUAL Stock OK?', 'Prophet_Ld_Acc': 'Prophet LT Accuracy % vs actual data recorded from PO', 'Prophet_Cost_Acc': 'Prophet Cost Accuracy % vs actual data recorded from PO', 'RAG_Ld_Acc': 'RAG LT Accuracy % vs actual data recorded from PO', 'RAG_Cost_Acc': 'RAG Cost Accuracy %', 'AI_Ld_Acc': 'AI LT Accuracy % vs actual data recorded from PO', 'AI_Cost_Acc': 'AI Cost Accuracy % vs actual data recorded from PO'}
        self.predictions_tree = ttk.Treeview(pred_tree_frame, columns=self.pred_header, show="headings", height=10, selectmode="browse",                                              yscrollcommand=pred_vsb.set, xscrollcommand=pred_hsb.set)

        self.pred_column_tooltips = {}
        for col in self.pred_header:
            width = pred_col_widths.get(col, 75); align = pred_col_align.get(col, 'center'); heading_text = col.replace('_',' '); tooltip_text = pred_col_tooltips.get(col, heading_text)
            self.predictions_tree.heading(col, text=heading_text, anchor='center')
            self.predictions_tree.column(col, width=width, minwidth=10, stretch=False, anchor=align)
            self.pred_column_tooltips[col] = tooltip_text
        pred_vsb = ttk.Scrollbar(pred_tree_frame, orient="vertical", command=self.predictions_tree.yview); pred_hsb = ttk.Scrollbar(pred_tree_frame, orient="horizontal", command=self.predictions_tree.xview)
        self.predictions_tree.configure(yscrollcommand=pred_vsb.set, xscrollcommand=pred_hsb.set);
        pred_hsb.config(command=self.predictions_tree.xview)
        pred_vsb.config(command=self.predictions_tree.yview)
        pred_hsb.pack(side=tk.BOTTOM, fill=tk.X)
        pred_vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.predictions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        pred_actions_frame = ttk.Frame(pred_update_frame); pred_actions_frame.grid(row=2, column=0, columnspan=2, sticky='ew', pady=(10, 5))
        load_pred_button = ttk.Button(pred_actions_frame, text="Load / Refresh", command=self.load_predictions_to_gui); load_pred_button.pack(side=tk.LEFT, padx=(0, 10)); self.create_tooltip(load_pred_button, f"Load/Reload prediction data from {PREDICTION_FILE.name}.")
        update_inputs_frame = ttk.Frame(pred_actions_frame); update_inputs_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(update_inputs_frame, text="Update Actuals ->").pack(side=tk.LEFT, padx=(0,5))
        lbl_actual_lead = ttk.Label(update_inputs_frame, text="Lead:"); lbl_actual_lead.pack(side=tk.LEFT, padx=(0,2)); self.real_lead_entry = ttk.Entry(update_inputs_frame, width=6, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL)); self.real_lead_entry.pack(side=tk.LEFT, padx=(0,5)); self.create_tooltip(self.real_lead_entry, "Enter ACTUAL observed lead time (days). This should be data from actual purchase order created")
        lbl_actual_cost = ttk.Label(update_inputs_frame, text="Cost:"); lbl_actual_cost.pack(side=tk.LEFT, padx=(0,2)); self.real_cost_entry = ttk.Entry(update_inputs_frame, width=8, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL)); self.real_cost_entry.pack(side=tk.LEFT, padx=(0,5)); self.create_tooltip(self.real_cost_entry, "Enter ACTUAL unit cost ($). This should be date from actual purchase order created")
        lbl_actual_stock = ttk.Label(update_inputs_frame, text="Stock OK?:"); lbl_actual_stock.pack(side=tk.LEFT, padx=(0,2)); self.real_stock_var = tk.StringVar(value="?"); self.real_stock_combo = ttk.Combobox(update_inputs_frame, textvariable=self.real_stock_var, values=["?", "True", "False"], width=5, state='readonly', font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL)); self.real_stock_combo.pack(side=tk.LEFT, padx=(0,10)); self.create_tooltip(self.real_stock_combo, "Select if sufficient stock was ACTUALLY available when creating purchase order.")
        self.save_pred_update_btn = ttk.Button(update_inputs_frame, text="Save", command=self.save_prediction_updates, state="disabled"); self.save_pred_update_btn.pack(side=tk.LEFT); self.create_tooltip(self.save_pred_update_btn, "Save entered Actuals to the predictions CSV for the selected row.")
        self.selected_pred_id_label = ttk.Label(pred_actions_frame, text=" ", style="Hint.TLabel"); self.selected_pred_id_label.pack(side=tk.RIGHT, padx=(5, 0))
        self.predictions_tree.bind("<Motion>", self._on_predictions_tree_motion); self.predictions_tree.bind("<Leave>", self._on_predictions_tree_leave); self.predictions_tree.bind('<<TreeviewSelect>>', self.on_prediction_select)

        # --- 6. Accuracy Display Frame ---
        avg_frame = ttk.LabelFrame(self.predictive_tab, text="Average Prediction Accuracy (%)", padding=5)
        avg_frame.grid(row=5, column=0, sticky='nsew', pady=(5, 0)); # Grid to row 5
        avg_frame.columnconfigure((1, 2, 3, 4), weight=1)
        self.avg_acc_labels = {}
        headers = ["Model", "Ld Acc", "Cost Acc", "# Points"]
        models = ["Prophet", "RAG", "AI"]
        ttk.Label(avg_frame, text=" ", font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky='w', padx=5)
        ttk.Label(avg_frame, text="Lead Time", font=("Segoe UI", 9, "bold"), anchor='center').grid(row=0, column=1, columnspan=2, sticky='ew')
        ttk.Label(avg_frame, text="Cost", font=("Segoe UI", 9, "bold"), anchor='center').grid(row=0, column=3, columnspan=2, sticky='ew')
        ttk.Label(avg_frame, text="Model", font=("Segoe UI", 8, "bold")).grid(row=1, column=0, sticky='w', padx=5, pady=(0,3))
        ttk.Label(avg_frame, text="Avg Accuracy%", font=("Segoe UI", 8, "bold")).grid(row=1, column=1, sticky='ew', pady=(0,3))
        ttk.Label(avg_frame, text="# Data Pts in calculation", font=("Segoe UI", 8, "bold")).grid(row=1, column=2, sticky='ew', pady=(0,3))
        ttk.Label(avg_frame, text="Avg Accuracy%", font=("Segoe UI", 8, "bold")).grid(row=1, column=3, sticky='ew', pady=(0,3))
        ttk.Label(avg_frame, text="# Data Pts in calculation", font=("Segoe UI", 8, "bold")).grid(row=1, column=4, sticky='ew', pady=(0,3))
        
        for i, model in enumerate(models):
            row_num = i + 2
            ttk.Label(avg_frame, text=f"{model}:").grid(row=row_num, column=0, sticky='w', padx=5)
            ld_key = f"{model}_Ld"; ld_count_key = f"{model}_Ld_Count"; cost_key = f"{model}_Cost"; cost_count_key = f"{model}_Cost_Count"
            ld_label = ttk.Label(avg_frame, text="N/A", width=8, anchor='e', relief='sunken', background="#f8f9fa"); ld_label.grid(row=row_num, column=1, sticky='ew', padx=2); self.avg_acc_labels[ld_key] = ld_label; self.create_tooltip(ld_label, f"{model} LT Acc")
            ld_count_label = ttk.Label(avg_frame, text="0", width=6, anchor='e', relief='sunken', background="#f8f9fa"); ld_count_label.grid(row=row_num, column=2, sticky='ew', padx=2); self.avg_acc_labels[ld_count_key] = ld_count_label; self.create_tooltip(ld_count_label, f"{model} LT Pts")
            cost_label = ttk.Label(avg_frame, text="N/A", width=8, anchor='e', relief='sunken', background="#f8f9fa"); cost_label.grid(row=row_num, column=3, sticky='ew', padx=2); self.avg_acc_labels[cost_key] = cost_label; self.create_tooltip(cost_label, f"{model} Cost Acc")
            cost_count_label = ttk.Label(avg_frame, text="0", width=6, anchor='e', relief='sunken', background="#f8f9fa"); cost_count_label.grid(row=row_num, column=4, sticky='ew', padx=2); self.avg_acc_labels[cost_count_key] = cost_count_label; self.create_tooltip(cost_count_label, f"{model} Cost Pts")
            
        # --- Tab 3: Visualizations ---
        self.viz_tab = ttk.Frame(self.results_notebook, padding=10)
        self.results_notebook.add(self.viz_tab, text=" Visualizations ")
        self.viz_tab.grid_columnconfigure(0, weight=1); self.viz_tab.grid_rowconfigure(1, weight=1)
        viz_controls_frame = ttk.Frame(self.viz_tab); viz_controls_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(viz_controls_frame, text="Select Plot:").pack(side=tk.LEFT, padx=(0, 5))
        self.plot_type_var = tk.StringVar(); self.plot_combo = ttk.Combobox(viz_controls_frame, textvariable=self.plot_type_var, state="readonly", width=30); self.plot_combo.pack(side=tk.LEFT, padx=(0, 10)); self.plot_combo.bind("<<ComboboxSelected>>", self.update_visualization)
        self.plot_frame = ttk.Frame(self.viz_tab, relief="sunken", borderwidth=1); self.plot_frame.grid(row=1, column=0, sticky="nsew")
        self.fig_canvas = None; self.toolbar = None

        # --- Instance Variables ---
        self.bom_df = None; self.bom_filepath = None; self.analysis_results = {}; self.strategies_for_export = {};   self.historical_data_df = None
        self.predictions_df = None; self.digikey_token_data = None; self.nexar_token_data = None; self.mouser_requests_today = 0
        self.mouser_last_reset_date = None; self.mouser_daily_limit = 1000; self.thread_pool = ThreadPoolExecutor(max_workers=MAX_API_WORKERS, thread_name_prefix="BOMWorker")
        self.running_analysis = False; self._hts_cache = {}; self.prediction_tree_row_map = {}; self.tree_item_data_map = {}; self._active_tooltip_widget = None
        self.plot_annotation = None # Annotation object for plot hovering

        # --- Initial Setup Calls ---
        self.load_mouser_request_counter()
        self.update_rate_limit_display()
        self.load_digikey_token_from_cache()
        self.load_nexar_token_from_cache()
        self.initialize_data_files()
        self.load_predictions_to_gui()
        self.validate_inputs()
        initial_main_sash_pos = 470
        self.root.after(100, lambda: self.set_sash_pos(self.main_paned_window, 0, initial_main_sash_pos))
        initial_analysis_sash_pos = 400
        self.root.after(150, lambda: self.set_sash_pos(self.analysis_pane, 0, initial_analysis_sash_pos))
        self.root.after(200, self.show_startup_guide_popup)
        self.update_export_buttons_state()
        self.load_mouser_request_counter()
        self.update_rate_limit_display()
        logger.info("GUI initialization complete.")

        

    # --- >>> ADD NEW HELPER METHOD <<< ---
    def set_sash_pos(self, pane, index, position):
        """Safely sets the sash position after the window is mapped."""
        try:
            if pane.winfo_exists():
                # Allow geometry manager to update first
                pane.update_idletasks()
                pane.sashpos(index, position)
                logger.debug(f"Set sash position for pane {pane} index {index} to {position}")
            else:
                logger.warning(f"Attempted to set sashpos on non-existent pane: {pane}")
        except tk.TclError as e:
            logger.warning(f"TclError setting sash position (will retry once): {e}")
            self.root.after(250, lambda: self.set_sash_pos(pane, index, position)) # Retry after longer delay
        except Exception as e:
            logger.error(f"Unexpected error setting sash position: {e}", exc_info=True)
    # --- END HELPER METHOD ---
    

    def show_startup_guide_popup(self):
        """Shows the initial startup guide if configured to do so."""
        app_config = load_app_config()
        if not app_config.get("show_startup_guide", True):
            logger.debug("Skipping startup guide based on config.")
            return

        popup = tk.Toplevel(self.root)
        popup.title("Welcome to NPI BOM Analyzer!")
        popup.transient(self.root)
        popup.geometry("550x350") # Adjust size
        popup.resizable(False, False)

        main_frame = ttk.Frame(popup, padding=15)
        main_frame.pack(fill=tk.BOTH, expand=True)

        title_label = ttk.Label(main_frame, text="Getting Started Guide", font=("Segoe UI", 14, "bold"))
        title_label.pack(pady=(0, 10))

        # Use a ScrolledText for potentially longer instructions
        guide_text_widget = scrolledtext.ScrolledText(main_frame, wrap=tk.WORD, height=10, font=("Segoe UI", 12), relief="flat", state="normal", background=popup.cget('background'))
        guide_text = """
    Welcome! Here's a quick guide:

    **WARNING!!!  BOM Analyzer will NOT run without API keys.  API keys need to be entered into the 'keys.env' file in the same directory as the python run file.  This file can be created with the following format:
    
                    DIGIKEY_CLIENT_ID=***USER API KEY***
                    DIGIKEY_CLIENT_SECRET=***USER API KEY***
                    MOUSER_API_KEY=***USER API KEY***
                    CHATGPT_API_KEY=***USER API KEY***
                    GITHUB_TOKEN=***USER API KEY***
                    NEXAR_CLIENT_ID=***USER API KEY***
                    NEXAR_CLIENT_SECRET=***USER API KEY***
                    ARROW_CLIENT_LOGIN=***USER API KEY***
                    ARROW_SECRET=***USER API KEY***
                    AVNET_CLIENT=***USER API KEY***
                    AVNET_CLIENT_SECRET=***USER API KEY***   

    1.  **Load BOM:** Click 'Load BOM...' to select your Bill of Materials CSV file. Ensure it has columns like 'Part Number' and 'Quantity'. Manufacturer is helpful but optional.

    2.  **Configure: Optimized Strategy Configuration**
        *   Set 'Total Units to Build'.
        *   Adjust 'Optimized Strategy Configurations' parameters (Max Premium %, Target Lead Time, Weights, Buy-Up Threshold) to define your priorities. Hover over inputs for details.
        *   (Optional) Enter any 'Custom Tariff Rates'. (Default country tariff rates used based on part origin if left blank.)

    3.  **Run Analysis:** Click 'Run Analysis'. The app will fetch data from enabled suppliers (check API Status), calculate costs, lead times, risks, and strategies.  The Results are available on the 'BOM Analysis' tab.  Double-click a row for alternates. Export strategies or the full view.

    4.  **Run Predictions and AI Summary:**
            *   Click 'Run Predictions' to forecast future cost/lead time based on history.
            *   Enter actual results ('Real Lead', 'Real Cost', 'Real Stock') and click 'Save Actuals' to improve future accuracy calculations
            *   Click 'AI Summary' (requires OpenAI key) for insights. (Predictions must be run first)      
        **Visualizations Tab:** Select different plots to visualize analysis results.

    5.  **Tooltips:** Hover over table headers and configuration inputs for definitions. Tooltip text appears in the status bar at the bottom.
        """
        guide_text_widget.insert(tk.END, guide_text)
        guide_text_widget.configure(state="disabled")
        guide_text_widget.pack(fill=tk.BOTH, expand=True, pady=5)


        # --- >>> Define bottom_frame BEFORE using it <<< ---
        bottom_frame = ttk.Frame(main_frame)
        bottom_frame.pack(fill=tk.X, pady=(10, 0))
        # --- >>> END Definition <<< ---

        dont_show_var = tk.BooleanVar()
        # Now it's safe to use bottom_frame as the parent
        dont_show_check = ttk.Checkbutton(bottom_frame, text="Don't show this again", variable=dont_show_var)
        dont_show_check.pack(side=tk.LEFT)

        def close_popup():
            if dont_show_var.get():
                logger.info("Updating config to hide startup guide next time.")
                # Need to access app_config defined earlier in this method
                current_config = load_app_config() # Reload just in case
                current_config["show_startup_guide"] = False
                save_app_config(current_config)
            popup.destroy()

        # Now it's safe to use bottom_frame as the parent
        ok_button = ttk.Button(bottom_frame, text="OK", command=close_popup, style="Accent.TButton")
        ok_button.pack(side=tk.RIGHT)

        # Center popup and make modal
        self.center_window(popup)
        #popup.grab_set() # Make modal
        #popup.wait_window() # Wait until closed

   
    def setup_plot_options(self):
        """Populates the plot selection dropdown."""
        # Define available plots based on available data
        plot_options = ["-- Select Plot --"]
        if self.analysis_results and self.analysis_results.get("gui_entries"):
            plot_options.extend([
                "Risk Score Distribution",
                "Cost Distribution (Optimized)",
                "Lead Time Distribution (Optimized)",
                "Cost vs Lead Time (Optimized)",
            ])
        if self.predictions_df is not None and not self.predictions_df.empty:
             plot_options.extend([
                 "Prediction Accuracy Comparison",
             ])
        # Add more plot types...

        self.plot_combo['values'] = plot_options
        if len(plot_options) > 1:
            self.plot_combo.current(0)
        else: # No data yet
             self.plot_combo.set("-- No Data for Plots --")

    def _on_plot_hover(self, event):
        """Handles mouse hover events on the plot canvas for annotations."""
        # Ensure annotation object and canvas exist
        if not self.plot_annotation or not self.fig_canvas:
            return
        
        # Ensure stored axes and plot-specific data are available
        if not hasattr(self, '_current_plot_ax') or not hasattr(self, '_current_plot_specific_df') or self._current_plot_specific_df.empty:
            if self.plot_annotation.get_visible():
                self.plot_annotation.set_visible(False)
                if self.fig_canvas: self.fig_canvas.draw_idle()
            return
        
        ax = self._current_plot_ax
        plot_df = self._current_plot_specific_df
        visible = self.plot_annotation.get_visible()
    
        # Check if mouse is inside the axes bounds
        if event.inaxes == ax:
            logger.debug(f"Hover: In axes at ({event.xdata:.2f}, {event.ydata:.2f}).")
            
            # --- Improved point detection ---
            # Get mouse position
            mouse_x, mouse_y = event.xdata, event.ydata
            
            # Define a threshold for point detection (in data coordinates)
            detection_threshold = 5.0  # Adjust based on your plot density
            
            # Initialize variables
            closest_point_idx = None
            min_distance = float('inf')
            target_collection = None
            
            # Check all collections (scatter plots)
            for i, collection in enumerate(ax.collections):
                # Skip collections without proper data
                if not hasattr(collection, 'get_offsets') or len(collection.get_offsets()) == 0:
                    continue
                    
                # Get all point coordinates
                points = collection.get_offsets()
                logger.debug(f"Hover: Collection {i} contains {len(points)} points")
                
                # Check standard contains method first
                cont, ind_dict = collection.contains(event)
                if cont and 'ind' in ind_dict and len(ind_dict['ind']) > 0:
                    point_idx = ind_dict['ind'][0]
                    logger.debug(f"Hover: Standard detection found point at index {point_idx}")
                    closest_point_idx = point_idx
                    target_collection = collection
                    break
                
                # If standard detection fails, use manual distance calculation
                # Especially useful for points with extreme values
                for j, point in enumerate(points):
                    # Calculate distance between mouse and point
                    dx = mouse_x - point[0]
                    dy = mouse_y - point[1]
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    # Check if this point is closer than previous ones
                    if distance < min_distance and distance < detection_threshold:
                        min_distance = distance
                        closest_point_idx = j
                        target_collection = collection
            
            # Process the found point
            if closest_point_idx is not None and closest_point_idx < len(plot_df):
                logger.debug(f"Hover: Found closest point at index {closest_point_idx}, distance: {min_distance:.2f}")
                
                # Initialize variables
                part_num = 'N/A'
                mfg_pn = 'N/A'
                cost_val = np.nan
                lt_val = np.nan
                
                try:
                    # Retrieve data for the point
                    if 'PartNumber' in plot_df.columns:
                        part_num = plot_df['PartNumber'].values[closest_point_idx]
                    if 'MfgPN' in plot_df.columns:
                        mfg_pn = plot_df['MfgPN'].values[closest_point_idx]
                    if 'BestTotalCost' in plot_df.columns:
                        cost_val = plot_df['BestTotalCost'].values[closest_point_idx]
                    if 'BestCostLT' in plot_df.columns:
                        lt_val = plot_df['BestCostLT'].values[closest_point_idx]
                    
                    logger.debug(f"Hover: Retrieved data for index {closest_point_idx}: PN={part_num}, Cost={cost_val}, LT={lt_val}")
                    
                    # Format annotation text
                    cost_val_fmt = float(cost_val) if pd.notna(cost_val) else np.nan
                    lt_val_fmt = float(lt_val) if pd.notna(lt_val) else np.nan
                    text = f"PN: {part_num}\nMfgPN: {mfg_pn}\nCost: ${cost_val_fmt:.2f}\nLT: {lt_val_fmt:.0f}d"
                    self.plot_annotation.set_text(text)
                    
                    # Get point coordinates
                    point_coords = target_collection.get_offsets()[closest_point_idx]
                    
                    # Get axes dimensions in data coordinates
                    x_min, x_max = ax.get_xlim()
                    y_min, y_max = ax.get_ylim()
                    x_range = x_max - x_min
                    y_range = y_max - y_min
                    
                    # Set annotation position
                    self.plot_annotation.xy = point_coords
                    
                    # Smart positioning based on point location in axes
                    # This ensures annotations for edge points remain visible
                    rel_x = (point_coords[0] - x_min) / x_range
                    rel_y = (point_coords[1] - y_min) / y_range
                    
                    # Position annotation to ensure it stays visible
                    if rel_y > 0.7:  # High point (top 30% of plot)
                        self.plot_annotation.set_position((10, -40))  # Below point
                    elif rel_y < 0.3:  # Low point (bottom 30% of plot)
                        self.plot_annotation.set_position((10, 20))  # Above point
                    elif rel_x > 0.7:  # Right side point
                        self.plot_annotation.set_position((-80, 10))  # Left of point
                    else:  # Default position
                        self.plot_annotation.set_position((10, 10))  # Right of point
                    
                    # Ensure annotation box style makes text readable
                    self.plot_annotation.set_bbox(dict(
                        boxstyle="round,pad=0.5",
                        fc="white",
                        ec="gray",
                        alpha=0.9
                    ))
                    
                    # Make annotation visible
                    if not visible:
                        logger.debug("Hover: Setting annotation visible.")
                    self.plot_annotation.set_visible(True)
                    self.fig_canvas.draw_idle()
                    
                except Exception as e:
                    logger.error(f"Hover: Error processing point data: {str(e)}", exc_info=True)
                    if visible:
                        self.plot_annotation.set_visible(False)
                        self.fig_canvas.draw_idle()
            else:
                # No point found near mouse position
                if visible:
                    logger.debug("Hover: Hiding annotation (no point found).")
                    self.plot_annotation.set_visible(False)
                    self.fig_canvas.draw_idle()
        else:
            # Mouse not in axes
            if visible:
                logger.debug("Hover: Hiding annotation (mouse left axes).")
                self.plot_annotation.set_visible(False)
                self.fig_canvas.draw_idle()
                
    def update_visualization(self, event=None):
        """Clears old plot and draws the selected new one."""
        selected_plot = self.plot_type_var.get()
        if not selected_plot or selected_plot.startswith("--"):
            self.clear_visualization()
            return

        # Get data (might need specific processing per plot)
        plot_data = None
        if self.analysis_results and self.analysis_results.get("gui_entries"):
             # Make a copy to avoid modifying the original analysis results
             plot_data = pd.DataFrame(self.analysis_results.get("gui_entries", [])).copy()
             # Convert numeric columns correctly FOR PLOTTING (do this inside plot functions ideally)
             # For hover data, we ensure columns exist in the df passed to the plot function
             # Let's move the conversion to numeric inside the specific plot functions where needed for plotting axes.
             # logger.debug(f"Initial plot_data for '{selected_plot}':\n{plot_data.head().to_string()}") # Optional: Check data before plotting

        # Clear previous plot widgets (Canvas, Toolbar, Annotation)
        self.clear_visualization()
        self.plot_annotation = None # Ensure annotation is reset

        # --- Start Change: Vertical Layout using GridSpec ---
        fig = Figure(figsize=(8, 5), dpi=100, facecolor=self.COLOR_BACKGROUND)

        # Create a GridSpec: 2 rows, 1 column. Give plot more height.
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.3) # Adjust ratio/hspace as needed
        ax_main_plot = fig.add_subplot(gs[0, 0]) # Main plot in the top row
        ax_main_plot.set_facecolor(self.COLOR_BACKGROUND)
        # --- End Change ---

        # Call specific plot function based on selection
        try:
            high_risk_list_text = None # Variable to store text from plot function
            self._current_plot_specific_df = pd.DataFrame() # Reset specific df

            # --- Log the plot_data being used ---
            if plot_data is not None:
                logger.debug(f"DataFrame 'plot_data' PASSED to plot functions (head):\n{plot_data.head().to_string()}")
                logger.debug(f"Columns in 'plot_data': {plot_data.columns.tolist()}")
                logger.debug(f"Data types in 'plot_data':\n{plot_data.dtypes}")

            # --- Plot Function Calls ---
            if selected_plot == "Risk Score Distribution" and plot_data is not None:
                # Ensure RiskScore is numeric before passing for filtering/plotting
                plot_data['RiskScore'] = pd.to_numeric(plot_data['RiskScore'], errors='coerce') # Coerce directly here
                high_risk_list_text = self.plot_risk_distribution(ax_main_plot, plot_data)
                # For risk plot, hover isn't set up, so store the base data potentially used
                self._current_plot_specific_df = plot_data.copy()

            elif selected_plot == "Cost Distribution (Optimized)" and plot_data is not None:
                 plot_data['BestTotalCost'] = pd.to_numeric(plot_data['BestTotalCost'], errors='coerce') # Convert for plot
                 self.plot_cost_distribution(ax_main_plot, plot_data)

            elif selected_plot == "Lead Time Distribution (Optimized)" and plot_data is not None:
                 plot_data['BestCostLT'] = pd.to_numeric(plot_data['BestCostLT'], errors='coerce') # Convert for plot
                 self.plot_lead_time_distribution(ax_main_plot, plot_data)

            elif selected_plot == "Cost vs Lead Time (Optimized)" and plot_data is not None:
                 # --- Start Change: Convert columns BEFORE calling plot function ---
                 plot_data['BestTotalCost'] = pd.to_numeric(plot_data['BestTotalCost'], errors='coerce')
                 plot_data['BestCostLT'] = pd.to_numeric(plot_data['BestCostLT'], errors='coerce')
                 # --- End Change ---
                 # Conversion to numeric happens inside plot_cost_vs_lead_time before dropna
                 returned_df = self.plot_cost_vs_lead_time(ax_main_plot, plot_data) # Pass converted data
                 self._current_plot_specific_df = returned_df if returned_df is not None else pd.DataFrame()

            elif selected_plot == "Country of Origin Distribution" and plot_data is not None:
                 self.plot_coo_distribution(ax_main_plot, plot_data)
                 self._current_plot_specific_df = plot_data.copy() # Store base data

            elif selected_plot == "Prediction Accuracy Comparison":
                if self.predictions_df is not None and not self.predictions_df.empty:
                    self.plot_prediction_accuracy(ax_main_plot)
                    self._current_plot_specific_df = self.predictions_df.copy() # Store predictions df
                else:
                     ax_main_plot.text(0.5, 0.5, 'No prediction data available.', ha='center', va='center', transform=ax_main_plot.transAxes)

            else: # No specific plot matched or plot_data was None
                 ax_main_plot.text(0.5, 0.5, 'Selected plot cannot be generated\n(Check data availability)', \
                         horizontalalignment='center', verticalalignment='center', transform=ax_main_plot.transAxes)

            # --- Add High Risk List Text Subplot (if generated) ---
            if high_risk_list_text is not None:
                ax_list = fig.add_subplot(gs[1, 0]) # Create subplot in the bottom row
                ax_list.set_facecolor(self.COLOR_FRAME_BG)
                ax_list.set_xticks([])
                ax_list.set_yticks([])
                for spine in ax_list.spines.values(): spine.set_visible(False)
                ax_list.text(0.01, 0.98, high_risk_list_text,
                             ha='left', va='top', fontsize=12, family='monospace', wrap=False,
                             transform=ax_list.transAxes)

            # --- Layout Adjustments ---
            try:
                # Adjust spacing, especially hspace for vertical gap
                fig.subplots_adjust(left=0.1, right=0.98, top=0.92, bottom=0.1, hspace=0.4) # Increased hspace
            except ValueError as layout_err:
                 logger.warning(f"Layout adjustment failed: {layout_err}")

            # --- Embed Canvas ---
            self.fig_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            self.fig_canvas.draw()
            canvas_widget = self.fig_canvas.get_tk_widget()
            canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

            # --- Configure Hover Annotation (only if Cost vs LT plot) ---
            if selected_plot == "Cost vs Lead Time (Optimized)" and not self._current_plot_specific_df.empty:
                self.plot_annotation = ax_main_plot.annotate("", xy=(0,0), xytext=(10,10),
                                                              textcoords="offset points",
                                                              bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.75),
                                                              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                                                              visible=False, fontsize=10, backgroundcolor='#FFFFE0')
                self._current_plot_ax = ax_main_plot
                self.fig_canvas.mpl_connect('motion_notify_event', self._on_plot_hover)
            else:
                 self.plot_annotation = None # Ensure annotation is None for other plots
                 self._current_plot_ax = None


            # --- Add Toolbar ---
            self.toolbar = NavigationToolbar2Tk(self.fig_canvas, self.plot_frame)
            self.toolbar.update()
            toolbar_widget = self.toolbar
            toolbar_widget.pack(side=tk.BOTTOM, fill=tk.X)

        except Exception as e:
             logger.error(f"Failed to generate plot '{selected_plot}': {e}", exc_info=True)
             # Use ax_main_plot for error message if it exists, else create basic axes
             ax_err = ax_main_plot if 'ax_main_plot' in locals() else fig.add_subplot(111)
             ax_err.text(0.5, 0.5, f'Error generating plot:\n{e}', color='red', \
                     horizontalalignment='center', verticalalignment='center', transform=ax_err.transAxes)
             # Still draw the canvas to show the error
             if not self.fig_canvas: # Ensure canvas exists even on error
                 self.fig_canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
                 self.fig_canvas.draw()
                 self.fig_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


    def plot_lead_time_distribution(self, ax, data):
        """Plots a histogram or boxplot of Optimized Lead Times."""
        # Use BestCostLT as it corresponds to the Optimized Cost data usually shown
        lead_data = data['BestCostLT'].dropna()
        if lead_data.empty:
             ax.text(0.5, 0.5, 'No valid Lead Time data available.', ha='center', va='center', transform=ax.transAxes)
             return

        # Consider using a boxplot for lead times as they can vary widely
        # sns.boxplot(y=lead_data, ax=ax, color=self.COLOR_ACCENT)
        # Or a histogram, potentially with log scale if range is huge
        sns.histplot(lead_data, kde=False, ax=ax, color=self.COLOR_ACCENT, bins=20) # Adjust bins as needed

        ax.set_title('Distribution of Part Lead Times (Optimized Strategy)', fontsize=10)
        ax.set_xlabel('Lead Time (Days)', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)


    def plot_cost_vs_lead_time(self, ax, data):
        """Plots a scatter plot of Optimized Cost vs Lead Time."""
        cost_col = 'BestTotalCost'
        lt_col = 'BestCostLT'
        
        # Check for required data
        if cost_col not in data.columns or lt_col not in data.columns or \
           data[cost_col].isnull().all() or data[lt_col].isnull().all():
            ax.text(0.5, 0.5, 'Insufficient Cost/Lead Time data.', ha='center', va='center', transform=ax.transAxes)
            return pd.DataFrame()  # Return empty if no valid data to plot
    
        # Define required columns for plotting and hover
        required_cols_for_hover = ['PartNumber', 'MfgPN', cost_col, lt_col]
        cols_to_select = [col for col in required_cols_for_hover if col in data.columns]
        
        if cost_col not in cols_to_select or lt_col not in cols_to_select:
            logger.error("Cost/LT columns missing for hover data prep.")
            return pd.DataFrame()
    
        # Filter to valid rows only
        valid_indices_df = data[[cost_col, lt_col]].dropna()
        if valid_indices_df.empty:
            ax.text(0.5, 0.5, 'No rows with valid Cost & Lead Time pairs.', ha='center', va='center', transform=ax.transAxes)
            return pd.DataFrame()
    
        valid_idx = valid_indices_df.index
        plot_df = data.loc[valid_idx, cols_to_select].reset_index(drop=True)
        
        # Log data statistics for diagnostic purposes
        cost_values = plot_df[cost_col].dropna()
        if not cost_values.empty:
            max_cost = cost_values.max()
            min_cost = cost_values.min()
            median_cost = cost_values.median()
            logger.debug(f"Cost stats: min={min_cost}, max={max_cost}, median={median_cost}")
            
        lt_values = plot_df[lt_col].dropna()
        if not lt_values.empty:
            max_lt = lt_values.max()
            min_lt = lt_values.min()
            median_lt = lt_values.median()
            logger.debug(f"Lead Time stats: min={min_lt}, max={max_lt}, median={median_lt}")
        
        # Determine if log scale is appropriate based on data range
        use_log_scale = False
        if not cost_values.empty:
            cost_range_ratio = max_cost / min_cost if min_cost > 0 else 10
            if cost_range_ratio > 10:
                use_log_scale = True
                logger.debug(f"Using log scale due to large cost range ratio: {cost_range_ratio:.1f}")
        
        # Create scatter plot with increased point size and picker radius
        sns.scatterplot(
            data=plot_df, 
            x=lt_col, 
            y=cost_col, 
            ax=ax, 
            alpha=0.7,  # Slightly increase alpha for better visibility
            color=self.COLOR_ACCENT, 
            s=120,      # Larger point size
            picker=8    # Larger picker radius
        )
        
        # Apply log scale if needed
        if use_log_scale:
            ax.set_yscale('log')
        
        # Add more space around the plot
        ax.margins(x=0.15, y=0.15)  # 15% margin on all sides
        
        # Set titles and labels
        ax.set_title('Part Cost vs Lead Time (Optimized Strategy)', fontsize=10)
        ax.set_xlabel('Lead Time (Days)', fontsize=9)
        ax.set_ylabel('Total Cost per Part ($)', fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Return the filtered DataFrame for hover functionality
        return plot_df
  

    def clear_visualization(self):
        """Removes the current plot canvas and toolbar."""
        if self.toolbar:
            self.toolbar.destroy()
            self.toolbar = None
        if self.fig_canvas:
            self.fig_canvas.get_tk_widget().destroy()
            self.fig_canvas = None
        # Clear any matplotlib figures to release memory (optional but good practice)
        plt.close('all')

    
    def plot_risk_distribution(self, ax_main, data): # Argument name is ax_main
        """Plots a histogram of Risk Scores and lists high-risk parts."""

        if 'RiskScore' not in data.columns or data['RiskScore'].isnull().all():
             ax_main.text(0.5, 0.5, 'No valid Risk Score data available.', ha='center', va='center', transform=ax_main.transAxes)
             return None # Return None if no data

        # Make a copy to avoid SettingWithCopyWarning later
        data = data.copy()

        # --- Convert RiskScore to numeric early ---
        # Convert to numeric, coercing errors (like 'N/A') to NaN
        data['RiskScoreNumeric'] = pd.to_numeric(data['RiskScore'], errors='coerce')
        # Drop rows where numeric conversion failed BEFORE calculating valid series for hist
        risk_scores_valid_series = data['RiskScoreNumeric'].dropna()

        if risk_scores_valid_series.empty:
             ax_main.text(0.5, 0.5, 'No valid Risk Score data points found after dropping N/A.', ha='center', va='center', transform=ax_main.transAxes)
             return None # Return None if no valid scores

        # --- Histogram Plotting ---
        bins = [self.RISK_CATEGORIES['low'][0], self.RISK_CATEGORIES['low'][1] + 0.1,
                self.RISK_CATEGORIES['moderate'][1] + 0.1, self.RISK_CATEGORIES['high'][1]]
        n, bins_out, patches = ax_main.hist(risk_scores_valid_series, bins=bins, color=self.COLOR_ACCENT) # Use the numeric series
        colors = [self.COLOR_SUCCESS, self.COLOR_WARN, self.COLOR_ERROR]
        bin_centers = 0.5 * (bins_out[1:] + bins_out[:-1])

        for count, patch, center in zip(n, patches, bin_centers):
             if center <= self.RISK_CATEGORIES['low'][1]: patch.set_facecolor(self.COLOR_SUCCESS)
             elif center <= self.RISK_CATEGORIES['moderate'][1]: patch.set_facecolor(self.COLOR_WARN)
             else: patch.set_facecolor(self.COLOR_ERROR)

        ax_main.set_title('Part Risk Score Distribution', fontsize=10)
        ax_main.set_xlabel('Risk Score (0-10)', fontsize=9)
        ax_main.set_ylabel('Number of Parts', fontsize=9)
        ax_main.tick_params(axis='both', which='major', labelsize=8)
        ax_main.grid(axis='y', linestyle='--', alpha=0.7)

        # --- High Risk Part List Calculation ---
        high_risk_threshold = self.RISK_CATEGORIES['high'][0]

        # --- Start Change: Debug Filter Data ---
        logger.debug(f"Data BEFORE high-risk filtering (showing RiskScoreNumeric and Status):\n{data[['PartNumber', 'RiskScore', 'RiskScoreNumeric', 'Status']].to_string()}")

        # Define filter conditions separately for logging
        # Use the 'RiskScoreNumeric' column we created
        numeric_risk_filter = (data['RiskScoreNumeric'] >= high_risk_threshold)
        unknown_status_filter = (data['Status'] == 'Unknown') # Check original Status column
        logger.debug(f"Numeric Risk Filter (>={high_risk_threshold}) results:\n{numeric_risk_filter.to_string()}")
        logger.debug(f"Unknown Status Filter results:\n{unknown_status_filter.to_string()}")

        # Apply combined filter using boolean OR
        high_risk_filter = numeric_risk_filter | unknown_status_filter
        logger.debug(f"Combined Filter results:\n{high_risk_filter.to_string()}")
        # Apply filter to the DataFrame using .loc for boolean indexing
        high_risk_df = data.loc[high_risk_filter].copy()
        logger.debug(f"DataFrame AFTER high-risk filtering:\n{high_risk_df[['PartNumber', 'RiskScoreNumeric', 'Status']].to_string()}")
        # --- End Change ---

        # Prepare text for the list
        list_text = "High Risk Parts (>={:.1f} or Unknown):\n".format(high_risk_threshold)
        list_text += "--------------------------\n"
        if high_risk_df.empty:
            list_text += "(None)"
            logger.debug("High risk list is empty after filtering.")
        else:
            # Sort by risk score descending for the list (use the numeric column)
            high_risk_df = high_risk_df.sort_values('RiskScoreNumeric', ascending=False, na_position='last')
            max_list_items = 15 # Limit the number of items shown
            logger.debug(f"Iterating through high_risk_df (showing max {max_list_items}):\n{high_risk_df.head(max_list_items).to_string()}") # ADDED LOG
            for idx, row in high_risk_df.head(max_list_items).iterrows():
                 logger.debug(f"  Processing row index {idx}, Raw PN: {row.get('PartNumber')}, Raw MfgPN: {row.get('MfgPN')}") # ADDED LOG
                 # --- Start Change: Prioritize BOM PN if MfgPN is 'NOT FOUND' ---
                 mfg_pn_val = row.get('MfgPN', 'N/A')
                 bom_pn_val = row.get('PartNumber', 'N/A')
                 # Assign part_id based on availability and 'NOT FOUND' status
                 part_id = bom_pn_val if mfg_pn_val == 'NOT FOUND' or mfg_pn_val == 'N/A' or pd.isna(mfg_pn_val) else mfg_pn_val
                 # --- End Change ---

                 # --- REMOVED the explicit skip for 'NOT FOUND' ---

                 score_numeric = row.get('RiskScoreNumeric', np.nan)
                 score_display = f"{score_numeric:.1f}" if pd.notna(score_numeric) else row.get('RiskScore', 'N/A') # Use numeric score for display
                 status = row.get('Status', 'N/A')
                 # Truncate long part numbers
                 part_id_display = (str(part_id)[:18] + '..') if len(str(part_id)) > 20 else str(part_id) # Ensure part_id is str
                 list_text += f"{part_id_display:<20} ({score_display}) [{status}]\n" # Format into columns
            if len(high_risk_df) > max_list_items:
                 list_text += f"... ({len(high_risk_df) - max_list_items} more)"

        # Drop the temporary numeric column - doing this on the original 'data' was wrong,
        # it doesn't affect the caller. No need to drop here.
        # data.drop(columns=['RiskScoreNumeric'], inplace=True, errors='ignore')

        return list_text # Return the generated text list
    
    def plot_cost_distribution(self, ax, data):
        """Plots a histogram or boxplot of Optimized Total Costs."""
        cost_data = data['BestTotalCost'].dropna()
        if cost_data.empty:
             ax.text(0.5, 0.5, 'No valid Cost data available.', ha='center', va='center', transform=ax.transAxes)
             return
        sns.histplot(cost_data, kde=True, ax=ax, color=self.COLOR_ACCENT)
        # Or: sns.boxplot(y=cost_data, ax=ax)
        ax.set_title('Distribution of Part Costs (Optimized Strategy)', fontsize=10)
        ax.set_xlabel('Total Cost per Part ($)', fontsize=9)
        ax.set_ylabel('Frequency / Density', fontsize=9)
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.grid(axis='y', linestyle='--', alpha=0.7)


    def plot_prediction_accuracy(self, ax):
         """ Plots average prediction accuracies """
         if self.predictions_df is None or self.predictions_df.empty:
             ax.text(0.5, 0.5, 'No prediction data available.', ha='center', va='center', transform=ax.transAxes)
             return

         models = ["Prophet", "RAG", "AI"]
         avg_acc = {'Lead Time': [], 'Cost': []}
         counts = {'Lead Time': [], 'Cost': []}
         valid_models_lt = []
         valid_models_cost = []

         for model in models:
             ld_col = f'{model}_Ld_Acc'
             cost_col = f'{model}_Cost_Acc'

             # Calculate mean and count for Lead Time
             if ld_col in self.predictions_df.columns:
                 ld_acc_series = pd.to_numeric(self.predictions_df[ld_col], errors='coerce')
                 if ld_acc_series.count() > 0: # Check if there are any valid accuracy points
                      avg_acc['Lead Time'].append(ld_acc_series.mean(skipna=True))
                      counts['Lead Time'].append(ld_acc_series.count())
                      valid_models_lt.append(model)
             # Calculate mean and count for Cost
             if cost_col in self.predictions_df.columns:
                  cost_acc_series = pd.to_numeric(self.predictions_df[cost_col], errors='coerce')
                  if cost_acc_series.count() > 0:
                       avg_acc['Cost'].append(cost_acc_series.mean(skipna=True))
                       counts['Cost'].append(cost_acc_series.count())
                       valid_models_cost.append(model)

         bar_width = 0.35
         x_lt = np.arange(len(valid_models_lt))
         x_cost = np.arange(len(valid_models_cost))

         # Plot Lead Time Accuracy
         rects1 = ax.bar(x_lt - bar_width/2, avg_acc['Lead Time'], bar_width, label='Avg Lead Time Acc (%)', color=self.COLOR_ACCENT)
         # Plot Cost Accuracy
         rects2 = ax.bar(x_cost + bar_width/2, avg_acc['Cost'], bar_width, label='Avg Cost Acc (%)', color=self.COLOR_WARN)

         ax.set_ylabel('Average Accuracy (%)', fontsize=9)
         ax.set_title('Prediction Model Accuracy Comparison', fontsize=10)
         # Use combined ticks if possible, ensure labels match bars
         ax.set_xticks(np.arange(len(models))) # Position ticks for all models potentially
         ax.set_xticklabels(models, fontsize=8) # Label with all model names
         ax.legend(fontsize=8)

         # Add counts as text labels (optional)
         def autolabel(rects, model_list, count_list):
             for i, rect in enumerate(rects):
                 height = rect.get_height()
                 if pd.notna(height) and i < len(count_list): # Ensure index exists
                     ax.annotate(f'n={count_list[i]}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha='center', va='bottom', fontsize=7)

         if valid_models_lt: autolabel(rects1, valid_models_lt, counts['Lead Time'])
         if valid_models_cost: autolabel(rects2, valid_models_cost, counts['Cost'])

         ax.tick_params(axis='both', which='major', labelsize=8)
         ax.grid(axis='y', linestyle='--', alpha=0.7)
         ax.set_ylim(0, 110) # Accuracy is 0-100

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
            # --- Start Change: Replace newlines ---
            # Explicitly replace any newline characters in the incoming text with spaces
            single_line_text = text.replace('\n', ' ')
            # --- End Change ---
            # Keep wraplength=0 in the label's creation in __init__
            self.universal_tooltip_label.config(text=single_line_text) # Use the modified single-line text
        except tk.TclError: pass

    def _hide_universal_tooltip(self):
        """Internal method to clear the universal tooltip label."""
        try:
            if hasattr(self, 'universal_tooltip_label') and self.universal_tooltip_label.winfo_exists():
                 self.universal_tooltip_label.config(text=" ")
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

    def export_ai_recommended_strategy(self):
        """Exports the specific strategy recommended by the AI."""
        logger.info("Export AI Recommended Strategy button clicked.")
        if not self.ai_recommended_strategy_key:
            messagebox.showwarning("No Recommendation", "AI summary has not been run or no valid strategy recommendation was found.")
            return
        if not self.strategies_for_export:
             messagebox.showerror("Export Error", "Strategy data is not available (run analysis first).")
             return

        logger.info(f"Attempting to export recommended strategy: '{self.ai_recommended_strategy_key}'")
        # Call the existing export function with the parsed key
        self.export_strategy_gui(self.ai_recommended_strategy_key)
        

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
        """ Enables/disables export buttons based on analysis results. Runs on main thread. """
        if not is_main_thread():
            self.root.after(0, self.update_export_buttons_state)
            return

        button_attr_names = [
            'lowest_cost_strict_btn',
            'in_stock_btn',           
            'with_lt_btn',            
            'fastest_btn',
            'optimized_strategy_btn',
            'export_parts_list_btn'
        ]

        for btn_name in button_attr_names:
            self._configure_button_state(btn_name, "disabled")

        try:
            has_summary_metrics = bool(self.analysis_results and self.analysis_results.get("summary_metrics"))
            strategies_dict = self.strategies_for_export

            if has_summary_metrics and isinstance(strategies_dict, dict):

                # Enable "Strict Lowest Cost" button
                if "Strict Lowest Cost" in strategies_dict and bool(strategies_dict["Strict Lowest Cost"]):
                    self._configure_button_state('lowest_cost_strict_btn', 'normal')

                
                if "Lowest Cost In Stock" in strategies_dict and bool(strategies_dict["Lowest Cost In Stock"]):
                    # Additional check: Ensure the corresponding summary metric is not N/A
                    instock_summary_value = "N/A"
                    summary_metrics_data = self.analysis_results.get("summary_metrics", [])
                    if isinstance(summary_metrics_data, list) and summary_metrics_data:
                         summary_as_dict = dict(summary_metrics_data)
                         instock_key = "Lowest Cost In Stock / LT ($ / Days)" # Check exact key from summary_list
                         instock_summary_value = summary_as_dict.get(instock_key, "N/A")
                    if "N/A" not in instock_summary_value:
                         self._configure_button_state('in_stock_btn', 'normal')

                
                if "Lowest Cost with Lead Time" in strategies_dict and bool(strategies_dict["Lowest Cost with Lead Time"]):
                    # Additional check: Ensure the corresponding summary metric is not N/A
                    withlt_summary_value = "N/A"
                    summary_metrics_data = self.analysis_results.get("summary_metrics", [])
                    if isinstance(summary_metrics_data, list) and summary_metrics_data:
                         summary_as_dict = dict(summary_metrics_data)
                         withlt_key = "Lowest Cost w/ LT / LT ($ / Days)" # Check exact key from summary_list
                         withlt_summary_value = summary_as_dict.get(withlt_key, "N/A")
                    if "N/A" not in withlt_summary_value:
                        self._configure_button_state('with_lt_btn', 'normal')

                # Enable "Fastest" button
                if "Fastest" in strategies_dict and bool(strategies_dict["Fastest"]):
                    self._configure_button_state('fastest_btn', 'normal')

                # Enable "Optimized Strategy" button
                if "Optimized Strategy" in strategies_dict and bool(strategies_dict["Optimized Strategy"]):
                    optimized_summary_value = "N/A"
                    summary_metrics_data = self.analysis_results.get("summary_metrics", [])
                    if isinstance(summary_metrics_data, list) and summary_metrics_data:
                         summary_as_dict = dict(summary_metrics_data)
                         opt_key = "Optimized Strategy / LT ($ / Days)"
                         optimized_summary_value = summary_as_dict.get(opt_key, "N/A")
                    if "N/A" not in optimized_summary_value:
                         self._configure_button_state('optimized_strategy_btn', 'normal')


            # Check main parts list tree
            parts_list_state = "disabled"
            if hasattr(self, 'tree') and self.tree and self.tree.winfo_exists():
                if self.tree.get_children():
                     parts_list_state = "normal"
            self._configure_button_state('export_parts_list_btn', parts_list_state)

        except tk.TclError: pass
        except Exception as e:
            logger.error(f"Error updating export button states: {e}", exc_info=True)


    def _configure_button_state(self, button_attr_name, state):
        """ Safely configures a button's state if it exists. """
        try:
            button_widget = getattr(self, button_attr_name, None)
            if button_widget and button_widget.winfo_exists():
                button_widget.config(state=state)
            # Optional: Add logging if widget is None or doesn't exist
            # elif button_widget is None: logger.debug(f"Button '{button_attr_name}' None in _configure_button_state.")
            # else: logger.debug(f"Button '{button_attr_name}' widget destroyed in _configure_button_state.")
        except tk.TclError: pass
        except Exception as e:
             logger.warning(f"Could not configure state for button '{button_attr_name}': {e}")

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


    # --- Start Change: Add method to open keys.env ---
    def edit_keys_file(self):
        """Attempts to open the keys.env file in the default text editor."""
        keys_file_path = SCRIPT_DIR / 'keys.env'
        logger.info(f"Attempting to open keys file: {keys_file_path}")
    
        if not keys_file_path.exists():
            msg = f"{keys_file_path.name} not found in the script directory ({SCRIPT_DIR}).\nPlease create it manually and add your API keys."
            logger.warning(msg)
            messagebox.showwarning("File Not Found", msg)
            return
    
        try:
            if sys.platform == "win32":
                # os.startfile(keys_file_path) # Might work but less reliable for default editor
                subprocess.run(['notepad.exe', str(keys_file_path)], check=False) # Try notepad first
            elif sys.platform == "darwin": # macOS
                subprocess.run(['open', '-t', str(keys_file_path)], check=True) # '-t' forces text editor
            else: # Linux and other POSIX
                subprocess.run(['xdg-open', str(keys_file_path)], check=True) # Relies on xdg-utils
            logger.info(f"Successfully launched editor for {keys_file_path.name}")
            messagebox.showinfo("Edit API Keys", f"Opened {keys_file_path.name} in default editor.\n\nRestart the application after saving changes for them to take effect.")
    
        except FileNotFoundError as e:
             # Specific error if 'open' or 'xdg-open' or 'notepad.exe' isn't found
             err_msg = f"Could not find command to open the file.\nPlatform: {sys.platform}\nError: {e}"
             logger.error(err_msg, exc_info=True)
             messagebox.showerror("Error Opening File", err_msg)
        except subprocess.CalledProcessError as e:
             # Error if the command ran but returned an error code
             err_msg = f"Error returned by editor command.\nPlatform: {sys.platform}\nError: {e}"
             logger.error(err_msg, exc_info=True)
             messagebox.showerror("Error Opening File", err_msg)
        except Exception as e:
            # Catch-all for other unexpected errors
            err_msg = f"Could not open {keys_file_path.name}.\nError: {e}"
            logger.error(err_msg, exc_info=True)
            messagebox.showerror("Error Opening File", err_msg)

    # --- BOM Loading ---
    def load_bom(self):
        """Loads BOM data from a CSV file, performs cleaning and validation."""
        logger.info("load_bom method entered.")
        
        if self.running_analysis:
            logger.warning("Load BOM attempted while analysis running.")
            messagebox.showwarning("Busy", "Analysis is currently running. Please wait.")
            return

        filepath = filedialog.askopenfilename(
            title="Select BOM CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        logger.debug(f"File dialog returned: {filepath}") # Log after dialog
        
        if not filepath: 
            logger.info("Load BOM cancelled by user (no filepath selected).")
            return # User cancelled

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
            if hasattr(self, 'export_recommended_btn') and self.export_recommended_btn:
                 self.export_recommended_btn.config(state="disabled")
            self.ai_recommended_strategy_key = None 

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
        
        # Ensure only one popup exists
        if self.alt_popup and self.alt_popup.winfo_exists():
            self.alt_popup.lift()
            self.alt_popup.focus_set()
            return
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
            # Fetch alternates directly from item_data
            alternates_list = item_data.get("AlternatesList", [])
            logger.debug(f"Item data for {bom_pn}: {item_data}")
            logger.debug(f"AlternatesList for {bom_pn}: {alternates_list}")
    
        logger.info(f"Showing alternates pop-up for BOM P/N: {bom_pn} (Mfg P/N: {mfg_pn})")
    
        # --- Create Pop-up --- (Standard Tkinter Toplevel)
        popup = tk.Toplevel(self.root)
        popup.title(f"Alternates for {mfg_pn}")
        popup.geometry("1024x768")
        popup.transient(self.root)  # Tie to parent window
        popup.grab_set()  # Modal behavior
        popup.lift()  # Bring to front
        popup.attributes('-topmost', True)  # Keep on top
        self.alt_popup = popup
        popup.protocol("WM_DELETE_WINDOW", lambda: self.close_alternates_popup(popup))
    
        popup_frame = ttk.Frame(popup, padding=10)
        popup_frame.pack(fill=tk.BOTH, expand=True)
        popup_frame.rowconfigure(2, weight=1) # Adjust row for text area to expand
        popup_frame.columnconfigure(0, weight=1)
    
        # Add a label at the top to display alternates details if they exist
        if alternates_list:
            alternates_details = "Alternate Parts Details:\n"
            row_offset = 0
    
        ttk.Label(popup_frame, text=f"Potential Substitutes/Alternates:", font=("Segoe UI", 16, "bold")).grid(row=row_offset, column=0, sticky='w', pady=(0, 2))
        ttk.Label(popup_frame, text=f"Mfg P/N: {mfg_pn}", style="Hint.TLabel").grid(row=row_offset + 1, column=0, sticky='w', pady=(0, 10))
    
        alt_text_area = scrolledtext.ScrolledText(popup_frame, wrap=tk.WORD, height=18, width=90, font=("Courier New", 14), relief="solid", borderwidth=1)
        alt_text_area.grid(row=row_offset + 2, column=0, sticky="nsew", pady=(5, 5))
        alt_text_area.configure(state='disabled')
    
        # Populate Content
        content = ""
        if not alternates_list:
            content = "No alternate parts found for this Manufacturer Part Number."
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
        close_button.grid(row=row_offset + 3, column=0, pady=(10, 0))
    
        # Center the popup
        popup.update_idletasks()
        self.center_window(popup)

    def close_alternates_popup(self, popup):
        if popup and popup.winfo_exists():
            popup.destroy()
        self.alt_popup = None
        

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

    def get_digikey_token(self, force_refresh=False):
            """Gets a valid DigiKey token, handling expiry and refresh. Uses HTTPS (modern SSLContext) for localhost callback."""
            function_start_time = time.time()
            logger.info("--- get_digikey_token START (HTTPS Mode - SSLContext) ---") # Mark start

            if not API_KEYS["DigiKey"]:
                self.root.after(0, self.update_status_threadsafe, "DigiKey API keys not set.", "error")
                logger.error("get_digikey_token: API keys not set.")
                return None

            # Check cache first
            if not force_refresh and self.digikey_token_data:
                expires_at = self.digikey_token_data.get('expires_at', 0)
                if time.time() < expires_at:
                    logger.info("get_digikey_token: Using valid cached token.")
                    return self.digikey_token_data['access_token']
                else:
                    logger.info("get_digikey_token: Cached token expired. Attempting refresh.")
                    if self.refresh_digikey_token():
                        logger.info("get_digikey_token: Refresh successful.")
                        return self.digikey_token_data['access_token']
                    else:
                        logger.warning("get_digikey_token: Refresh failed, forcing re-authentication.")
                        force_refresh = True

            # If no valid token/refresh failed, start OAuth flow
            if force_refresh or not self.digikey_token_data:
                logger.info("get_digikey_token: Triggering full DigiKey OAuth flow (HTTPS Redirect).")
                self.root.after(0, self.update_status_threadsafe, "DigiKey authentication required (Check browser & accept HTTPS warning)", "warning")

                redirect_uri = "https://localhost:8000"
                auth_port = 8000

                auth_url = f"https://api.digikey.com/v1/oauth2/authorize?client_id={DIGIKEY_CLIENT_ID}&response_type=code&redirect_uri={urllib.parse.quote(redirect_uri)}"
                logger.debug(f"get_digikey_token: Opening browser to: {auth_url}")

                try:
                     webbrowser.open(auth_url)
                except Exception as e:
                     logger.error(f"get_digikey_token: Failed to open browser: {e}")
                     self.root.after(0, self.update_status_threadsafe, f"Failed to open browser: {e}", "error")
                     return None

                # Start HTTPS server to catch redirect
                server = None
                auth_code = None
                server_start_time = time.time()
                certfile_path = Path("./localhost.pem") # Assumes cert is in same dir as script

                try:
                    if not certfile_path.exists():
                        logger.error(f"SSL Certificate file not found: {certfile_path}")
                        raise FileNotFoundError("localhost.pem not found. Please generate it using the openssl command.")

                    logger.info(f"Attempting to bind HTTPS server to localhost:{auth_port}")
                    server_address = ('localhost', auth_port)
                    httpd = HTTPServer(server_address, OAuthHandler)
                    httpd.auth_code = None
                    httpd.timeout = 300

                    # --- MODERN SSL CONTEXT ---
                    logger.info(f"Creating SSLContext and loading cert/key from {certfile_path}")
                    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                    context.load_cert_chain(certfile=str(certfile_path)) # Automatically loads key from same file if cert contains both
                    # ---

                    # --- WRAP SOCKET USING SSL CONTEXT ---
                    logger.info("Wrapping server socket using SSLContext...")
                    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
                    logger.info("SSL context applied successfully using wrap_socket.")
                    # ---

                    server = httpd # Assign the wrapped server

                    logger.info(f"Waiting for DigiKey OAuth callback on https://localhost:{auth_port} ...")
                    server.handle_request() # Wait for one GET request on the SSL socket
                    auth_code = getattr(server, 'auth_code', None)
                    server_duration = time.time() - server_start_time
                    logger.info(f"get_digikey_token: Local server finished after {server_duration:.1f}s.")

                except FileNotFoundError as e:
                     logger.error(f"get_digikey_token: {e}")
                     self.root.after(0, self.update_status_threadsafe, f"SSL Error: {e}", "error")
                     self.root.after(0, messagebox.showerror, "SSL Error", f"Could not find certificate file:\n{certfile_path}\n\nPlease generate it using the provided openssl command.")
                     return None
                except ssl.SSLError as e:
                     logger.error(f"get_digikey_token: SSL context/handshake error: {e}", exc_info=True)
                     self.root.after(0, self.update_status_threadsafe, f"SSL Error: {e}", "error")
                     self.root.after(0, messagebox.showerror, "SSL Error", f"Failed to create secure HTTPS server.\n\nError: {e}\n\nCheck certificate file and permissions.")
                     return None
                except OSError as e:
                     logger.error(f"get_digikey_token: OAuth server bind error (Port {auth_port} likely in use): {e}")
                     self.root.after(0, self.update_status_threadsafe, f"OAuth Port Error: {e}", "error")
                     self.root.after(0, messagebox.showerror, "OAuth Error", f"Could not start callback server on port {auth_port}.\nEnsure no other application is using it.\n\n{e}")
                     return None
                except Exception as e:
                     logger.error(f"get_digikey_token: OAuth callback server error: {e}", exc_info=True)
                     self.root.after(0, self.update_status_threadsafe, f"OAuth server error: {e}", "error")
                     return None
                finally:
                    if server:
                        # Use a simple check, maybe the thread pool is already shut down?
                        try:
                            if not self.thread_pool._shutdown: # Check if pool is active
                                 self.thread_pool.submit(server.server_close)
                                 logger.info("get_digikey_token: Submitted server close request.")
                            else:
                                 logger.warning("get_digikey_token: Thread pool shutdown, cannot submit server close.")
                                 # Attempt direct close, might block briefly if server hung
                                 try: server.server_close()
                                 except: pass
                        except Exception: # Catch potential errors checking pool status
                             logger.warning("get_digikey_token: Error submitting/closing server, attempting direct close.")
                             try: server.server_close()
                             except: pass


                # --- Check if auth_code was received ---
                if auth_code:
                    logger.info(f"get_digikey_token: Received auth_code: {auth_code[:10]}...")
                    logger.info("get_digikey_token: Attempting to exchange code for token...")
                    token_exchange_start_time = time.time()
                    try:
                        token_url = "https://api.digikey.com/v1/oauth2/token"
                        payload = {
                            'client_id': DIGIKEY_CLIENT_ID,
                            'client_secret': DIGIKEY_CLIENT_SECRET,
                            'grant_type': 'authorization_code',
                            'code': auth_code,
                            'redirect_uri': redirect_uri # MUST be https://localhost:8000
                        }
                        logger.debug(f"get_digikey_token: POSTing to {token_url} with grant_type=authorization_code and redirect_uri={redirect_uri}")
                        response = requests.post(token_url, data=payload,
                                                 headers={'Content-Type': 'application/x-www-form-urlencoded'},
                                                 timeout=API_TIMEOUT_SECONDS + 10)

                        logger.info(f"get_digikey_token: Token exchange response status: {response.status_code}")
                        if not response.ok:
                             logger.error(f"get_digikey_token: Token exchange failed! Response Text: {response.text}")

                        response.raise_for_status()
                        token_data = response.json()
                        logger.info("get_digikey_token: Token exchange successful!")
                        token_exchange_duration = time.time() - token_exchange_start_time
                        logger.info(f"get_digikey_token: Token exchange took {token_exchange_duration:.1f}s.")

                        expires_in = token_data.get('expires_in', 1800)
                        token_data['expires_at'] = time.time() + expires_in - 60
                        self.digikey_token_data = token_data

                        logger.info(f"get_digikey_token: Attempting to write token to {TOKEN_FILE}...")
                        try:
                            os.makedirs(CACHE_DIR, exist_ok=True)
                            with open(TOKEN_FILE, 'w') as f:
                                json.dump(self.digikey_token_data, f, indent=2)
                            logger.info("get_digikey_token: Token successfully cached.")
                        except IOError as e:
                            logger.error(f"get_digikey_token: FAILED to cache token! IOError: {e}")
                            self.root.after(0, messagebox.showerror, "Cache Error", f"Could not write token cache file:\n{TOKEN_FILE}\n\nError: {e}")
                        except Exception as e:
                             logger.error(f"get_digikey_token: FAILED to cache token! Unexpected Error: {e}", exc_info=True)
                             self.root.after(0, messagebox.showerror, "Cache Error", f"Unexpected error writing token cache file:\n{TOKEN_FILE}\n\nError: {e}")

                        self.root.after(0, self.update_status_threadsafe, "DigiKey authentication successful.", "success")
                        self.root.after(0, self.update_rate_limit_display)
                        if hasattr(self, '_digikey_refresh_after_id'): self.root.after_cancel(self._digikey_refresh_after_id)
                        self._digikey_refresh_after_id = self.root.after(int((expires_in - 300) * 1000), self.refresh_digikey_token)

                        logger.info(f"--- get_digikey_token END (Success HTTPS) - Total Time: {time.time() - function_start_time:.1f}s ---")
                        return self.digikey_token_data.get('access_token')

                    except requests.RequestException as e:
                        logger.error(f"get_digikey_token: Token exchange POST failed: {e}", exc_info=True)
                        error_detail_msg = f"Status {e.response.status_code}" if hasattr(e, 'response') and e.response is not None else str(e)
                        try:
                             if hasattr(e, 'response') and e.response is not None:
                                  error_json = e.response.json()
                                  error_detail_msg = f"Status {e.response.status_code} - {error_json.get('error', '')}: {error_json.get('error_description', '')}"
                        except: pass
                        self.root.after(0, self.update_status_threadsafe, f"DigiKey token exchange error: {error_detail_msg}", "error")
                        self.root.after(0, messagebox.showerror, "DigiKey Auth Error", f"Failed to get token: {error_detail_msg}\n\nCheck Client ID/Secret and DigiKey App Redirect URI (using HTTPS).")
                        self.digikey_token_data = None
                        logger.info(f"--- get_digikey_token END (Token Exchange Failed HTTPS) - Total Time: {time.time() - function_start_time:.1f}s ---")
                        return None
                    except Exception as e:
                         logger.error(f"get_digikey_token: Unexpected error during token exchange/caching (HTTPS): {e}", exc_info=True)
                         self.root.after(0, self.update_status_threadsafe, f"Unexpected Auth Error: {e}", "error")
                         self.root.after(0, messagebox.showerror, "DigiKey Auth Error", f"Unexpected error during authentication (HTTPS):\n\n{e}")
                         self.digikey_token_data = None
                         logger.info(f"--- get_digikey_token END (Unexpected Error HTTPS) - Total Time: {time.time() - function_start_time:.1f}s ---")
                         return None
                else: # auth_code is None
                    logger.error("get_digikey_token: Did not receive auth_code from local HTTPS server. Authentication likely timed out or failed in browser/redirect/warning bypass.")
                    self.root.after(0, self.update_status_threadsafe, "DigiKey authentication timed out or failed (HTTPS).", "error")
                    self.root.after(0, messagebox.showerror, "DigiKey Auth Error", "Did not receive authorization code from DigiKey callback.\nPlease ensure you complete the login/authorization and **accept any browser security warnings** for https://localhost:8000.")
                    logger.info(f"--- get_digikey_token END (No Auth Code HTTPS) - Total Time: {time.time() - function_start_time:.1f}s ---")
                    return None

            # Fallthrough - should not be reached
            logger.error("get_digikey_token: Reached unexpected end of function.")
            logger.info(f"--- get_digikey_token END (Error Fallthrough HTTPS) - Total Time: {time.time() - function_start_time:.1f}s ---")
            return None
        
    def refresh_digikey_token(self):
            """Refreshes the DigiKey access token using the refresh token."""
            logger.info("Attempting to refresh DigiKey token...")
            if not self.digikey_token_data or 'refresh_token' not in self.digikey_token_data:
                # Schedule status update on main thread
                self.root.after(0, self.update_status_threadsafe, "No refresh token available. Manual re-authentication needed.", "warning")
                self.digikey_token_data = None
                try:
                    if TOKEN_FILE.exists(): os.remove(TOKEN_FILE)
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
                logger.debug("refresh_digikey_token: POSTing with grant_type=refresh_token")
                response = requests.post(token_url, data=payload,
                                         headers={'Content-Type': 'application/x-www-form-urlencoded'},
                                         timeout=API_TIMEOUT_SECONDS)

                logger.info(f"refresh_digikey_token: Response status: {response.status_code}")
                if not response.ok:
                     logger.error(f"refresh_digikey_token: Refresh failed! Response Text: {response.text}")

                response.raise_for_status()
                new_token_data = response.json()
                logger.info("refresh_digikey_token: Token refresh successful!")

                expires_in = new_token_data.get('expires_in', 1800)
                new_token_data['expires_at'] = time.time() + expires_in - 60

                # Preserve the *new* refresh token if provided, otherwise keep the old one
                if 'refresh_token' not in new_token_data and 'refresh_token' in self.digikey_token_data:
                    new_token_data['refresh_token'] = self.digikey_token_data['refresh_token'] # Keep old if none provided
                    logger.debug("refresh_digikey_token: Kept existing refresh token.")
                elif 'refresh_token' in new_token_data:
                     logger.debug("refresh_digikey_token: Received a new refresh token.")

                self.digikey_token_data = new_token_data
                try:
                    # Ensure cache dir exists before writing
                    os.makedirs(CACHE_DIR, exist_ok=True)
                    with open(TOKEN_FILE, 'w') as f:
                        json.dump(self.digikey_token_data, f, indent=2)
                    logger.info("refresh_digikey_token: Refreshed token cached successfully.")
                except IOError as e:
                    logger.error(f"refresh_digikey_token: FAILED to cache refreshed token! IOError: {e}")
                    # Maybe show a non-blocking warning? Showing messagebox might be too intrusive for background refresh
                    self.root.after(0, self.update_status_threadsafe, f"Warning: Failed to cache refreshed token: {e}", "warning")

                self.root.after(0, self.update_status_threadsafe, "DigiKey token refreshed.", "info")
                self.root.after(0, self.update_rate_limit_display)
                # Schedule next refresh
                if hasattr(self, '_digikey_refresh_after_id'):
                     try: self.root.after_cancel(self._digikey_refresh_after_id)
                     except tk.TclError: pass # Ignore if timer doesn't exist
                self._digikey_refresh_after_id = self.root.after(int((expires_in - 300) * 1000), self.refresh_digikey_token)
                return True

            except requests.RequestException as e:
                logger.error(f"refresh_digikey_token: Token refresh POST failed: {e}", exc_info=True)
                error_detail_msg = f"Status {e.response.status_code}" if hasattr(e, 'response') and e.response is not None else str(e)
                self.root.after(0, self.update_status_threadsafe, f"Token refresh failed: {error_detail_msg}", "error")
                self.digikey_token_data = None
                try:
                    if TOKEN_FILE.exists(): os.remove(TOKEN_FILE)
                except OSError: pass
                # Avoid showing popup on background refresh failure, just log and require manual re-auth next time
                # self.root.after(0, messagebox.showerror, "DigiKey Error", f"Token refresh failed. You may need to 'Run Analysis' again to re-authenticate.\nError: {error_detail_msg}")
                logger.error(f"Token refresh failed. User may need to re-authenticate manually. Error: {error_detail_msg}")
                return False
            except Exception as e:
                 logger.error(f"refresh_digikey_token: Unexpected error during token refresh: {e}", exc_info=True)
                 self.root.after(0, self.update_status_threadsafe, f"Unexpected token refresh error: {e}", "error")
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
        """
        Searches DigiKey using Keyword, filters results, extracts data defensively.
        Combines working structure with robustness fixes. V12 - Added MPN Debug
        """
        # ... (Token check and API call setup) ...
        if not API_KEYS["DigiKey"]: return None
        access_token = self.get_digikey_token() # This might block/trigger OAuth
        if not access_token:
             logger.warning(f"DigiKey search skipped for {part_number}: No valid token.")
             return None

        url = "https://api.digikey.com/products/v4/search/keyword"
        headers = {
            'Authorization': f"Bearer {access_token}", 'X-DIGIKEY-Client-Id': DIGIKEY_CLIENT_ID,
            'X-DIGIKEY-Locale-Site': 'US', 'X-DIGIKEY-Locale-Language': 'en',
            'X-DIGIKEY-Locale-Currency': 'USD', 'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        keywords = f"{manufacturer} {part_number}".strip() if manufacturer else part_number
        payload = {"Keywords": keywords, "Limit": 10, "Offset": 0}
        logger.debug(f"DigiKey Searching (Keyword API) for: '{keywords}' with Limit=10")

        try:
            response = self._make_api_request("POST", url, headers=headers, json=payload)
            data = response.json()
            # Shorten log snippet slightly if too verbose
            logger.debug(f"DigiKey Raw Response Snippet for '{keywords}': {str(data)[:300]}")

            # Schedule rate limit update
            if self.digikey_token_data:
                 self.digikey_token_data["rate_limit_remaining"] = response.headers.get('X-RateLimit-Remaining', 'NA')
                 self.digikey_token_data["rate_limit"] = response.headers.get('X-RateLimit-Limit', 'NA')
                 self.root.after(0, self.update_rate_limit_display)

            products = data.get("Products", [])
            if not products or not isinstance(products, list):
                logger.info(f"DigiKey: No valid 'Products' list found for '{keywords}'.")
                return None

            # --- Find Best Match (Revised Logic - Incorporating Fallback) ---
            best_match_dict = None
            exact_pn_upper = part_number.upper()
            mfg_upper = manufacturer.upper().strip() if manufacturer else None
            potential_matches = [] # Stores exact MPN matches

            for p in products:
                if isinstance(p, dict):
                    mpn_api = p.get("ManufacturerProductNumber", "").upper()
                    if mpn_api == exact_pn_upper:
                        potential_matches.append(p)
                else:
                    logger.warning(f"DigiKey: Skipped non-dict item in Products list for '{keywords}'.")

            if potential_matches:
                # We found at least one exact MPN match
                if mfg_upper and len(potential_matches) > 1:
                    # Try to filter by manufacturer if multiple exact MPNs found
                    filtered_by_mfg = [
                        p for p in potential_matches
                        if isinstance(p.get("Manufacturer"), dict) and
                           p.get("Manufacturer", {}).get("Name", "").upper().strip() == mfg_upper
                    ]
                    if filtered_by_mfg:
                        best_match_dict = filtered_by_mfg[0]
                        logger.debug(f"DigiKey: Found exact MPN + Mfg match for '{keywords}'.")
                    else:
                        # Fallback to first exact MPN match if manufacturer doesn't match
                        best_match_dict = potential_matches[0]
                        logger.debug(f"DigiKey: Found exact MPN but no Mfg match for '{keywords}'. Using first exact MPN match.")
                else:
                    # Use the first exact MPN match if only one found or no manufacturer provided
                    best_match_dict = potential_matches[0]
                    logger.debug(f"DigiKey: Using first/only exact MPN match for '{keywords}'.")
            # --- START: Added Fallback Logic ---
            elif products:
                 # Fallback: No exact MPN match found, use the VERY FIRST result from the API
                 first_product = products[0]
                 if isinstance(first_product, dict):
                     best_match_dict = first_product
                     api_mpn_fallback = best_match_dict.get('ManufacturerProductNumber', 'N/A')
                     logger.warning(f"DigiKey: No exact MPN match for '{keywords}'. FALLING BACK to first result (MPN: {api_mpn_fallback}). Results may be less accurate.")
                 else:
                     logger.error(f"DigiKey: No exact MPN match and first product item is not a dict for '{keywords}'. Cannot proceed.")
                     return None # Cannot proceed if first item isn't a dict
            # --- END: Added Fallback Logic ---
            else:
                # No products returned at all
                logger.info(f"DigiKey: No products array in response for '{keywords}'.")
                return None

            # --- CRITICAL CHECK + DEBUG: Ensure best_match_dict is valid AND log MPN ---
            if not best_match_dict or not isinstance(best_match_dict, dict):
                logger.error(f"DigiKey: Final selected best_match is not a dictionary for '{keywords}'. Type: {type(best_match_dict)}. Cannot extract data.")
                return None
            # --- ADDED DEBUG ---
            mpn_from_selected_dict = best_match_dict.get("ManufacturerProductNumber", "---MPN MISSING FROM DICT---")
            logger.debug(f"DigiKey: MPN DIRECTLY from selected best_match_dict: '{mpn_from_selected_dict}'")
            # --- END ADDED DEBUG ---
            # --- End Check ---

            # --- Extract Data (Highly Defensive) ---
            mfg_data = best_match_dict.get("Manufacturer", {})
            mfg_name = mfg_data.get("Name", "N/A") if isinstance(mfg_data, dict) else "N/A"

            # --- Re-assign mfg_pn HERE using the confirmed best_match_dict ---
            mfg_pn = best_match_dict.get("ManufacturerProductNumber", "N/A")
            if mfg_pn == "N/A" or mfg_pn == "---MPN MISSING FROM DICT---": # Add extra check based on debug log
                 logger.error(f"DigiKey: CRITICAL - MPN still N/A or missing after selecting best_match_dict for {keywords}. Aborting extraction.")
                 return None # Abort if MPN somehow still invalid at this point
            # --- End Re-assignment ---

            desc_data = best_match_dict.get("Description", {})
            description = desc_data.get("Value", "N/A") if isinstance(desc_data, dict) else "N/A"
            datasheet_url = best_match_dict.get("DatasheetUrl", "N/A")
            digikey_pn_base = best_match_dict.get("DigiKeyProductNumber", "N/A")

            # ... (Variation Handling, Lead Time, Pricing, Status, COO/HTS, Parameters) ---
            stock = int(safe_float(best_match_dict.get("QuantityAvailable", 0), default=0))
            min_order_qty = int(safe_float(best_match_dict.get("MinimumOrderQuantity", 0), default=1)) # Default MOQ 1
            package_type = "N/A"
            digikey_pn = digikey_pn_base # Start with base DKPN

            variations = best_match_dict.get("ProductVariations", [])
            target_variation_data = None
            if isinstance(variations, list) and variations:
                # Try to find variation matching the base product's DKPN
                target_variation_data = next((v for v in variations if isinstance(v, dict) and v.get("DigiKeyProductNumber") == digikey_pn_base), None)
                # If not found, fallback to the first valid variation dict
                if not target_variation_data:
                     first_valid_variation = next((v for v in variations if isinstance(v, dict)), None)
                     if first_valid_variation:
                          target_variation_data = first_valid_variation
                          logger.debug(f"DigiKey: Variation matching base DKPN not found for {mfg_pn}. Using first valid variation.")

            if target_variation_data:
                logger.debug(f"DigiKey: Using data from variation for {mfg_pn}")
                stock = int(safe_float(target_variation_data.get("QuantityAvailable", stock), default=stock)) # Fallback to base stock
                min_order_qty = int(safe_float(target_variation_data.get("MinimumOrderQuantity", min_order_qty), default=min_order_qty))
                pkg_data = target_variation_data.get("PackageType", {})
                package_type = pkg_data.get("Name", "N/A") if isinstance(pkg_data, dict) else "N/A"
                variation_dkpn = target_variation_data.get("DigiKeyProductNumber");
                if variation_dkpn and variation_dkpn != "N/A":
                     digikey_pn = variation_dkpn # Update DKPN if valid one found in variation
            else:
                logger.debug(f"DigiKey: No valid variation; using product level stock/MOQ for {mfg_pn}")
            # --- End Variation Handling ---

            lead_time_weeks_str = best_match_dict.get("ManufacturerLeadWeeks")
            lead_time_days = convert_lead_time_to_days(lead_time_weeks_str)
            logger.debug(f"DigiKey lead time for {mfg_pn}: Raw='{lead_time_weeks_str}', Converted={lead_time_days} days")

            # --- Pricing (Robust) ---
            pricing_raw = best_match_dict.get("StandardPricing", [])
            pricing = []
            if isinstance(pricing_raw, list):
                 # Use safe_float and check validity within the loop
                 for pb in pricing_raw:
                      if isinstance(pb, dict):
                           qty = int(safe_float(pb.get("BreakQuantity", 0), default=0))
                           price = safe_float(pb.get("UnitPrice"))
                           if qty > 0 and pd.notna(price) and price >= 0:
                                pricing.append({"qty": qty, "price": price})
                 if pricing: pricing.sort(key=lambda x: x['qty']) # Sort only if valid breaks found
                 else: logger.warning(f"DigiKey: 'StandardPricing' list found for {mfg_pn} but contained no valid breaks.")
            else:
                 logger.warning(f"DigiKey: 'StandardPricing' is not a list for {mfg_pn}. Type: {type(pricing_raw)}")

            # Fallback to single unit price ONLY if standard pricing failed
            if not pricing:
                 unit_price_single = best_match_dict.get("UnitPrice")
                 single_price = safe_float(unit_price_single)
                 if pd.notna(single_price) and single_price >= 0:
                      pricing = [{"qty": 1, "price": single_price}]
                      logger.warning(f"DigiKey: Using top-level 'UnitPrice' fallback for {mfg_pn}.")
                 else:
                      logger.warning(f"DigiKey: Could not find ANY valid pricing for {mfg_pn}.")
            # --- End Pricing ---

            status_data = best_match_dict.get("ProductStatus", {})
            status_str = status_data.get("Status", "").lower() if isinstance(status_data, dict) else ""

            coo = "N/A"; hts = "N/A"
            classifications_data = best_match_dict.get("Classifications")
            if isinstance(classifications_data, dict):
                coo = classifications_data.get("CountryOfOrigin", "N/A")
                hts = classifications_data.get("HtsusCode", "N/A")

            # --- Parameters / Normally Stocking ---
            is_normally_stocking = False # Default false
            parameters_data = best_match_dict.get("Parameters")
            try:
                if isinstance(parameters_data, list):
                    norm_stock_param = next((p for p in parameters_data if isinstance(p, dict) and p.get("Parameter") == "Normally Stocking"), None)
                    if norm_stock_param:
                         is_normally_stocking = norm_stock_param.get("Value", "").lower() in ["yes", "true", "1"]
                # Add other ways 'normally stocking' might be represented if needed
            except Exception as param_err:
                logger.warning(f"DigiKey: Error parsing 'Parameters' for {mfg_pn}: {param_err}")
            # --- End Parameters ---

            # --- Final Result Dict ---
            result = {
                "Source": "DigiKey",
                "SourcePartNumber": digikey_pn, # Use potentially updated DKPN
                "ManufacturerPartNumber": mfg_pn, # Use mfg_pn extracted correctly now
                "Manufacturer": mfg_name,
                "Description": description,
                "Stock": stock,
                "LeadTimeDays": lead_time_days,
                "MinOrderQty": min_order_qty,
                "Packaging": package_type,
                "Pricing": pricing,
                "CountryOfOrigin": coo,
                "TariffCode": hts,
                "NormallyStocking": is_normally_stocking,
                "Discontinued": status_str == 'discontinued',
                # More specific EOL check based on common DK statuses
                "EndOfLife": status_str in ['obsolete', 'last time buy', 'nrnd'],
                "DatasheetUrl": datasheet_url,
                "ApiTimestamp": datetime.now(timezone.utc).isoformat(timespec='seconds'),
            }
            # Use the CORRECT mfg_pn in the success log
            logger.info(f"DigiKey search SUCCESS for '{keywords}'. MPN={result['ManufacturerPartNumber']}, Stock={result['Stock']}, LT={result['LeadTimeDays'] if pd.notna(result['LeadTimeDays']) else 'N/A'}")
            return result

        except requests.HTTPError as e:
            # ... (Error handling - 401, 404, etc.) ...
             # --- Handle 401 Unauthorized ---
            if e.response is not None and e.response.status_code == 401:
                logger.warning(f"DigiKey 401 Unauthorized for {part_number}. Token likely invalid. Will attempt refresh/re-auth on next call.")
                # DO NOT remove token file here. Let refresh/get_token handle it.
                # self.digikey_token_data = None # Signal that token is bad locally
            # --- Handle 404 Not Found ---
            elif e.response is not None and e.response.status_code == 404:
                 logger.info(f"DigiKey 404 Not Found for '{keywords}'.")
            # --- Handle Other HTTP Errors ---
            else:
                logger.error(f"DigiKey API HTTP Error for '{keywords}': {e}", exc_info=True)
            return None # Return None on HTTP errors
        except Exception as e:
            logger.error(f"DigiKey search failed unexpectedly for '{keywords}': {e}", exc_info=True)
            return None
        

    def search_mouser(self, part_number, manufacturer=""):
        """Searches Mouser using Keyword Search. Returns processed dict or None."""
        if not API_KEYS["Mouser"]: return None
        if not self.check_and_wait_mouser_rate_limit(): return None # Check limit

        # --- Use Keyword Search Endpoint ---
        url = "https://api.mouser.com/api/v1/search/keyword" # Changed endpoint
        headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
        params = {'apiKey': MOUSER_API_KEY}
        # Construct keyword including manufacturer if provided
        keyword = f"{manufacturer} {part_number}".strip() if manufacturer else part_number
        # Request multiple records to potentially find the best match if keyword is ambiguous
        body = {'SearchByKeywordRequest': {'keyword': keyword, 'records': 5, 'startingRecord': 0, 'searchOptions': 'RohsAndReach'}}
        logger.debug(f"Mouser Searching (Keyword API) for: '{keyword}'")
        # --- End Endpoint/Body Change ---

        try:
            response = self._make_api_request("POST", url, headers=headers, params=params, json=body)
            raw_response_text = response.text

            # Increment count *after* successful request OR non-401/429 error
            self.mouser_requests_today += 1
            self.save_mouser_request_counter()
            self.root.after(0, self.update_rate_limit_display)

            try: data = response.json()
            except json.JSONDecodeError as json_err:
                 logger.error(f"Mouser JSON Decode Error for '{keyword}': {json_err}. Response: {raw_response_text[:500]}")
                 return None

            if 'Errors' in data and data['Errors']:
                err_msg = data['Errors'][0].get('Message', 'Unknown Mouser Error') if data['Errors'] else 'Unknown'
                if "not found" in err_msg.lower(): logger.debug(f"Mouser: Part '{keyword}' not found via Keyword API.")
                else: logger.error(f"Mouser API Error for '{keyword}': {err_msg}")
                return None

            parts_list = data.get('SearchResults', {}).get('Parts', [])
            if not parts_list:
                logger.debug(f"Mouser: No parts found for '{keyword}' via Keyword API.")
                return None

            # --- Find Best Match (Prioritize Exact MPN & Manufacturer if provided) ---
            best_match = None
            exact_pn_upper = part_number.upper()
            mfg_upper = manufacturer.upper().strip() if manufacturer else None

            potential_matches = []
            for p in parts_list:
                if isinstance(p, dict) and p.get("ManufacturerPartNumber", "").upper() == exact_pn_upper:
                    potential_matches.append(p)

            if not potential_matches:
                 logger.warning(f"Mouser: No exact MPN match found for '{keyword}' in results. Using first result.")
                 best_match = parts_list[0] if isinstance(parts_list[0], dict) else None
            elif mfg_upper and len(potential_matches) > 1:
                 filtered_by_mfg = [p for p in potential_matches if p.get("Manufacturer", "").upper().strip() == mfg_upper]
                 if filtered_by_mfg:
                     best_match = filtered_by_mfg[0]
                     logger.debug(f"Mouser: Found exact MPN+Mfg match for '{keyword}'.")
                 else:
                     best_match = potential_matches[0]
                     logger.debug(f"Mouser: Found exact MPN but no Mfg match for '{keyword}'. Using first exact MPN match.")
            else: # Only one exact MPN match or no manufacturer provided
                best_match = potential_matches[0]
                logger.debug(f"Mouser: Using first/only exact MPN match for '{keyword}'.")

            if not isinstance(best_match, dict):
                 logger.error(f"Mouser: Failed to select a valid part dictionary for '{keyword}'.")
                 return None
            # --- End Match Finding ---


            # --- Extract Data ---
            lead_time_str = best_match.get('LeadTime')
            lead_time_days = convert_lead_time_to_days(lead_time_str)

            pricing_raw = best_match.get('PriceBreaks', [])
            pricing = [{"qty": int(p["Quantity"]), "price": safe_float(p["Price"].replace('$',''))} for p in pricing_raw if isinstance(p, dict) and safe_float(p.get("Price")) is not None and int(p.get("Quantity", 0)) > 0]
            if pricing: pricing.sort(key=lambda x: x['qty'])

            lifecycle_status = best_match.get('LifecycleStatus', '') or ""

            result = {
                "Source": "Mouser",
                "SourcePartNumber": best_match.get('MouserPartNumber', "N/A"),
                "ManufacturerPartNumber": best_match.get('ManufacturerPartNumber', "N/A"),
                "Manufacturer": best_match.get('Manufacturer', "N/A"),
                "Description": best_match.get('Description', "N/A"),
                "Stock": int(safe_float(best_match.get('AvailabilityInStock', 0), default=0)), # Use AvailabilityInStock
                "LeadTimeDays": lead_time_days,
                "MinOrderQty": int(safe_float(best_match.get('Min', 0), default=1)), # Default MOQ 1
                "Packaging": best_match.get('Packaging', "N/A"),
                "Pricing": pricing,
                "CountryOfOrigin": best_match.get("CountryOfOrigin", "N/A"),
                "TariffCode": "N/A",
                "NormallyStocking": True, # Assumption
                "Discontinued": "discontinued" in lifecycle_status.lower(),
                "EndOfLife": any(s in lifecycle_status.lower() for s in ["obsolete", "nrnd", "not recommended"]),
                "DatasheetUrl": best_match.get('DataSheetUrl', "N/A"),
                "ApiTimestamp": datetime.now(timezone.utc).isoformat(timespec='seconds'),
            }
            logger.info(f"Mouser search SUCCESS for '{keyword}'. MPN={result['ManufacturerPartNumber']}, Stock={result['Stock']}, LT={result['LeadTimeDays'] if pd.notna(result['LeadTimeDays']) else 'N/A'}")
            return result

        except requests.HTTPError as e:
             # ... (Error handling) ...
              # Check specifically for 401 Unauthorized / 403 Forbidden
             if e.response is not None and e.response.status_code in [401, 403]:
                 logger.error(f"Mouser API Key Invalid or Unauthorized ({e.response.status_code}). Disabling.", exc_info=False)
                 API_KEYS["Mouser"] = False
                 # Use lambda or functools.partial for arguments in after()
                 self.root.after(0, lambda: self.api_status_labels["Mouser"].config(text="Mouser: Invalid Key", foreground="red"))
                 self.root.after(0, self.update_rate_limit_display)
             elif e.response is not None and e.response.status_code == 404:
                  logger.debug(f"Mouser 404 Not Found for '{keyword}'.") # Use keyword here
             else:
                  logger.error(f"Mouser API HTTP Error for '{keyword}': {e}", exc_info=True) # Use keyword here

             # Avoid decrementing count if it was already incremented successfully before error
             # Safety check - decrement only if status code suggests request didn't count towards limit (e.g., auth errors)
             if e.response is not None and e.response.status_code not in [429, 500, 502, 503, 504]: # Don't decrement for server-side/rate limit issues
                  # Check if counter was already incremented in this call attempt (tricky, maybe needs a flag)
                  # Simpler: just don't decrement here, accept slight overcount on some errors.
                  pass
             return None
        except (TimeoutError, ConnectionError, RuntimeError, Exception) as e:
            logger.error(f"Mouser search failed for '{keyword}': {e}", exc_info=True) # Use keyword
            # Avoid decrementing count here as request likely failed before incrementing
            return None


    def search_octopart_nexar(self, part_number, manufacturer=""):
        """Searches Octopart/Nexar using GraphQL. Returns processed dict or None."""
        # ... (token check logic remains the same) ...
        if not API_KEYS["Octopart (Nexar)"]: return None
        access_token = self.get_nexar_token()
        if not access_token:
            logger.warning("Cannot search Nexar without access token.")
            return None

        headers = { 'Authorization': f'Bearer {access_token}', 'Content-Type': 'application/json'}
        search_term = part_number # Using MPN directly

        # Corrected GraphQL Query - Removed limit from offers
        graphql_query = f"""
        query MinimalPartSearch {{
          supSearchMpn(q: "{search_term}", limit: 1, country: "US", currency: "USD") {{
            hits
            results {{
              part {{
                mpn
                manufacturer {{ name }}
                shortDescription
                bestDatasheet {{ url }} # Correct syntax: field {{ subfield }}
                sellers(authorizedOnly: false) {{ # No limit on sellers
                  company {{ name }}
                  isAuthorized
                  offers {{ # No limit on offers
                    sku
                    inventoryLevel
                    moq
                    packaging
                    factoryLeadDays
                    prices {{ quantity price currency }}
                  }}
                }}
              }}
            }}
          }}
        }}
        """
        # Also added bestDatasheet query field
        logger.debug(f"Nexar GraphQL Query (Corrected - no offers limit) for MPN '{search_term}'")

        try:
            response = self._make_api_request("POST", NEXAR_API_URL, headers=headers, json={'query': graphql_query})
            data = response.json()
            # ... (rest of the error handling and data processing remains the same) ...

            if "errors" in data:
                # ... (error handling) ...
                return None

            search_results = data.get("data", {}).get("supSearchMpn", {}).get("results", [])
            if not search_results:
                logger.info(f"Nexar: No results found via supSearchMpn for '{search_term}'.")
                return None

            part_data = search_results[0].get("part", {})
            if not part_data:
                 logger.warning(f"Nexar: Found result but no 'part' data for '{search_term}'.")
                 return None

             # --- Select Best Offer ---
            potential_offers = []
            sellers = part_data.get("sellers", [])
            if sellers and isinstance(sellers, list):
                for seller in sellers:
                   # Correctly handle nested company name
                    seller_name = seller.get("company", {}).get("name", "Unknown Seller") if isinstance(seller.get("company"), dict) else "Unknown Seller"
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
                 best_offer_data = {}
                 best_seller_name = "Nexar Aggregate"
            else:
                 # Sort offers: Prioritize authorized, then stock, then MOQ
                 potential_offers.sort(key=lambda x: (
                     -int(x.get('is_authorized', False)), # Authorized first (True > False)
                     -(x.get('inventoryLevel', 0) or 0), # Higher stock first
                     x.get('moq', 0) or float('inf'),    # Lower MOQ first
                 ))
                 best_offer_data = potential_offers[0]
                 best_seller_name = best_offer_data.get('seller_name', "Nexar Aggregate")
                 logger.debug(f"Nexar: Selected best offer for {search_term} from '{best_seller_name}'...")


            # --- Extract Data ---
            # ... (pricing extraction remains the same) ...
            prices_raw = best_offer_data.get("prices", [])
            pricing = [{"qty": int(p["quantity"]), "price": safe_float(p["price"])} for p in prices_raw if isinstance(p, dict) and p.get("currency", "USD") == "USD" and safe_float(p.get("price")) is not None and int(p.get("quantity", 0)) > 0]
            if pricing: pricing.sort(key=lambda x: x['qty'])

            raw_lead_value = best_offer_data.get("factoryLeadDays")
            logger.debug(f"Nexar [{search_term}]: Raw factoryLeadDays from best offer ({best_seller_name}): '{raw_lead_value}' (Type: {type(raw_lead_value)})")
            
            # Ensure lead time is treated as integer days if valid
            lead_time_days_float = safe_float(raw_lead_value) # Use safe_float first
            lead_time_days = int(lead_time_days_float) if pd.notna(lead_time_days_float) else np.nan # Convert to int if valid float, else nan

            mfg_name = part_data.get('manufacturer', {}).get('name', manufacturer or "N/A")
            mpn = part_data.get('mpn', part_number)
            # Use the queried datasheet URL
            datasheet_url = part_data.get('bestDatasheet', {}).get('url', 'N/A') if isinstance(part_data.get('bestDatasheet'), dict) else 'N/A'

            result = {
                "Source": "Octopart (Nexar)",
                "SourcePartNumber": best_offer_data.get('sku', "N/A"),
                "ManufacturerPartNumber": mpn,
                "Manufacturer": mfg_name,
                "Description": part_data.get('shortDescription', "N/A"),
                "Stock": int(safe_float(best_offer_data.get('inventoryLevel', 0), default=0)),
                "LeadTimeDays": lead_time_days,
                "MinOrderQty": int(safe_float(best_offer_data.get('moq', 0), default=0)),
                "Packaging": best_offer_data.get('packaging', "N/A"),
                "Pricing": pricing,
                "CountryOfOrigin": "N/A", # Not reliably available in this query structure
                "TariffCode": "N/A",
                "NormallyStocking": True, # Assumption for Octopart listings
                "Discontinued": False, # Lifecycle not easily available here
                "EndOfLife": False,
                "DatasheetUrl": datasheet_url, # Use queried URL
                "ApiTimestamp": datetime.now(timezone.utc).isoformat(timespec='seconds'),
            }
            logger.info(f"Nexar search SUCCESS for '{search_term}'. MPN={result['ManufacturerPartNumber']}, Stock={result['Stock']}, LT={result['LeadTimeDays'] if pd.notna(result['LeadTimeDays']) else 'N/A'}")
            return result

        except requests.HTTPError as e:
            logger.error(f"Nexar API HTTP Error for {part_number}: {e}", exc_info=True) # Log full trace for HTTP errors
            return None
        except Exception as e:
            logger.error(f"Nexar search failed unexpectedly for {part_number}: {e}", exc_info=True)
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

            # --- START Robustness Check ---
            content_type = response.headers.get('Content-Type', '').lower()
            if 'application/json' not in content_type:
                logger.error(f"USITC API did not return JSON for HTS {hts_code}. Content-Type: {content_type}. Response text (first 200): {response.text[:200]}")
                self._hts_cache[hts_clean] = None # Cache failure
                return None
            # --- END Robustness Check ---

            data = response.json() # Safe to parse now

            # Parse the response structure (this might change based on USITC API updates)
            # Example: Assuming results are in a list, and we want 'general_rate'
            if isinstance(data, dict) and 'results' in data and isinstance(data['results'], list) and data['results']:
                # Find exact match or best result
                found_article = None
                for article in data['results']:
                     if not isinstance(article, dict): continue # Skip non-dict items
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

        except requests.RequestException as e:
             logger.error(f"USITC request failed for HTS {hts_code}: {e}")
        except (json.JSONDecodeError, ValueError, KeyError, AttributeError) as e:
            # JSONDecodeError will likely be caught by the Content-Type check now, but keep for safety
             logger.error(f"Error parsing USITC response for HTS {hts_code}: {e}")
        except Exception as e:
             logger.error(f"Unexpected error fetching USITC tariff for HTS {hts_code}: {e}", exc_info=True)

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
    def export_strategy_gui(self, strategy_key_internal): # Changed parameter name for clarity
        """Handles button click and exports the selected strategy to CSV."""
        logger.info(f"Exporting strategy using internal key: '{strategy_key_internal}'") # Log the key being used

        if not self.strategies_for_export:
            messagebox.showerror("Export Error", "No strategy data available. Please run analysis.")
            logger.warning("Export failed: self.strategies_for_export is empty.")
            return

        # Use the internal key directly to get the data
        strategy_dict = self.strategies_for_export.get(strategy_key_internal)
        if not strategy_dict:
            messagebox.showerror("Export Error", f"Data for '{strategy_key_internal}' strategy not found or is empty.")
            logger.warning(f"Export failed: Strategy key '{strategy_key_internal}' not found or dict is empty in self.strategies_for_export.")
            return

        output_data = []
        # Define a standard header - adjust specific columns if needed later
        # Using a mostly consistent header simplifies things
        output_header = [
             "BOM Part Number", "Manufacturer", "Manufacturer PN", "Qty Per Unit", "Total Qty Needed",
             "Chosen Source", "Source PN",
             "Unit Cost ($)", "Total Cost ($)", "Actual Qty Ordered", # Use these consistent names
             "Lead Time (Days)", "Stock", "Notes/Score" # Combine notes/score
         ]
        output_data.append(output_header)

        parts_exported = 0
        for bom_pn, chosen_option in strategy_dict.items():
            if not isinstance(chosen_option, dict):
                 logger.warning(f"Skipping export for {bom_pn} in strategy '{strategy_key_internal}': Invalid option format {type(chosen_option)}.")
                 continue

            # Extract data using .get() - create_strategy_entry ensures keys exist mostly
            lead_time_val = chosen_option.get('lead_time', np.nan)
            lead_time_str = f"{lead_time_val:.0f}" if pd.notna(lead_time_val) and lead_time_val != np.inf else "N/A"

            # Get cost/qty based on strategy type if strict cost was different
            # For simplicity now, we'll export the 'cost' and 'actual_order_qty' stored,
            # assuming create_strategy_entry correctly stored the relevant values.
            # If you need separate "strict" columns, more logic is needed here.
            unit_cost = chosen_option.get('unit_cost', np.nan)
            total_cost = chosen_option.get('cost', np.nan)
            actual_qty = chosen_option.get("actual_order_qty", chosen_option.get("total_qty_needed", "N/A"))

            unit_cost_str = f"{unit_cost:.4f}" if pd.notna(unit_cost) else "N/A"
            total_cost_str = f"{total_cost:.2f}" if pd.notna(total_cost) else "N/A"
            actual_qty_str = str(actual_qty)

            # Combine notes and score
            notes = str(chosen_option.get('notes', ''))
            score = str(chosen_option.get('optimized_strategy_score', ''))
            notes_score_str = notes
            if score and score != "N/A" and strategy_key_internal == 'Optimized Strategy':
                 notes_score_str = f"{notes}; Score: {score}".strip('; ')

            output_data.append([
                 chosen_option.get("bom_pn", "N/A"),
                 chosen_option.get("Manufacturer", "N/A"),
                 chosen_option.get("ManufacturerPartNumber", "N/A"),
                 chosen_option.get("original_qty_per_unit", "N/A"),
                 chosen_option.get("total_qty_needed", "N/A"),
                 chosen_option.get("source", "N/A"),
                 chosen_option.get("SourcePartNumber", "N/A"),
                 unit_cost_str,
                 total_cost_str,
                 actual_qty_str,
                 lead_time_str,
                 chosen_option.get("stock", 0),
                 notes_score_str # Combined notes/score
            ])
            parts_exported += 1

        if parts_exported == 0:
             messagebox.showinfo("Export Info", f"No valid part data found to export for the '{strategy_key_internal}' strategy.")
             return

        # Ask user for filename
        # Use the internal key for the filename for consistency
        safe_filename_key = strategy_key_internal.replace(' ', '_').replace('/', '')
        default_filename = f"BOM_Strategy_{safe_filename_key}_{datetime.now():%Y%m%d_%H%M}.csv"
        filepath = filedialog.asksaveasfilename(
            title=f"Save {strategy_key_internal} Strategy As", defaultextension=".csv", # Use internal key in title
            initialfile=default_filename, filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not filepath:
            logger.info("User cancelled export.")
            return # User cancelled

        # Write to CSV
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
                writer.writerows(output_data) # Write header + data rows
            self.update_status_threadsafe(f"Exported '{strategy_key_internal}' ({parts_exported} parts) to {Path(filepath).name}", "success")
            messagebox.showinfo("Export Successful", f"Successfully exported {parts_exported} parts for strategy '{strategy_key_internal}' to:\n{filepath}")
        except IOError as e:
            logger.error(f"Failed to export strategy CSV: {e}", exc_info=True)
            messagebox.showerror("Export Error", f"Failed to write CSV file:\n{filepath}\n\nError: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during strategy export: {e}", exc_info=True)
            messagebox.showerror("Export Error", f"An unexpected error occurred during export:\n\n{e}")

    
    # --- Main Analysis Function (Per Part) ---
    def analyze_single_part(self, bom_part_number, bom_manufacturer, bom_qty_per_unit, config):
        logger.debug(f"--- Analyzing Part Start: {bom_part_number} (Mfg: {bom_manufacturer}, Qty/Unit: {bom_qty_per_unit}) ---")
        # --- Initial Setup ---
        total_units = config.get('total_units', 1)
        buy_up_threshold_pct = config.get('buy_up_threshold', 1.0)
        total_qty_needed = int(bom_qty_per_unit * total_units)
        if total_qty_needed <= 0:
            logger.warning(f"Skipping analysis for {bom_part_number}: Calculated total_qty_needed is zero or negative.")
            gui_entry = {
                "PartNumber": bom_part_number,
                "Notes": "Invalid quantity",
                **{k: "N/A" for k in ["Manufacturer", "MfgPN", "QtyNeed", "Status", "Sources", "StockAvail", "COO", "RiskScore", "TariffPct", "BestCostPer", "BestTotalCost", "ActualBuyQty", "BestCostLT", "BestCostSrc"]},
                "Alternates": "No",
                "AlternatesList": []  # Ensure AlternatesList is set
            }
            logger.debug(f"gui_entry for invalid quantity: {gui_entry}")
            return [gui_entry], [], {}
    
        # --- Fetch Data ---
        try:
            part_results_by_supplier = self.get_part_data_parallel(bom_part_number, bom_manufacturer)
        except Exception as e:
            logger.error(f"Failed to fetch supplier data for {bom_part_number}: {str(e)}")
            gui_entry = {
                "PartNumber": bom_part_number,
                "Manufacturer": bom_manufacturer or "N/A",
                "MfgPN": "NOT FOUND",
                "QtyNeed": total_qty_needed,
                "Status": "Error",
                "Sources": "0",
                "StockAvail": "N/A",
                "COO": "N/A",
                "RiskScore": "10.0",
                "TariffPct": "N/A",
                "BestCostPer": "N/A",
                "BestTotalCost": "N/A",
                "ActualBuyQty": "N/A",
                "BestCostLT": "N/A",
                "BestCostSrc": "N/A",
                "Alternates": "No",
                "AlternatesList": []  # Ensure AlternatesList is set
            }
            logger.debug(f"gui_entry for fetch error: {gui_entry}")
            return [gui_entry], [], {}
    
        # --- Initialize Return Structures ---
        historical_entries = []
        part_summary = {
            "bom_pn": bom_part_number, "bom_mfg": bom_manufacturer,
            "original_qty_per_unit": bom_qty_per_unit,
            "total_qty_needed": total_qty_needed,
            "options": [], "alternates": []
        }
    
        # --- Check for No Supplier Data (Unknown) ---
        if not part_results_by_supplier or all(not isinstance(data, dict) for data in part_results_by_supplier.values()):
            logger.warning(f"Part {bom_part_number}: No supplier data returned. Marking as Unknown.")
            gui_entry = {
                "PartNumber": bom_part_number,
                "Manufacturer": bom_manufacturer or "N/A",
                "MfgPN": "NOT FOUND",
                "QtyNeed": total_qty_needed,
                "Status": "Unknown",
                "Sources": "0",
                "StockAvail": "N/A",
                "COO": "N/A",
                "RiskScore": "10.0",
                "TariffPct": "N/A",
                "BestCostPer": "N/A",
                "BestTotalCost": "N/A",
                "ActualBuyQty": "N/A",
                "BestCostLT": "N/A",
                "BestCostSrc": "N/A",
                "Alternates": "No",
                "AlternatesList": []  # Ensure AlternatesList is set
            }
            logger.debug(f"gui_entry for unknown part: {gui_entry}")
            return [gui_entry], [], {}
    
        # --- Check Lifecycle Status Early ---
        lifecycle_notes = set()
        consolidated_mfg = bom_manufacturer or "N/A"; mfg_source = "BOM"
        consolidated_mpn = bom_part_number; mpn_source = "BOM"
    
        # Consolidate Mfg/MPN and check lifecycle
        for source, data in part_results_by_supplier.items():
            if not isinstance(data, dict): continue
            # Consolidate Mfg/MPN for alternates lookup
            api_mfg = data.get('Manufacturer')
            api_mpn = data.get('ManufacturerPartNumber')
            if api_mfg and api_mfg != "N/A" and consolidated_mfg == "N/A":
                consolidated_mfg = api_mfg; mfg_source = source
            if api_mpn and api_mpn != "N/A" and consolidated_mpn == "N/A":
                consolidated_mpn = api_mpn; mpn_source = source
            # Collect lifecycle notes
            if data.get('Discontinued'): lifecycle_notes.add("DISC")
            if data.get('EndOfLife'): lifecycle_notes.add("EOL")
    
        # --- Handle EOL/Discontinued Parts ---
        if "EOL" in lifecycle_notes or "DISC" in lifecycle_notes:
            status = "EOL" if "EOL" in lifecycle_notes else "Discontinued"
            logger.info(f"Part {bom_part_number} is {status}. Fetching alternates but excluding from calculations.")
    
            # Fetch alternates for EOL/Discontinued part
            substitutes = []
            if API_KEYS["DigiKey"] and consolidated_mpn != "NOT FOUND":
                substitutes = self.get_digikey_substitutions(consolidated_mpn)
            part_summary['alternates'] = substitutes
    
            gui_entry = {
                "PartNumber": bom_part_number,
                "Manufacturer": consolidated_mfg,
                "MfgPN": consolidated_mpn,
                "QtyNeed": total_qty_needed,
                "Status": status,
                "Sources": f"{len(part_results_by_supplier)}",
                "StockAvail": "N/A",
                "COO": "N/A",
                "RiskScore": "10.0",
                "TariffPct": "N/A",
                "BestCostPer": "N/A",
                "BestTotalCost": "N/A",
                "ActualBuyQty": "N/A",
                "BestCostLT": "N/A",
                "BestCostSrc": "N/A",
                "Alternates": "Yes" if substitutes else "No",
                "AlternatesList": substitutes if substitutes else [],  # Ensure AlternatesList is set
                "Notes": f"Excluded ({status})"
            }
            logger.debug(f"gui_entry for EOL/Discontinued part: {gui_entry}")
            return [gui_entry], [], part_summary
    
        # --- Process Valid Parts ---
        all_options_data = []
        for source, data in part_results_by_supplier.items():
            if not isinstance(data, dict): continue
            api_mfg = data.get('Manufacturer')
            api_mpn = data.get('ManufacturerPartNumber')
            if api_mfg and api_mfg != "N/A" and consolidated_mfg == "N/A":
                consolidated_mfg = api_mfg; mfg_source = source
            if api_mpn and api_mpn != "N/A" and consolidated_mpn == "N/A":
                consolidated_mpn = api_mpn; mpn_source = source
    
            # Calculate optimal cost
            supplier_pricing_breaks = data.get('Pricing', [])
            unit_cost, total_cost, actual_order_qty, cost_notes = self.get_optimal_cost(
                total_qty_needed, supplier_pricing_breaks, data.get('MinOrderQty', 0), buy_up_threshold_pct
            )
    
            processed_lead_time = np.inf if pd.isna(data.get('LeadTimeDays')) else float(data.get('LeadTimeDays'))
            option_dict = {
                "source": source,
                "cost": total_cost if pd.notna(total_cost) else np.inf,
                "lead_time": processed_lead_time,
                "stock": data.get('Stock', 0),
                "unit_cost": unit_cost,
                "actual_order_qty": actual_order_qty,
                "moq": data.get('MinOrderQty', 0),
                "discontinued": data.get('Discontinued', False),
                "eol": data.get('EndOfLife', False),
                'bom_pn': bom_part_number,
                'original_qty_per_unit': bom_qty_per_unit,
                'total_qty_needed': total_qty_needed,
                'Manufacturer': data.get('Manufacturer', 'N/A'),
                'ManufacturerPartNumber': data.get('ManufacturerPartNumber', 'N/A'),
                'SourcePartNumber': data.get('SourcePartNumber', 'N/A'),
                'Pricing': supplier_pricing_breaks,
                'TariffCode': data.get('TariffCode'),
                'CountryOfOrigin': data.get('CountryOfOrigin'),
                'ApiTimestamp': data.get('ApiTimestamp'),
                'tariff_rate': None,
                'stock_prob': 0.0,
                'notes': cost_notes,
            }
            all_options_data.append(option_dict)
    
        if not all_options_data:
            logger.error(f"Part {bom_part_number}: Had API results but all_options_data is empty after processing loop. Check logs for errors during processing.")
            gui_entry = {
                "PartNumber": bom_part_number,
                "Notes": "Data Processing Error",
                **{k: "ERR" for k in ["Manufacturer", "MfgPN", "QtyNeed", "Status", "Sources", "StockAvail", "COO", "RiskScore", "TariffPct", "BestCostPer", "BestTotalCost", "ActualBuyQty", "BestCostLT", "BestCostSrc", "Alternates"]},
                "AlternatesList": []  # Ensure AlternatesList is set
            }
            logger.debug(f"gui_entry for data processing error: {gui_entry}")
            return [gui_entry], [], {}
    
        part_summary["options"] = all_options_data
    
        # --- Consolidate Part Info (Mfg, MPN, COO, HTS, Alternates) ---
        logger.debug(f"Part {bom_part_number}: Final Consolidated Mfg='{consolidated_mfg}' (Source: {mfg_source}), MPN='{consolidated_mpn}' (Source: {mpn_source})")
        final_component_name = f"{consolidated_mfg} {consolidated_mpn}".strip()
    
        # Get alternates based on consolidated MPN
        substitutes = []
        if API_KEYS["DigiKey"] and consolidated_mpn != "NOT FOUND": # Check if MPN is valid before searching
             substitutes = self.get_digikey_substitutions(consolidated_mpn)
        part_summary['alternates'] = substitutes
    
        # --- Consolidate COO/HTS (Revised with logging) ---
        consolidated_coo = "N/A"; coo_source_log = "None Found"
        consolidated_hts = "N/A"; hts_source_log = "None Found"
    
        for option in all_options_data:
            api_coo = option.get('CountryOfOrigin')
            logger.debug(f"Checking COO from {option['source']}: {api_coo}")
            if api_coo and isinstance(api_coo, str) and api_coo.strip().upper() not in ["N/A", "", "UNKNOWN", "AGGREGATE"]:
                consolidated_coo = api_coo.strip()
                coo_source_log = f"API ({option['source']})"
                break
    
        if consolidated_coo == "N/A":
            for option in all_options_data:
                api_hts = option.get('TariffCode')
                logger.debug(f"Checking HTS from {option['source']}: {api_hts}")
                if api_hts and isinstance(api_hts, str) and api_hts.strip().lower() not in ['n/a', '']:
                    consolidated_hts = api_hts.strip()
                    hts_source_log = f"API ({option['source']})"
                    inferred_coo = self.infer_coo_from_hts(consolidated_hts)
                    if inferred_coo != "Unknown":
                        consolidated_coo = inferred_coo
                        coo_source_log = f"Inferred from HTS ({consolidated_hts} via {option['source']})"
                    break
    
        logger.info(f"Final COO for {bom_part_number}: {consolidated_coo} (Source: {coo_source_log})")
    
        # --- Calculate Consolidated Metrics & Final GUI Row ---
        options_with_valid_cost = [opt for opt in all_options_data if opt.get('cost', np.inf) != np.inf]
        options_with_valid_lt = [opt for opt in all_options_data if opt.get('lead_time', np.inf) != np.inf] # Includes 0 days
    
        # --- Determine BEST COST Option (Prioritize Stock) ---
        best_cost_option = None
        in_stock_options_cost = [opt for opt in options_with_valid_cost if opt.get('stock', 0) >= total_qty_needed]
    
        if in_stock_options_cost:
            # Among in-stock, find the cheapest (tie-break with source name for stability if needed)
            best_cost_option = min(in_stock_options_cost, key=lambda x: (x.get('cost', np.inf), x.get('source', '')))
            logger.debug(f"Best Cost for {bom_part_number}: Selected IN-STOCK option from {best_cost_option['source']}")
        elif options_with_valid_cost:
            # If no stock, find the absolute cheapest among all options with a valid cost
            best_cost_option = min(options_with_valid_cost, key=lambda x: (x.get('cost', np.inf), x.get('lead_time', np.inf), x.get('source', '')))
            logger.debug(f"Best Cost for {bom_part_number}: No in-stock options meet need. Selected cheapest overall from {best_cost_option['source']}")
        else:
            logger.warning(f"Best Cost for {bom_part_number}: No options with valid cost found.")
            # Keep best_cost_option as None
    
        # --- Determine FASTEST Option (Prioritize Stock) ---
        fastest_lt_option = None
        in_stock_options_lt = [opt for opt in all_options_data if opt.get('stock', 0) >= total_qty_needed] # Check all options for stock
    
        if in_stock_options_lt:
            # Among in-stock, find the cheapest (as LT is effectively 0)
            fastest_lt_option = min(in_stock_options_lt, key=lambda x: (x.get('cost', np.inf), x.get('source', '')))
            logger.debug(f"Fastest LT for {bom_part_number}: Selected IN-STOCK option from {fastest_lt_option['source']}")
        elif options_with_valid_lt:
            # If no stock, find the one with the shortest finite lead time (tie-break with cost)
            fastest_lt_option = min(options_with_valid_lt, key=lambda x: (x.get('lead_time', np.inf), x.get('cost', np.inf), x.get('source', '')))
            logger.debug(f"Fastest LT for {bom_part_number}: No in-stock options meet need. Selected shortest LT from {fastest_lt_option['source']}")
        else:
            logger.warning(f"Fastest LT for {bom_part_number}: No options with valid lead time found.")
    
        # Calculate overall metrics using all available data
        consolidated_tariff_rate, tariff_source_info = self.get_tariff_info(consolidated_hts, consolidated_coo, config.get('custom_tariff_rates', {}))
        stock_prob = self.calculate_stock_probability_simple(all_options_data, total_qty_needed)
        total_stock_available = 0
        min_lead_no_stock = np.inf
        lifecycle_notes = set()
        has_stock_gap = False # Recalculate based on ALL options
    
        for option in all_options_data:
            option['tariff_rate'] = consolidated_tariff_rate # Store tariff rate used
            option['stock_prob'] = stock_prob # Store calculated probability
    
            # Add to historical entries
            historical_entries.append([
                final_component_name,
                option.get('Manufacturer', 'N/A'), option.get('ManufacturerPartNumber', 'N/A'),
                option.get('source'),
                option.get('lead_time', np.nan) if option.get('lead_time') != np.inf else np.nan, # Convert inf back to nan for CSV
                option.get('unit_cost', np.nan), 
                option.get('stock', 0),
                stock_prob,
                option.get('ApiTimestamp', datetime.now(timezone.utc).isoformat(timespec='seconds'))
            ])
    
            # Update aggregate info
            total_stock_available += option.get('stock', 0)
            if option.get('discontinued'): lifecycle_notes.add("DISC")
            if option.get('eol'): lifecycle_notes.add("EOL")
            if option.get('stock', 0) < total_qty_needed and option.get('lead_time', np.inf) != np.inf:
                 min_lead_no_stock = min(min_lead_no_stock, option['lead_time'])
    
        # Determine stock gap based on ALL options
        has_stock_gap = not any(opt.get('stock', 0) >= total_qty_needed for opt in all_options_data)
    
        # --- Calculate Risk Factors & Score ---
        risk_factors = {'Sourcing': 0, 'Stock': 0, 'LeadTime': 0, 'Lifecycle': 0, 'Geographic': 0}
        num_sources_found = len(all_options_data)
        if num_sources_found <= 1: risk_factors['Sourcing'] = 10
        elif num_sources_found == 2: risk_factors['Sourcing'] = 5
        else: risk_factors['Sourcing'] = 1
    
        if has_stock_gap: risk_factors['Stock'] = 8
        elif total_stock_available < 1.5 * total_qty_needed: risk_factors['Stock'] = 4
        else: risk_factors['Stock'] = 0
    
        # Base lead time risk on FASTEST available option's lead time
        fastest_overall_lead = fastest_lt_option.get('lead_time', np.inf) if fastest_lt_option else np.inf
        if fastest_overall_lead == 0: risk_factors['LeadTime'] = 0 # In stock
        elif fastest_overall_lead == np.inf: risk_factors['LeadTime'] = 9 # No LT info
        elif fastest_overall_lead > 90: risk_factors['LeadTime'] = 7
        elif fastest_overall_lead > 45: risk_factors['LeadTime'] = 4
        else: risk_factors['LeadTime'] = 1
    
        if "EOL" in lifecycle_notes or "DISC" in lifecycle_notes: risk_factors['Lifecycle'] = 10
        else: risk_factors['Lifecycle'] = 0
        risk_factors['Geographic'] = self.GEO_RISK_TIERS.get(consolidated_coo, self.GEO_RISK_TIERS["_DEFAULT_"])
        overall_risk_score = sum(risk_factors[factor] * self.RISK_WEIGHTS[factor] for factor in self.RISK_WEIGHTS)
        overall_risk_score = round(max(0, min(10, overall_risk_score)), 1)
    
        # --- Determine Status and Notes for GUI ---
        status = "Active"; notes_str_list = []
        if "EOL" in lifecycle_notes: status = "EOL"
        elif "DISC" in lifecycle_notes: status = "Discontinued"
        if has_stock_gap: notes_str_list.append("Stock Gap")
        # Get buy-up notes from the BEST COST option chosen
        best_cost_notes = best_cost_option.get('notes', '') if best_cost_option else ''
        if best_cost_notes: notes_str_list.append(best_cost_notes)
        notes_str = "; ".join(notes_str_list)
    
        # --- Create Final GUI Entry using best_cost_option and fastest_lt_option ---
        gui_entry = {
            "PartNumber": bom_part_number,
            "Manufacturer": consolidated_mfg,
            "MfgPN": consolidated_mpn,
            "QtyNeed": total_qty_needed,
            "Status": status,
            "Sources": f"{len(all_options_data)}",
            "StockAvail": f"{total_stock_available}",
            "COO": consolidated_coo,
            "RiskScore": f"{overall_risk_score:.1f}" if pd.notna(overall_risk_score) else "N/A",
            "TariffPct": f"{consolidated_tariff_rate * 100:.1f}%" if pd.notna(consolidated_tariff_rate) else "N/A",
            "BestCostPer": f"{best_cost_option.get('unit_cost', np.nan):.4f}" if best_cost_option and pd.notna(best_cost_option.get('unit_cost', np.nan)) else "N/A",
            "BestTotalCost": f"{best_cost_option.get('cost', np.inf):.2f}" if best_cost_option and best_cost_option.get('cost', np.inf) != np.inf else "N/A",
            "ActualBuyQty": f"{best_cost_option.get('actual_order_qty', 'N/A')}" if best_cost_option else "N/A",
            "BestCostLT": f"{best_cost_option.get('lead_time', np.inf):.0f}" if best_cost_option and best_cost_option.get('lead_time', np.inf) != np.inf else ("0" if best_cost_option and best_cost_option.get('stock', 0) >= total_qty_needed else "N/A"),
            "BestCostSrc": best_cost_option.get('source', "N/A") if best_cost_option else "N/A",
            "Alternates": "Yes" if part_summary.get('alternates') else "No",
            "AlternatesList": part_summary.get('alternates', []),  # Ensure AlternatesList is set
            "Notes": notes_str,
            "RiskFactors": risk_factors
        }
        logger.debug(f"gui_entry for valid part: {gui_entry}")
    
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
        start_time = time.time()
        try:
            if self.bom_df is None or self.bom_df.empty:
                self.update_status_threadsafe("Error: BOM became invalid before analysis start.", "error")
                return
    
            if API_KEYS["DigiKey"]:
                self.update_status_threadsafe("Checking DigiKey token...", "info")
                access_token = self.get_digikey_token()
                if not access_token:
                    logger.error("Analysis cancelled: Failed to obtain DigiKey token.")
                    self.update_status_threadsafe("Analysis Cancelled: DigiKey Auth Failed.", "error")
                    self.root.after(0, messagebox.showerror, "Auth Error", "Could not get/refresh DigiKey token. Analysis cancelled.")
                    return
    
            self._hts_cache = {}
            all_analysis_entries = []
            all_historical_entries = []
            all_part_summaries = []
            total_parts_in_bom = len(self.bom_df)
            self.update_progress_threadsafe(0, total_parts_in_bom, "Initializing...")
    
            for i, row in self.bom_df.iterrows():
                if not self.running_analysis: logger.info("Analysis run cancelled."); return
                bom_pn = row['Part Number']
                bom_mfg = row.get('Manufacturer', '')
                bom_qty_per = row['Quantity']
                self.update_progress_threadsafe(i, total_parts_in_bom, f"Analyzing {bom_pn[:25]}...")
                part_gui_rows, part_hist_rows, part_summary = self.analyze_single_part(
                    bom_pn, bom_mfg, bom_qty_per, config
                )
                all_analysis_entries.extend(part_gui_rows)
                all_historical_entries.extend(part_hist_rows)
                if part_summary.get('options'):
                    all_part_summaries.append(part_summary)
    
            if not self.running_analysis: return
    
            self.update_progress_threadsafe(total_parts_in_bom, total_parts_in_bom, "Aggregating results...")
            self.analysis_results["config"] = config
            self.analysis_results["part_summaries"] = all_part_summaries
            self.analysis_results["gui_entries"] = all_analysis_entries
            self.root.after(0, self.populate_treeview, self.tree, all_analysis_entries)
    
            if all_historical_entries:
                logger.info(f"Submitting append of {len(all_historical_entries)} rows to {HISTORICAL_DATA_FILE.name}")
                self.thread_pool.submit(append_to_csv, HISTORICAL_DATA_FILE, all_historical_entries)
    
            summary_metrics = self.calculate_summary_metrics(all_part_summaries, config)
            self.analysis_results["summary_metrics"] = summary_metrics
            logger.debug(f"Data being sent to populate_treeview for analysis_table:\nType: {type(summary_metrics)}\nContent: {summary_metrics}")
            self.root.after(0, self.populate_treeview, self.analysis_table, summary_metrics)
    
            self.analysis_results["strategies"] = self.strategies_for_export
            self.root.after(100, self.update_export_buttons_state)
    
            elapsed_time = time.time() - start_time
            self.update_status_threadsafe(f"Analysis complete ({len(all_part_summaries)} valid parts in {elapsed_time:.1f}s).", "success")
            self.update_progress_threadsafe(total_parts_in_bom, total_parts_in_bom, "Done")
    
        except Exception as e:
            logger.error(f"Analysis failed: {str(e)}", exc_info=True)
            self.update_status_threadsafe(f"Analysis Failed: {str(e)}", "error")
            self.root.after(0, messagebox.showerror, "Analysis Error", f"An error occurred during analysis:\n{str(e)}")
        finally:
            self.running_analysis = False
            self.root.after(50, self.update_analysis_controls_state, False)
            self.root.after(100, self.setup_plot_options)


    def calculate_summary_metrics(self, part_summaries, config):
        logger.info(f"Calculating summary metrics for {len(part_summaries)} parts...")
        # --- Initialize Aggregates & Flags ---
        total_bom_cost_strict_min = 0.0; total_bom_cost_min = 0.0; total_bom_cost_max = 0.0
        total_bom_cost_fastest = 0.0; total_bom_cost_optimized = 0.0
        total_bom_cost_in_stock = 0.0; total_bom_cost_with_lt = 0.0
        max_lead_time_strict_min = 0; max_lead_time_min = 0; max_lead_time_fastest = 0
        max_lead_time_optimized = 0; max_lead_time_in_stock = 0; max_lead_time_with_lt = 0
        strict_parts_count = 0; min_parts_count = 0; fastest_parts_count = 0
        in_stock_parts_count = 0; with_lt_parts_count = 0; optimized_parts_count = 0
        max_cost_parts_count = 0
        invalid_strict_min = False; invalid_min = False; invalid_max = False
        invalid_fastest = False; invalid_optimized = False; invalid_in_stock = False
        invalid_with_lt = False
        clear_to_build_stock_only = True
        time_to_acquire_all_parts = 0
        stock_gap_parts = []; parts_with_stock_avail = 0
        total_parts_analyzed = len(part_summaries)
        strict_lowest_cost_strategy = {}; lowest_cost_strategy = {}; fastest_strategy = {}
        optimized_strategy = {}; in_stock_strategy = {}; with_lt_strategy = {}
        near_miss_info = {}
        invalid_parts_list = []
        ignored_parts_list = []
        unknown_parts_list = []
        valid_part_names = []
    
        def _calculate_strict_cost(qty_needed, pricing, moq):
            try: qty_needed_int = int(qty_needed)
            except (ValueError, TypeError): return np.inf, qty_needed
            if not pricing: return np.inf, max(qty_needed_int, int(safe_float(moq, default=1)))
    
            moq_int = max(1, int(safe_float(moq, default=1)))
            strict_order_qty = max(qty_needed_int, moq_int)
            unit_price = np.inf
            applicable_break = None
    
            sorted_pricing = sorted([pb for pb in pricing if isinstance(pb, dict)], key=lambda p: int(safe_float(p.get('qty'), default=0)))
            for pb in sorted_pricing:
                pb_qty = int(safe_float(pb.get('qty'), default=0))
                price = safe_float(pb.get('price'), default=np.inf)
                if pb_qty <= 0 or pd.isna(price) or price == np.inf: continue
                if strict_order_qty >= pb_qty:
                    applicable_break = pb
                else:
                    break
    
            if applicable_break:
                unit_price = safe_float(applicable_break.get('price'), default=np.inf)
            elif sorted_pricing:
                first_break = sorted_pricing[0]
                first_break_qty = int(safe_float(first_break.get('qty'), default=1))
                first_break_price = safe_float(first_break.get('price'), default=np.inf)
                if first_break_qty > 0 and pd.notna(first_break_price) and first_break_price != np.inf:
                    strict_order_qty = max(strict_order_qty, first_break_qty)
                    unit_price = first_break_price
    
            final_cost = float(unit_price) * strict_order_qty if pd.notna(unit_price) and unit_price != np.inf else np.inf
            return final_cost, strict_order_qty

        parts_with_sufficient_single_source_stock = 0 
        total_active_parts_considered = 0 
    
        # --- Iterate Through Each Part Summary ---
        for i, summary in enumerate(part_summaries):
            bom_pn = summary.get('bom_pn', f'UnknownPart_{i}')
            logger.debug(f"--- Start Summary Calcs for Part {i+1}: {bom_pn} ---")
            try:
                qty_needed = int(summary.get('total_qty_needed', 0))
                if qty_needed <= 0: raise ValueError("Quantity needed must be positive")
            except (ValueError, TypeError):
                logger.warning(f"Invalid or zero quantity needed for {bom_pn}. Skipping part.")
                invalid_parts_list.append(f"{bom_pn} (Invalid Qty)")
                invalid_strict_min = invalid_min = invalid_max = invalid_fastest = invalid_optimized = invalid_in_stock = invalid_with_lt = True
                clear_to_build_stock_only = False
                stock_gap_parts.append(f"{bom_pn}: Invalid Qty")
                time_to_acquire_all_parts = np.inf
                placeholder = self.create_strategy_entry({'notes': 'Invalid Quantity Needed'})
                for strategy_dict in [strict_lowest_cost_strategy, lowest_cost_strategy, fastest_strategy,
                                      optimized_strategy, in_stock_strategy, with_lt_strategy]:
                    strategy_dict[bom_pn] = placeholder
                continue
    
            options = summary.get('options', [])
            if not options:
                logger.info(f"No options for {bom_pn}. Skipping part.")
                invalid_parts_list.append(f"{bom_pn} (No Data)")
                invalid_strict_min = invalid_min = invalid_max = invalid_fastest = invalid_optimized = invalid_in_stock = invalid_with_lt = True
                clear_to_build_stock_only = False
                stock_gap_parts.append(f"{bom_pn}: No Valid Data")
                time_to_acquire_all_parts = np.inf
                placeholder = self.create_strategy_entry({'notes': 'No Valid Data Found', 'bom_pn': bom_pn, 'total_qty_needed': qty_needed})
                for strategy_dict in [strict_lowest_cost_strategy, lowest_cost_strategy, fastest_strategy,
                                      optimized_strategy, in_stock_strategy, with_lt_strategy]:
                    strategy_dict[bom_pn] = placeholder
                continue
    
            valid_part_names.append(bom_pn)
            for opt in options:
                if isinstance(opt, dict):
                    lt = opt.get('lead_time')
                    if lt is None or pd.isna(lt):
                        opt['lead_time'] = np.inf
                    else:
                        try:
                            opt['lead_time'] = float(lt)
                            if opt['lead_time'] < 0: opt['lead_time'] = np.inf
                        except (ValueError, TypeError):
                            opt['lead_time'] = np.inf
    
            valid_options_for_calc = [
                opt for opt in options
                if isinstance(opt, dict) and
                   ((safe_float(opt.get('cost'), default=np.inf) != np.inf) or
                    (opt.get('lead_time', np.inf) != np.inf))
            ]
    
            part_invalid_strict = False; part_invalid_min = False; part_invalid_max = False
            part_invalid_fastest = False; part_invalid_optimized = False
            part_invalid_in_stock = False; part_invalid_with_lt = False
    
            if not valid_options_for_calc:
                logger.warning(f"No valid options with finite cost OR finite lead time for {bom_pn}. Marking as unavailable for ALL calculations.")
                invalid_parts_list.append(bom_pn)
                clear_to_build_stock_only = False
                stock_gap_parts.append(f"{bom_pn}: No Valid Data")
                time_to_acquire_all_parts = np.inf
                placeholder = self.create_strategy_entry({'notes': 'No Valid Data Found', 'bom_pn': bom_pn, 'total_qty_needed': qty_needed})
                for strategy_dict in [strict_lowest_cost_strategy, lowest_cost_strategy, fastest_strategy,
                                      optimized_strategy, in_stock_strategy, with_lt_strategy]:
                    strategy_dict[bom_pn] = placeholder
                continue
    
            best_cost_option_strict_ref = None; part_min_cost_strict = np.inf
            best_cost_option_optimized_ref = None; part_min_cost_optimized = np.inf
            fastest_option_ref = None; part_fastest_cost = np.inf; part_min_lead = np.inf
            best_in_stock_option_ref = None; part_min_cost_in_stock = np.inf
            best_with_lt_option_ref = None; part_min_cost_with_lt = np.inf
    
            # --- Strict Cost ---
            current_min_strict_cost = np.inf
            current_best_lead_for_strict = np.inf
            calculated_strict_values = {}
            for option in valid_options_for_calc:
                source = option.get('source', f'UnknownSource_{valid_options_for_calc.index(option)}')
                pricing_data = option.get('Pricing', [])
                moq_data = option.get('MinOrderQty', 0)
    
                strict_cost_val, strict_qty_val = _calculate_strict_cost(qty_needed, pricing_data, moq_data)
                calculated_strict_values[source] = {'cost': strict_cost_val, 'qty': strict_qty_val}
    
                option_lead = option.get('lead_time', np.inf)
                if strict_cost_val < current_min_strict_cost:
                    current_min_strict_cost = strict_cost_val
                    current_best_lead_for_strict = option_lead
                    best_cost_option_strict_ref = option
                elif strict_cost_val == current_min_strict_cost and option_lead < current_best_lead_for_strict:
                    current_best_lead_for_strict = option_lead
                    best_cost_option_strict_ref = option
    
            part_min_cost_strict = current_min_strict_cost
    
            # --- Lowest Optimized Cost (Baseline) ---
            try:
                best_cost_option_optimized_ref = min(
                    valid_options_for_calc,
                    key=lambda x: (
                        safe_float(x.get('cost'), default=np.inf),
                        x.get('lead_time', np.inf)
                    )
                )
                part_min_cost_optimized = safe_float(best_cost_option_optimized_ref.get('cost'), default=np.inf)
            except ValueError:
                best_cost_option_optimized_ref = None
                part_min_cost_optimized = np.inf
    
            # --- Fastest Option ---
            fastest_option_ref = None
            part_min_lead = np.inf
            options_in_stock_f = [
                opt for opt in valid_options_for_calc
                if opt.get('stock', 0) >= qty_needed
            ]
    
            if options_in_stock_f:
                try:
                    fastest_option_ref = min(
                        options_in_stock_f,
                        key=lambda x: (
                            safe_float(x.get('cost'), default=np.inf),
                            x.get('source', '')
                        )
                    )
                    part_min_lead = 0
                    part_fastest_cost = safe_float(fastest_option_ref.get('cost'), default=np.inf)
                except ValueError:
                    part_fastest_cost = np.inf
            else:
                options_with_finite_lt = [
                    opt for opt in valid_options_for_calc
                    if opt.get('lead_time', np.inf) != np.inf
                ]
                if options_with_finite_lt:
                    try:
                        fastest_option_ref = min(
                            options_with_finite_lt,
                            key=lambda x: (
                                x.get('lead_time'),
                                safe_float(x.get('cost'), default=np.inf),
                                x.get('source', '')
                            )
                        )
                        part_min_lead = fastest_option_ref.get('lead_time')
                        part_fastest_cost = safe_float(fastest_option_ref.get('cost'), default=np.inf)
                    except ValueError:
                        part_fastest_cost = np.inf
                else:
                    part_fastest_cost = np.inf
    
            # --- Lowest Cost In Stock ---
            best_in_stock_option_ref = None
            part_min_cost_in_stock = np.inf
            if options_in_stock_f:
                try:
                    best_in_stock_option_ref = min(
                        options_in_stock_f,
                        key=lambda x: (
                            safe_float(x.get('cost'), default=np.inf),
                            x.get('source', '')
                        )
                    )
                    part_min_cost_in_stock = safe_float(best_in_stock_option_ref.get('cost'), default=np.inf)
                except ValueError:
                    pass
    
            # --- Lowest Cost with Lead Time ---
            best_with_lt_option_ref = None
            part_min_cost_with_lt = np.inf
            options_for_with_lt = [
                opt for opt in valid_options_for_calc
                if opt.get('stock', 0) >= qty_needed or opt.get('lead_time', np.inf) != np.inf
            ]
            if options_for_with_lt:
                try:
                    best_with_lt_option_ref = min(
                        options_for_with_lt,
                        key=lambda x: (
                            safe_float(x.get('cost'), default=np.inf),
                            x.get('lead_time', np.inf),
                            x.get('source', '')
                        )
                    )
                    part_min_cost_with_lt = safe_float(best_with_lt_option_ref.get('cost'), default=np.inf)
                except ValueError:
                    pass
    
            # --- Set Part Invalidity Flags ---
            part_invalid_strict = (best_cost_option_strict_ref is None or part_min_cost_strict == np.inf)
            part_invalid_min = (best_cost_option_optimized_ref is None or part_min_cost_optimized == np.inf)
            part_invalid_fastest = (fastest_option_ref is None or part_fastest_cost == np.inf)
            part_invalid_in_stock = (best_in_stock_option_ref is None or part_min_cost_in_stock == np.inf)
            part_invalid_with_lt = (best_with_lt_option_ref is None or part_min_cost_with_lt == np.inf)
    
            # --- Store Strategy Details ---
            if not part_invalid_strict:
                strict_source = best_cost_option_strict_ref.get('source', 'UnknownSource_Strict')
                strict_vals = calculated_strict_values.get(strict_source)
                if strict_vals and strict_vals['cost'] != np.inf:
                    strict_entry_dict = self.create_strategy_entry(best_cost_option_strict_ref)
                    strict_entry_dict.update({
                        'cost': strict_vals['cost'],
                        'actual_order_qty': strict_vals['qty'],
                        'unit_cost': (strict_vals['cost'] / strict_vals['qty']) if strict_vals['qty'] > 0 else np.nan,
                        'notes': f"Strict Cost Calc; {strict_entry_dict.get('notes', '')}".strip('; '),
                    })
                    strict_lowest_cost_strategy[bom_pn] = strict_entry_dict
                else:
                    part_invalid_strict = True
            if part_invalid_strict:
                strict_lowest_cost_strategy[bom_pn] = self.create_strategy_entry({'notes': 'No Valid (Strict) Cost Option Found'})
    
            lowest_cost_strategy[bom_pn] = self.create_strategy_entry(best_cost_option_optimized_ref) if not part_invalid_min else self.create_strategy_entry({'notes': 'No Valid Optimized Cost Option Found'})
            fastest_strategy[bom_pn] = self.create_strategy_entry(fastest_option_ref) if not part_invalid_fastest else self.create_strategy_entry({'notes': 'No Valid Finite Lead Time Option Found'})
            in_stock_strategy[bom_pn] = self.create_strategy_entry(best_in_stock_option_ref) if not part_invalid_in_stock else self.create_strategy_entry({'notes': 'No In Stock Option Found'})
            with_lt_strategy[bom_pn] = self.create_strategy_entry(best_with_lt_option_ref) if not part_invalid_with_lt else self.create_strategy_entry({'notes': 'No Option Found w/ Stock or Finite LT'})
    
            # --- Aggregate Totals ---
            def get_actual_lead(strategy_dict, bom_pn_key, qty_needed_val):
                if bom_pn_key not in strategy_dict: return np.inf
                chosen_option = strategy_dict[bom_pn_key]
                if not isinstance(chosen_option, dict): return np.inf
                chosen_stock = safe_float(chosen_option.get('stock'), default=0)
                chosen_qty = safe_float(chosen_option.get('actual_order_qty'), default=qty_needed_val)
                if chosen_qty <= 0: chosen_qty = qty_needed_val
                chosen_listed_lead = safe_float(chosen_option.get('lead_time'), default=np.inf)
                return 0 if chosen_stock >= chosen_qty else chosen_listed_lead
    
            # Strict
            if not part_invalid_strict and not invalid_strict_min:
                cost_to_add = safe_float(strict_lowest_cost_strategy[bom_pn].get('cost'), default=np.inf)
                if cost_to_add != np.inf:
                    total_bom_cost_strict_min += cost_to_add
                    actual_lead_strict = get_actual_lead(strict_lowest_cost_strategy, bom_pn, qty_needed)
                    if actual_lead_strict != np.inf:
                        max_lead_time_strict_min = max(max_lead_time_strict_min, actual_lead_strict)
                    strict_parts_count += 1
                else:
                    logger.warning(f"Part {bom_pn} added infinite cost to Strict strategy. Invalidating.")
                    invalid_strict_min = True
            logger.debug(f"Part {bom_pn} - Post-Strict Agg: invalid_strict_min = {invalid_strict_min}, Total Cost = {total_bom_cost_strict_min}, Max LT = {max_lead_time_strict_min}")
    
            # Optimized Baseline Cost
            if not part_invalid_min and not invalid_min:
                cost_to_add = safe_float(lowest_cost_strategy[bom_pn].get('cost'), default=np.inf)
                if cost_to_add != np.inf:
                    total_bom_cost_min += cost_to_add
                    actual_lead_min = get_actual_lead(lowest_cost_strategy, bom_pn, qty_needed)
                    if actual_lead_min != np.inf:
                        max_lead_time_min = max(max_lead_time_min, actual_lead_min)
                    min_parts_count += 1
                else:
                    logger.warning(f"Part {bom_pn} added infinite cost to Min Cost strategy. Invalidating.")
                    invalid_min = True
            logger.debug(f"Part {bom_pn} - Post-MinBase Agg: invalid_min = {invalid_min}, Total Cost = {total_bom_cost_min}, Max LT = {max_lead_time_min}")
    
            # Fastest
            if not part_invalid_fastest and not invalid_fastest:
                cost_to_add = part_fastest_cost
                if cost_to_add != np.inf:
                    total_bom_cost_fastest += cost_to_add
                    actual_lead_fastest = part_min_lead
                    if actual_lead_fastest != np.inf:
                        max_lead_time_fastest = max(max_lead_time_fastest, actual_lead_fastest)
                    fastest_parts_count += 1
                else:
                    logger.warning(f"Part {bom_pn} added infinite cost to Fastest strategy. Invalidating.")
                    invalid_fastest = True
            logger.debug(f"Part {bom_pn} - Post-Fastest Agg: invalid_fastest = {invalid_fastest}, Total Cost = {total_bom_cost_fastest}, Max LT = {max_lead_time_fastest}")
    
            # In Stock
            if not part_invalid_in_stock and not invalid_in_stock:
                cost_to_add = part_min_cost_in_stock
                if cost_to_add != np.inf:
                    total_bom_cost_in_stock += cost_to_add
                    in_stock_parts_count += 1
                else:
                    logger.warning(f"Part {bom_pn} added infinite cost to In Stock strategy. Invalidating.")
                    invalid_in_stock = True
            logger.debug(f"Part {bom_pn} - Post-InStock Agg: invalid_in_stock = {invalid_in_stock}, Total Cost = {total_bom_cost_in_stock}")
    
            # With LT
            if not part_invalid_with_lt and not invalid_with_lt:
                cost_to_add = part_min_cost_with_lt
                if cost_to_add != np.inf:
                    total_bom_cost_with_lt += cost_to_add
                    actual_lead_with_lt = get_actual_lead(with_lt_strategy, bom_pn, qty_needed)
                    if actual_lead_with_lt != np.inf:
                        max_lead_time_with_lt = max(max_lead_time_with_lt, actual_lead_with_lt)
                    with_lt_parts_count += 1
                else:
                    logger.warning(f"Part {bom_pn} added infinite cost to With LT strategy. Invalidating.")
                    invalid_with_lt = True
            logger.debug(f"Part {bom_pn} - Post-WithLT Agg: invalid_with_lt = {invalid_with_lt}, Total Cost = {total_bom_cost_with_lt}, Max LT = {max_lead_time_with_lt}")
    
            # Max Cost
            part_max_cost = 0.0
            valid_costs_part = [safe_float(opt.get('cost')) for opt in valid_options_for_calc if pd.notna(safe_float(opt.get('cost'), default=np.nan)) and safe_float(opt.get('cost'), default=np.inf) != np.inf]
            if valid_costs_part:
                part_max_cost = max(valid_costs_part)
                if not invalid_max:
                    total_bom_cost_max += part_max_cost
                max_cost_parts_count += 1
            else:
                logger.warning(f"Part {bom_pn} has no valid cost options. Invalidating overall Max Cost calculation.")
                invalid_max = True
            logger.debug(f"Part {bom_pn} - Post-Max Agg: invalid_max = {invalid_max}, Total Cost = {total_bom_cost_max}")
    
            # --- Optimized Strategy Calculation ---
            target_lt_days = config.get('target_lead_time_days', np.inf)
            max_prem_pct = config.get('max_premium', np.inf)
            cost_weight = config.get('cost_weight', 0.5)
            lead_weight = config.get('lead_time_weight', 0.5)
    
            chosen_option_opt_ref = None
            opt_notes = ""
            best_score = np.inf
            part_invalid_optimized = False
    
            if part_invalid_min:
                logger.warning(f"Optimized Strategy: Skipping {bom_pn} as base lowest cost is invalid for this part.")
                opt_notes = "N/A (Invalid Base Cost)"
                part_invalid_optimized = True
            else:
                baseline_cost = part_min_cost_optimized
                constrained_options = []
    
                for opt_idx, option in enumerate(valid_options_for_calc):
                    cost = safe_float(option.get('cost'), default=np.inf)
                    lead_time = option.get('lead_time', np.inf)
                    stock = option.get('stock', 0)
                    effective_lead_time = 0 if stock >= qty_needed else lead_time
    
                    if cost == np.inf: continue
    
                    if effective_lead_time == np.inf or effective_lead_time > target_lt_days:
                        continue
    
                    cost_premium_pct_calc = 0.0
                    if baseline_cost > 1e-9:
                        cost_premium_pct_calc = ((cost - baseline_cost) / baseline_cost * 100.0)
                    else:
                        if cost > 1e-9: cost_premium_pct_calc = np.inf
    
                    if cost_premium_pct_calc > max_prem_pct:
                        continue
    
                    constrained_options.append(option)
    
                if constrained_options:
                    constrained_costs = [safe_float(opt.get('cost')) for opt in constrained_options]
                    constrained_effective_lts = [0 if opt.get('stock', 0) >= qty_needed else opt.get('lead_time', np.inf) for opt in constrained_options if (0 if opt.get('stock', 0) >= qty_needed else opt.get('lead_time', np.inf)) != np.inf]
    
                    min_viable_cost = min(constrained_costs) if constrained_costs else np.inf
                    max_viable_cost = max(constrained_costs) if constrained_costs else np.inf
                    min_viable_lt = min(constrained_effective_lts) if constrained_effective_lts else 0
                    max_viable_lt = max(constrained_effective_lts) if constrained_effective_lts else 0
    
                    cost_range = (max_viable_cost - min_viable_cost) if max_viable_cost > min_viable_cost else 1.0
                    lead_range = (max_viable_lt - min_viable_lt) if max_viable_lt > min_viable_lt else 1.0
                    if cost_range < 1e-9: cost_range = 1.0
                    if lead_range < 1e-9: lead_range = 1.0
    
                    for viable_opt in constrained_options:
                        cost = safe_float(viable_opt.get('cost'))
                        stock = viable_opt.get('stock', 0)
                        lead_time = viable_opt.get('lead_time', np.inf)
                        effective_lead_time = 0 if stock >= qty_needed else lead_time
    
                        norm_cost = (cost - min_viable_cost) / cost_range if cost_range > 1e-9 else 0
                        norm_lead = (effective_lead_time - min_viable_lt) / lead_range if lead_range > 1e-9 else 0
                        score = (cost_weight * norm_cost) + (lead_weight * norm_lead)
    
                        if viable_opt.get('discontinued') or viable_opt.get('eol'): score += 0.5
                        if stock < qty_needed: score += 0.1
    
                        viable_opt['_temp_score'] = score
                        if score < best_score:
                            best_score = score
                            chosen_option_opt_ref = viable_opt
    
                    if chosen_option_opt_ref:
                        opt_notes = f"Score: {best_score:.3f}"
                        if '_temp_score' in chosen_option_opt_ref: del chosen_option_opt_ref['_temp_score']
                        for opt in constrained_options:
                            if '_temp_score' in opt: del opt['_temp_score']
                    else:
                        logger.error(f"Optimized Strategy: LOGIC ERROR - No option selected for {bom_pn} from {len(constrained_options)} constrained options.")
                        part_invalid_optimized = True
                        opt_notes = "Error during scoring"
    
                if not chosen_option_opt_ref:
                    logger.warning(f"Optimized Strategy: No option met constraints or scoring failed for {bom_pn}.")
                    strict_fallback_entry = strict_lowest_cost_strategy.get(bom_pn)
                    strict_fallback_valid = (strict_fallback_entry and safe_float(strict_fallback_entry.get('cost'), default=np.inf) != np.inf)
    
                    fastest_fallback_entry = fastest_strategy.get(bom_pn)
                    fastest_fallback_valid = (fastest_fallback_entry and safe_float(fastest_fallback_entry.get('cost'), default=np.inf) != np.inf and safe_float(fastest_fallback_entry.get('lead_time'), default=np.inf) != np.inf)
    
                    chosen_fallback = None
                    fallback_note = ""
    
                    if strict_fallback_valid:
                        strict_fallback_lt = get_actual_lead(strict_lowest_cost_strategy, bom_pn, qty_needed)
                        is_strict_too_slow = (strict_fallback_lt > target_lt_days * 1.5) and (target_lt_days > 0)
    
                        if is_strict_too_slow and fastest_fallback_valid:
                            chosen_fallback = fastest_fallback_entry
                            fallback_note = "Constraints Failed. Fallback to Fastest (Strict LT too long)."
                        else:
                            chosen_fallback = strict_fallback_entry
                            fallback_note = "Constraints Failed. Fallback to Strict."
                    elif fastest_fallback_valid:
                        chosen_fallback = fastest_fallback_entry
                        fallback_note = "Constraints Failed & Strict Invalid. Fallback to Fastest."
                    else:
                        fallback_note = "Constraints Failed. No valid fallback found."
                        part_invalid_optimized = True
    
                    if chosen_fallback:
                        optimized_strategy[bom_pn] = chosen_fallback
                        optimized_strategy[bom_pn]['notes'] = f"{fallback_note}; {optimized_strategy[bom_pn].get('notes', '')}".strip('; ')
                        optimized_strategy[bom_pn]['optimized_strategy_score'] = 'N/A (Fallback)'
                        cost_to_add_fallback = safe_float(chosen_fallback.get('cost'), default=np.inf)
                        if cost_to_add_fallback == np.inf: part_invalid_optimized = True
                    else:
                        optimized_strategy[bom_pn] = self.create_strategy_entry({'notes': fallback_note})
                        part_invalid_optimized = True
    
                    best_score = np.nan
    
            if chosen_option_opt_ref and not part_invalid_optimized:
                opt_entry = self.create_strategy_entry(chosen_option_opt_ref)
                opt_entry['optimized_strategy_score'] = f"{best_score:.3f}" if pd.notna(best_score) else "N/A"
                opt_entry['notes'] = f"{opt_notes}; {opt_entry.get('notes', '')}".strip('; ')
                optimized_strategy[bom_pn] = opt_entry
            elif not part_invalid_optimized and bom_pn not in optimized_strategy:
                logger.error(f"Optimized Strategy: Reached end for {bom_pn} without selection or fallback decision.")
                optimized_strategy[bom_pn] = self.create_strategy_entry({'notes': 'N/A (Processing Error)'})
                part_invalid_optimized = True
    
            if not invalid_optimized and not part_invalid_optimized:
                chosen_opt_entry = optimized_strategy.get(bom_pn)
                if isinstance(chosen_opt_entry, dict):
                    cost_to_add_opt = safe_float(chosen_opt_entry.get('cost'), default=np.inf)
                    if cost_to_add_opt != np.inf:
                        total_bom_cost_optimized += cost_to_add_opt
                        actual_lead_opt = get_actual_lead(optimized_strategy, bom_pn, qty_needed)
                        if actual_lead_opt != np.inf:
                            max_lead_time_optimized = max(max_lead_time_optimized, actual_lead_opt)
                        optimized_parts_count += 1
                    else:
                        logger.warning(f"Optimized strategy for {bom_pn} resulted in INF cost. Invalidating.")
                        invalid_optimized = True
                else:
                    logger.warning(f"Missing or invalid optimized strategy entry for {bom_pn}. Invalidating.")
                    invalid_optimized = True
            logger.debug(f"Part {bom_pn} - Post-Optimized Agg: invalid_optimized = {invalid_optimized}, Total Cost = {total_bom_cost_optimized}, Max LT = {max_lead_time_optimized}")
    
            # --- Near Miss Calculation ---
            part_near_misses = {}
            if not part_invalid_min and not part_invalid_optimized:
                baseline_cost = part_min_cost_optimized
                options_that_failed_constraints = []
                for opt in valid_options_for_calc:
                    is_chosen_opt = (chosen_option_opt_ref is not None and opt == chosen_option_opt_ref)
                    was_constrained = opt in constrained_options
                    if not is_chosen_opt and not was_constrained:
                        cost = safe_float(opt.get('cost'), default=np.inf)
                        lead_time = opt.get('lead_time', np.inf)
                        stock = opt.get('stock', 0)
                        effective_lead_time = 0 if stock >= qty_needed else lead_time
                        if cost == np.inf or effective_lead_time == np.inf: continue
    
                        failed_lt = effective_lead_time > target_lt_days
                        cost_premium_pct_calc = 0.0
                        if baseline_cost > 1e-9: cost_premium_pct_calc = ((cost - baseline_cost) / baseline_cost * 100.0)
                        elif cost > 1e-9: cost_premium_pct_calc = np.inf
                        failed_prem = cost_premium_pct_calc > max_prem_pct
    
                        if failed_lt or failed_prem:
                            options_that_failed_constraints.append({
                                'option_ref': opt,
                                'cost': cost,
                                'effective_lead_time': effective_lead_time,
                                'premium_pct': cost_premium_pct_calc,
                                'failed_lt': failed_lt,
                                'failed_prem': failed_prem
                            })
    
                if options_that_failed_constraints:
                    over_lt_candidates = [
                        cand for cand in options_that_failed_constraints
                        if cand['failed_lt'] and not cand['failed_prem']
                    ]
                    if over_lt_candidates:
                        over_lt_candidates.sort(key=lambda x: (
                            x['effective_lead_time'] - target_lt_days,
                            x['cost']
                        ))
                        best_over_lt_cand = over_lt_candidates[0]
                        over_by_days = best_over_lt_cand['effective_lead_time'] - target_lt_days
                        if pd.notna(over_by_days) and over_by_days > 0 and over_by_days <= 14:
                            part_near_misses['slightly_over_lt'] = {
                                'option': self.create_strategy_entry(best_over_lt_cand['option_ref']),
                                'over_by_days': round(over_by_days, 1)
                            }
    
                    over_cost_candidates = [
                        cand for cand in options_that_failed_constraints
                        if cand['failed_prem'] and not cand['failed_lt']
                    ]
                    if over_cost_candidates:
                        over_cost_candidates.sort(key=lambda x: (
                            x['premium_pct'] - max_prem_pct,
                            x['effective_lead_time']
                        ))
                        best_over_cost_cand = over_cost_candidates[0]
                        over_by_pct = best_over_cost_cand['premium_pct'] - max_prem_pct
                        if pd.notna(over_by_pct) and over_by_pct > 0 and over_by_pct <= 5.0:
                            part_near_misses['slightly_over_cost'] = {
                                'option': self.create_strategy_entry(best_over_cost_cand['option_ref']),
                                'over_by_pct': round(over_by_pct, 2)
                            }
    
            if part_near_misses:
                near_miss_info[bom_pn] = part_near_misses
    
            # --- Check Stock Availability & Acquisition Time ---
            part_acquire_time = 0
            total_stock_for_part = sum(safe_float(opt.get('stock'), default=0) for opt in valid_options_for_calc if safe_float(opt.get('stock'), default=0) > 0)
            part_can_be_sourced_from_stock = (total_stock_for_part >= qty_needed)
    
            if part_can_be_sourced_from_stock:
                parts_with_stock_avail += 1
                part_acquire_time = 0
            else:
                clear_to_build_stock_only = False
                min_finite_lead_overall = np.inf
                for opt in valid_options_for_calc:
                    lt = opt.get('lead_time', np.inf)
                    if lt != np.inf:
                        min_finite_lead_overall = min(min_finite_lead_overall, lt)
    
                stock_gap_note = f"{bom_pn} (Need:{qty_needed}, Have:{total_stock_for_part:.0f}"
                if min_finite_lead_overall != np.inf:
                    stock_gap_note += f", Min LT:{min_finite_lead_overall:.0f}d)"
                    part_acquire_time = min_finite_lead_overall
                else:
                    stock_gap_note += ", No Finite LT Found)"
                    part_acquire_time = np.inf
    
                stock_gap_parts.append(stock_gap_note)
    
            if part_acquire_time != np.inf:
                time_to_acquire_all_parts = max(time_to_acquire_all_parts, float(part_acquire_time))

            total_active_parts_considered += 1
            part_has_sufficient_stock_option = False
            for opt in valid_options_for_calc: # Use the list already filtered for valid options
                 if opt.get('stock', 0) >= qty_needed:
                     part_has_sufficient_stock_option = True
                     break # Found at least one, no need to check others for this part

            if part_has_sufficient_stock_option:
                parts_with_sufficient_single_source_stock += 1 # Increment the new counter
            else:
                clear_to_build_stock_only = False # Set flag to False if this part fails
    
        # --- End of Part Loop ---

        # --- Correctly Count Parts Based on GUI Entries ---
        gui_entries = self.analysis_results.get("gui_entries", [])
        total_parts_in_bom = len(gui_entries)

        # Categorize parts based on their status and notes
        invalid_qty_parts = [entry.get("PartNumber") for entry in gui_entries if entry.get("Notes") == "Invalid quantity"]
        eol_discontinued_parts = [entry.get("PartNumber") for entry in gui_entries if entry.get("Status") in ["EOL", "Discontinued"]]
        unknown_parts = [entry.get("PartNumber") for entry in gui_entries if entry.get("Status") == "Unknown"]
        if 'active_parts' not in locals():
            active_parts = [entry.get("PartNumber") for entry in gui_entries if entry.get("Status") == "Active"]
        final_denominator_count = len(active_parts)

        # Handle edge case where no active parts exist
        if final_denominator_count == 0:
             clear_to_build_stock_only = False # Cannot build if no active parts
             parts_with_sufficient_single_source_stock = 0
    
        # --- Adjust Counts for Ignored Parts ---
        if self.analysis_results.get("gui_entries"):
            for entry in self.analysis_results["gui_entries"]:
                status = entry.get("Status")
                bom_pn = entry.get("PartNumber", "Unknown")
                if status in ["EOL", "Discontinued"]:
                    ignored_parts_list.append(f"{bom_pn} ({status})")
                elif status == "Unknown":
                    unknown_parts_list.append(f"{bom_pn} ({status})")
    
        valid_parts_processed = total_parts_analyzed - len(invalid_parts_list) - len(ignored_parts_list) - len(unknown_parts_list)
    
        if valid_parts_processed <= 0 and total_parts_analyzed > 0:
            logger.warning("All parts were invalid, ignored, or unknown. Setting all totals to N/A.")
            total_bom_cost_strict_min = total_bom_cost_min = total_bom_cost_max = total_bom_cost_fastest = np.nan
            total_bom_cost_optimized = total_bom_cost_in_stock = total_bom_cost_with_lt = np.nan
            max_lead_time_strict_min = max_lead_time_min = max_lead_time_fastest = np.inf
            max_lead_time_optimized = max_lead_time_with_lt = np.inf
            max_lead_time_in_stock = 0
            invalid_strict_min = invalid_min = invalid_max = invalid_fastest = invalid_optimized = invalid_in_stock = invalid_with_lt = True
    
        # --- Finalize Aggregates ---
        if invalid_strict_min or strict_parts_count == 0:
            total_bom_cost_strict_min = np.nan
            max_lead_time_strict_min = np.inf
        elif strict_parts_count > 0 and max_lead_time_strict_min == 0:
            logger.debug(f"Strict strategy: All {strict_parts_count} parts in stock, confirming max LT = 0")
            max_lead_time_strict_min = 0
    
        if invalid_min or min_parts_count == 0:
            total_bom_cost_min = np.nan
            max_lead_time_min = np.inf
        elif min_parts_count > 0 and max_lead_time_min == 0:
            logger.debug(f"MinBase strategy: All {min_parts_count} parts in stock, confirming max LT = 0")
            max_lead_time_min = 0
    
        if invalid_max or max_cost_parts_count == 0:
            total_bom_cost_max = np.nan
    
        if invalid_fastest or fastest_parts_count == 0:
            total_bom_cost_fastest = np.nan
            max_lead_time_fastest = np.inf
        elif fastest_parts_count > 0 and max_lead_time_fastest == 0:
            logger.debug(f"Fastest strategy: All {fastest_parts_count} parts in stock, confirming max LT = 0")
            max_lead_time_fastest = 0
    
        if invalid_optimized or optimized_parts_count == 0:
            total_bom_cost_optimized = np.nan
            max_lead_time_optimized = np.inf
        elif optimized_parts_count > 0 and max_lead_time_optimized == 0:
            logger.debug(f"Optimized strategy: All {optimized_parts_count} parts in stock, confirming max LT = 0")
            max_lead_time_optimized = 0
    
        if invalid_in_stock or in_stock_parts_count == 0:
            total_bom_cost_in_stock = np.nan
        max_lead_time_in_stock = 0
    
        if invalid_with_lt or with_lt_parts_count == 0:
            total_bom_cost_with_lt = np.nan
            max_lead_time_with_lt = np.inf
        elif with_lt_parts_count > 0 and max_lead_time_with_lt == 0:
            logger.debug(f"With LT strategy: All {with_lt_parts_count} parts in stock, confirming max LT = 0")
            max_lead_time_with_lt = 0
    
        # --- Store Strategies for Export ---
        self.strategies_for_export = {
            "Strict Lowest Cost": strict_lowest_cost_strategy,
            "Fastest": fastest_strategy,
            "Optimized Strategy": optimized_strategy,
            "Lowest Cost In Stock": in_stock_strategy,
            "Lowest Cost with Lead Time": with_lt_strategy,
            "Baseline Lowest Cost": lowest_cost_strategy,
        }
        self.analysis_results['near_miss_info'] = near_miss_info
    
        logger.debug(f"Final Strategy Status (Cost NaN?): Strict={pd.isna(total_bom_cost_strict_min)}, MinBase={pd.isna(total_bom_cost_min)}, Max={pd.isna(total_bom_cost_max)}, Fastest={pd.isna(total_bom_cost_fastest)}, Optimized={pd.isna(total_bom_cost_optimized)}, InStock={pd.isna(total_bom_cost_in_stock)}, WithLT={pd.isna(total_bom_cost_with_lt)}")
    
        # --- Format Summary Data for GUI Table ---
        def format_num(val, precision=2, suffix="", nan_inf_placeholder="N/A"):
            if pd.isna(val) or val == np.inf or val == -np.inf:
                return nan_inf_placeholder
            try:
                numeric_val = float(val)
                return f"{numeric_val:.{precision}f}{suffix}"
            except (ValueError, TypeError):
                return nan_inf_placeholder
    
        valid_parts_count = total_parts_analyzed - len(invalid_parts_list) - len(ignored_parts_list) - len(unknown_parts_list)
        parts_with_stock_avail = min(parts_with_stock_avail, valid_parts_count) if valid_parts_count >= 0 else 0
        total_parts_str = f"{total_parts_analyzed}"
        valid_parts_str = f"{valid_parts_count}" if valid_parts_count >= 0 else "0"
    
        summary_list = [
            ("Total Parts in BOM", str(total_parts_in_bom)),
            ("Parts Ignored (Invalid Qty/Data)", str(len(invalid_qty_parts))),
            ("Parts Ignored (EOL/Discontinued)", str(len(eol_discontinued_parts))),
            ("Parts Ignored (Unknown)", str(len(unknown_parts))),
            ("Parts Used in Calculations", f"{final_denominator_count}"),
            ("Immediate Kit Possible (Single Source Stock)", f"{clear_to_build_stock_only} ({parts_with_sufficient_single_source_stock} of {final_denominator_count} active parts meetable from stock)"),
            ("Est. Time to Full Kit (Days)", format_num(time_to_acquire_all_parts, 0)),
            ("Parts Requiring Lead Time Buys", "; ".join(stock_gap_parts) if stock_gap_parts else "None"),
            ("Potential Cost Range ($)", f"${format_num(total_bom_cost_min)} / ${format_num(total_bom_cost_max)}"),
            ("Strict Lowest Cost / LT ($ / Days)", f"${format_num(total_bom_cost_strict_min)} / {format_num(max_lead_time_strict_min, 0, 'd')}"),
            ("Lowest Cost In Stock / LT ($ / Days)", f"${format_num(total_bom_cost_in_stock)} / {format_num(max_lead_time_in_stock, 0, 'd')}"),
            ("Lowest Cost w/ LT / LT ($ / Days)", f"${format_num(total_bom_cost_with_lt)} / {format_num(max_lead_time_with_lt, 0, 'd')}"),
            ("Fastest Strategy / LT ($ / Days)", f"${format_num(total_bom_cost_fastest)} / {format_num(max_lead_time_fastest, 0, 'd')}"),
            ("Optimized Strategy / LT ($ / Days)", f"{'$'+format_num(total_bom_cost_optimized) if not invalid_optimized else 'N/A'} / {format_num(max_lead_time_optimized, 0, 'd')}{' (Fallback/Invalid)' if invalid_optimized else ''}"),
        ]
    
        # Store detailed lists for popup display
        self.summary_details = {}
        if invalid_parts_list:
            summary_list.append(("Ignored Part Details (Invalid)", f"{len(invalid_parts_list)} parts excluded"))
            self.summary_details["Ignored Part Details (Invalid)"] = invalid_parts_list
        if ignored_parts_list:
            summary_list.append(("Ignored Part Details (EOL/Discontinued)", f"{len(ignored_parts_list)} parts excluded"))
            self.summary_details["Ignored Part Details (EOL/Discontinued)"] = ignored_parts_list
        if unknown_parts_list:
            summary_list.append(("Ignored Part Details (Unknown)", f"{len(unknown_parts_list)} parts excluded"))
            self.summary_details["Ignored Part Details (Unknown)"] = unknown_parts_list
    
        # --- Tariff Calculation ---
        total_tariff_cost = 0.0
        calculated_bom_cost_for_tariff = 0.0
        total_tariff_pct = 0.0
        chosen_strategy_for_tariff_calc = {}
        tariff_basis_name = "N/A"
        strategy_valid_for_tariff = False
    
        if not invalid_optimized and optimized_parts_count > 0:
            chosen_strategy_for_tariff_calc = optimized_strategy
            tariff_basis_name = "Optimized"
            strategy_valid_for_tariff = True
        elif not invalid_strict_min and strict_parts_count > 0:
            chosen_strategy_for_tariff_calc = strict_lowest_cost_strategy
            tariff_basis_name = "Strict Lowest Cost"
            strategy_valid_for_tariff = True
    
        if strategy_valid_for_tariff:
            for bom_pn, chosen_option in chosen_strategy_for_tariff_calc.items():
                if isinstance(chosen_option, dict):
                    part_cost_basis = safe_float(chosen_option.get('cost'), default=np.inf)
                    if part_cost_basis != np.inf:
                        calculated_bom_cost_for_tariff += part_cost_basis
                        tariff_rate = safe_float(chosen_option.get('tariff_rate'), default=0.0)
                        if tariff_rate > 0:
                            total_tariff_cost += part_cost_basis * tariff_rate
    
            if calculated_bom_cost_for_tariff > 1e-9:
                total_tariff_pct = (total_tariff_cost / calculated_bom_cost_for_tariff * 100)
            else:
                total_tariff_pct = 0.0
    
            summary_list.append((f"Est. Total Tariff Cost ({tariff_basis_name})", f"${format_num(total_tariff_cost)}"))
            summary_list.append((f"Est. Total Tariff % ({tariff_basis_name})", f"{format_num(total_tariff_pct)}%"))
        else:
            summary_list.append(("Est. Total Tariff Cost (N/A)", "N/A"))
            summary_list.append(("Est. Total Tariff % (N/A)", "N/A"))

        # --- Log Ignored Parts Details ---
        if invalid_qty_parts:
            logger.info(f"Invalid Qty/Data Parts: {invalid_qty_parts}")
        if eol_discontinued_parts:
            logger.info(f"EOL/Discontinued Parts: {eol_discontinued_parts}")
        if unknown_parts:
            logger.info(f"Unknown Parts: {unknown_parts}")
        
        logger.info("Summary metrics calculation complete.")
        return summary_list

     
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
        """Combines Prophet and RAG Mock predictions using a weighted average."""
        ai_lead, ai_cost = prophet_lead, prophet_cost # Defaults to Prophet values
        ai_stock_prob = stock_prob # Pass through stock probability initially
        rag_mid_lead = np.nan
        rag_mid_cost = np.nan
        
        if isinstance(rag_lead_range, str) and rag_lead_range != "N/A" and '-' in rag_lead_range:
            try:
                parts = [safe_float(p) for p in rag_lead_range.split('-')]
                if len(parts) == 2 and not any(pd.isna(p) for p in parts):
                    rag_mid_lead = (parts[0] + parts[1]) / 2.0 # Calculate midpoint
                    logger.debug(f"Parsed RAG lead range '{rag_lead_range}' to midpoint: {rag_mid_lead}")
            except Exception as e:
                 logger.warning(f"Could not parse RAG lead range '{rag_lead_range}': {e}")

        # Safely parse RAG cost range string
        if isinstance(rag_cost_range, str) and rag_cost_range != "N/A" and '-' in rag_cost_range:
            try:
                parts = [safe_float(p) for p in rag_cost_range.split('-')]
                if len(parts) == 2 and not any(pd.isna(p) for p in parts):
                    rag_mid_cost = (parts[0] + parts[1]) / 2.0 # Calculate midpoint
                    logger.debug(f"Parsed RAG cost range '{rag_cost_range}' to midpoint: {rag_mid_cost}")
            except Exception as e:
                 logger.warning(f"Could not parse RAG cost range '{rag_cost_range}': {e}")
                
        prophet_weight = 0.7 # Example: Give Prophet 70% weight initially as the RAG is mock data
        rag_weight = 1.0 - prophet_weight

        # Calculate AI Lead
        if pd.notna(prophet_lead) and pd.notna(rag_mid_lead):
             ai_lead = (prophet_lead * prophet_weight) + (rag_mid_lead * rag_weight)
        elif pd.notna(prophet_lead):
             ai_lead = prophet_lead # Fallback to Prophet if RAG midpoint invalid
        elif pd.notna(rag_mid_lead):
             ai_lead = rag_mid_lead # Fallback to RAG midpoint if Prophet invalid
        else:
             ai_lead = np.nan # No valid lead prediction available

        # Calculate AI Cost
        if pd.notna(prophet_cost) and pd.notna(rag_mid_cost):
             ai_cost = (prophet_cost * prophet_weight) + (rag_mid_cost * rag_weight)
        elif pd.notna(prophet_cost):
             ai_cost = prophet_cost # Fallback to Prophet if RAG midpoint invalid
        elif pd.notna(rag_mid_cost):
             ai_cost = rag_mid_cost # Fallback to RAG midpoint if Prophet invalid
        else:
             ai_cost = np.nan # No valid cost prediction available

        # --- Apply Bounds ---
        # Ensure lead time is non-negative
        ai_lead = max(0, ai_lead) if pd.notna(ai_lead) else np.nan
        # Ensure cost is positive (e.g., minimum $0.001)
        ai_cost = max(0.001, ai_cost) if pd.notna(ai_cost) else np.nan

        # Currently, AI stock probability just mirrors RAG's adjusted probability

        logger.debug(f"AI Comparison Result: Lead={ai_lead}, Cost={ai_cost}, StockProb={ai_stock_prob}")
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
            if 'Date' in df_display.columns:
                df_display['Date_dt'] = pd.to_datetime(df_display['Date'], errors='coerce')
                # Format as YYYY-MM-DD, leave as empty string if conversion failed
                df_display['Date'] = df_display['Date_dt'].dt.strftime('%Y-%m-%d').fillna('')
                df_display = df_display.drop(columns=['Date_dt'])

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
            self.setup_plot_options()

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
        self.update_progress_threadsafe(0, 100, "Preparing data for AI Summary...")
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
            prompt += "Provide concise, strategic recommendations for executive review. Use Markdown formatting. **CRITICAL:** Clearly highlight any parts excluded due to EOL, Discontinuation, or being Unknown/Not Found. Explicitly state that these exclusions impact the accuracy of overall metrics like 'Est. Time to Full Kit'. Use **bold red** markers like `**CRITICAL ACTION:**` or `**WARNING:**` for these high-priority issues and related recommendations.\n\n"
            self.update_progress_threadsafe(20, 100, "Constructed prompt for AI...")
            prompt += "Focus on:\n"
            prompt += "1.  **Optimal Sourcing Strategy:** Start this section ONLY with the exact line `RECOMMENDED_STRATEGY: [Strategy Name]` where [Strategy Name] is one of: 'Strict Lowest Cost', 'Lowest Cost In Stock', 'Lowest Cost with Lead Time', 'Fastest', 'Optimized Strategy'. Then, justify the recommendation.\n"
            prompt += "2.  **CRITICAL PART ISSUES (Highlight Urgency!):**\n"
            prompt += "    *   Identify ALL EOL/Discontinued/Unknown parts. Use **ALL CAPS** and prefix with `CRITICAL:`. State explicitly these parts are **EXCLUDED** from overall cost/lead time calculations.\n"
            prompt += "    *   For these critical parts, recommend **immediate actions** (e.g., `CRITICAL: Validate alternate for EOL part [MfgPN]`, `CRITICAL: Identify source for Unknown part [BOM PN]`, `CRITICAL: Initiate redesign if no alternate found for [MfgPN]`).\n"
            prompt += "    *   Identify any other **High Risk** (Score > 6.5) Active parts. Prefix with `WARNING:`. Recommend actions (e.g., `WARNING: Secure second source for [MfgPN]`, `WARNING: Increase buffer stock for sole-source [MfgPN]`).\n"
            prompt += "3.  **Schedule & Stock Risk:**\n"
            prompt += "    *   State clearly if the 'Est. Time to Full Kit' metric is potentially **MISLEADING** because critical parts were excluded. Use **bold text**.\n"
            prompt += "    *   If there are parts requiring lead time buys ('Stock Gap Parts'), prefix this finding with `WARNING:`. Recommend confirming lead times.\n"

            prompt += "4.  **Potential Plan B / Trade-offs:** Analyze 'Near Miss' data. If applicable, highlight specific parts where minor constraint relaxation offers significant benefits.\n"
            prompt += "5.  **Overall Risk Mitigation:** Suggest 1-2 high-level strategic actions.\n"
            prompt += "6.  **Buy-Up Decisions:** Comment on significant over-buys in the recommended strategy.\n"
            prompt += "7.  **Financial Summary:** Summarize recommended strategy cost/LT and estimated tariff impact.\n"
            prompt += "**Formatting Instructions:** Use Markdown. Use **bold** for key recommendations/metrics. Use ALL CAPS and prefixes `CRITICAL:` or `WARNING:` as instructed above for high-priority issues. Use bullet points.\n\n"
            prompt += "Keep language clear, direct, and action-oriented. Use bullet points for recommendations.\n"
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
            logger.info("AI Summary Thread: Prompt built (with near-miss section), preparing API call.")

            # --- Call OpenAI ---
            self.update_progress_threadsafe(30, 100, "Generating AI Summary with OpenAI...")
            ai_response = call_chatgpt(prompt, model="gpt-4o") # Use appropriate model

            if ai_response and "OpenAI" not in ai_response: # Basic check for error messages from call_chatgpt
                 logger.info(f"AI Summary Thread: Received response from OpenAI (Length: {len(ai_response)} chars).")
                 self.update_progress_threadsafe(50, 100, "Received AI response, processing...")
            else:
                 logger.error(f"AI Summary Thread: Received error or no response from OpenAI. Response: {ai_response}")
                 ai_response = f"Error: Failed to get summary from AI.\n\nDetails: {ai_response}" # Provide error in GUI
                 self.update_progress_threadsafe(0, 100, "Error generating AI summary.")

            # --- Schedule GUI Update ---
            def update_gui():
                """ Inner function to update GUI elements from the AI summary thread. """
                logger.debug("AI Summary Thread: Executing update_gui (with tagging and separation).")
                try:
                    # Ensure necessary widgets exist
                    if not all(hasattr(self, w) and getattr(self, w) and getattr(self, w).winfo_exists()
                               for w in ['ai_summary_text', 'ai_recommendation_label', 'export_recommended_btn']):
                        logger.error("One or more required AI summary widgets missing during GUI update.")
                        # Attempt to continue if possible, but button/label updates might fail
                        export_button_exists = False # Assume button doesn't exist if check fails
                    else:
                        export_button_exists = True

                    # --- Start Parsing, Separation, Tagging, and Button Logic ---
                    # Configure main text area for writing
                    self.ai_summary_text.configure(state='normal')
                    self.ai_summary_text.delete(1.0, tk.END)
                    # Clear previous recommendation label and set default style
                    self.ai_recommendation_label.config(text="Processing recommendation...",
                                                         bg=self.COLOR_FRAME_BG, # Default BG
                                                         fg=self.COLOR_TEXT) # Default FG

                    # Define keywords for tagging (case-insensitive) - Adjusted keywords
                    critical_keywords = ["critical:", "critical action:", "eol part", "discontinued part", "unknown part", "not found part", "excluded part", "must replace", "redesign required", "immediate action"]
                    warning_keywords = ["warning:", "moderate risk:", "stock gap", "exceeds target", "potentially misleading", "n/a", "fallback strategy"]

                    # Reset recommended strategy key before parsing
                    self.ai_recommended_strategy_key = None
                    recommendation_section_active = False # Flag to track if we are inside the recommended section
                    valid_strategy_keys = list(self.strategies_for_export.keys()) # Get valid keys at runtime

                    recommendation_lines = []
                    other_analysis_lines = []
                    recommendation_found = False
                    recommendation_header_line = ""

                    response_lines = ai_response.splitlines()

                    # --- First Pass: Extract Recommendation & Separate Lines ---
                    for i, line in enumerate(response_lines):
                        line_upper_strip = line.strip().upper() # Check case-insensitive
                        line_original_strip = line.strip() # Keep original case for parsing
                        recommendation_marker = "RECOMMENDED_STRATEGY:" # The target marker

                        marker_pos = line_upper_strip.find(recommendation_marker)

                        if marker_pos != -1 and not recommendation_found: # Find first occurrence
                            recommendation_found = True
                            try:
                                # Extract text *after* the marker
                                extracted_key_raw = line_original_strip[marker_pos + len(recommendation_marker):].strip()
                                # Attempt to clean up
                                possible_key = re.split(r'\s*[:\-(]\s*', extracted_key_raw, 1)[0].strip()
                                found_key = next((vk for vk in valid_strategy_keys if possible_key.lower() == vk.lower()), None)

                                if found_key:
                                    self.ai_recommended_strategy_key = found_key
                                    # Format header nicely for the label
                                    recommendation_header_line = f"Recommended: {found_key}"
                                    # Add the rest of the line (if any) to the justification body
                                    justification_part = extracted_key_raw[len(possible_key):].strip(' :-')
                                    if justification_part:
                                        recommendation_lines.append(justification_part)
                                else:
                                    recommendation_header_line = f"Recommendation Found (Unknown Strategy: '{extracted_key_raw}')"
                                    logger.warning(f"AI provided strategy key '{extracted_key_raw}' not matched.")
                            except Exception as parse_err:
                                recommendation_header_line = "Error Parsing Recommendation"
                                logger.error(f"Error parsing recommendation line: {parse_err}")
                            # Don't add the marker line itself to either list

                        elif recommendation_found:
                            # If we've found the marker, add subsequent lines to recommendation list
                            if line_original_strip.startswith("===") or line_original_strip.startswith("##") or line_original_strip.startswith("---"):
                                recommendation_found = False # End of this section
                                other_analysis_lines.append(line) # Add section header to other lines
                            else:
                                recommendation_lines.append(line)
                        else:
                            # Lines before or after the recommendation section
                            other_analysis_lines.append(line)

                    # --- Update Recommendation Label ---
                    full_recommendation_text = recommendation_header_line
                    if recommendation_lines:
                        justification = "\n".join(recommendation_lines).strip()
                        justification = justification.replace("**", "") # Basic markdown removal
                        full_recommendation_text += "\n\n" + justification

                    if self.ai_recommended_strategy_key: # If valid key found, apply green style
                         self.ai_recommendation_label.config(text=full_recommendation_text, bg=self.COLOR_SUCCESS, fg="white")
                    elif recommendation_header_line: # If marker found but key invalid, show header with default style
                         self.ai_recommendation_label.config(text=full_recommendation_text, bg=self.COLOR_FRAME_BG, fg=self.COLOR_TEXT)
                    else: # No recommendation marker found at all
                         self.ai_recommendation_label.config(text="No specific strategy recommendation found in AI response.", bg=self.COLOR_FRAME_BG, fg=self.COLOR_TEXT)


                    # --- Populate Main ScrolledText with Other Lines & Tags ---
                    for line in other_analysis_lines:
                        tags_to_apply = ()
                        line_lower = line.lower().strip()
                        line_original_strip = line.strip()

                        # Apply tags based on keywords or markdown headers
                        if line_original_strip.startswith("===") or line_original_strip.startswith("##") or line_original_strip.startswith("---"):
                             tags_to_apply = ('bold',)
                        elif any(keyword in line_lower for keyword in critical_keywords):
                             tags_to_apply = ('critical',)
                        elif any(keyword in line_lower for keyword in warning_keywords):
                             tags_to_apply = ('warning',)

                        self.ai_summary_text.insert(tk.END, line + "\n", tags_to_apply)

                    # --- Enable/Disable Export Button ---
                    if export_button_exists:
                         state_to_set = "normal" if self.ai_recommended_strategy_key else "disabled"
                         self.export_recommended_btn.config(state=state_to_set)
                         logger.debug(f"Set 'Export Recommended Strategy' button state to: {state_to_set}")

                    self.ai_summary_text.configure(state='disabled')
                    # --- End Parsing, Separation, Tagging, and Button Logic ---

                    logger.info("AI Summary widgets updated.")

                    # Tab Switching and Status Update
                    if hasattr(self, 'results_notebook') and self.results_notebook.winfo_exists() and \
                       hasattr(self, 'predictive_tab') and self.predictive_tab.winfo_exists():
                         try:
                              self.results_notebook.select(self.predictive_tab)
                              logger.debug("Switched to Predictive Analysis tab.")
                         except tk.TclError as tab_err:
                              logger.error(f"Failed to switch to Predictive Analysis tab: {tab_err}")

                    if "Error:" not in ai_response:
                         self.update_status_threadsafe("AI summary generated.", "success")
                    else:
                         self.update_status_threadsafe("Error generating AI summary.", "error")

                except tk.TclError as tk_err:
                     logger.error(f"Tkinter Error updating AI summary GUI: {tk_err}", exc_info=True)
                     self.update_status_threadsafe(f"GUI Error updating AI Summary: {tk_err}", "error")
                except Exception as e:
                     logger.error(f"Error during AI summary display/tagging: {e}", exc_info=True)
                     try:
                         self.ai_summary_text.configure(state='normal')
                         self.ai_summary_text.delete(1.0, tk.END)
                         self.ai_summary_text.insert(tk.END, "Error applying formatting. Raw response:\n\n" + ai_response)
                         self.ai_summary_text.configure(state='disabled')
                         self.update_status_threadsafe(f"Error displaying/formatting AI Summary: {e}", "error")
                         if hasattr(self, 'export_recommended_btn') and self.export_recommended_btn.winfo_exists(): self.export_recommended_btn.config(state="disabled")
                     except Exception as fallback_e:
                          logger.error(f"Failed even to display raw AI response: {fallback_e}")

            self.root.after(0, update_gui) 
            logger.info("AI Summary Thread: GUI update scheduled.") 

        except Exception as e:
            # Catch errors during data extraction or prompt building
            logger.error(f"AI summary generation thread failed before API call: {e}", exc_info=True)
            self.root.after(0, self.update_status_threadsafe, f"AI Summary Prep Error: {e}", "error")
            # Schedule messagebox on main thread
            self.root.after(0, messagebox.showerror, "AI Summary Error", f"Failed preparing data for AI summary:\n\n{e}")
        finally:
            def configure_tags_main_thread():
                if hasattr(self, 'ai_summary_text') and self.ai_summary_text.winfo_exists():
                    try:
                        # Define critical tag (bold red)
                        self.ai_summary_text.tag_configure("critical", foreground=self.COLOR_ERROR, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL, "bold"))
                        # Define warning tag (bold orange)
                        self.ai_summary_text.tag_configure("warning", foreground=self.COLOR_WARN, font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL, "bold"))
                        # Define highlight tag (yellow background - use sparingly)
                        self.ai_summary_text.tag_configure("highlight", background="#FFFFE0") # Light yellow
                        # Define standard bold tag (if needed, beyond AI's markdown)
                        self.ai_summary_text.tag_configure("bold", font=(self.FONT_FAMILY, self.FONT_SIZE_NORMAL, "bold"))
                        logger.debug("AI Summary text tags configured.")
                    except tk.TclError as tag_err:
                        logger.error(f"Error configuring AI summary text tags: {tag_err}")
    
            if is_main_thread():
                configure_tags_main_thread()
            else:
                 self.root.after(0, configure_tags_main_thread)
                
            logger.info("AI Summary Thread: Finishing.")
            self.update_progress_threadsafe(100, 100, "AI Summary Complete")
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

             if isinstance(data, list) and data:
                first_item = data[0]

                # --- Handling for Main Parts Tree (self.tree) ---
                if tree == self.tree and isinstance(first_item, dict):
                    if not hasattr(self, 'tree_item_data_map'): self.tree_item_data_map = {}
                    self.tree_item_data_map.clear()  # Clear map before repopulating
                
                    for i, item_dict in enumerate(data):
                        if not isinstance(item_dict, dict): continue
                        values = [str(item_dict.get(col, '')).replace('nan', 'N/A') for col in cols]
                        tags = ()
                        # Apply Risk Tag based on RiskScore
                        risk_score_str = item_dict.get('RiskScore', 'N/A')
                        risk_score = safe_float(risk_score_str, default=np.nan)
                        if pd.notna(risk_score):
                            if risk_score >= self.RISK_CATEGORIES['high'][0]: tags = ('high_risk',)
                            elif risk_score >= self.RISK_CATEGORIES['moderate'][0]: tags = ('moderate_risk',)
                            else: tags = ('low_risk',)
                        else:
                            tags = ('na_risk',)  # Tag for N/A risk score
                
                        item_id = tree.insert("", "end", values=values, tags=tags)
                        self.tree_item_data_map[item_id] = item_dict  # Store gui_entry in tree_item_data_map

                # --- Handling for Summary Metrics Table (self.analysis_table) ---
                elif tree == self.analysis_table and isinstance(first_item, (list, tuple)) and len(first_item) == 2:
                    for item_tuple in data:
                        if not isinstance(item_tuple, (list, tuple)) or len(item_tuple) != 2: continue
                        metric_name, value = item_tuple
                        # Ensure value is string for display
                        value_str = str(value).replace('nan', 'N/A') if value is not None else ''

                        # --- Apply Conditional Tags based on Metric/Value ---
                        tags = () # Default: no special tag
                        value_lower = value_str.lower()

                        # Example conditions (adjust as needed)
                        if metric_name == "Immediate Stock Availability" and value_lower.startswith("false"):
                            tags = ('error_metric',) # Use error style tag
                        elif metric_name == "Parts with Stock Gaps" and value_str != "None":
                            tags = ('warn_metric',) # Use warning style tag
                        elif "Est. Time to Full Kit" in metric_name and "N/A" in value_str:
                             tags = ('error_metric',)
                        elif "Strategy Cost / LT" in metric_name and "N/A" in value_str:
                             tags = ('warn_metric',) # Mark unavailable strategies as warnings

                        # --- END Apply Conditional Tags ---

                        # Insert row with tags
                        tree.insert("", "end", values=[metric_name, value_str], tags=tags)

                # --- Handling for Predictions Table (self.predictions_tree) ---
                elif tree == self.predictions_tree and isinstance(first_item, (list, tuple)):
                     # Assuming data for predictions tree is already correctly formatted list of lists/tuples
                     for item_values in data:
                          if not isinstance(item_values, (list, tuple)): continue
                          # Just insert directly, no special styling here yet
                          # Ensure values are strings
                          str_values = [str(v) if v is not None else '' for v in item_values]
                          tree.insert("", "end", values=str_values) # Need mapping logic if saving edits

                # --- Fallback/General Handling ---
                elif isinstance(first_item, (list, tuple)): # General case if type not specific tree
                     for item_tuple in data:
                          if not isinstance(item_tuple, (list, tuple)): continue
                          values = [str(v).replace('nan', 'N/A') if v is not None else '' for v in item_tuple]
                          if len(values) == len(cols):
                               tree.insert("", "end", values=values)
                          else: logger.warning(f"Row length mismatch...") # Keep warning

            # ... (rest of function, error handling) ...
        except tk.TclError as e:
            logger.warning(f"Ignoring Tkinter error during populate_treeview: {e}")
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

            try:
                plt.close('all')
                logger.info("Closed Matplotlib figures.")
            except Exception as e:
                 logger.warning(f"Could not close Matplotlib figures during shutdown: {e}")
            # --- End Optional ---

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