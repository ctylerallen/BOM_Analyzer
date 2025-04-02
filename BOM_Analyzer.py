import os
import sys
from pathlib import Path
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
from dotenv import load_dotenv
import openai
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeoutError
from datetime import datetime, timedelta, timezone
import csv
import ssl
print(f"DEBUG: Imported ssl module from: {ssl.__file__}") # <-- ADD THIS LINE

# --- Dependency Check ---
try:
    from prophet import Prophet
except ImportError:
    # Use messagebox directly here as root window doesn't exist yet
    messagebox.showerror("Dependency Error",
                         "Prophet library not found. Please install it: \n"
                         "pip install prophet")
    sys.exit(1)

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
logger = logging.getLogger(__name__)
# Quieten noisy libraries
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING) # Further reduced prophet noise
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)


# --- Configuration & Constants ---
CACHE_DIR = Path('./cache')
os.makedirs(CACHE_DIR, exist_ok=True)
TOKEN_FILE = CACHE_DIR / 'digikey_oauth2_token.json'
MOUSER_COUNTER_FILE = CACHE_DIR / 'mouser_request_counter.json'
HISTORICAL_DATA_FILE = Path('bom_historical_data.csv')
PREDICTION_FILE = Path('supply_chain_predictions.csv')
DEFAULT_TARIFF_RATE = 0.035
API_TIMEOUT_SECONDS = 15 # Timeout for individual API calls
MAX_API_WORKERS = 8 # Max concurrent API calls

# --- Load Environment Variables ---
logger.info("Attempting to load keys.env...")
if not load_dotenv('keys.env'):
    logger.warning("Could not find keys.env file.")
else:
    logger.info(f"Loaded .env from: {os.getenv('DOTENV_PATH', 'keys.env')}")

# --- API Key Validation and Loading ---
DIGIKEY_CLIENT_ID = os.getenv('DIGIKEY_CLIENT_ID')
DIGIKEY_CLIENT_SECRET = os.getenv('DIGIKEY_CLIENT_SECRET')
MOUSER_API_KEY = os.getenv('MOUSER_API_KEY')
OPENAI_API_KEY = os.getenv('CHATGPT_API_KEY') # Renamed for clarity
# Add placeholders for other keys - GET THESE FROM THE RESPECTIVE SERVICES
ARROW_API_KEY = os.getenv('ARROW_API_KEY') # Placeholder
AVNET_API_KEY = os.getenv('AVNET_API_KEY') # Placeholder
OCTOPART_API_KEY = os.getenv('OCTOPART_API_KEY') # Placeholder

API_KEYS = {
    "DigiKey": bool(DIGIKEY_CLIENT_ID and DIGIKEY_CLIENT_SECRET),
    "Mouser": bool(MOUSER_API_KEY),
    "OpenAI": bool(OPENAI_API_KEY),
    "Arrow": bool(ARROW_API_KEY),
    "Avnet": bool(AVNET_API_KEY),
    "Octopart": bool(OCTOPART_API_KEY),
}

logger.info(f"DigiKey Keys: {'Set' if API_KEYS['DigiKey'] else 'Not set'}")
logger.info(f"Mouser Key: {'Set' if API_KEYS['Mouser'] else 'Not set'}")
logger.info(f"OpenAI Key: {'Set' if API_KEYS['OpenAI'] else 'Not set'}")
# Logging for mock status (corrected logic is in GUI build)
mockable_apis_log_check = ["Arrow", "Avnet", "Octopart"]
for api in mockable_apis_log_check:
     logger.info(f"{api} Key: {'Set' if API_KEYS[api] else 'Not set (Mocked)'}")


# Configure OpenAI
if API_KEYS["OpenAI"]:
    openai.api_key = OPENAI_API_KEY
else:
    logger.warning("CHATGPT_API_KEY not set - AI analysis features will be disabled.")

# --- Utility Functions ---

@lru_cache(maxsize=128) # Keep caching
def call_chatgpt(prompt, model="gpt-3.5-turbo", max_tokens=1500):
    """Calls the OpenAI API with caching and retry logic for rate limits."""
    if not API_KEYS["OpenAI"]:
        logger.warning("OpenAI API key not available. Skipping ChatGPT call.")
        return "OpenAI API key not configured."

    max_retries = 3 # Number of times to retry after hitting a rate limit
    base_wait_time = 2 # Initial wait time in seconds

    for attempt in range(max_retries + 1): # Try once + max_retries
        try:
            logger.debug(f"Attempt {attempt + 1} calling OpenAI model {model}...")
            response = openai.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a supply chain analysis expert specializing in electronic components, pricing, lead times, and risk assessment."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.5
            )
            logger.debug(f"OpenAI call successful on attempt {attempt + 1}.")
            return response.choices[0].message.content.strip() # Return on success

        except openai.RateLimitError as e:
            logger.warning(f"OpenAI Rate Limit Error encountered (Attempt {attempt + 1}/{max_retries + 1}): {e}")
            if attempt < max_retries:
                wait_time = base_wait_time * (2 ** attempt) # Exponential backoff (2s, 4s, 8s)
                logger.info(f"Waiting {wait_time} seconds before retrying OpenAI call...")
                # Optionally update status bar via root.after if called from a context that has self.root
                # Example: if 'app_instance' in globals(): # Crude check if running in app context
                #     app_instance.root.after(0, app_instance.update_status, f"OpenAI Rate Limit Hit. Retrying in {wait_time}s...", "warning")
                time.sleep(wait_time)
            else:
                logger.error("OpenAI Rate Limit Error: Max retries exceeded.")
                return f"OpenAI Rate Limit Error: Max retries exceeded. Please try again later. ({e})" # Return error after max retries

        except openai.AuthenticationError as e:
            logger.error(f"OpenAI Authentication Error: {e}. Check your API key.")
            # Optionally update status bar
            # if 'app_instance' in globals():
            #     app_instance.root.after(0, app_instance.update_status, "OpenAI Auth Error. Check Key.", "error")
            return "OpenAI Authentication Error. Check Key." # Don't retry auth errors

        except Exception as e:
            logger.error(f"ChatGPT API error (Attempt {attempt + 1}): {str(e)}")
            # Maybe retry certain transient errors? For now, fail fast on other errors.
            return f"ChatGPT API error: {str(e)}" # Return error

    # Should not be reached if logic is correct, but acts as a fallback
    return "ChatGPT call failed after multiple attempts."

def init_csv_file(filepath, header):
    """Initializes a CSV file with a header if it doesn't exist."""
    if not filepath.exists():
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
            logger.info(f"Initialized CSV file: {filepath}")
        except IOError as e:
            logger.error(f"Failed to initialize CSV file {filepath}: {e}")
            # Show error popup if initialization fails critically
            messagebox.showerror("File Error", f"Could not create required data file:\n{filepath}\n\nError: {e}\n\nApplication might not function correctly.")

def append_to_csv(filepath, data_rows):
    """Appends rows of data to a CSV file."""
    if not data_rows:
        return
    try:
        # Ensure data_rows contains lists/tuples, not complex objects
        cleaned_rows = []
        for row in data_rows:
             if isinstance(row, (list, tuple)):
                 # Convert all elements to string to avoid CSV writer issues
                 cleaned_rows.append([str(item) if item is not None else '' for item in row])
             else:
                  logger.warning(f"Skipping invalid row type during CSV append: {type(row)}")

        if not cleaned_rows:
             logger.warning(f"No valid rows to append to {filepath} after cleaning.")
             return

        with open(filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(cleaned_rows)
    except IOError as e:
        logger.error(f"Failed to append to CSV file {filepath}: {e}")
    except Exception as e:
        logger.error(f"Unexpected error writing to CSV {filepath}: {e}")

def safe_float(value, default=np.nan):
    """Safely convert value to float, handling common invalid inputs."""
    if value is None or isinstance(value, (bool)): return default
    if isinstance(value, (int, float)): return float(value)
    try:
        # Handle strings like 'N/A', '$1.23', '1,000', empty strings
        s_val = str(value).strip().replace('$', '').replace(',', '').replace('%', '')
        if not s_val or s_val.lower() in ['n/a', 'none', 'inf', '-inf', 'na']: # Added 'na'
            return default
        return float(s_val)
    except (ValueError, TypeError):
        return default

def convert_lead_time_to_days(lead_time_str):
    """Converts various lead time strings (weeks, days) to days."""
    if lead_time_str is None or pd.isna(lead_time_str):
        return np.nan
    if isinstance(lead_time_str, (int, float)): # Assume weeks if numeric
         if pd.isna(lead_time_str) or np.isinf(lead_time_str): return np.nan
         return int(lead_time_str * 7)

    s = str(lead_time_str).lower().strip()
    if s in ['n/a', 'unknown', '', 'na', 'none']:
        return np.nan
    try:
        # Improved parsing: find first number sequence
        import re
        match = re.search(r'\d+', s)
        if not match: return np.nan
        num = int(match.group(0))

        if 'week' in s:
            return int(num * 7)
        elif 'day' in s:
            return int(num)
        else: # Assume weeks if no unit, typical for DigiKey 'X Weeks' format
            return int(num * 7)

    except Exception as e:
        logger.warning(f"Failed to convert lead time '{lead_time_str}': {e}")
        return np.nan

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
    def __init__(self, root):
        self.root = root
        self.root.title("NPI BOM Analyzer - Rev 2 (Integrated)")
        self.root.geometry("1400x850") # Increased size
        self.root.minsize(1000, 600)
        self.root.resizable(True, True)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing) # Handle closing
        self.root.configure(bg='#f0f0f0')
        logger.info("Initializing GUI...")

        # --- Define Headers Early ---
        self.hist_header = ['Component', 'Manufacturer', 'Part_Number', 'Distributor',
               'Lead_Time_Days', 'Cost', 'Inventory', 'Stock_Probability', 'Fetch_Timestamp']
        self.pred_header = ['Component', 'Date', 'Prophet_Lead', 'Prophet_Cost',
                       'RAG_Lead', 'RAG_Cost', 'AI_Lead', 'AI_Cost',
                       'Stock_Probability', 'Human_Lead', 'Human_Cost',
                       'Real_Lead', 'Real_Cost', 'Real_Stock',
                       'Prophet_Acc', 'RAG_Acc', 'AI_Acc', 'Human_Acc']

        self.style = ttk.Style()
        self.style.theme_use('clam') # Or 'alt', 'default', 'classic'
        # Configure styles for better appearance
        self.style.configure("TLabel", background="#f0f0f0", font=("Segoe UI", 9))
        self.style.configure("TButton", font=("Segoe UI", 9), padding=5)
        self.style.configure("TFrame", background="#f0f0f0")
        self.style.configure("Treeview", font=("Segoe UI", 9), rowheight=25)
        self.style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        self.style.configure("TNotebook.Tab", font=("Segoe UI", 9, "bold"), padding=[10, 5])
        self.style.configure("TLabelframe", background="#f0f0f0", font=("Segoe UI", 10, "bold"))
        self.style.configure("TLabelframe.Label", background="#f0f0f0", font=("Segoe UI", 10, "bold"))

        self.main_paned_window = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_paned_window.pack(fill="both", expand=True, padx=10, pady=10)

        # --- Left Pane: Configuration ---
        self.config_frame_outer = ttk.Frame(self.main_paned_window, padding=10, width=350) # Increased width
        self.main_paned_window.add(self.config_frame_outer, weight=1)

        self.config_scroll_canvas = tk.Canvas(self.config_frame_outer, borderwidth=0, background="#f0f0f0")
        self.config_scrollbar = ttk.Scrollbar(self.config_frame_outer, orient="vertical", command=self.config_scroll_canvas.yview)
        self.config_frame = ttk.Frame(self.config_scroll_canvas, padding=(10,0), style="TFrame") # Inner frame for content

        self.config_frame.bind("<Configure>", lambda e: self.config_scroll_canvas.configure(scrollregion=self.config_scroll_canvas.bbox("all")))
        self.config_scroll_canvas.create_window((0, 0), window=self.config_frame, anchor="nw")
        self.config_scroll_canvas.configure(yscrollcommand=self.config_scrollbar.set)

        self.config_scroll_canvas.pack(side="left", fill="both", expand=True)
        self.config_scrollbar.pack(side="right", fill="y")


        # --- Configuration Widgets ---
        ttk.Label(self.config_frame, text="BOM Analysis Configuration", font=("Segoe UI", 12, "bold")).pack(fill="x", pady=(0, 15))

        # Load BOM
        load_bom_frame = ttk.Frame(self.config_frame)
        load_bom_frame.pack(fill="x", pady=(0, 10))
        self.load_button = ttk.Button(load_bom_frame, text="Load BOM CSV", command=self.load_bom)
        self.load_button.pack(side=tk.LEFT, padx=(0, 5))
        self.file_label = ttk.Label(load_bom_frame, text="No BOM loaded.", wraplength=200)
        self.file_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

        # Analysis Controls
        run_frame = ttk.Frame(self.config_frame)
        run_frame.pack(fill="x", pady=(10, 10))
        self.run_button = ttk.Button(run_frame, text="Run Analysis", command=self.validate_and_run_analysis, state="enabled")
        self.run_button.pack(side=tk.LEFT, padx=(0,5))
        self.predict_button = ttk.Button(run_frame, text="Run Predictions", command=self.run_predictive_analysis_gui, state="enabled")
        self.predict_button.pack(side=tk.LEFT, padx=(0,5))
        self.ai_summary_button = ttk.Button(run_frame, text="AI Summary", command=self.generate_ai_summary_gui, state="enabled")
        self.ai_summary_button.pack(side=tk.LEFT)

        # Sweet Spot Config
        sweet_spot_frame = ttk.LabelFrame(self.config_frame, text="Sweet Spot Configuration", padding=10)
        sweet_spot_frame.pack(fill="x", pady=(10, 5))

        config_entries = [
            ("Total Units to Build:", "total_units", "1", "Number of finished goods to build"),
            ("Max Cost Premium (%):", "max_premium", "15", "Max % above cheapest available price"),
            ("Target Lead Time (days):", "target_lead_time_days", "56", "Max acceptable lead time in days (e.g., 8 weeks = 56 days)"),
            ("Cost Weight (0-1):", "cost_weight", "0.5", "Priority of cost (lower is better)"),
            ("Lead Time Weight (0-1):", "lead_time_weight", "0.5", "Priority of lead time (lower is better)"),
        ]

        self.config_vars = {}
        for label, attr, default, hint in config_entries:
            frame = ttk.Frame(sweet_spot_frame)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=label, width=20).pack(side="left")
            entry = ttk.Entry(frame, width=10)
            entry.pack(side="left", padx=5)
            entry.insert(0, default)
            # Add tooltip (requires a library like `tkinter.tix` or manual implementation)
            # Simple label hint for now:
            hint_widget = ttk.Label(frame, text="?", foreground="blue", cursor="question_arrow")
            hint_widget.pack(side=tk.LEFT)
            hint_widget.bind("<Enter>", lambda e, h=hint: self.show_hint(h))
            hint_widget.bind("<Leave>", lambda e: self.hide_hint())


            self.config_vars[attr] = entry
            entry.bind("<KeyRelease>", self.validate_inputs)

        # Tariff Config
        self.tariff_frame = ttk.LabelFrame(self.config_frame, text="Custom Tariff Rates (%)", padding=10)
        self.tariff_frame.pack(fill="x", pady=(10, 5))
        self.tariff_entries = {}
        top_countries = ["China", "Mexico", "India", "Vietnam", "Taiwan", "Japan", "Malaysia", "Germany"] # Expanded list
        for i, country in enumerate(top_countries):
            frame = ttk.Frame(self.tariff_frame)
            # Grid layout for alignment
            frame.grid(row=i // 2, column=i % 2, sticky="w", padx=5, pady=2)
            ttk.Label(frame, text=f"{country}: ").pack(side="left")
            entry = ttk.Entry(frame, width=6)
            entry.insert(0, "")  # Default blank = use default/predicted
            entry.pack(side="left")
            self.tariff_entries[country] = entry
        ttk.Label(self.tariff_frame, text="(Blank = default/predicted)", font=("Segoe UI", 8)).grid(row=(len(top_countries)+1)//2, column=0, columnspan=2, pady=(5,0))

        self.validation_label = ttk.Label(self.config_frame, text="", foreground="red", wraplength=300)
        self.validation_label.pack(fill="x", pady=5)

        self.hint_label = ttk.Label(self.config_frame, text="", foreground="blue", wraplength=300, font=("Segoe UI", 8))
        self.hint_label.pack(fill="x", pady=5)

        # API Status
        api_status_frame = ttk.LabelFrame(self.config_frame, text="API Status", padding=10)
        api_status_frame.pack(fill="x", pady=(10, 5))
        self.api_status_labels = {}
        mockable_apis = ["Arrow", "Avnet", "Octopart"] # Define list for checking mocks
        for api_name, is_set in API_KEYS.items():
             status_text = "OK" if is_set else ("Not Set" if api_name != "OpenAI" else "Not Set (Optional)")
             color = "green" if is_set else ("red" if status_text == "Not Set" else "orange")
             # Corrected Mock Check
             if not is_set and api_name in mockable_apis:
                 status_text += " (Mocked)"
                 color = "orange"

             lbl = ttk.Label(api_status_frame, text=f"{api_name}: {status_text}", foreground=color)
             lbl.pack(anchor="w")
             self.api_status_labels[api_name] = lbl


        # --- Right Pane: Results ---
        self.results_frame = ttk.Frame(self.main_paned_window, padding=5)
        self.main_paned_window.add(self.results_frame, weight=4) # Give more weight
        self.results_frame.grid_rowconfigure(1, weight=1)
        self.results_frame.grid_columnconfigure(0, weight=1)

        # Status Bar at the top of results
        status_progress_frame = ttk.Frame(self.results_frame)
        status_progress_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        status_progress_frame.grid_columnconfigure(1, weight=1) # Make progress bar expand

        self.status_label = ttk.Label(status_progress_frame, text="Ready", relief="sunken", anchor="w", padding=3, width=30) # Added width
        self.status_label.grid(row=0, column=0, padx=(0, 5), sticky="w")

        self.progress = ttk.Progressbar(status_progress_frame, orient="horizontal", length=300, mode="determinate")
        self.progress.grid(row=0, column=1, padx=5, sticky="ew")
        self.progress_label = ttk.Label(status_progress_frame, text="0%", width=10) # Added width
        self.progress_label.grid(row=0, column=2, padx=(5, 0), sticky="e")

        self.rate_label = ttk.Label(status_progress_frame, text="API Rates: DK NA/NA | M NA/NA | ...", relief="sunken", anchor="e", padding=3, width=35) # Added width
        self.rate_label.grid(row=0, column=3, padx=(10, 0), sticky="e")


        # Notebook for different result views
        self.results_notebook = ttk.Notebook(self.results_frame)
        self.results_notebook.grid(row=1, column=0, sticky="nsew")

        # --- Tab 1: Parts List & Analysis ---
        self.tree_frame = ttk.Frame(self.results_notebook, padding=(0, 5, 0, 0)) # No top padding
        self.results_notebook.add(self.tree_frame, text="BOM Analysis Summary")
        self.tree_frame.grid_columnconfigure(0, weight=1)
        self.tree_frame.grid_rowconfigure(0, weight=3) # Treeview takes most space
        self.tree_frame.grid_rowconfigure(2, weight=1) # Analysis table smaller (was row 1, now row 2)

        # Parts Treeview
        columns = [
            "PartNumber", "Manufacturer", "MfgPN", "QtyNeed", "Status", "Sources", "StockAvail",
            "COO", # ADDED
            "BestCost", "BestCostLT", "BestCostSrc",
            "FastestLT", "FastestCost", "FastestLTSrc",
            "TariffPct", # ADDED
            "Notes"
        ]
        headings = [
            "BOM P/N", "Manufacturer", "Mfg P/N", "Need", "Lifecycle", "Sources", "Stock Avail",
            "COO", # ADDED
            "Best Cost ($)", "LT (d)", "Src",
            "Fastest LT (d)", "Cost ($)", "Src",
            "Tariff (%)", # ADDED
            "Notes/Risks"
        ]
        col_widths = { # Adjusted widths
            "PartNumber": 140, "Manufacturer": 110, "MfgPN": 140, "QtyNeed": 50, "Status": 70, "Sources": 50, "StockAvail": 70,
            "COO": 50, # ADDED
            "BestCost": 70, "BestCostLT": 50, "BestCostSrc": 50,
            "FastestLT": 50, "FastestCost": 70, "FastestLTSrc": 50,
            "TariffPct": 60, # ADDED
            "Notes": 120 # Adjusted
        }
        col_align = { # Added alignment for new columns
            "QtyNeed": 'center', "Status": 'center', "Sources": 'center', "StockAvail": 'e',
            "COO": 'center', # ADDED
            "BestCost": 'e', "BestCostLT": 'center', "BestCostSrc": 'center',
            "FastestLT": 'center', "FastestCost": 'e', "FastestLTSrc": 'center',
            "TariffPct": 'e', # ADDED
        }

        self.tree = ttk.Treeview(self.tree_frame, columns=columns, show="headings", height=18)
        for col, heading in zip(columns, headings):
            width = col_widths.get(col, 90)
            align = col_align.get(col, 'w') # Default left
            self.tree.heading(col, text=heading, command=lambda c=col: self.sort_treeview(c, False))
            self.tree.column(col, width=width, minwidth=40, stretch=True, anchor=align) # Use anchor for alignment

        self.tree_vsb = ttk.Scrollbar(self.tree_frame, orient="vertical", command=self.tree.yview)
        self.tree_hsb = ttk.Scrollbar(self.tree_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=self.tree_vsb.set, xscrollcommand=self.tree_hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree_vsb.grid(row=0, column=1, sticky="ns")
        self.tree_hsb.grid(row=1, column=0, columnspan=2, sticky="ew")

        # Analysis Summary Table (Below Parts List)
        self.analysis_table_frame = ttk.LabelFrame(self.tree_frame, text="BOM Summary", padding=5)
        self.analysis_table_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        self.analysis_table_frame.grid_columnconfigure(0, weight=1)
        self.analysis_table_frame.grid_rowconfigure(0, weight=1) # Allow table to resize vertically

        self.analysis_table = ttk.Treeview(self.analysis_table_frame, columns=["Metric", "Value"], show="headings", height=6) # Reduced height
        self.analysis_table.heading("Metric", text="Metric")
        self.analysis_table.heading("Value", text="Value")
        self.analysis_table.column("Metric", width=250, stretch=False, anchor='w')
        self.analysis_table.column("Value", width=400, stretch=True, anchor='w')
        self.analysis_table_scrollbar = ttk.Scrollbar(self.analysis_table_frame, orient="vertical", command=self.analysis_table.yview)
        self.analysis_table.configure(yscrollcommand=self.analysis_table_scrollbar.set)

        self.analysis_table.grid(row=0, column=0, sticky="nsew")
        self.analysis_table_scrollbar.grid(row=0, column=1, sticky="ns")

        self.analysis_export_frame = ttk.Frame(self.analysis_table_frame)
        self.analysis_export_frame.grid(row=1, column=0, columnspan=2, pady=5, sticky='ew') # Below table

        # Add buttons (disable initially)
        self.export_cheapest_btn = ttk.Button(self.analysis_export_frame, text="Export Cheapest", command=lambda:   self.export_strategy_gui("Cheapest"), state="disabled")
        self.export_cheapest_btn.pack(side=tk.LEFT, padx=5)
        self.export_fastest_btn = ttk.Button(self.analysis_export_frame, text="Export Fastest", command=lambda: self.export_strategy_gui("Fastest"), state="disabled")
        self.export_fastest_btn.pack(side=tk.LEFT, padx=5)
        self.export_sweetspot_btn = ttk.Button(self.analysis_export_frame, text="Export Sweet Spot", command=lambda: self.export_strategy_gui("Sweet Spot"), state="disabled")
        self.export_sweetspot_btn.pack(side=tk.LEFT, padx=5)

        # --- Tab 2: AI & Predictive Analysis ---
        self.predictive_frame = ttk.Frame(self.results_notebook, padding=10)
        self.results_notebook.add(self.predictive_frame, text="AI & Predictive Analysis")
        self.predictive_frame.grid_rowconfigure(0, weight=1) # Text area
        self.predictive_frame.grid_rowconfigure(1, weight=2) # Prediction/Update table gets more space
        self.predictive_frame.grid_columnconfigure(0, weight=1)

        # AI Summary Text Area
        ai_frame = ttk.LabelFrame(self.predictive_frame, text="AI Analysis & Recommendations", padding=5)
        ai_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        ai_frame.grid_rowconfigure(0, weight=1)
        ai_frame.grid_columnconfigure(0, weight=1)
        self.ai_summary_text = scrolledtext.ScrolledText(ai_frame, wrap=tk.WORD, height=10, font=("Segoe UI", 9), relief="solid", borderwidth=1)
        self.ai_summary_text.grid(row=0, column=0, sticky="nsew")
        self.ai_summary_text.insert(tk.END, "Run analysis and then click 'AI Summary' to populate.")
        self.ai_summary_text.configure(state='disabled')

        # Predictions and Human Input Table
        pred_update_frame = ttk.LabelFrame(self.predictive_frame, text="Predictions vs Actuals", padding=5)
        pred_update_frame.grid(row=1, column=0, sticky="nsew")
        pred_update_frame.grid_columnconfigure(0, weight=1)
        pred_update_frame.grid_rowconfigure(0, weight=1) # Table takes available space

        # Column definitions moved to __init__ (self.pred_cols, self.pred_headings)
        pred_col_widths = {c: 70 for c in self.pred_header} # Use header for keys
        pred_col_widths['Component'] = 150
        pred_col_widths['Date'] = 80
        pred_col_widths['RAG_Lead'] = 80 # Wider for range
        pred_col_widths['RAG_Cost'] = 80 # Wider for range

        self.predictions_tree = ttk.Treeview(pred_update_frame, columns=self.pred_header, show="headings", height=10)
        for col, head in zip(self.pred_header, self.pred_header): # Use header for both
            self.predictions_tree.heading(col, text=head)
            self.predictions_tree.column(col, width=pred_col_widths.get(col, 70), minwidth=40, stretch=False, anchor='center')

        self.pred_vsb = ttk.Scrollbar(pred_update_frame, orient="vertical", command=self.predictions_tree.yview)
        self.pred_hsb = ttk.Scrollbar(pred_update_frame, orient="horizontal", command=self.predictions_tree.xview)
        self.predictions_tree.configure(yscrollcommand=self.pred_vsb.set, xscrollcommand=self.pred_hsb.set)

        self.predictions_tree.grid(row=0, column=0, sticky="nsew")
        self.pred_vsb.grid(row=0, column=1, sticky="ns")
        self.pred_hsb.grid(row=1, column=0, columnspan=2, sticky="ew") # Span horizontal scrollbar

        # Add Button to Load/Refresh Predictions Table
        load_pred_button = ttk.Button(pred_update_frame, text="Load/Refresh Predictions", command=self.load_predictions_to_gui)
        load_pred_button.grid(row=2, column=0, columnspan=2, pady=5)
        # TODO: Add controls for selecting row and updating real values

        # --- Instance Variables ---
        self.bom_df = None
        self.bom_filepath = None
        self.analysis_results = {} # Store detailed results for AI summary etc. Reset to dict
        self.part_data_cache = {} # Cache API results during a run: {part_number: {supplier: data}}
        self.historical_data_df = None
        self.predictions_df = None
        self.digikey_token_data = None
        self.mouser_requests_today = 0
        self.mouser_last_reset_date = None
        self.mouser_daily_limit = 1000 # Assume limit
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_API_WORKERS)
        self.running_analysis = False # Flag to prevent concurrent runs
        self._hts_cache = {} # HTS cache per run

        # --- Initial Setup Calls ---
        self.load_mouser_request_counter()
        self.update_rate_limit_display() # Call after loading counter
        self.load_digikey_token_from_cache()
        self.initialize_data_files() # Call before loading predictions
        self.load_predictions_to_gui() # Load existing predictions on start
        self.validate_inputs() # Initial validation after everything is set up

        logger.info("GUI initialization complete.")

    # --- GUI Helpers ---
    def show_hint(self, hint_text):
        self.hint_label.config(text=hint_text)

    def hide_hint(self):
        self.hint_label.config(text="")

    def update_status(self, message, level="info"):
        """Updates the status bar and logs the message. MUST BE CALLED FROM MAIN THREAD."""
        if not hasattr(self, 'status_label'): return # Avoid errors during shutdown
        try:
            color_map = {"info": "black", "warning": "orange", "error": "red", "success": "green"}
            # Ensure status_label exists before configuring
            if self.status_label.winfo_exists():
                 self.status_label.config(text=message, foreground=color_map.get(level, "black")) # Removed background color change
            log_func = getattr(logger, level, logger.info)
            log_func(message)
            # self.root.update_idletasks() # Avoid calling this directly from background threads
        except tk.TclError:
             logger.warning("Ignoring Tkinter update error, likely during shutdown.")
        except Exception as e:
             logger.error(f"Error updating status: {e}", exc_info=True)

    
    def update_progress(self, value, maximum, label_text=""):
        """Updates the progress bar. MUST BE CALLED FROM MAIN THREAD."""
        if not hasattr(self, 'progress') or not hasattr(self, 'progress_label'): return
        try:
            # Ensure widgets exist
            if not self.progress.winfo_exists() or not self.progress_label.winfo_exists():
                 return

            if maximum > 0:
                self.progress["maximum"] = maximum
                self.progress["value"] = value
                percentage = min(100, (value / maximum) * 100) # Cap at 100
                self.progress_label.config(text=f"{percentage:.0f}% {label_text}")
            else:
                self.progress["value"] = 0
                self.progress_label.config(text="0%")
            if value >= maximum:
                 self.progress_label.config(text=f"100% {label_text}")
            # self.root.update_idletasks() # Avoid calling this directly
        except tk.TclError:
             logger.warning("Ignoring Tkinter update error, likely during shutdown.")
        except Exception as e:
            logger.error(f"Error updating progress: {e}", exc_info=True)


    def update_rate_limit_display(self):
        """Updates the API rate limit label. MUST BE CALLED FROM MAIN THREAD."""
        if not hasattr(self, 'rate_label'): return
        try:
             # Ensure widget exists
            if not self.rate_label.winfo_exists():
                 return

            dk_remain = self.digikey_token_data.get("rate_limit_remaining", "NA") if self.digikey_token_data else "NA"
            dk_limit = self.digikey_token_data.get("rate_limit", "NA") if self.digikey_token_data else "NA"
            m_remain = self.mouser_daily_limit - self.mouser_requests_today
            m_limit = self.mouser_daily_limit
            # Add others...
            self.rate_label.config(text=f"API Rates: DK {dk_remain}/{dk_limit} | M {m_remain}/{m_limit} | ...")
            # self.root.update_idletasks() # Avoid calling this directly
        except tk.TclError:
             logger.warning("Ignoring Tkinter update error, likely during shutdown.")
        except Exception as e:
            logger.error(f"Error updating rate limit display: {e}", exc_info=True)

    def update_export_buttons_state(self):
        """Enables/disables export buttons based on analysis results."""
        if not hasattr(self, 'export_cheapest_btn'): return # Avoid error if called before init complete
        has_results = bool(self.analysis_results and self.analysis_results.get("strategies")) # Check strategies dict
        sweet_spot_summary_value = "N/A"
        summary_metrics_data = self.analysis_results.get("summary_metrics", [])
        if has_results and isinstance(summary_metrics_data, list) and summary_metrics_data:
             summary_dict = dict(self.analysis_results["summary_metrics"])
             sweet_spot_summary_value = summary_dict.get("Sweet Spot BOM Cost / Max Lead", "N/A")
        sweet_spot_valid = has_results and "N/A" not in sweet_spot_summary_value

        try:
            if self.export_cheapest_btn.winfo_exists():
                 self.export_cheapest_btn.config(state="normal" if has_results else "disabled")
            if self.export_fastest_btn.winfo_exists():
                 self.export_fastest_btn.config(state="normal" if has_results else "disabled")
            if self.export_sweetspot_btn.winfo_exists():
                 self.export_sweetspot_btn.config(state="normal" if sweet_spot_valid else "disabled")
        except tk.TclError:
            logger.warning("Ignoring Tkinter state update error, likely during shutdown.")
        except Exception as e:
            logger.error(f"Error updating export button state: {e}")


    def sort_treeview(self, col, reverse):
        """Sorts a treeview column."""
        # Determine which treeview is currently active/visible? Or sort both?
        # Simple approach: Sort the main parts list tree only for now.
        tree = self.tree
        try:
            # Attempt numeric sort first, converting non-numeric to a value that sorts last/first
            data = []
            for item in tree.get_children(''):
                 val_str = tree.set(item, col)
                 num_val = safe_float(val_str, default=None) # Use None for non-numeric
                 # Assign a very large/small number for sorting N/A etc.
                 sort_key = num_val if num_val is not None else (float('inf') if reverse else float('-inf'))
                 data.append((sort_key, item))

            data.sort(reverse=reverse)

        except (tk.TclError, ValueError): # Fallback to case-insensitive string sort
             logger.warning(f"Falling back to string sort for column {col}")
             try:
                 data = [(tree.set(item, col).lower(), item) for item in tree.get_children('')]
                 data.sort(reverse=reverse)
             except tk.TclError as e:
                 logger.error(f"TclError during string sort for {col}: {e}")
                 return # Abort sort if error

        # Rearrange items in the treeview
        try:
            for index, (sort_key, item) in enumerate(data):
                tree.move(item, '', index)
        except tk.TclError as e:
             logger.error(f"TclError moving items during sort for {col}: {e}")


        # Reverse sort direction for next click
        tree.heading(col, command=lambda: self.sort_treeview(col, not reverse))

    def infer_coo_from_hts(self, hts_code):
        """Infers likely Country of Origin from HTS code using a basic mapping."""
        if not hts_code or pd.isna(hts_code) or str(hts_code).strip().lower() in ['n/a', '']:
            return "Unknown"

        # Basic HTS Prefix -> Likely COO Mapping (Expand this)
        hts_map = {
            '8542': 'Malaysia', # General ICs often Malaysia/Taiwan/China
            '8541': 'China',    # Diodes, Transistors often China/Taiwan/Korea
            '8533': 'Japan',    # Resistors often Japan/China
            '8532': 'China',    # Capacitors often China/Taiwan/Japan
            '8504': 'Germany',  # Inductors often Germany/Japan
            '90':   'USA',      # Measuring Instruments often US/Germany
            # Add many more based on common components...
        }
        # Check prefixes
        hts_clean = str(hts_code).replace('.', '')
        for prefix, country in hts_map.items():
            if hts_clean.startswith(prefix):
                return country
       
        return "Unknown" # Default if no prefix matches
    # --- End helper function ---

    # --- Add Helper Function to Class (Simplified) ---
    def calculate_stock_probability_simple(self, options_list, qty_needed):
     """Calculates a simple stock probability score based on availability."""
     if not options_list: return 0.0

     score = 0.0
     suppliers_with_stock = 0
     total_stock = 0
     max_lead_with_stock = 0
     min_lead_no_stock = np.inf

     for option in options_list:
         stock = option.get('stock', 0)
         lead = option.get('lead_time', np.inf)
         if stock >= qty_needed:
              suppliers_with_stock += 1
              total_stock += stock
              max_lead_with_stock = max(max_lead_with_stock, lead if lead != np.inf else 0)
         else:
              min_lead_no_stock = min(min_lead_no_stock, lead)

     # Base score on number of suppliers with full stock
     if suppliers_with_stock >= 2: score = 85.0
     elif suppliers_with_stock == 1: score = 65.0
     else: score = 10.0 # Low base score if no one has full stock

     # Adjust based on lead times
     if suppliers_with_stock > 0:
          if max_lead_with_stock <= 14: score += 10.0 # Bonus for fast stock
          elif max_lead_with_stock > 56: score -= 15.0 # Penalty for long LT stock
     elif min_lead_no_stock <= 28 : score += 5.0 # Small bonus if LT isn't terrible even without stock
     elif min_lead_no_stock > 84 : score -= 10.0 # Penalty for very long LT when no stock

     # Adjust based on total stock amount (simple)
     if total_stock > qty_needed * 10: score += 5.0
     elif total_stock < qty_needed * 1.5 and suppliers_with_stock > 0: score -= 5.0

     return round(max(0.0, min(100.0, score)), 1)
        
    # --- Data File Handling ---
    def initialize_data_files(self):
        """Ensure historical and prediction CSV files exist and load initial data."""
        # Initialize CSV files with headers if they don't exist
        init_csv_file(HISTORICAL_DATA_FILE, self.hist_header)
        init_csv_file(PREDICTION_FILE, self.pred_header)

        # --- Load Historical Data ---
        try:
            # Try reading with dtype=str first to avoid initial parsing errors
            self.historical_data_df = pd.read_csv(HISTORICAL_DATA_FILE, dtype=str, keep_default_na=False)

            # Check if expected columns exist BEFORE conversion
            # Use self.hist_header which is defined in __init__
            missing_hist_cols = [col for col in self.hist_header if col not in self.historical_data_df.columns]
            if missing_hist_cols:
                 # Log the specific missing columns
                 raise KeyError(f"Column(s) missing in historical CSV header: {', '.join(missing_hist_cols)}")

            # Convert types after loading and checking header
            # Ensure Fetch_Timestamp conversion happens first
            if 'Fetch_Timestamp' in self.historical_data_df.columns:
                self.historical_data_df['Fetch_Timestamp'] = pd.to_datetime(self.historical_data_df['Fetch_Timestamp'], errors='coerce')
            else: # Should have been caught by KeyError check, but for safety
                 logger.warning("Initialize: 'Fetch_Timestamp' column unexpectedly missing after loading historical data.")

            numeric_cols_hist = ['Lead_Time_Days', 'Cost', 'Inventory', 'Stock_Probability']
            for col in numeric_cols_hist:
                 if col in self.historical_data_df.columns: # Check column exists before converting
                      self.historical_data_df[col] = pd.to_numeric(self.historical_data_df[col], errors='coerce')
                 else:
                      logger.warning(f"Initialize: Historical data missing expected numeric column: {col}")
            logger.info(f"Loaded {len(self.historical_data_df)} historical records.")

        except FileNotFoundError:
             logger.info(f"{HISTORICAL_DATA_FILE} not found. Initializing empty DataFrame.")
             self.historical_data_df = pd.DataFrame(columns=self.hist_header)
        except KeyError as e: # Catch missing column error during load/check
            logger.error(f"Failed to load historical data - {e}. Check CSV header matches self.hist_header definition. Initializing empty DataFrame.")
            # Optionally show warning, but might be annoying on first run if file is empty
            # messagebox.showwarning("File Warning", f"Historical data file ({HISTORICAL_DATA_FILE.name}) has incorrect headers.\nOld data might be ignored. Check console logs.")
            self.historical_data_df = pd.DataFrame(columns=self.hist_header) # Reset to empty with correct structure
        except Exception as e:
            logger.error(f"Failed to load historical data: {e}", exc_info=True) # Log full traceback for unexpected errors
            self.historical_data_df = pd.DataFrame(columns=self.hist_header) # Reset to empty

        # --- Load Prediction Data ---
        try:
            # Read as string first, handle NaN later
            self.predictions_df = pd.read_csv(PREDICTION_FILE, dtype=str, keep_default_na=False)

            # Check if expected columns exist BEFORE conversion
            missing_pred_cols = [col for col in self.pred_header if col not in self.predictions_df.columns]
            if missing_pred_cols:
                 raise KeyError(f"Column(s) missing in prediction CSV header: {', '.join(missing_pred_cols)}")


            # Convert relevant columns after load and header check
            if 'Date' in self.predictions_df.columns:
                 # Try specific format first, fallback to letting pandas infer
                 try:
                      self.predictions_df['Date'] = pd.to_datetime(self.predictions_df['Date'], format='%Y-%m-%d', errors='coerce')
                 except ValueError:
                      logger.warning("Could not parse 'Date' column with format %Y-%m-%d, attempting inference.")
                      self.predictions_df['Date'] = pd.to_datetime(self.predictions_df['Date'], errors='coerce')

            numeric_cols_pred = ['Prophet_Lead', 'Prophet_Cost', 'AI_Lead', 'AI_Cost', 'Stock_Probability',
                                 'Human_Lead', 'Human_Cost', 'Real_Lead', 'Real_Cost',
                                 'Prophet_Acc', 'RAG_Acc', 'AI_Acc', 'Human_Acc']
            for col in numeric_cols_pred:
                 if col in self.predictions_df.columns:
                      # Handle ranges in RAG columns before converting - RAG values kept as strings for now
                      if col not in ['RAG_Lead', 'RAG_Cost']:
                           self.predictions_df[col] = pd.to_numeric(self.predictions_df[col], errors='coerce')
                 else: logger.warning(f"Initialize: Prediction data missing expected numeric column: {col}")

            # Convert Real_Stock to boolean, handle various string representations
            if 'Real_Stock' in self.predictions_df.columns:
                 self.predictions_df['Real_Stock'] = self.predictions_df['Real_Stock'].str.lower().map(
                     {'true': True, 't': True, '1': True, 'yes': True,
                      'false': False, 'f': False, '0': False, 'no': False, '': pd.NA} # Map empty to NA
                 ).astype('boolean') # Use pandas nullable boolean type
            else: logger.warning("Initialize: Prediction data missing 'Real_Stock' column.")


            logger.info(f"Loaded {len(self.predictions_df)} prediction records.")

        except FileNotFoundError:
            logger.info(f"{PREDICTION_FILE} not found. Initializing empty DataFrame.")
            self.predictions_df = pd.DataFrame(columns=self.pred_header)
        except KeyError as e: # Catch missing column error during load/check
            logger.error(f"Failed to load prediction data - {e}. Check CSV header matches self.pred_header definition. Initializing empty DataFrame.")
            # messagebox.showwarning("File Warning", f"Prediction data file ({PREDICTION_FILE.name}) has incorrect headers.\nOld data might be ignored. Check console logs.")
            self.predictions_df = pd.DataFrame(columns=self.pred_header) # Reset to empty with correct structure
        except Exception as e:
            logger.error(f"Failed to load prediction data: {e}", exc_info=True)
            self.predictions_df = pd.DataFrame(columns=self.pred_header) # Reset to empty


    # --- Input Validation ---
    def validate_inputs(self, event=None):
        """Validates configuration inputs and updates button states."""
        is_valid = False # Default to invalid
        try:
            # --- Perform Input Validation (same as before) ---
            total_units = safe_float(self.config_vars["total_units"].get(), default=-1)
            max_premium = safe_float(self.config_vars["max_premium"].get(), default=-1)
            target_lead_time = safe_float(self.config_vars["target_lead_time_days"].get(), default=-1)
            cost_weight = safe_float(self.config_vars["cost_weight"].get(), default=-1)
            lead_time_weight = safe_float(self.config_vars["lead_time_weight"].get(), default=-1)

            errors = []
            if total_units <= 0: errors.append("Total Units must be > 0.")
            if max_premium < 0: errors.append("Max Premium must be >= 0.")
            if target_lead_time < 0: errors.append("Target Lead Time must be >= 0.")
            if not (0 <= cost_weight <= 1): errors.append("Cost Weight must be 0-1.")
            if not (0 <= lead_time_weight <= 1): errors.append("Lead Time Weight must be 0-1.")
            if not np.isclose(cost_weight + lead_time_weight, 1.0, atol=0.01):
                errors.append("Cost + Lead Time Weights must sum to 1.0.")

            for country, entry in self.tariff_entries.items():
                val = entry.get()
                if val:
                    rate = safe_float(val, default=-999)
                    if not (0 <= rate <= 1000):
                        errors.append(f"Tariff for {country} invalid (must be % >= 0).")

            if errors:
                self.validation_label.config(text="Invalid: " + " ".join(errors), foreground="red")
            else:
                self.validation_label.config(text="Configuration Valid", foreground="green")
                is_valid = True
            # --- End Input Validation ---

        except Exception as e:
            # Handle exceptions during validation itself
            if hasattr(self, 'validation_label') and self.validation_label.winfo_exists():
                 self.validation_label.config(text=f"Validation Error: {e}", foreground="red")
            logger.error(f"Input validation error: {e}", exc_info=True)
            is_valid = False # Ensure invalid on error
        finally:
            # --- Update Button States ---
            # This part runs regardless of validation exceptions to ensure buttons reflect current state
            try:
                  # Check conditions for enabling buttons
                  bom_loaded = self.bom_df is not None and not self.bom_df.empty
                  hist_loaded = self.historical_data_df is not None and not self.historical_data_df.empty
                  # Check if analysis_results dict is populated and has summaries
                  summary_exists = bool(self.analysis_results and self.analysis_results.get("summary_metrics"))

                  can_run = is_valid and bom_loaded and not self.running_analysis
                  can_predict = hist_loaded and not self.running_analysis
                  can_summarize = bool(self.analysis_results and self.analysis_results.get("summary_metrics")) and not self.running_analysis

                  # Update main control buttons
                  if hasattr(self, 'run_button') and self.run_button.winfo_exists():
                       self.run_button.config(state="normal" if can_run else "disabled")
                  if hasattr(self, 'predict_button') and self.predict_button.winfo_exists():
                       self.predict_button.config(state="normal" if can_predict else "disabled")
                  if hasattr(self, 'ai_summary_button') and self.ai_summary_button.winfo_exists():
                       self.ai_summary_button.config(state="normal" if can_summarize else "disabled")

                  # Update export buttons (call the separate helper method)
                  # Note: We don't call this directly here anymore, it's called after analysis finishes
                  # self.update_export_buttons_state() # REMOVED FROM HERE

            except tk.TclError:
                  # Ignore errors during shutdown
                  logger.warning("Ignoring Tkinter state update error, likely during shutdown.")
            except AttributeError:
                 # Ignore errors if widgets haven't been created yet during initial startup
                 logger.debug("Ignoring AttributeError during initial button state update.")
            except Exception as e:
                 logger.error(f"Unexpected error updating button states: {e}")

            # Return the validity status determined in the try block
            return is_valid
            
    # --- BOM Loading ---
    def load_bom(self):
        """Loads BOM data from a CSV file."""
        if self.running_analysis:
            messagebox.showwarning("Busy", "Analysis is currently running. Please wait.")
            return

        filepath = filedialog.askopenfilename(
            title="Select BOM CSV File",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )
        if not filepath:
            return

        try:
            df = pd.read_csv(filepath, dtype=str, keep_default_na=False) # Read all as string, keep blanks as blanks
            required_cols = {"Part Number", "Quantity"} # Minimum required

            # Find columns case-insensitively
            col_map = {c.lower().replace(" ","").replace("_",""): c for c in df.columns}
            mapped_pn = col_map.get("partnumber")
            mapped_qty = col_map.get("quantity") or col_map.get("qty")
            mapped_mfg = col_map.get("manufacturer") or col_map.get("mfg")

            rename_dict = {}
            if not mapped_pn: raise ValueError("Missing required column 'Part Number'")
            if mapped_pn != "Part Number": rename_dict[mapped_pn] = "Part Number"

            if not mapped_qty: raise ValueError("Missing required column 'Quantity'")
            if mapped_qty != "Quantity": rename_dict[mapped_qty] = "Quantity"

            if mapped_mfg and mapped_mfg != "Manufacturer":
                 rename_dict[mapped_mfg] = "Manufacturer"

            if rename_dict:
                 logger.warning(f"Renaming BOM columns: {rename_dict}")
                 df = df.rename(columns=rename_dict)

            # Basic Cleaning & Type Conversion
            df['Part Number'] = df['Part Number'].astype(str).str.strip()
            # Remove rows with blank Part Number
            df = df[df['Part Number'] != '']

            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
            df = df.dropna(subset=['Quantity']) # Drop rows where Quantity couldn't be converted
            df['Quantity'] = df['Quantity'].astype(int)

            if 'Manufacturer' not in df.columns:
                df['Manufacturer'] = '' # Add empty Manufacturer if missing
                logger.warning("BOM missing 'Manufacturer' column. Added empty column. Results may be less accurate.")
            else:
                 df['Manufacturer'] = df['Manufacturer'].astype(str).str.strip()

            # Filter invalid rows
            original_count = len(df)
            df = df[(df['Quantity'] > 0) & (df['Part Number'] != '')]
            removed_count = original_count - len(df)
            if removed_count > 0:
                 logger.warning(f"Removed {removed_count} rows from BOM due to zero/negative quantity or blank part number.")


            if df.empty:
                 raise ValueError("No valid parts found in the BOM after cleaning.")

            self.bom_df = df
            self.bom_filepath = Path(filepath)
            self.file_label.config(text=f"Loaded: {self.bom_filepath.name} ({len(self.bom_df)} parts)")
            self.update_status(f"BOM loaded successfully: {len(self.bom_df)} valid parts.", level="success")
            self.analysis_results = {} # Clear previous results (reset to dict)
            # Clear display tables
            self.clear_treeview(self.tree)
            self.clear_treeview(self.analysis_table)
            self.validate_inputs() # Re-validate to enable run button

        except Exception as e:
            self.bom_df = None
            self.bom_filepath = None
            self.file_label.config(text="BOM Load Failed!")
            self.update_status(f"Failed to load or parse BOM: {e}", level="error")
            logger.error(f"BOM Load Error: {e}", exc_info=True)
            messagebox.showerror("BOM Load Error", f"Could not load or process the BOM file.\n\nError: {e}\n\nPlease ensure it's a valid CSV with 'Part Number' and 'Quantity' columns.")
            self.validate_inputs()


    # --- DigiKey Authentication ---
    def load_digikey_token_from_cache(self):
        """Loads DigiKey token from cache file if valid."""
        try:
            with open(TOKEN_FILE, 'r') as f:
                self.digikey_token_data = json.load(f)
            expires_at = self.digikey_token_data.get('expires_at', 0)
            now = time.time()

            if now < expires_at and self.digikey_token_data.get('access_token'):
                 logger.info("Valid DigiKey token loaded from cache.")
                 refresh_in = expires_at - now - 300 # Refresh 5 mins before expiry
                 if refresh_in > 0:
                      # Cancel previous scheduled refresh if exists
                      if hasattr(self, '_digikey_refresh_after_id'):
                           self.root.after_cancel(self._digikey_refresh_after_id)
                      self._digikey_refresh_after_id = self.root.after(int(refresh_in * 1000), self.refresh_digikey_token)
                      logger.debug(f"Scheduled token refresh in {refresh_in:.0f}s")
                 return True
            elif self.digikey_token_data.get('refresh_token'):
                 logger.info("Cached DigiKey token expired, refresh needed.")
                 return True # Allow proceeding, refresh will be attempted on first API call
            else:
                 logger.warning("Cached DigiKey token invalid or missing refresh token.")
                 self.digikey_token_data = None
                 return False
        except FileNotFoundError:
            logger.info("No DigiKey token cache file found.")
            self.digikey_token_data = None
            return False
        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.error(f"Failed to load or validate cached DigiKey token: {e}")
            self.digikey_token_data = None
            try:
                if TOKEN_FILE.exists():
                     os.remove(TOKEN_FILE)
                     logger.info(f"Removed potentially corrupted token file: {TOKEN_FILE}")
            except OSError: pass
            return False

    def get_digikey_token(self, force_refresh=False):
            """Gets a valid DigiKey token, handling expiry and refresh. Uses HTTPS (modern SSLContext) for localhost callback."""
            function_start_time = time.time()
            logger.info("--- get_digikey_token START (HTTPS Mode - SSLContext) ---") # Mark start

            if not API_KEYS["DigiKey"]:
                self.root.after(0, self.update_status, "DigiKey API keys not set.", "error")
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
                self.root.after(0, self.update_status, "DigiKey authentication required (Check browser & accept HTTPS warning)", "warning")

                redirect_uri = "https://localhost:8000"
                auth_port = 8000

                auth_url = f"https://api.digikey.com/v1/oauth2/authorize?client_id={DIGIKEY_CLIENT_ID}&response_type=code&redirect_uri={urllib.parse.quote(redirect_uri)}"
                logger.debug(f"get_digikey_token: Opening browser to: {auth_url}")

                try:
                     webbrowser.open(auth_url)
                except Exception as e:
                     logger.error(f"get_digikey_token: Failed to open browser: {e}")
                     self.root.after(0, self.update_status, f"Failed to open browser: {e}", "error")
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
                     self.root.after(0, self.update_status, f"SSL Error: {e}", "error")
                     self.root.after(0, messagebox.showerror, "SSL Error", f"Could not find certificate file:\n{certfile_path}\n\nPlease generate it using the provided openssl command.")
                     return None
                except ssl.SSLError as e:
                     logger.error(f"get_digikey_token: SSL context/handshake error: {e}", exc_info=True)
                     self.root.after(0, self.update_status, f"SSL Error: {e}", "error")
                     self.root.after(0, messagebox.showerror, "SSL Error", f"Failed to create secure HTTPS server.\n\nError: {e}\n\nCheck certificate file and permissions.")
                     return None
                except OSError as e:
                     logger.error(f"get_digikey_token: OAuth server bind error (Port {auth_port} likely in use): {e}")
                     self.root.after(0, self.update_status, f"OAuth Port Error: {e}", "error")
                     self.root.after(0, messagebox.showerror, "OAuth Error", f"Could not start callback server on port {auth_port}.\nEnsure no other application is using it.\n\n{e}")
                     return None
                except Exception as e:
                     logger.error(f"get_digikey_token: OAuth callback server error: {e}", exc_info=True)
                     self.root.after(0, self.update_status, f"OAuth server error: {e}", "error")
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

                        self.root.after(0, self.update_status, "DigiKey authentication successful.", "success")
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
                        self.root.after(0, self.update_status, f"DigiKey token exchange error: {error_detail_msg}", "error")
                        self.root.after(0, messagebox.showerror, "DigiKey Auth Error", f"Failed to get token: {error_detail_msg}\n\nCheck Client ID/Secret and DigiKey App Redirect URI (using HTTPS).")
                        self.digikey_token_data = None
                        logger.info(f"--- get_digikey_token END (Token Exchange Failed HTTPS) - Total Time: {time.time() - function_start_time:.1f}s ---")
                        return None
                    except Exception as e:
                         logger.error(f"get_digikey_token: Unexpected error during token exchange/caching (HTTPS): {e}", exc_info=True)
                         self.root.after(0, self.update_status, f"Unexpected Auth Error: {e}", "error")
                         self.root.after(0, messagebox.showerror, "DigiKey Auth Error", f"Unexpected error during authentication (HTTPS):\n\n{e}")
                         self.digikey_token_data = None
                         logger.info(f"--- get_digikey_token END (Unexpected Error HTTPS) - Total Time: {time.time() - function_start_time:.1f}s ---")
                         return None
                else: # auth_code is None
                    logger.error("get_digikey_token: Did not receive auth_code from local HTTPS server. Authentication likely timed out or failed in browser/redirect/warning bypass.")
                    self.root.after(0, self.update_status, "DigiKey authentication timed out or failed (HTTPS).", "error")
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
                self.root.after(0, self.update_status, "No refresh token available. Manual re-authentication needed.", "warning")
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
                    self.root.after(0, self.update_status, f"Warning: Failed to cache refreshed token: {e}", "warning")

                self.root.after(0, self.update_status, "DigiKey token refreshed.", "info")
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
                self.root.after(0, self.update_status, f"Token refresh failed: {error_detail_msg}", "error")
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
                 self.root.after(0, self.update_status, f"Unexpected token refresh error: {e}", "error")
                 return False

    # --- Mouser Rate Limiting ---
    def load_mouser_request_counter(self):
        """Loads the Mouser API request counter."""
        today = datetime.now(timezone.utc).date()
        if MOUSER_COUNTER_FILE.exists():
            try:
                with open(MOUSER_COUNTER_FILE, 'r') as f:
                    data = json.load(f)
                last_reset_iso = data.get('last_reset_date')
                if last_reset_iso:
                    last_reset_date = datetime.fromisoformat(last_reset_iso).date()
                    if last_reset_date == today:
                        self.mouser_requests_today = data.get('requests', 0)
                        self.mouser_last_reset_date = last_reset_date
                    else: # Reset counter if it's a new day
                        self.mouser_requests_today = 0
                        self.mouser_last_reset_date = today
                else: # If date missing, reset
                     self.mouser_requests_today = 0
                     self.mouser_last_reset_date = today

            except (json.JSONDecodeError, KeyError, ValueError, Exception) as e: # Added ValueError
                logger.error(f"Failed to load Mouser request counter: {e}. Resetting count.")
                self.mouser_requests_today = 0
                self.mouser_last_reset_date = today
        else: # File doesn't exist
            self.mouser_requests_today = 0
            self.mouser_last_reset_date = today
        logger.info(f"Mouser request count loaded: {self.mouser_requests_today} for {self.mouser_last_reset_date}")
        self.save_mouser_request_counter() # Save potentially reset date/count

    def save_mouser_request_counter(self):
        """Saves the Mouser API request counter."""
        if self.mouser_last_reset_date is None: return # Don't save if not initialized
        try:
            with open(MOUSER_COUNTER_FILE, 'w') as f:
                json.dump({
                    'requests': self.mouser_requests_today,
                    'last_reset_date': self.mouser_last_reset_date.isoformat()
                }, f)
        except IOError as e:
            logger.error(f"Failed to save Mouser request counter: {e}")

    def check_and_wait_mouser_rate_limit(self):
        """Checks Mouser rate limit and waits if necessary. Runs in background thread."""
        if not API_KEYS["Mouser"]: return True # Skip if no key

        today = datetime.now(timezone.utc).date()
        if self.mouser_last_reset_date != today:
            logger.info(f"Mouser counter reset for new day: {today}")
            self.mouser_requests_today = 0
            self.mouser_last_reset_date = today
            self.save_mouser_request_counter()
            self.root.after(0, self.update_rate_limit_display) # Schedule GUI update

        if self.mouser_requests_today >= self.mouser_daily_limit:
            now = datetime.now(timezone.utc)
            next_reset_dt = datetime.combine(today + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
            wait_seconds = (next_reset_dt - now).total_seconds()

            if wait_seconds > 0:
                wait_hours = wait_seconds / 3600
                msg = f"Mouser API limit ({self.mouser_daily_limit}) reached. Waiting {wait_hours:.1f}h until reset."
                logger.warning(f"Mouser API daily limit reached. Waiting {wait_seconds:.0f} seconds.")
                self.root.after(0, self.update_status, msg, "warning") # Schedule GUI update

                # Sleep in intervals to allow GUI updates/potential cancellation
                wait_interval = 5 # seconds
                end_time = time.time() + wait_seconds
                while time.time() < end_time:
                     if not self.running_analysis: # Check if analysis was cancelled
                          logger.info("Mouser wait interrupted by analysis cancellation.")
                          return False # Indicate interruption

                     remaining = end_time - time.time()
                     wait_msg = f"Mouser limit reached. Waiting {remaining // 60:.0f}m {remaining % 60:.0f}s..."
                     self.root.after(0, self.update_status, wait_msg, "warning") # Schedule GUI update
                     time.sleep(min(wait_interval, remaining + 0.1)) # Sleep a bit longer than interval


                # After waiting (if not interrupted)
                self.mouser_requests_today = 0
                self.mouser_last_reset_date = datetime.now(timezone.utc).date()
                self.save_mouser_request_counter()
                self.root.after(0, self.update_status, "Mouser API limit reset. Resuming...", "info") # Schedule GUI update
                logger.info("Mouser API limit reset after waiting.")
                self.root.after(0, self.update_rate_limit_display) # Schedule GUI update
                return True
            else: # wait_seconds <= 0
                 self.mouser_requests_today = 0
                 self.mouser_last_reset_date = today
                 self.save_mouser_request_counter()
                 self.root.after(0, self.update_rate_limit_display)
                 return True
        else: # Limit not reached
            return True


    # --- Supplier API Functions ---

    def _make_api_request(self, method, url, **kwargs):
        """Wrapper for requests with timeout and error handling. Runs in background thread."""
        # Note: This doesn't schedule GUI updates itself, relies on caller context.
        try:
            response = requests.request(method, url, timeout=API_TIMEOUT_SECONDS, **kwargs)
            # Simple rate limit check (can be enhanced)
            if response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', '10')) # Default 10s
                logger.warning(f"Rate limit hit ({url}). Waiting {retry_after}s.")
                # Schedule status update from the caller thread context using root.after if needed
                # self.root.after(0, self.update_status, f"API rate limit hit. Pausing {retry_after}s...", "warning")
                time.sleep(retry_after)
                # Retry the request once after waiting
                response = requests.request(method, url, timeout=API_TIMEOUT_SECONDS, **kwargs)

            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            return response
        except requests.Timeout:
            logger.error(f"API request timed out: {method} {url}")
            raise TimeoutError(f"API request timed out ({API_TIMEOUT_SECONDS}s)") from None
        except requests.ConnectionError as e:
             logger.error(f"API connection error: {method} {url} - {e}")
             raise ConnectionError(f"API connection error: {e}") from e
        except requests.HTTPError as e:
            logger.error(f"API HTTP error: {method} {url} - Status {e.response.status_code} - Response: {e.response.text[:200]}")
            raise requests.HTTPError(f"HTTP {e.response.status_code}: {e.response.text[:500]}", response=e.response) from e
        except requests.RequestException as e:
            logger.error(f"API request failed: {method} {url} - {e}")
            raise RuntimeError(f"API request failed: {e}") from e

    def search_digikey(self, part_number, manufacturer=""):
        """Searches DigiKey for a part number. Runs in background thread."""
        if not API_KEYS["DigiKey"]: return None
        access_token = self.get_digikey_token() # This might block for auth
        if not access_token:
             logger.error("Failed to get DigiKey token for search.")
             return None

        url = "https://api.digikey.com/products/v4/search/keyword"
        headers = {
            'Authorization': f"Bearer {access_token}",
            'X-DIGIKEY-Client-Id': DIGIKEY_CLIENT_ID,
            'X-DIGIKEY-Locale-Site': 'US',
            'X-DIGIKEY-Locale-Language': 'en',
            'X-DIGIKEY-Locale-Currency': 'USD',
            'Content-Type': 'application/json'
        }
        keywords = f"{manufacturer} {part_number}" if manufacturer else part_number
        payload = {"Keywords": keywords.strip(), "Limit": 5, "Offset": 0} # Get a few results

        try:
            response = self._make_api_request("POST", url, headers=headers, json=payload)
            data = response.json()

            # Store rate limit info if available
            # Check if token_data exists before modifying
            if self.digikey_token_data:
                 self.digikey_token_data["rate_limit_remaining"] = response.headers.get('X-RateLimit-Remaining', 'NA')
                 self.digikey_token_data["rate_limit"] = response.headers.get('X-RateLimit-Limit', 'NA')
                 self.root.after(0, self.update_rate_limit_display) # Schedule GUI update

            products = data.get("Products", [])
            if not products:
                logger.debug(f"DigiKey: No exact match found for '{keywords}'.")
                return None

            # Find best match
            best_match = None
            exact_pn = part_number.upper()
            for p in products:
                 mpn = p.get("ManufacturerProductNumber", "").upper()
                 if mpn == exact_pn:
                      best_match = p
                      break
            if not best_match:
                 best_match = products[0]
                 logger.debug(f"DigiKey: Using first result for '{keywords}' as no exact MPN match found.")

            lead_time_weeks_str = best_match.get("ManufacturerLeadWeeks")
            logger.debug(f"DK Raw Lead Weeks for {part_number}: '{lead_time_weeks_str}' (Type: {type(lead_time_weeks_str)})")
            lead_time_days = convert_lead_time_to_days(lead_time_weeks_str)
            logger.debug(f"DK Converted Lead Days: {lead_time_days}")
            pricing_raw = best_match.get("StandardPricing", [])
            unit_price_single = best_match.get("UnitPrice")
            pricing = []
            if pricing_raw:
                 pricing = [{"qty": p["BreakQuantity"], "price": safe_float(p["UnitPrice"])} for p in pricing_raw if safe_float(p.get("UnitPrice")) is not None]
            elif unit_price_single: # Fallback to single unit price
                 single_price = safe_float(unit_price_single)
                 if single_price is not None:
                      pricing = [{"qty": 1, "price": single_price}]
            digikey_pn = "N/A"
            product_variations = best_match.get("ProductVariations", [])
            if isinstance(product_variations, list) and len(product_variations) > 0:
                first_variation = product_variations[0]
            if isinstance(first_variation, dict):
                digikey_pn = first_variation.get("DigiKeyProductNumber", "N/A")

            result = {
                "Source": "DigiKey",
                "SourcePartNumber": digikey_pn, # Use the correctly extracted DKPN
                "ManufacturerPartNumber": best_match.get("ManufacturerProductNumber", "N/A"), # Should get MPN from top-level product object
                "LeadTimeDays": lead_time_days,
                "Manufacturer": best_match.get("Manufacturer", {}).get("Name", "N/A"),
                # ... other fields using best_match or first_variation as appropriate ...
                "Stock": int(safe_float(first_variation.get("QuantityAvailableforPackageType", 0))) if digikey_pn != "N/A" else best_match.get("QuantityAvailable", 0), # Prefer variation stock if variation found
                "MinOrderQty": int(safe_float(first_variation.get("MinimumOrderQuantity", 0))) if digikey_pn != "N/A" else best_match.get("MinimumOrderQuantity", 0), # Prefer variation MOQ
                "PackageType": first_variation.get("PackageType", {}).get("Name", "N/A") if digikey_pn != "N/A" else "N/A", # Get Package Type name
                "Pricing": pricing, # Pricing often relates to variations too - check if StandardPricing/MyPricing are in variation
                "StandardPricing": first_variation.get("StandardPricing", []) if digikey_pn != "N/A" else [], # Example: Prefer variation pricing
                "MyPricing": first_variation.get("MyPricing", []) if digikey_pn != "N/A" else [], # Example: Prefer variation pricing
                # ... Rest of the fields from best_match (like COO, TariffCode etc.) ...
                 "CountryOfOrigin": best_match.get("Classifications", {}).get("CountryOfOrigin", "N/A"), # Classification might be top level
                 "TariffCode": best_match.get("Classifications", {}).get("HtsusCode", "N/A"),
                 "NormallyStocking": best_match.get("NormallyStocking", False),
                 "Discontinued": best_match.get("Discontinued", False),
                 "EndOfLife": best_match.get("ProductStatus",{}).get("Value", "").lower() in ['obsolete', 'last time buy', 'not recommended for new designs', 'nrnd'],
                 "DatasheetUrl": best_match.get("DatasheetUrl", "N/A"),
                 "ApiTimestamp": datetime.now(timezone.utc).isoformat(),
            }
            # Debug logging added previously...
            logger.debug(f"DigiKey Result Dict for {part_number}:")
            logger.debug(f"  >> MPN Field: {result.get('ManufacturerPartNumber')}") # Check this log output!
            logger.debug(f"  DKPN Field (SourcePN): {result.get('SourcePartNumber')}")
            logger.debug(f"  MFG Field: {result.get('Manufacturer')}")

            return result

        except requests.HTTPError as e:
            if e.response.status_code == 401:
                logger.warning("DigiKey 401 Unauthorized. Token might be invalid. Re-authentication may be needed.")
                # Don't retry here, let the main loop handle re-auth trigger if necessary
            else:
                 logger.error(f"DigiKey API HTTP Error for {part_number}: {e}")
            return None # Indicate error
        except (TimeoutError, ConnectionError, RuntimeError, Exception) as e:
            logger.error(f"DigiKey search failed for {part_number}: {e}")
            return None # Indicate error

    def search_mouser(self, part_number, manufacturer=""):
            """Searches Mouser for a part number. Runs in background thread."""
            if not API_KEYS["Mouser"]: return None

            # Check rate limit *before* request
            if not self.check_and_wait_mouser_rate_limit():
                 logger.warning(f"Mouser search skipped for {part_number} due to rate limit or interruption.")
                 return None

            url = "https://api.mouser.com/api/v1/search/keyword"
            headers = {'Accept': 'application/json', 'Content-Type': 'application/json'}
            params = {'apiKey': MOUSER_API_KEY}
            keyword = f"{manufacturer} {part_number}" if manufacturer else part_number
            # Request fewer records initially to reduce response size for debugging
            body = {'SearchByKeywordRequest': {'keyword': keyword.strip(), 'records': 1, 'searchOptions': 'RohsAndReach'}}

            logger.info(f"--- Mouser Search START for: {keyword} ---") # <<< Log Start

            try:
                response = self._make_api_request("POST", url, headers=headers, params=params, json=body)
                # Log status immediately
                logger.info(f"Mouser API response status for {keyword}: {response.status_code}")

                # Log raw text BEFORE trying to parse JSON
                raw_response_text = response.text
                logger.debug(f"Mouser API raw response text for {keyword} (first 500 chars): {raw_response_text[:500]}")

                # Increment count *after* successful request or handled error below
                self.mouser_requests_today += 1
                self.save_mouser_request_counter()
                self.root.after(0, self.update_rate_limit_display)

                # Try parsing JSON
                try:
                    data = response.json()
                    logger.debug(f"Mouser API JSON parsed successfully for {keyword}.")
                except json.JSONDecodeError as json_err:
                    logger.error(f"Mouser JSON Decode Error for {keyword}: {json_err}")
                    logger.error(f"Mouser Raw Response Text that failed JSON decode: {raw_response_text}")
                    return None # Cannot proceed without valid JSON

                # Check for API-level errors within the JSON
                if 'Errors' in data and data['Errors']:
                    # Ensure Errors is a list and non-empty before accessing [0]
                    if isinstance(data['Errors'], list) and len(data['Errors']) > 0:
                         err_msg = data['Errors'][0].get('Message', 'Unknown Mouser Error')
                         logger.error(f"Mouser API error in JSON for {keyword}: {err_msg}")
                         if "Unauthorized" in err_msg or "Invalid API Key" in err_msg:
                              self.root.after(0, self.update_status,"Mouser API Key Invalid or Unauthorized.", "error")
                              API_KEYS["Mouser"] = False
                              self.root.after(0, lambda: self.api_status_labels["Mouser"].config(text="Mouser: Invalid Key", foreground="red"))
                    else:
                         logger.error(f"Mouser API returned 'Errors' but format is unexpected: {data['Errors']}")
                    return None # Return None on API error

                # Process SearchResults - Add detailed logging here!
                logger.debug(f"Processing SearchResults for {keyword}...")
                search_results = data.get('SearchResults', {})
                if not search_results: # Check if SearchResults dict exists
                     logger.warning(f"Mouser: 'SearchResults' key missing or empty in response for {keyword}.")
                     return None

                parts = search_results.get('Parts', [])
                logger.debug(f"Mouser: Found {len(parts)} parts in response for {keyword}.")

                if not parts: # Check if Parts list is empty
                    logger.info(f"Mouser: No parts found for '{keyword}'.")
                    return None

                # --- Find best match with extra logging ---
                best_match = None
                exact_pn = part_number.upper()
                logger.debug(f"Mouser: Searching for exact MPN '{exact_pn}' within results...")
                for idx, p in enumerate(parts):
                     # Check if p is a dictionary before proceeding
                     if not isinstance(p, dict):
                          logger.warning(f"Mouser: Item at index {idx} in 'Parts' list is not a dictionary: {p}")
                          continue
                     mpn = p.get("ManufacturerPartNumber", "").upper()
                     logger.debug(f"Mouser: Checking part index {idx}, MPN '{mpn}'")
                     if mpn == exact_pn:
                          best_match = p
                          logger.debug(f"Mouser: Found exact match at index {idx}.")
                          break

                if not best_match:
                     best_match = parts[0] # Fallback to first result
                     # Ensure first part is a dict
                     if not isinstance(best_match, dict):
                          logger.error(f"Mouser: First item in 'Parts' list is not a dictionary for {keyword}. Cannot proceed.")
                          return None
                     logger.debug(f"Mouser: Using first result for '{keyword}' as no exact MPN match found.")
                # --- End match finding ---

                # --- Extract data with try/except around the whole block ---
                try:
                    logger.debug(f"Mouser: Extracting data from best_match for {keyword}...")
                    lead_time_str = best_match.get('LeadTime')
                    lead_time_days = convert_lead_time_to_days(lead_time_str)
                    logger.debug(f"Mouser: Extracted LeadTimeDays: {lead_time_days}")

                    pricing_raw = best_match.get('PriceBreaks', [])
                    pricing = []
                    if isinstance(pricing_raw, list):
                         for pb_idx, p in enumerate(pricing_raw):
                              if isinstance(p, dict):
                                  price_val = safe_float(p.get("Price"))
                                  qty_val = p.get("Quantity", 0)
                                  if price_val is not None:
                                      pricing.append({"qty": qty_val, "price": price_val})
                              else:
                                   logger.warning(f"Mouser: Item in PriceBreaks at index {pb_idx} is not a dict: {p}")
                    logger.debug(f"Mouser: Extracted Pricing: {pricing}")

                    lifecycle_status = best_match.get('LifecycleStatus', '') # Get value, default to empty string
                    logger.debug(f"Mouser: Extracted LifecycleStatus: {lifecycle_status}")

                    result = {
                        "Source": "Mouser",
                        "SourcePartNumber": best_match.get('MouserPartNumber', "N/A"),
                        "ManufacturerPartNumber": best_match.get('ManufacturerPartNumber', "N/A"),
                        "Manufacturer": best_match.get('Manufacturer', "N/A"),
                        "Description": best_match.get('Description', "N/A"),
                        "Stock": int(safe_float(best_match.get('AvailabilityInStock', 0), default=0)),
                        "LeadTimeDays": lead_time_days,
                        "MinOrderQty": best_match.get('Min', 0),
                        "Packaging": best_match.get('Packaging', "N/A"),
                        "Pricing": pricing,
                        "CountryOfOrigin": best_match.get("CountryOfOrigin", "N/A"), # Corrected default
                        "TariffCode": "N/A",
                        "NormallyStocking": True, # Assume true if stocked
                        "Discontinued": isinstance(lifecycle_status, str) and "Discontinued" in lifecycle_status,
                        "EndOfLife": isinstance(lifecycle_status, str) and ("Obsolete" in lifecycle_status or "NRND" in lifecycle_status),
                        "DatasheetUrl": best_match.get('DataSheetUrl', "N/A"),
                        "ApiTimestamp": datetime.now(timezone.utc).isoformat(),
                    }
                    logger.info(f"--- Mouser Search SUCCESS for: {keyword} ---")
                    return result

                except Exception as extract_err:
                     # Log the error AND the data structure that caused it
                     logger.error(f"!!! Mouser: Error during data extraction for {keyword}: {extract_err}", exc_info=True)
                     logger.error(f"!!! Mouser: Data causing extraction error (best_match): {best_match}")
                     return None # Indicate failure during extraction
                # --- End data extraction block ---

            except (TimeoutError, ConnectionError, RuntimeError, requests.HTTPError, Exception) as e:
                logger.error(f"Mouser search failed for {part_number} (Outer Try/Except): {e}", exc_info=False)
                # Increment count only if error happened *outside* the main processing block
                # Avoid double counting if it was incremented after successful request earlier
                # This logic is tricky - maybe safer to just let it overcount slightly on error
                # self.mouser_requests_today += 1 # Incrementing here might be safer
                # self.save_mouser_request_counter()
                # self.root.after(0, self.update_rate_limit_display)
                logger.info(f"--- Mouser Search FAILED (Outer Exception) for: {keyword} ---")
                return None

    # --- Mock/Placeholder APIs for Others ---
    # These run in background thread, don't need root.after
    def search_arrow(self, part_number, manufacturer=""):
        if not API_KEYS["Arrow"]: return None
        logger.debug(f"[MOCK] Searching Arrow for {manufacturer} {part_number}")
        time.sleep(0.1)
        if np.random.rand() < 0.8:
             stock = np.random.randint(0, 5000)
             lead = np.random.choice([0, 0, 7, 14, 28, 42, 56, 84, np.nan], p=[0.1, 0.1, 0.15, 0.15, 0.2, 0.1, 0.1, 0.05, 0.05]) if stock == 0 else 0
             price = round(np.random.uniform(0.05, 5.0), 3)
             return {
                 "Source": "Arrow", "SourcePartNumber": f"ARROW-{part_number}", "ManufacturerPartNumber": part_number,
                 "Manufacturer": manufacturer or "Unknown Mfg", "Description": f"Mock Arrow Desc for {part_number}",
                 "Stock": stock, "LeadTimeDays": lead if stock==0 else 0, "MinOrderQty": np.random.choice([1, 10, 25, 100]),
                 "Pricing": [{"qty": 1, "price": price}, {"qty": 100, "price": round(price*0.9, 3)}, {"qty": 1000, "price": round(price*0.8, 3)}],
                 "CountryOfOrigin": np.random.choice(["China", "Malaysia", "USA", "Mexico", "N/A"]), "TariffCode": "8542.XX.YYYY",
                 "NormallyStocking": stock > 100, "Discontinued": False, "EndOfLife": False,
                 "DatasheetUrl": "N/A", "ApiTimestamp": datetime.now(timezone.utc).isoformat(),
             }
        else: return None

    def search_avnet(self, part_number, manufacturer=""):
        if not API_KEYS["Avnet"]: return None
        logger.debug(f"[MOCK] Searching Avnet for {manufacturer} {part_number}")
        time.sleep(0.1)
        if np.random.rand() < 0.75:
             stock = np.random.randint(0, 3000)
             lead = np.random.choice([7, 14, 28, 42, 56, 84, 112, np.nan], p=[0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05, 0.05]) if stock == 0 else 0
             price = round(np.random.uniform(0.1, 6.0), 3)
             return {
                 "Source": "Avnet", "SourcePartNumber": f"AVNET-{part_number}", "ManufacturerPartNumber": part_number,
                 "Manufacturer": manufacturer or "Unknown Mfg", "Description": f"Mock Avnet Desc for {part_number}",
                 "Stock": stock, "LeadTimeDays": lead if stock==0 else 0, "MinOrderQty": np.random.choice([1, 5, 10, 50]),
                 "Pricing": [{"qty": 1, "price": price}, {"qty": 50, "price": round(price*0.95, 3)}, {"qty": 500, "price": round(price*0.85, 3)}],
                 "CountryOfOrigin": np.random.choice(["China", "Taiwan", "USA", "EU", "N/A"]), "TariffCode": "8533.XX.YYYY",
                 "NormallyStocking": stock > 50, "Discontinued": np.random.rand() < 0.05, "EndOfLife": np.random.rand() < 0.02,
                 "DatasheetUrl": "N/A", "ApiTimestamp": datetime.now(timezone.utc).isoformat(),
             }
        else: return None

    def search_octopart(self, part_number, manufacturer=""):
        if not API_KEYS["Octopart"]: return None
        logger.debug(f"[MOCK] Searching Octopart for {manufacturer} {part_number}")
        time.sleep(0.15)
        if np.random.rand() < 0.9:
             stock = np.random.randint(0, 10000)
             lead = np.random.choice([0, 7, 14, 21, 28, 42, 56, np.nan], p=[0.2, 0.15, 0.15, 0.15, 0.15, 0.1, 0.05, 0.05]) if stock == 0 else 0
             price = round(np.random.uniform(0.02, 4.0), 3)
             return {
                 "Source": "Octopart", "SourcePartNumber": f"OCTO-{part_number}", "ManufacturerPartNumber": part_number,
                 "Manufacturer": manufacturer or "Unknown Mfg", "Description": f"Mock Octopart Aggregated Desc for {part_number}",
                 "Stock": stock, "LeadTimeDays": lead if stock==0 else 0, "MinOrderQty": np.random.choice([1, 10, 100]),
                 "Pricing": [{"qty": 1, "price": price}, {"qty": 100, "price": round(price*0.92, 3)}, {"qty": 1000, "price": round(price*0.82, 3)}],
                 "CountryOfOrigin": "Aggregate", "TariffCode": "Aggregate",
                 "NormallyStocking": stock > 200, "Discontinued": False, "EndOfLife": False,
                 "DatasheetUrl": "N/A", "ApiTimestamp": datetime.now(timezone.utc).isoformat(),
             }
        else: return None


    # --- Core Analysis Logic ---

    def get_part_data_parallel(self, part_number, manufacturer):
        """Fetches data for a single part from all enabled suppliers in parallel. Runs in background thread."""
        part_number = str(part_number).strip()
        manufacturer = str(manufacturer).strip()
        if not part_number: return {}

        cache_key = (part_number, manufacturer)
        # Check instance cache first
        # if cache_key in self.part_data_cache:
        #      logger.debug(f"Using cache for {part_number}")
        #      return self.part_data_cache[cache_key]

        futures = {}
        results = {}
        supplier_funcs = {
            "DigiKey": self.search_digikey,
            "Mouser": self.search_mouser,
            "Arrow": self.search_arrow,
            "Avnet": self.search_avnet,
            "Octopart": self.search_octopart,
        }
        mockable_apis = ["Arrow", "Avnet", "Octopart"]

        # Submit tasks
        for name, func in supplier_funcs.items():
            key_is_set = API_KEYS.get(name, False)
            run_mock = (not key_is_set) and (name in mockable_apis)
            should_run = key_is_set or run_mock

            if should_run:
                # logger.debug(f"Submitting task for {name} ({'Real' if key_is_set else 'Mock'}) for {part_number}")
                futures[self.thread_pool.submit(func, part_number, manufacturer)] = name
            # else: logger.debug(f"Skipping {name} for {part_number}")

        # Process results
        for future in as_completed(futures):
            supplier_name = futures[future]
            try:
                result = future.result() # Get result from completed future
                if result:
                    results[supplier_name] = result
                # else: logger.debug(f"{supplier_name}: No data found for {part_number}")
            except Exception as e:
                # Log error but continue processing other suppliers
                logger.error(f"Error processing result from {supplier_name} for {part_number}: {e}", exc_info=False)

        # Don't cache results across runs for now, rely on API caching if implemented by them
        # self.part_data_cache[cache_key] = results
        return results


    def get_optimal_cost(self, qty_needed, pricing_breaks, min_order_qty=0):
        """Calculates the optimal unit and total cost for a given quantity."""
        logger.debug(f"--- get_optimal_cost ---")
        logger.debug(f"Inputs: qty_needed={qty_needed}, min_order_qty={min_order_qty}")
        logger.debug(f"Raw pricing_breaks: {pricing_breaks}")
        # Input validation
        if not isinstance(pricing_breaks, list) or qty_needed <= 0:
             logger.debug(f"Invalid input for get_optimal_cost: qty={qty_needed}, breaks={pricing_breaks}")
             return np.nan, np.nan

        if not pricing_breaks: # Handle case with no price breaks
             logger.debug(f"No pricing breaks provided for qty {qty_needed}")
             return np.nan, np.nan

        # Ensure breaks are sorted and MOQ is sensible
        try:
            # Ensure pricing_breaks is a list of dicts with 'qty' and 'price'
            valid_breaks = []
            if isinstance(pricing_breaks, list):
                for pb in pricing_breaks:
                    if isinstance(pb, dict) and 'qty' in pb and 'price' in pb and not pd.isna(pb['price']):
                            # Attempt to convert qty to int here for sorting
                            try:
                                pb['qty'] = int(pb['qty'])
                                valid_breaks.append(pb)
                            except (ValueError, TypeError):
                                logger.warning(f"Skipping price break with non-integer qty: {pb}")
                    else:
                        logger.warning(f"Skipping invalid price break structure: {pb}")
            else:
                logger.warning(f"pricing_breaks is not a list: {pricing_breaks}")

            if not valid_breaks:
                logger.warning(f"No valid price breaks found after cleaning.")
                return np.nan, np.nan

            pricing_breaks = sorted(valid_breaks, key=lambda x: x['qty']) # Sort valid breaks
            min_order_qty = int(min_order_qty) if min_order_qty else 0
            logger.debug(f"Sorted valid breaks: {pricing_breaks}") # Log sorted breaks
        except (ValueError, TypeError, KeyError) as e:
            logger.error(f"Error sorting/processing pricing breaks or MOQ: {e}. Breaks: {pricing_breaks}, MOQ: {min_order_qty}")
            return np.nan, np.nan

        order_qty = max(qty_needed, min_order_qty)
        unit_price = np.nan
        found_break = False

        # Iterate through sorted breaks
        for pb in pricing_breaks:
            break_qty = pb['qty'] # Already int
            pb_price = pb['price'] # Already float/nan

            if order_qty >= break_qty:
                    unit_price = pb_price
                    found_break = True
            elif found_break:
                    break

         # Handle case where needed qty is less than the smallest break qty > 0
        if not found_break and pricing_breaks:
                first_valid_break = pricing_breaks[0] # Already sorted, first is lowest qty break
                first_break_qty = first_valid_break['qty']
                if order_qty < first_break_qty and first_break_qty > 0 : # Check > 0
                    unit_price = first_valid_break['price']
                    order_qty_for_calc = max(order_qty, first_break_qty, min_order_qty)
                    logger.warning(f"Qty needed ({qty_needed}) < lowest break ({first_break_qty}). Using price {unit_price:.4f} for calculation qty {order_qty_for_calc}.")
                    total_cost = unit_price * order_qty_for_calc if not pd.isna(unit_price) else np.nan
                    logger.debug(f"Result (Low Qty Fallback): unit_price={unit_price}, total_cost={total_cost}, order_qty_for_calc={order_qty_for_calc}")
                    return unit_price, total_cost # Return early for this specific case

        # Calculate final total cost
        total_cost = unit_price * order_qty if not pd.isna(unit_price) else np.nan
        logger.debug(f"Result: unit_price={unit_price}, total_cost={total_cost}, final order_qty={order_qty}")
        logger.debug(f"--- END get_optimal_cost ---")
        return unit_price, total_cost


    def fetch_usitc_tariff_rate(self, hts_code):
        """Fetches tariff rate from USITC HTS Search (best effort). Runs in background thread."""
        if not hts_code or pd.isna(hts_code) or str(hts_code).strip().lower() in ['n/a', '']:
            return None

        hts_code_clean = str(hts_code).strip().replace(".", "")
        if not hts_code_clean.isdigit():
             logger.warning(f"Invalid HTS code format for lookup: {hts_code}")
             return None

        if hts_code_clean in self._hts_cache:
             return self._hts_cache[hts_code_clean]

        rate = None # Default if fetch fails
        try:
            # Using keyword search approach (adjust if a direct HTS lookup URL is found)
            search_url = f"https://hts.usitc.gov/reststop/search?keyword={hts_code_clean}"
            logger.debug(f"Fetching USITC tariff for HTS: {hts_code_clean}")
            # This request should ideally have its own timeout/retry logic if needed
            response = requests.get(search_url, timeout=10) # Shorter timeout for this informational API
            response.raise_for_status()
            data = response.json()

            if data and isinstance(data, list) and len(data) > 0:
                article = data[0]
                general_rate_str = article.get('general_rate')
                if general_rate_str and isinstance(general_rate_str, str):
                    rate_val = safe_float(general_rate_str) # Handles '%'
                    if rate_val is not None:
                         rate = rate_val / 100.0
                         logger.debug(f"USITC rate found for {hts_code}: {rate*100:.2f}%")

            if rate is None:
                 logger.warning(f"No general tariff rate found for HTS code {hts_code} in USITC response.")

        except requests.RequestException as e:
            logger.error(f"Failed to fetch USITC tariff rate for HTS {hts_code}: {e}")
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Error parsing USITC tariff response for HTS {hts_code}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error fetching USITC tariff for HTS {hts_code}: {e}")

        self._hts_cache[hts_code_clean] = rate # Cache result (even if None)
        return rate


    def get_tariff_info(self, hts_code, country_of_origin, custom_tariff_rates):
        """Determines the applicable tariff rate. Runs in background thread."""
        base_tariff_rate = None
        source = "N/A"

        # 1. Custom Rates
        coo_clean = str(country_of_origin).strip() if country_of_origin else ""
        if coo_clean and coo_clean != "N/A":
             custom_rate = custom_tariff_rates.get(coo_clean)
             if custom_rate is not None:
                  base_tariff_rate = custom_rate
                  source = f"Custom ({coo_clean})"
                  # logger.debug(f"Using custom tariff rate for {coo_clean}: {base_tariff_rate:.2%}")
                  return base_tariff_rate, source

        # 2. USITC Lookup
        if base_tariff_rate is None and hts_code:
             fetched_rate = self.fetch_usitc_tariff_rate(hts_code)
             if fetched_rate is not None:
                  base_tariff_rate = fetched_rate
                  source = f"USITC ({hts_code})"
                  # logger.debug(f"Using USITC tariff rate for HTS {hts_code}: {base_tariff_rate:.2%}")
                  return base_tariff_rate, source

        # 3. Fallback Prediction
        if base_tariff_rate is None and coo_clean and coo_clean != "N/A":
             predictive_increase = {
                 'China': 0.15, 'Mexico': 0.02, 'India': 0.08, 'Vietnam': 0.05,
                 'Taiwan': 0.03, 'Malaysia': 0.03, 'Japan': 0.01, 'Germany': 0.01,
                 'USA': 0.0, 'United States': 0.0,
             }
             country_rate = predictive_increase.get(coo_clean, DEFAULT_TARIFF_RATE)
             base_tariff_rate = country_rate
             source = f"Predicted ({coo_clean})"
             # logger.debug(f"Using predicted tariff rate for {coo_clean}: {base_tariff_rate:.2%}")
             return base_tariff_rate, source

        # 4. Absolute Fallback
        if base_tariff_rate is None:
             base_tariff_rate = DEFAULT_TARIFF_RATE
             source = "Default"
             # logger.debug(f"Using default tariff rate: {base_tariff_rate:.2%}")

        return base_tariff_rate, source

    def export_strategy_gui(self, strategy_name):
        """Handles button click and exports the selected strategy to a CSV file."""
        logger.info(f"Export button clicked for strategy: {strategy_name}")
        logger.debug(f"Current analysis_results keys: {self.analysis_results.keys() if self.analysis_results else 'None'}")
        # Access strategies stored in self.analysis_results
        if not self.analysis_results or strategy_name not in self.analysis_results.get("strategies", {}):
            messagebox.showerror("Export Error", f"No analysis data found for '{strategy_name}' strategy.\nPlease run analysis first.")
            logger.warning(f"Export failed: Strategy '{strategy_name}' not found in analysis_results.")
            logger.debug(f"Strategy keys available: {self.analysis_results['strategies'].keys()}")
            logger.debug(f"Content of '{strategy_name}' strategy (first 2 items): {list(self.analysis_results['strategies'].get(strategy_name, {}).items())[:2]}")
            return

        strategy_dict = self.analysis_results["strategies"][strategy_name]
        if not strategy_dict:
             messagebox.showinfo("Export Info", f"The '{strategy_name}' strategy is empty or invalid (e.g., constraints failed for all parts). No data to export.")
             logger.warning(f"Export skipped: Strategy '{strategy_name}' dict is empty.")
             return

        output_data = []
        output_header = ["BOM Part Number", "Manufacturer", "Manufacturer PN", "Qty Per Unit", "Total Qty Needed",
                         "Chosen Source", "Source PN", "Unit Cost ($)", "Total Cost ($)", "Lead Time (Days)", "Stock", "Notes"]

        for bom_pn, chosen_option in strategy_dict.items():
            # chosen_option should be the dictionary stored previously
            if not isinstance(chosen_option, dict):
                 logger.warning(f"Skipping export for {bom_pn}: Invalid option format {type(chosen_option)} in strategy.")
                 continue

            # Extract data using .get() for safety
            mfg = chosen_option.get("Manufacturer", "N/A")
            mfg_pn = chosen_option.get("ManufacturerPartNumber", bom_pn)
            qty_per = chosen_option.get("original_qty_per_unit", "N/A")
            total_need = chosen_option.get("total_qty_needed", "N/A")
            source = chosen_option.get("source", "N/A")
            source_pn = chosen_option.get("SourcePartNumber", "N/A")
            unit_cost = chosen_option.get('unit_cost', np.nan)
            total_cost = chosen_option.get('cost', np.nan)
            lead_time = chosen_option.get('lead_time', np.inf)
            stock = chosen_option.get("stock", 0)
            notes = chosen_option.get('sweet_spot_score', '') # Add score/notes if available

            output_data.append([
                bom_pn, mfg, mfg_pn, qty_per, total_need,
                source, source_pn,
                f"{unit_cost:.4f}" if not pd.isna(unit_cost) else "N/A",
                f"{total_cost:.2f}" if not pd.isna(total_cost) else "N/A",
                f"{lead_time:.0f}" if lead_time != np.inf else "N/A",
                stock, notes
            ])

        if not output_data:
            messagebox.showinfo("Export Info", "No valid part data found to export for this strategy.")
            return

        # Ask user for filename
        default_filename = f"BOM_Strategy_{strategy_name.replace(' ', '')}_{datetime.now():%Y%m%d}.csv"
        filepath = filedialog.asksaveasfilename(
            title=f"Save {strategy_name} Strategy As",
            defaultextension=".csv",
            initialfile=default_filename,
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")]
        )

        if not filepath:
            logger.info("User cancelled export.")
            return # User cancelled

        # Write to CSV
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL) # Ensure quoting
                writer.writerow(output_header)
                writer.writerows(output_data)
            # Schedule GUI updates for feedback
            self.root.after(0, self.update_status, f"Exported '{strategy_name}' strategy to {Path(filepath).name}", "success")
            messagebox.showinfo("Export Successful", f"Successfully exported strategy to:\n{filepath}") # This blocks, but OK after save
        except IOError as e:
            logger.error(f"Failed to export strategy CSV: {e}")
            messagebox.showerror("Export Error", f"Failed to write CSV file:\n{filepath}\n\nError: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during strategy export: {e}", exc_info=True)
            messagebox.showerror("Export Error", f"An unexpected error occurred during export:\n\n{e}")

    def analyze_single_part(self, bom_part_number, bom_manufacturer, bom_qty_needed, config):
        """
        Analyzes a single BOM line item, returns ONE consolidated GUI row,
        historical entries with consistent Component key, and detailed options.
        """
        part_results_by_supplier = self.get_part_data_parallel(bom_part_number, bom_manufacturer)

        analysis_entry = {} # For consolidated GUI row
        historical_entries = [] # For historical CSV
        part_summary = { # For strategy calculation & export
            "bom_pn": bom_part_number, "bom_mfg": bom_manufacturer,
            "bom_qty_needed": bom_qty_needed, "options": []
        }
        temp_supplier_data = {} # For intermediate processing

        if not part_results_by_supplier:
             # Create placeholder consolidated row
             analysis_entry = {
                 "PartNumber": bom_part_number, "Manufacturer": bom_manufacturer or "N/A", "MfgPN": "NOT FOUND",
                 "QtyNeed": bom_qty_needed, "Status": "Unknown", "Sources": 0, "StockAvail": "N/A",
                 "BestCost": "N/A", "BestCostLT": "N/A", "BestCostSrc": "N/A",
                 "FastestLT": "N/A", "FastestCost": "N/A", "FastestLTSrc": "N/A", "Notes": "No suppliers found"
             }
             return [analysis_entry], historical_entries, part_summary

        # --- First Pass: Process suppliers, store temp data AND create detailed options ---
        bom_qty_map = self.bom_df.set_index('Part Number')['Quantity'].to_dict() if self.bom_df is not None else {}
        original_qty_per_unit = bom_qty_map.get(bom_part_number, "N/A")

        all_options_data = [] # Temp list to hold dicts for part_summary['options']

        for source, data in part_results_by_supplier.items():
            if not isinstance(data, dict): continue
            temp_supplier_data[source] = data # Store raw API data for consolidation

            # Calculate cost/LT
            unit_cost, total_cost = self.get_optimal_cost(bom_qty_needed, data.get('Pricing', []), data.get('MinOrderQty', 0))
            lead_time_days = data.get('LeadTimeDays', np.nan) # Already converted if needed
            lead_time_inf = lead_time_days if not pd.isna(lead_time_days) else np.inf
            total_cost_inf = total_cost if not pd.isna(total_cost) else np.inf
            stock = int(data.get('Stock', 0))
            min_order_qty = int(data.get('MinOrderQty', 0))

            # Create the detailed 'option' dictionary for calculations/export
            option_dict = {
                "source": source,
                "cost": total_cost_inf, # Use inf for sorting/calculations
                "lead_time": lead_time_inf, # Use inf for sorting/calculations
                "stock": stock,
                "unit_cost": unit_cost, # Store calculated unit cost (can be NaN)
                "moq": min_order_qty,
                "discontinued": data.get('Discontinued', False),
                "eol": data.get('EndOfLife', False),
                # Add context fields
                'bom_pn': bom_part_number,
                'original_qty_per_unit': original_qty_per_unit,
                'total_qty_needed': bom_qty_needed,
                # Add identifiers directly from this source's API data
                'Manufacturer': data.get('Manufacturer', bom_manufacturer), # Use source's Mfg, fallback to BOM
                'ManufacturerPartNumber': data.get('ManufacturerPartNumber', bom_part_number), # Use source's MPN, fallback to BOM PN
                'SourcePartNumber': data.get('SourcePartNumber', 'N/A'), # Use source's specific PN
                # Store fields needed for later consolidation/calculation
                'TariffCode': data.get('TariffCode'),
                'CountryOfOrigin': data.get('CountryOfOrigin'),
                'ApiTimestamp': data.get('ApiTimestamp'),
                'tariff_rate': None # Calculated later after consolidation
            }
            # --- ADD LOGGING TO CHECK THIS DICT ---
            logger.debug(f"Option Dict Created - Source: {source}, Mfg: '{option_dict.get('Manufacturer')}', MPN: '{option_dict.get('ManufacturerPartNumber')}'")
            # ---
            all_options_data.append(option_dict)

        part_summary["options"] = all_options_data # Assign collected options

        # --- Second Pass: Consolidate COO, HTS, Manufacturer, MPN ---
        consolidated_coo = "Unknown"; consolidated_hts = "N/A"
        consolidated_mfg = bom_manufacturer or "Unknown Mfg"
        consolidated_mpn = bom_part_number
        coo_source_log = "None Found"; mpn_source_log = "BOM Input"

        # Use all_options_data which contains API results for consolidation
        # Find best Manufacturer Name
        for option in all_options_data:
            api_mfg = option.get('Manufacturer')
            if api_mfg and str(api_mfg).strip().upper() not in ["N/A", "UNKNOWN MFG", ""]:
                consolidated_mfg = str(api_mfg).strip()
                mpn_source_log += f", Mfg from {option['source']}"
                break
        # Find best Manufacturer PN
        for option in all_options_data:
            api_mpn = option.get('ManufacturerPartNumber')
            if api_mpn and str(api_mpn).strip() not in ["N/A", ""] and str(api_mpn).strip() != bom_part_number:
                 # If MPN from API is different & non-empty, prefer it
                 consolidated_mpn = str(api_mpn).strip()
                 mpn_source_log += f", MPN from {option['source']}"
                 break

        final_component_name = f"{consolidated_mfg} {consolidated_mpn}".strip()
        logger.debug(f"Consolidated Component Name for {bom_part_number}: '{final_component_name}' (Source: {mpn_source_log})")

        # Consolidate COO/HTS (using all_options_data)
        for option in all_options_data: # Find best COO
            api_coo = option.get('CountryOfOrigin')
            if api_coo and str(api_coo).strip().upper() not in ["N/A", "", "UNKNOWN", "AGGREGATE"]:
                consolidated_coo = str(api_coo).strip()
                coo_source_log = f"API ({option['source']})"
                break
        if consolidated_coo == "Unknown": # Find best HTS if no COO
            for option in all_options_data:
                 api_hts = option.get('TariffCode')
                 if api_hts and str(api_hts).strip().lower() not in ['n/a', '']:
                      consolidated_hts = str(api_hts).strip()
                      inferred_coo = self.infer_coo_from_hts(consolidated_hts)
                      if inferred_coo != "Unknown":
                           consolidated_coo = inferred_coo
                           coo_source_log = f"Inferred from HTS ({consolidated_hts} via {option['source']})"
                           break
            if consolidated_hts == "N/A": # Find any HTS for display
                 for option in all_options_data:
                      api_hts = option.get('TariffCode')
                      if api_hts and str(api_hts).strip().lower() not in ['n/a', '']:
                           consolidated_hts = str(api_hts).strip()
                           break
        logger.debug(f"Consolidated COO for {bom_part_number}: {consolidated_coo} (Source: {coo_source_log})")

        # --- Third Pass: Final Metrics, GUI Row, Historical Rows ---
        best_cost_option = None; fastest_lt_option = None
        total_stock_available = 0
        sources_found = len(part_summary["options"])
        lifecycle_notes = set()

        valid_options = [opt for opt in part_summary["options"] if opt['cost'] != np.inf or opt['lead_time'] != np.inf]

        if not valid_options:
             # Create placeholder consolidated row (same as no results case)
             analysis_entry = {} # ... (placeholder as before) ...}
             analysis_entry["Manufacturer"] = consolidated_mfg # Use consolidated even for placeholder
             analysis_entry["MfgPN"] = consolidated_mpn if consolidated_mpn != bom_part_number else "NOT FOUND"
        else:
            for option in valid_options: # Iterate through valid options to log history & gather final details
                # Calculate tariff using consolidated COO/HTS
                tariff_rate, _ = self.get_tariff_info(consolidated_hts, consolidated_coo, config['custom_tariff_rates'])
                option['tariff_rate'] = tariff_rate # Store back into option dict

                total_stock_available += option.get('stock', 0)
                if option.get('discontinued'): lifecycle_notes.add("DISC")
                if option.get('eol'): lifecycle_notes.add("EOL")

            # --- Calculate Stock Probability ---
                stock_prob = self.calculate_stock_probability_simple(valid_options, bom_qty_needed)
                option['stock_prob'] = stock_prob # Store back into option dict
                logger.debug(f"Calculated Stock Probability for {bom_part_number}: {stock_prob}%")

                # --- Historical Logging ---
                # Use FINAL consolidated component name, and SOURCE-SPECIFIC Mfg/MPN from option dict
                historical_entries.append([
                    final_component_name, # Consolidated Key
                    option.get('Manufacturer', 'N/A'), # Source Specific Mfg
                    option.get('ManufacturerPartNumber', 'N/A'), # Source Specific MPN
                    option.get('source'),
                    option.get('lead_time', np.inf) if option.get('lead_time') != np.inf else np.nan,
                    option.get('unit_cost', np.nan),
                    option.get('stock', 0),
                    stock_prob, #use calculated probability
                    option.get('ApiTimestamp', datetime.now(timezone.utc).isoformat())
                ])

            # Find best/fastest from valid options
            best_cost_option = min(valid_options, key=lambda x: x['cost'])
            fastest_lt_option = min(valid_options, key=lambda x: x['lead_time'])

            status = "Active"; notes = ""
            if "EOL" in lifecycle_notes: status = "EOL"
            elif "DISC" in lifecycle_notes: status = "Discontinued"
            if total_stock_available < bom_qty_needed: notes = "Stock Gap"
            notes = (notes + "; " + "; ".join(lc for lc in lifecycle_notes if lc not in status)).strip('; ')


            # Create the single consolidated row for the GUI table
            analysis_entry = {
                "PartNumber": bom_part_number, # Original BOM PN
                "Manufacturer": consolidated_mfg, # Consolidated Best Mfg
                "MfgPN": consolidated_mpn, # Consolidated Best MPN
                "QtyNeed": bom_qty_needed,
                "Status": status,
                "Sources": f"{sources_found}",
                "StockAvail": f"{total_stock_available}",
                "COO": consolidated_coo,
                "BestCost": f"{best_cost_option['cost']:.2f}" if best_cost_option['cost'] != np.inf else "N/A",
                "BestCostLT": f"{best_cost_option['lead_time']:.0f}" if best_cost_option['lead_time'] != np.inf else "N/A",
                "BestCostSrc": best_cost_option['source'],
                "FastestLT": f"{fastest_lt_option['lead_time']:.0f}" if fastest_lt_option['lead_time'] != np.inf else "N/A",
                "FastestCost": f"{fastest_lt_option['cost']:.2f}" if fastest_lt_option['cost'] != np.inf else "N/A",
                "FastestLTSrc": fastest_lt_option['source'],
                "TariffPct": f"{best_cost_option.get('tariff_rate', np.nan) * 100:.1f}" if best_cost_option and not pd.isna(best_cost_option.get('tariff_rate')) else "N/A",
                "Notes": notes
            }

        # Return list containing the single consolidated entry for the GUI
        return [analysis_entry], historical_entries, part_summary

    def run_analysis_thread(self, config):
        """The function that runs in a separate thread to perform analysis."""
        try:
             self.running_analysis = True
             self.root.after(0, self.validate_inputs)
             self.root.after(0, self.update_status, "Starting analysis...", "info")
             self.root.after(0, self.update_progress, 0, len(self.bom_df), "Initializing...")

             if API_KEYS["DigiKey"]:
                 self.root.after(0, self.update_status, "Checking DigiKey token...", "info")
                 access_token = self.get_digikey_token()
                 if not access_token:
                      logger.error("Analysis cancelled: Failed to obtain DigiKey token.")
                      self.root.after(0, self.update_status, "Analysis cancelled: DigiKey Auth Failed.", "error")
                      self.root.after(0, messagebox.showerror, "Auth Error", "Could not get DigiKey token. Analysis cancelled.")
                      return
                 else:
                      self.root.after(0, self.update_status, "DigiKey token OK.", "info")

             self.root.after(0, self.clear_treeview, self.tree)
             self.root.after(0, self.clear_treeview, self.analysis_table)
             self.part_data_cache.clear() # Clear API cache for new run
             self._hts_cache = {} # Clear HTS cache
             all_analysis_entries = []
             all_historical_entries = []
             all_part_summaries = []

             # Process each BOM line
             for i, row in self.bom_df.iterrows():
                 if not self.running_analysis:
                      logger.info("Analysis run cancelled by user.")
                      return

                 bom_pn = row['Part Number']
                 bom_mfg = row.get('Manufacturer', '')
                 bom_qty_per_unit = row['Quantity']
                 if pd.isna(bom_qty_per_unit) or bom_qty_per_unit <= 0 : # Added check for <=0
                     logger.warning(f"Skipping BOM row {i+1} ({bom_pn}) due to invalid quantity.")
                     self.root.after(0, self.update_progress, i + 1, len(self.bom_df), f"Skipped {bom_pn}")
                     continue
                 total_qty_needed = int(bom_qty_per_unit * config['total_units']) # Ensure int

                 self.root.after(0, self.update_progress, i, len(self.bom_df), f"Processing {bom_pn}...")

                 # analyze_single_part now populates self.part_data_cache internally
                 part_analysis_rows, part_historical_rows, part_summary = self.analyze_single_part(
                     bom_pn, bom_mfg, total_qty_needed, config
                 )

                 if not self.running_analysis:
                      logger.info("Analysis run cancelled during part processing.")
                      return

                 all_analysis_entries.extend(part_analysis_rows)
                 all_historical_entries.extend(part_historical_rows)
                 if part_summary and part_summary.get('options'): # Only add summaries with valid options
                    all_part_summaries.append(part_summary)


             if not self.running_analysis: return

             # --- Post-Processing ---
             self.root.after(0, self.update_progress, len(self.bom_df), len(self.bom_df), "Aggregating...")

             # 1. Populate Main Treeview
             self.root.after(0, self.populate_treeview, self.tree, all_analysis_entries)

             # 2. Save Historical Data
             if all_historical_entries:
                  append_to_csv(HISTORICAL_DATA_FILE, all_historical_entries)
                  logger.info(f"Appended {len(all_historical_entries)} rows to {HISTORICAL_DATA_FILE}")
                  try:
                       # Reload historical data after append (careful with types)
                       self.historical_data_df = pd.read_csv(HISTORICAL_DATA_FILE, dtype=str, keep_default_na=False)
                       if 'Fetch_Timestamp' not in self.historical_data_df.columns:
                            raise KeyError("'Fetch_Timestamp' column missing in historical CSV header after reload.")
                       self.historical_data_df['Fetch_Timestamp'] = pd.to_datetime(self.historical_data_df['Fetch_Timestamp'], errors='coerce')
                       numeric_cols_hist = ['Lead_Time_Days', 'Cost', 'Inventory', 'Stock_Probability']
                       for col in numeric_cols_hist:
                         if col in self.historical_data_df.columns: 
                           self.historical_data_df[col] = pd.to_numeric(self.historical_data_df[col], errors='coerce')
                         else: logger.warning(f"Reloaded Historical data missing expected numeric column: {col}")
                       logger.info("Reloaded historical data after append.")
                  except KeyError as e: # Catch KeyError specifically
                       logger.error(f"Failed to reload historical data - Column Missing: {e}. Check CSV header matches self.hist_header.")
                  except Exception as e:
                       logger.error(f"Failed to reload historical data after append: {e}")

             # 3. Calculate Summary Metrics (also stores strategies in self.strategies_for_export)
             summary_metrics = self.calculate_summary_metrics(all_part_summaries, config)
             logger.debug(f"Calculated Summary Metrics for Table: {summary_metrics}")
            
             # 4. Populate Analysis Summary Table
             self.root.after(0, self.populate_treeview, self.analysis_table, summary_metrics)

             # 5. Store results for AI summary AND Export
             self.analysis_results = {
                  "config": config,
                  "part_summaries": all_part_summaries,
                  # Get strategies stored by calculate_summary_metrics
                  "strategies": getattr(self, 'strategies_for_export', {}),
                  "summary_metrics": summary_metrics # Store list of tuples directly
             }
             # Enable export buttons after results are stored
             self.root.after(0, self.update_export_buttons_state) # Schedule button update

             self.root.after(0, self.update_status, "Analysis complete.", "success")
             self.root.after(0, self.update_progress, len(self.bom_df), len(self.bom_df), "Done")

        except Exception as e:
             logger.error(f"Analysis thread failed: {e}", exc_info=True)
             self.root.after(0, self.update_status, f"Analysis Error: {e}", "error")
             self.root.after(0, messagebox.showerror, "Analysis Error", f"An error occurred during analysis:\n\n{e}")
        finally:
             self.running_analysis = False
             # Schedule button state update on main thread
             self.root.after(0, self.validate_inputs) # Ensures main buttons reset
             # Also ensure export buttons reset if analysis fails early
             # Check analysis_results again in the helper function update_export_buttons_state
             self.root.after(0, self.update_export_buttons_state)

    def calculate_summary_metrics(self, part_summaries, config):
        """
        Calculates aggregate BOM metrics and stores detailed strategy options for export.
        Runs in background thread. Expects part_summaries to be a list of dicts,
        where each dict has 'bom_pn', 'bom_mfg', 'bom_qty_needed', and 'options' keys.
        The 'options' list should contain detailed dictionaries for each supplier option,
        including calculated costs/lead times and identifiers like Manufacturer, MPN, SourcePN.
        """
        logger.debug(f"Starting calculate_summary_metrics for {len(part_summaries)} parts.")
        summary_data = [] # For the GUI summary table (list of tuples)
        total_bom_cost_min = 0.0
        total_bom_cost_max = 0.0
        total_bom_cost_fastest = 0.0
        max_lead_time_min_cost_strategy = 0 # Max lead time when selecting cheapest for each part
        max_lead_time_fastest_strategy = 0 # Max lead time when selecting fastest for each part
        total_bom_cost_sweet_spot = 0.0
        max_lead_time_sweet_spot = 0

        clear_to_build = True
        clear_to_build_issues = []
        parts_with_issues = 0
        total_parts = len(part_summaries) or 1 # Avoid division by zero if empty

        # --- Initialize Strategy Dictionaries ---
        # These will store {bom_pn: chosen_option_dict} for each strategy
        # The chosen_option_dict comes from part_summary['options'] created in analyze_single_part
        cheapest_strategy = {}
        fastest_strategy = {}
        sweet_spot_strategy = {}

        parts_processed_count = 0
        # --- Iterate through parts ---
        for i, summary in enumerate(part_summaries):
            parts_processed_count += 1
            bom_pn = summary.get('bom_pn', f'UnknownPart_{i}') # Use get for safety
            qty_needed = summary.get('bom_qty_needed', 0)
            options = summary.get('options', []) # Use get with default empty list

            # Ensure options is a list before proceeding
            if not isinstance(options, list):
                 logger.error(f"Invalid 'options' format for {bom_pn}: type {type(options)}. Skipping part.")
                 clear_to_build = False; parts_with_issues += 1
                 total_bom_cost_min=np.nan; total_bom_cost_max=np.nan; total_bom_cost_fastest=np.nan; total_bom_cost_sweet_spot=np.nan
                 continue

            if not options:
                logger.warning(f"No supplier options found for {bom_pn} in summary.")
                clear_to_build = False
                clear_to_build_issues.append(f"{bom_pn}: No suppliers found.")
                parts_with_issues += 1
                total_bom_cost_min=np.nan; total_bom_cost_max=np.nan; total_bom_cost_fastest=np.nan; total_bom_cost_sweet_spot=np.nan
                continue # Skip calculations for this part

            # --- Per-Part Calculations & Strategy Storage ---

            # Cheapest option for *this* part
            # Ensure sorting handles np.inf correctly (it usually does, inf sorts last)
            options.sort(key=lambda x: (x.get('cost', np.inf), x.get('lead_time', np.inf)))
            cheapest_option_part = options[0] if options else {'cost': np.inf, 'lead_time': np.inf}
            min_cost_part = cheapest_option_part.get('cost', np.inf)
            # Store the *entire chosen option dictionary* for export
            cheapest_strategy[bom_pn] = cheapest_option_part
            logger.debug(f"Storing Cheapest Option for {bom_pn}: {cheapest_strategy[bom_pn]}")

            # Aggregate total min cost
            if min_cost_part != np.inf and not pd.isna(total_bom_cost_min):
                 total_bom_cost_min += min_cost_part
            else: total_bom_cost_min = np.nan # If any part fails, total fails

            # Track max lead time for the cheapest strategy
            max_lead_time_min_cost_strategy = max(max_lead_time_min_cost_strategy, cheapest_option_part.get('lead_time', np.inf))

            # Max cost for *this* part
            valid_costs = [opt.get('cost', np.inf) for opt in options if opt.get('cost', np.inf) != np.inf]
            max_cost_part = max(valid_costs) if valid_costs else np.nan
            if not pd.isna(max_cost_part) and not pd.isna(total_bom_cost_max):
                 total_bom_cost_max += max_cost_part
            else: total_bom_cost_max = np.nan

            # Fastest option for *this* part
            options.sort(key=lambda x: (x.get('lead_time', np.inf), x.get('cost', np.inf)))
            fastest_option_part = options[0] if options else {'cost': np.inf, 'lead_time': np.inf}
            # Store the *entire chosen option dictionary* for export
            fastest_strategy[bom_pn] = fastest_option_part
            logger.debug(f"Storing Fastest Option for {bom_pn}: {fastest_strategy[bom_pn]}")

            # Aggregate total fastest cost
            fastest_cost_part = fastest_option_part.get('cost', np.inf)
            if fastest_cost_part != np.inf and not pd.isna(total_bom_cost_fastest):
                 total_bom_cost_fastest += fastest_cost_part
            else: total_bom_cost_fastest = np.nan

            # Track max lead time for the fastest strategy
            max_lead_time_fastest_strategy = max(max_lead_time_fastest_strategy, fastest_option_part.get('lead_time', np.inf))

            # --- Clear to Build Check ---
            stock_available = any(opt.get('stock', 0) >= qty_needed for opt in options)
            if not stock_available:
                clear_to_build = False
                min_lead_no_stock = min((opt.get('lead_time', np.inf) for opt in options if opt.get('stock', 0) < qty_needed), default=np.inf)
                issue = f"{bom_pn}: Insufficient stock ({qty_needed} needed). "
                issue += f"Min lead {min_lead_no_stock:.0f}d." if min_lead_no_stock != np.inf else "No LT info."
                clear_to_build_issues.append(issue)
                parts_with_issues += 1

            # --- Sweet Spot Calculation ---
            best_score = np.inf
            chosen_option_ss = None
            viable_options = []

            # Check if min_cost_part is valid before proceeding with sweet spot
            if min_cost_part == np.inf:
                 logger.warning(f"Sweet Spot: Skipping {bom_pn} as min cost is invalid (infinity).")
                 total_bom_cost_sweet_spot = np.nan # Cannot calculate sweet spot
                 # Store fallback (cheapest, which is also invalid) with note
                 sweet_spot_strategy[bom_pn] = {**cheapest_option_part, 'sweet_spot_score': 'N/A (Invalid Base Cost)'}
            else:
                # Find viable options meeting constraints
                for option in options: # Iterate through original options again
                     cost = option.get('cost', np.inf)
                     lead_time = option.get('lead_time', np.inf)
                     if cost == np.inf or lead_time == np.inf: continue # Skip options with no cost or lead time

                     cost_premium = (cost - min_cost_part) / min_cost_part if min_cost_part > 0 else 0
                     if cost_premium > config['max_premium']: continue
                     if lead_time > config['target_lead_time_days']: continue
                     viable_options.append(option)

                if not viable_options:
                     logger.warning(f"Sweet Spot: No option met constraints for {bom_pn}.")
                     if not pd.isna(total_bom_cost_sweet_spot): # Only mark as NaN if not already failed
                         total_bom_cost_sweet_spot = np.nan
                     # Store fallback (cheapest) with note
                     sweet_spot_strategy[bom_pn] = {**cheapest_option_part, 'sweet_spot_score': 'N/A (Constraints Failed)'}
                     logger.debug(f"Storing Sweet Spot FALLBACK Option for {bom_pn}: {sweet_spot_strategy[bom_pn]}")
                else:
                     # Calculate scores for viable options
                     min_viable_cost = min(opt['cost'] for opt in viable_options)
                     max_viable_cost = max(opt['cost'] for opt in viable_options) # Max cost among viable
                     min_viable_lt = min(opt['lead_time'] for opt in viable_options)
                     max_viable_lt = max(opt['lead_time'] for opt in viable_options) # Max lead time among viable
                     cost_range = max_viable_cost - min_viable_cost
                     lead_range = max_viable_lt - min_viable_lt

                     for option in viable_options:
                         cost = option['cost']
                         lead_time = option['lead_time']
                         stock = option.get('stock', 0)
                         norm_cost = (cost - min_viable_cost) / cost_range if cost_range > 0 else 0
                         norm_lead = (lead_time - min_viable_lt) / lead_range if lead_range > 0 else 0
                         score = (config['cost_weight'] * norm_cost) + (config['lead_time_weight'] * norm_lead)

                         if option.get('discontinued') or option.get('eol'): score += 0.5
                         if stock < qty_needed: score += 0.1

                         if score < best_score:
                              best_score = score
                              chosen_option_ss = option

                     if chosen_option_ss and not pd.isna(total_bom_cost_sweet_spot):
                         total_bom_cost_sweet_spot += chosen_option_ss['cost']
                         max_lead_time_sweet_spot = max(max_lead_time_sweet_spot, chosen_option_ss['lead_time'])
                         # Store the *entire chosen option dictionary* with score
                         sweet_spot_strategy[bom_pn] = {**chosen_option_ss, 'sweet_spot_score': f"{best_score:.3f}"}
                         logger.debug(f"Storing Sweet Spot Option for {bom_pn}: {sweet_spot_strategy[bom_pn]}")
                     elif not chosen_option_ss:
                          logger.error(f"Sweet Spot: Logic error - Failed to select viable option for {bom_pn}")
                          total_bom_cost_sweet_spot = np.nan
                          # Store fallback if something went wrong during selection
                          sweet_spot_strategy[bom_pn] = {**cheapest_option_part, 'sweet_spot_score': 'N/A (Selection Error)'}


        # --- Store Strategies for Export ---
        # Assign the collected dictionaries to an instance variable accessible by other methods
        self.strategies_for_export = {
             "Cheapest": cheapest_strategy,
             "Fastest": fastest_strategy,
             "Sweet Spot": sweet_spot_strategy,
        }
        logger.debug(f"Strategies stored. Keys: {list(self.strategies_for_export.keys())}")


        # --- Format Summary Data for Table ---
        summary_list = []
        summary_list.append(("Total Parts Analyzed", f"{parts_processed_count}"))
        ctb_value = f"{clear_to_build} ({total_parts - parts_with_issues}/{total_parts} parts OK)"
        summary_list.append(("Clear to Build (Stock Check)", ctb_value))
        if clear_to_build_issues:
             issues_str = "; ".join(clear_to_build_issues)
             summary_list.append(("Clear to Build Issues", issues_str[:500] + ('...' if len(issues_str) > 500 else '')))

        summary_list.append(("Min/Max Possible BOM Cost", f"${total_bom_cost_min:.2f} / ${total_bom_cost_max:.2f}" if not pd.isna(total_bom_cost_min) else "N/A"))

        # Calculate cheapest strategy totals safely using the stored strategy
        cheapest_total_cost = sum(v.get('cost', np.inf) for v in cheapest_strategy.values())
        cheapest_max_lt_actual = max((v.get('lead_time', np.inf) for v in cheapest_strategy.values()), default=np.inf)
        summary_list.append(("Cheapest BOM Strategy Cost / Max Lead", f"${cheapest_total_cost:.2f} / {cheapest_max_lt_actual:.0f} days" if cheapest_total_cost != np.inf else "N/A"))

        # Calculate fastest strategy totals safely using the stored strategy
        fastest_total_cost = sum(v.get('cost', np.inf) for v in fastest_strategy.values())
        # max_lead_time_fastest_strategy should already be calculated correctly above
        summary_list.append(("Fastest BOM Strategy Cost / Max Lead", f"${fastest_total_cost:.2f} / {max_lead_time_fastest_strategy:.0f} days" if fastest_total_cost != np.inf else "N/A"))


        if not pd.isna(total_bom_cost_sweet_spot):
             # Ensure max_lead_time_sweet_spot is finite if cost is finite
             if total_bom_cost_sweet_spot != np.inf and max_lead_time_sweet_spot == np.inf : max_lead_time_sweet_spot = 0 # Adjust if needed
             summary_list.append(("Sweet Spot BOM Cost / Max Lead", f"${total_bom_cost_sweet_spot:.2f} / {max_lead_time_sweet_spot:.0f} days"))
        else:
             summary_list.append(("Sweet Spot BOM Cost / Max Lead", "N/A (Constraints Failed or Invalid Data)"))

        # --- Tariff Calculation ---
        total_tariff_cost = 0.0
        calculated_bom_cost_for_tariff = 0.0
        # Choose which strategy's tariff to show (Sweet Spot if valid, else Cheapest)
        chosen_strategy_for_tariff = sweet_spot_strategy if not pd.isna(total_bom_cost_sweet_spot) else cheapest_strategy
        tariff_basis_name = "Sweet Spot" if not pd.isna(total_bom_cost_sweet_spot) else "Cheapest"

        for bom_pn, chosen_option in chosen_strategy_for_tariff.items():
            part_cost_basis = chosen_option.get('cost', np.inf)
            if chosen_option and part_cost_basis != np.inf:
                calculated_bom_cost_for_tariff += part_cost_basis
                # Use the tariff_rate stored in the option dict
                tariff_rate = chosen_option.get('tariff_rate') # Retrieve stored rate
                if tariff_rate is not None and not pd.isna(tariff_rate):
                     total_tariff_cost += part_cost_basis * tariff_rate

        total_tariff_pct = (total_tariff_cost / calculated_bom_cost_for_tariff * 100) if calculated_bom_cost_for_tariff > 0 else 0.0
        summary_list.append((f"Est. Total Tariff Cost ({tariff_basis_name})", f"${total_tariff_cost:.2f}"))
        summary_list.append((f"Est. Total Tariff % ({tariff_basis_name})", f"{total_tariff_pct:.2f}%"))

        logger.debug(f"Finished calculate_summary_metrics. Summary list: {summary_list}")
        return summary_list # Return list of tuples for the GUI table

    def validate_and_run_analysis(self):
        """Validates inputs and starts the analysis thread."""
        if self.running_analysis:
            messagebox.showwarning("Busy", "Analysis is already in progress.")
            return
        if not self.validate_inputs(): # Validation itself updates GUI, no need for root.after
            messagebox.showerror("Invalid Config", "Please fix the configuration errors before running analysis.")
            return
        if self.bom_df is None:
            messagebox.showerror("No BOM", "Please load a BOM file first.")
            return

        # Get config values safely
        try:
             config = {
                 "total_units": int(safe_float(self.config_vars["total_units"].get())),
                 "max_premium": safe_float(self.config_vars["max_premium"].get(), default=0) / 100.0,
                 "target_lead_time_days": int(safe_float(self.config_vars["target_lead_time_days"].get(), default=0)),
                 "cost_weight": safe_float(self.config_vars["cost_weight"].get(), default=0.5),
                 "lead_time_weight": safe_float(self.config_vars["lead_time_weight"].get(), default=0.5),
                 "custom_tariff_rates": {} # Initialize empty
             }
             # Process tariffs safely
             for country, entry in self.tariff_entries.items():
                 rate_str = entry.get()
                 if rate_str:
                      rate = safe_float(rate_str)
                      if rate is not None and rate >= 0:
                           config["custom_tariff_rates"][country] = rate / 100.0
                      else: # Invalid rate entered
                           raise ValueError(f"Invalid tariff rate for {country}: '{rate_str}'")

        except ValueError as e:
            messagebox.showerror("Config Error", f"Invalid value in configuration: {e}")
            return
        except Exception as e:
             messagebox.showerror("Config Error", f"Unexpected error reading configuration: {e}")
             logger.error("Unexpected config error", exc_info=True)
             return

        # Start the analysis in a separate thread
        logger.info("Submitting analysis task to thread pool.")
        self.thread_pool.submit(self.run_analysis_thread, config) # Pass config dict


    # --- Predictive Analysis (Prophet, RAG Mock) ---

    def run_prophet(self, component_historical_data, metric='Lead_Time_Days', periods=90, min_data_points=5): # Added min_data_points
        """Runs Prophet forecasting with outlier filtering. Runs in background thread."""
        if component_historical_data.empty: return None, None
        if metric not in component_historical_data.columns: return None, None

        df_prophet = component_historical_data[['Fetch_Timestamp', metric]].copy()
        df_prophet = df_prophet.dropna(subset=[metric])
        df_prophet.rename(columns={'Fetch_Timestamp': 'ds', metric: 'y'}, inplace=True)

        # --- Minimum Data Point Check ---
        if len(df_prophet) < min_data_points:
            logger.warning(f"Prophet ({metric}): Insufficient data points ({len(df_prophet)} < {min_data_points}). Skipping forecast.")
            return None, None
        # ---

        df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], errors='coerce')
        df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
        df_prophet = df_prophet.dropna(subset=['ds', 'y'])

        # --- Remove Timezone (Keep this) ---
        if pd.api.types.is_datetime64_any_dtype(df_prophet['ds']) and df_prophet['ds'].dt.tz is not None:
            logger.debug("Removing timezone info from 'ds' column for Prophet.")
            df_prophet['ds'] = df_prophet['ds'].dt.tz_localize(None)
        # ---

        # --- Basic Outlier Filtering (using IQR) ---
        q1 = df_prophet['y'].quantile(0.25)
        q3 = df_prophet['y'].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        initial_rows = len(df_prophet)
        df_prophet = df_prophet[(df_prophet['y'] >= lower_bound) & (df_prophet['y'] <= upper_bound)]
        removed_rows = initial_rows - len(df_prophet)
        if removed_rows > 0:
             logger.info(f"Prophet ({metric}): Removed {removed_rows} potential outliers using IQR filter.")
        # --- End Outlier Filtering ---

        # Check again after filtering/cleaning
        if len(df_prophet) < min_data_points:
             logger.warning(f"Prophet ({metric}): Insufficient data points after filtering ({len(df_prophet)} < {min_data_points}). Skipping forecast.")
             return None, None


        try:
            # Initialize and fit Prophet model
            model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                           changepoint_prior_scale=0.05) # Default is 0.05
            model.fit(df_prophet)

            # Create future dataframe and predict
            future = model.make_future_dataframe(periods=periods)
            forecast = model.predict(future)

            # Get the latest prediction
            latest_forecast = forecast.iloc[-1]
            predicted_value = latest_forecast['yhat']
            # Using bounds from forecast output is usually better than recalculating IQR
            lower_bound_pred = latest_forecast['yhat_lower']
            upper_bound_pred = latest_forecast['yhat_upper']

            # Apply reasonable bounds to final prediction
            if metric == 'Lead_Time_Days':
                predicted_value = max(0, predicted_value) # Cannot be negative
                lower_bound_pred = max(0, lower_bound_pred)
                upper_bound_pred = max(0, upper_bound_pred)
            elif metric == 'Cost':
                predicted_value = max(0.001, predicted_value) # Cost > 0
                lower_bound_pred = max(0.001, lower_bound_pred)
                upper_bound_pred = max(0.001, upper_bound_pred)

            logger.debug(f"Prophet forecast for {metric}: {predicted_value:.2f} [{lower_bound_pred:.2f}-{upper_bound_pred:.2f}]")
            # Return prediction and confidence interval range
            return predicted_value, (lower_bound_pred, upper_bound_pred)

        except Exception as e:
            logger.error(f"Prophet forecasting failed for {metric}: {e}", exc_info=False) # Less verbose log
            return None, None


    def run_rag_mock(self, prophet_lead, prophet_cost, stock_prob, context=""):
        """Mock RAG function. Runs in background thread."""
        rag_lead_range, rag_cost_range = "N/A", "N/A"
        adj_stock_prob = stock_prob

        context_lower = context.lower()
        has_issues = "shortage" in context_lower or "delay" in context_lower or "constrained" in context_lower

        if prophet_lead is not None:
            lead_min = prophet_lead
            lead_max = prophet_lead + 7 # Base variability +/- 1 week maybe?
            if has_issues:
                lead_min += 7
                lead_max += 21 # Wider upper range if issues
                adj_stock_prob *= 0.7
            rag_lead_range = f"{max(0, lead_min):.0f}-{max(0, lead_max):.0f}"

        if prophet_cost is not None:
             cost_min = prophet_cost * 0.98 # Allow slight decrease
             cost_max = prophet_cost * 1.05 # Base variability
             if has_issues or "increase" in context_lower:
                  cost_min = prophet_cost # Min unlikely to decrease if issues
                  cost_max = prophet_cost * 1.20 # Larger potential increase
             rag_cost_range = f"{max(0.001, cost_min):.3f}-{max(0.001, cost_max):.3f}"

        return rag_lead_range, rag_cost_range, round(max(0, min(100, adj_stock_prob)), 1)


    def run_ai_comparison(self, prophet_lead, prophet_cost, rag_lead_range, rag_cost_range, stock_prob):
        """Mock AI comparison. Runs in background thread."""
        ai_lead, ai_cost = prophet_lead, prophet_cost # Defaults
        ai_stock_prob = stock_prob

        try: # Safely parse RAG ranges
            if rag_lead_range != "N/A":
                 lead_parts = [safe_float(p) for p in rag_lead_range.split('-')]
                 if len(lead_parts) == 2 and not any(pd.isna(p) for p in lead_parts):
                     rag_mid_lead = (lead_parts[0] + lead_parts[1]) / 2
                     if prophet_lead is not None: ai_lead = (prophet_lead + rag_mid_lead) / 2
                     else: ai_lead = rag_mid_lead # Use RAG if Prophet failed
        except: pass

        try:
            if rag_cost_range != "N/A":
                 cost_parts = [safe_float(p) for p in rag_cost_range.split('-')]
                 if len(cost_parts) == 2 and not any(pd.isna(p) for p in cost_parts):
                      rag_mid_cost = (cost_parts[0] + cost_parts[1]) / 2
                      if prophet_cost is not None: ai_cost = (prophet_cost + rag_mid_cost) / 2
                      else: ai_cost = rag_mid_cost # Use RAG if Prophet failed
        except: pass

        # Bounds
        ai_lead = max(0, ai_lead) if ai_lead is not None else None
        ai_cost = max(0.001, ai_cost) if ai_cost is not None else None

        return ai_lead, ai_cost, ai_stock_prob # Stock prob just passed through


    def run_predictive_analysis_gui(self):
        """Handles 'Run Predictions' button. Schedules background task."""
        if self.running_analysis:
            messagebox.showwarning("Busy", "Analysis/Prediction is already in progress.")
            return
        if self.historical_data_df is None or self.historical_data_df.empty or 'Component' not in self.historical_data_df.columns:
            messagebox.showerror("No Data", "No historical data available. Run analysis first.")
            logger.error("Prediction start check: Historical data invalid or missing 'Component' column.")
            return

        # TODO: Add simple input dialog for context?
        context = "Current market conditions" # Placeholder

        self.running_analysis = True
        self.root.after(0, self.validate_inputs) # Disable buttons
        self.root.after(0, self.update_status, "Generating predictions...", "info") # Schedule GUI update
        logger.info("Submitting prediction task to thread pool.")
        self.thread_pool.submit(self.run_predictive_analysis_thread, context)


    def run_predictive_analysis_thread(self, context):
        """Thread function to generate predictions."""
        try:
            if self.historical_data_df is None or self.historical_data_df.empty or 'Component' not in self.historical_data_df.columns:
                 logger.error("Prediction thread: Historical data invalid or missing 'Component' column.")
                 self.root.after(0, self.update_status, "Prediction Error: No valid historical data.", "error")
                 return

            # Ensure 'Component' column exists and handle potential NaN components
            if 'Component' not in self.historical_data_df.columns:
                 logger.error("Prediction thread: 'Component' column missing in historical data.")
                 return
            unique_components = self.historical_data_df['Component'].dropna().unique()

            total_comps = len(unique_components)
            if total_comps == 0:
                 logger.warning("Prediction thread: No valid components found in historical data.")
                 return

            self.root.after(0, self.update_progress, 0, total_comps, "Predicting...") # Schedule GUI update
            new_predictions = []

            for i, component in enumerate(unique_components):
                 if not self.running_analysis: # Check for cancellation
                      logger.info("Prediction run cancelled.")
                      return

                 self.root.after(0, self.update_progress, i, total_comps, f"Predicting {component[:30]}...") # Schedule GUI update

                 component_data = self.historical_data_df[self.historical_data_df['Component'] == component].copy()
                 if component_data.empty: continue

                 # Sort by date to get latest stock probability reliably
                 component_data = component_data.sort_values('Fetch_Timestamp', ascending=False)
                 latest_stock_prob = component_data['Stock_Probability'].iloc[0] if not component_data.empty and 'Stock_Probability' in component_data.columns and not pd.isna(component_data['Stock_Probability'].iloc[0]) else 0.0

                 # Run predictions (these run in the background thread)
                 prophet_lead, _ = self.run_prophet(component_data, 'Lead_Time_Days')
                 prophet_cost, _ = self.run_prophet(component_data, 'Cost')
                 rag_lead, rag_cost, rag_stock_prob = self.run_rag_mock(prophet_lead, prophet_cost, latest_stock_prob, context)
                 ai_lead, ai_cost, ai_stock_prob = self.run_ai_comparison(prophet_lead, prophet_cost, rag_lead, rag_cost, rag_stock_prob)
                 # --- Sanitize Component Name ---
                 cleaned_component = str(component).replace('\n', ' ').replace('\r', '').replace('"', "'") # Replace newlines, quotes
    # ---
 
                 # Format prediction record
                 pred_row = [
                     cleaned_component, datetime.now().strftime('%Y-%m-%d'),
                     f"{prophet_lead:.1f}" if prophet_lead is not None else "N/A",
                     f"{prophet_cost:.3f}" if prophet_cost is not None else "N/A",
                     rag_lead, rag_cost,
                     f"{ai_lead:.1f}" if ai_lead is not None else "N/A",
                     f"{ai_cost:.3f}" if ai_cost is not None else "N/A",
                     f"{ai_stock_prob:.1f}",
                     "", "", "", "", "", "", "", "", "" # Blanks for Human/Real/Acc
                 ]
                 new_predictions.append(pred_row)

            if not self.running_analysis: return # Check again before writing

            # Append new predictions to file (can run in background)
            if new_predictions:
                 append_to_csv(PREDICTION_FILE, new_predictions)
                 logger.info(f"Appended {len(new_predictions)} predictions to {PREDICTION_FILE}")
                 # Schedule reload and GUI update on main thread
                 self.root.after(0, self.load_predictions_to_gui)

            self.root.after(0, self.update_status, "Predictive analysis complete.", "success") # Schedule GUI update
            self.root.after(0, self.update_progress, total_comps, total_comps, "Done") # Schedule GUI update

        except Exception as e:
            logger.error(f"Predictive analysis thread failed: {e}", exc_info=True)
            self.root.after(0, self.update_status, f"Prediction Error: {e}", "error") # Schedule GUI update
            self.root.after(0, messagebox.showerror, "Prediction Error", f"An error occurred during prediction:\n\n{e}") # Schedule GUI update
        finally:
            self.running_analysis = False
            self.root.after(0, self.validate_inputs) # Schedule button state update


    def load_predictions_to_gui(self):
         """Loads data from prediction CSV into the GUI table. Runs on main thread."""
         self.clear_treeview(self.predictions_tree) # Safe to call directly if this runs on main thread
         try:
              # Reload from file
              self.predictions_df = pd.read_csv(PREDICTION_FILE, dtype=str, keep_default_na=False, on_bad_lines='warn') # Keep blanks
              # Convert types needed for display logic, maybe not needed if just displaying strings
              # self.predictions_df['Date'] = pd.to_datetime(self.predictions_df['Date'], format='%Y-%m-%d', errors='coerce')

              # Fill NaN/empty strings properly before inserting
              # df_display = self.predictions_df.fillna('') # Use fillna('') instead of keep_default_na=False? Test this.
              df_display = self.predictions_df

              for _, row in df_display.iterrows():
                   # Ensure correct number of values matches columns
                   values = [str(row.get(col, '')) for col in self.pred_header] # Ensure string conversion
                   self.predictions_tree.insert("", "end", values=values)
              logger.info(f"Loaded {len(df_display)} predictions into GUI.")
         except FileNotFoundError:
              logger.info(f"{PREDICTION_FILE} not found. No predictions loaded.")
         except Exception as e:
              logger.error(f"Failed to load or display predictions: {e}", exc_info=True)
              messagebox.showerror("Load Error", f"Could not load predictions from {PREDICTION_FILE}:\n\n{e}")


    # --- AI Summary ---
    def generate_ai_summary_gui(self):
        """Handles 'AI Summary' button. Schedules background task."""
        if self.running_analysis:
            messagebox.showwarning("Busy", "Analysis/Prediction is already in progress.")
            return
        # Check if analysis_results dict is populated
        if not self.analysis_results or not self.analysis_results.get("part_summaries"):
             messagebox.showinfo("No Data", "Please run the main analysis first.")
             return
        if not API_KEYS["OpenAI"]:
             messagebox.showwarning("No API Key", "OpenAI API key is not configured.")
             return

        self.running_analysis = True
        self.root.after(0, self.validate_inputs) # Disable buttons
        self.root.after(0, self.update_status, "Generating AI summary...", "info") # Schedule GUI update
        logger.info("Submitting AI summary task to thread pool.")
        self.thread_pool.submit(self.generate_ai_summary_thread)

    def generate_ai_summary_thread(self):
        """Thread function to call OpenAI."""
        try:
            config = self.analysis_results.get("config", {})
            part_summaries = self.analysis_results.get("part_summaries", [])
            # Ensure summary_metrics is a list of tuples before converting
            summary_metrics_list = self.analysis_results.get("summary_metrics", [])
            if not isinstance(summary_metrics_list, list): summary_metrics_list = []
            summary_metrics_dict = dict(summary_metrics_list)

            # --- Build Prompt ---
            prompt = f"Analyze the following BOM analysis results for a build of {config.get('total_units', 'N/A')} units. Provide a concise executive summary, key risks, and actionable recommendations for a supply chain professional.\n\n"

            prompt += "--- Analysis Configuration ---\n"
            prompt += f"- Max Cost Premium: {config.get('max_premium', 0)*100:.1f}%\n"
            prompt += f"- Target Lead Time: {config.get('target_lead_time_days', 0)} days\n"
            prompt += f"- Weights (Cost/LT): {config.get('cost_weight', 0):.2f} / {config.get('lead_time_weight', 0):.2f}\n\n"

            prompt += "--- Overall Summary Metrics ---\n"
            for metric, value in summary_metrics_list: # Use the list directly
                prompt += f"- {metric}: {value}\n"
            prompt += "\n"

            prompt += "--- Key Part Details & Potential Issues (Limit to ~10 most critical) ---\n"
            critical_parts_info = []
            issues_count = 0
            MAX_PARTS_IN_PROMPT = 10

            for part_sum in part_summaries:
                if issues_count >= MAX_PARTS_IN_PROMPT: break

                bom_pn = part_sum['bom_pn']
                options = part_sum['options']
                part_prompt = f"- {bom_pn} (Need {part_sum['bom_qty_needed']}): "
                part_issues = []

                if not options:
                    part_prompt += "**CRITICAL - No suppliers found.**"
                    part_issues.append("No Source") # Flag for sorting
                else:
                    best_cost_opt = min(options, key=lambda x: x['cost'])
                    fastest_lt_opt = min(options, key=lambda x: x['lead_time'])
                    stocked_options = [o for o in options if o['stock'] >= part_sum['bom_qty_needed']]

                    if not stocked_options:
                        part_issues.append(f"NO STOCK (Min LT {fastest_lt_opt['lead_time']:.0f}d)")
                    # else: part_prompt += "Stock OK. " # Can omit for brevity

                    # Lead Time Issue
                    # Check lead time of sweet spot or cheapest if sweet spot failed
                    sweet_spot_part = next((d for d in self.analysis_results.get("sweet_spot_details", []) if d["bom_pn"] == bom_pn), None)
                    relevant_lead_time = np.inf
                    if sweet_spot_part: relevant_lead_time = sweet_spot_part['chosen_lead_time']
                    elif best_cost_opt: relevant_lead_time = best_cost_opt['lead_time']

                    if relevant_lead_time > config.get('target_lead_time_days', 90): # Use 90 day fallback
                        part_issues.append(f"LONG LEAD ({relevant_lead_time:.0f}d > {config.get('target_lead_time_days', 90)}d)")

                    # EOL/Discontinued Issue
                    if any(o.get('discontinued') or o.get('eol') for o in options):
                         part_issues.append("EOL/DISC RISK")

                    # Add basic cost/lt range
                    cost_range = f"${best_cost_opt['cost']:.2f}-${max(o['cost'] for o in options):.2f}" if best_cost_opt['cost']!=np.inf else "N/A"
                    lt_range = f"{fastest_lt_opt['lead_time']:.0f}d-{max(o['lead_time'] for o in options if o['lead_time'] != np.inf):.0f}d" if fastest_lt_opt['lead_time']!=np.inf else "N/A"
                    part_prompt += f"Cost {cost_range}, LT {lt_range}. "

                if part_issues:
                     part_prompt += "**Issues:** " + "; ".join(part_issues)
                     critical_parts_info.append((len(part_issues), part_prompt)) # Sort by number of issues
                     issues_count += 1
                # Optionally add non-critical parts if space allows?

            # Sort by severity (number of issues) and add to prompt
            critical_parts_info.sort(key=lambda x: x[0], reverse=True)
            for _, info in critical_parts_info:
                 prompt += info + "\n"

            if not critical_parts_info:
                 prompt += "- No major issues identified based on current constraints.\n"

            prompt += "\n--- Analysis Request ---\n"
            prompt += "Based ONLY on the data provided above:\n"
            prompt += "1. Executive summary (2-3 sentences) on BOM health (cost, lead time, stock).\n"
            prompt += "2. Top 3-5 key risks identified.\n"
            prompt += "3. Actionable recommendations for mitigation.\n"
            prompt += "4. Comment on the Sweet Spot option's viability vs. cheapest/fastest.\n"
            prompt += "Be professional and concise."

            # --- Call OpenAI ---
            logger.debug(f"Sending prompt to OpenAI (Length: {len(prompt)} chars)...")
            ai_response = call_chatgpt(prompt)

            # --- Schedule GUI Update ---
            def update_gui():
                 if not self.ai_summary_text.winfo_exists(): return # Check widget exists
                 try:
                      self.ai_summary_text.configure(state='normal')
                      self.ai_summary_text.delete(1.0, tk.END)
                      self.ai_summary_text.insert(tk.END, ai_response)
                      self.ai_summary_text.configure(state='disabled')
                      self.root.after(0, self.update_status, "AI summary generated.", "success")
                      self.results_notebook.select(self.predictive_frame) # Switch tab
                 except tk.TclError: logger.warning("Ignoring Tkinter update error, likely during shutdown.")

            self.root.after(0, update_gui) # Schedule the update

        except Exception as e:
            logger.error(f"AI summary generation failed: {e}", exc_info=True)
            self.root.after(0, self.update_status, f"AI Summary Error: {e}", "error")
            self.root.after(0, messagebox.showerror, "AI Summary Error", f"Failed to generate AI summary:\n\n{e}")
        finally:
            self.running_analysis = False
            self.root.after(0, self.validate_inputs) # Schedule button state update


    # --- GUI Table Helpers ---
    def clear_treeview(self, tree):
        """Clears all items from a ttk.Treeview. MUST run on main thread."""
        if not hasattr(tree, 'winfo_exists') or not tree.winfo_exists(): return
        try:
             # Check if treeview actually has children before trying to delete
             children = tree.get_children()
             if children:
                 tree.delete(*children) # More efficient way to delete all
        except tk.TclError as e:
             logger.warning(f"Ignoring error while clearing treeview (might be during shutdown): {e}")

    def populate_treeview(self, tree, data):
        """Populates a ttk.Treeview. MUST run on main thread."""
        if not hasattr(tree, 'winfo_exists') or not tree.winfo_exists(): return
        self.clear_treeview(tree) # Clear first
        if not data: return

        try:
             cols = tree['columns']
             if isinstance(data[0], dict):
                  for item_dict in data:
                       # Ensure values are strings for display, handle None/NaN
                       values = [str(item_dict.get(col, '')).replace('nan', 'N/A') for col in cols]
                       tree.insert("", "end", values=values)
             elif isinstance(data[0], (list, tuple)):
                  for item_tuple in data:
                       # Ensure values are strings, handle None/NaN
                       values = [str(v).replace('nan', 'N/A') if v is not None else '' for v in item_tuple]
                       # Ensure correct number of values
                       if len(values) == len(cols):
                            tree.insert("", "end", values=values)
                       else:
                            logger.warning(f"Row length mismatch: Expected {len(cols)}, Got {len(values)}. Row: {item_tuple}")

        except tk.TclError as e:
             logger.warning(f"Ignoring error while populating treeview (might be during shutdown): {e}")
        except Exception as e:
             logger.error(f"Failed to populate treeview: {e}", exc_info=True)
             # Avoid showing messagebox here as it might be called many times rapidly
             self.update_status(f"Error displaying results: {e}", "error")


    # --- Application Closing ---
    def on_closing(self):
        """Handles application close event."""
        logger.info("Close button clicked.")
        # Optional: Confirmation
        # if messagebox.askokcancel("Quit", "Do you want to quit?"):
        self.running_analysis = False # Attempt to signal threads to stop early if checked
        self.root.after(0, self.update_status, "Shutting down...", "info") # Schedule final status
        logger.info("Shutting down thread pool...")
        # Shutdown pool - give threads a moment but don't wait indefinitely
        self.thread_pool.shutdown(wait=True, cancel_futures=True) # Python 3.9+
        # self.thread_pool.shutdown(wait=True) # Python < 3.9
        logger.info("Destroying main window...")
        self.root.destroy()
        logger.info("Application closed.")


# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    app = BOMAnalyzerApp(root)
    logger.info("Starting mainloop...")
    root.mainloop()
    logger.info("Mainloop finished.") # Log when mainloop exits