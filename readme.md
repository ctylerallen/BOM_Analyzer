# BOM Analyzer - v1.0.0
**Streamline Electronic Component Sourcing, Optimize Costs, and Mitigate Supply Chain Risks during New Product Introduction (NPI).**

## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Features](#features)
* [Screenshots](#screenshots)
* [Setup](#setup)
* [Usage](#usage)
* [Project Status](#project-status)
* [Room for Improvement](#room-for-improvement)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)
* [License](#license) 


## General Information
The BOM Analyzer is a Python-based desktop application designed to automate the time-consuming process of analyzing Bill of Materials (BOMs) for new electronic products. It integrates directly with major supplier APIs to fetch real-time pricing, stock, and lead time data, calculates optimal purchasing strategies, assesses multi-factor risks, provides predictive insights, and generates AI-powered summaries to support faster, data-driven decision-making.


## Technologies Used
*   **Core:** Python 3
*   **GUI:** Tkinter, ttk (Standard Library)
*   **Data Handling:** Pandas, NumPy
*   **API Communication:** Requests
*   **Authentication:** python-dotenv, http.server, ssl (for DigiKey OAuth)
*   **Prediction:** Prophet (via `prophet` library)
*   **AI Summary:** OpenAI (`openai` library)
*   **Visualization:** Matplotlib, Seaborn
*   **Packaging (Optional):** PyInstaller (for creating executables)


## Features
*   **Automated Data Aggregation:** Fetches real-time data (pricing tiers, stock, lead time, COO, HTS, lifecycle) via official APIs (Digi-Key, Mouser, Octopart/Nexar, Arrow, Avnet).
*   **BOM Import & Validation:** Loads CSV BOMs, handles common column names, cleans basic data.
*   **Optimal Cost Calculation:** Determines lowest total cost considering MOQs and price breaks, including optional "buy-up" logic.
*   **Multiple Sourcing Strategies:** Calculates and compares Lowest Cost (Strict, In Stock, w/ LT), Fastest, and a user-configurable Optimized strategy (balancing cost/lead time).
*   **Multi-Factor Risk Assessment:** Scores each part (0-10) based on sourcing depth, stock, lead time, lifecycle, and geographic origin. Visually highlights risk levels.
*   **Tariff Estimation:** Incorporates custom tariff rates or attempts USITC HTS code lookup for estimated duty costs.
*   **Predictive Analytics:** Uses historical data (locally saved) and the Prophet library to forecast future cost and lead time trends. Allows tracking actuals vs. predictions.
*   **AI-Powered Summary (OpenAI):** Generates concise executive summaries, highlights critical risks (EOL, Unknown Parts, Stock Gaps), and provides actionable recommendations using GPT models.
*   **Data Export:** Exports analysis results and calculated purchasing strategies to CSV files.
*   **Interactive GUI:** Built with Tkinter/ttk, featuring sortable tables, tabbed results, interactive plots (powered by Matplotlib/Seaborn), and integrated tooltips.

## Screenshots
![Example screenshot](Real-Time-BOM-Optimizer-and-Predictor/blob/main/Screen%20Shot%202025-04-27%20at%2010.01.09%20PM.png)

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone [Your GitHub Repository URL]
    cd bom-analyzer # Or your repository name
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:**
    *   Prophet installation can sometimes require specific C++ build tools. Refer to the official [Prophet installation guide](https://facebook.github.io/prophet/docs/installation.html) if you encounter issues.
    *   Install libraries from `requirements.txt`:
        ```bash
        pip install -r requirements.txt
        ```
        *(You need to generate `requirements.txt` using `pip freeze > requirements.txt` in your activated virtual environment)*

4.  **API Keys (`keys.env` file):**
    *   Create a file named `keys.env` in the main project directory (where `test_local.py` resides).
    *   Add your API keys in the following format:
        ```dotenv
        # DigiKey API Credentials (OAuth - Required for DigiKey Search/Substitutions)
        DIGIKEY_CLIENT_ID=YOUR_DIGIKEY_CLIENT_ID
        DIGIKEY_CLIENT_SECRET=YOUR_DIGIKEY_CLIENT_SECRET

        # Mouser API Key (Required for Mouser Search)
        MOUSER_API_KEY=YOUR_MOUSER_API_KEY

        # Nexar/Octopart API Credentials (OAuth - Required for Octopart Search)
        NEXAR_CLIENT_ID=YOUR_NEXAR_CLIENT_ID
        NEXAR_CLIENT_SECRET=YOUR_NEXAR_CLIENT_SECRET

        # OpenAI API Key (Optional - Required for AI Summary Feature)
        OPENAI_API_KEY=YOUR_OPENAI_API_KEY

        # Arrow API Key (Optional - Not fully implemented yet)
        ARROW_API_KEY=YOUR_ARROW_API_KEY

        # Avnet API Key (Optional - Not fully implemented yet)
        AVNET_API_KEY=YOUR_AVNET_API_KEY
        ```
    *   **Important:** Ensure the Redirect URI for your DigiKey App registration is set to `https://localhost:8000`.

5.  **DigiKey OAuth Certificate (`localhost.pem`):**
    *   The DigiKey API requires a local HTTPS server for the OAuth callback. You need a self-signed certificate.
    *   If you have `openssl` installed, you can generate one using a command like this in your terminal (run from the project directory):
        ```bash
        openssl req -x509 -newkey rsa:4096 -keyout localhost.pem -out localhost.pem -sha256 -days 3650 -nodes -subj "/C=US/ST=State/L=City/O=Organization/OU=Unit/CN=localhost"
        ```
    *   This creates `localhost.pem` containing both the key and certificate, valid for 10 years. Place this file in the same directory as the script.

## Usage Instructions

1.  **Run the Application:**
    ```bash
    python BOM_Analyzer.py
    ```
    *(Ensure your virtual environment is activated)*

2.  **Load BOM:** Click "Load BOM..." and select your CSV file. Ensure it has 'Part Number' and 'Quantity' columns.
3.  **Configure Analysis:** Adjust 'Total Units to Build' and other parameters in the 'Optimized Strategy Configuration' and 'Custom Tariff Rates' sections.
4.  **Run Analysis:** Click "1. Run Analysis". API data will be fetched (may require browser interaction for first-time DigiKey OAuth).
5.  **Review Results:**
    *   **BOM Analysis Tab:** Examine the main table (sortable columns) and the summary metrics. Double-click rows for alternates.
    *   **AI & Predictions Tab:**
        *   Click "3. AI Summary" (if OpenAI key is configured) to view AI recommendations in the top box and full details below. Export the recommended strategy using the dedicated button.
        *   Click "2. Run Predictions" to generate forecasts based on historical data.
        *   Select rows in the "Predictions vs Actuals" table, enter real-world data, and click "Save" to track accuracy.
    *   **Visualizations Tab:** Select plots from the dropdown to visualize results. Hover over points in "Cost vs Lead Time" for details.
6.  **Export:** Use the buttons at the bottom of the "BOM Analysis" or "AI & Predictions" tabs to export data/strategies to CSV.


## Project Status
Project is: _complete_ / In process of prioritizing features on future roadmap.


## Future Roadmap Summary

*   Enhanced API integrations (Arrow, Avnet, more distributors).
*   Direct ERP/PLM integration for BOM import/export.
*   Automated Purchase Order draft generation.
*   Advanced AI/ML features (Real RAG, fine-tuning, explainability).
*   Accuracy-gated autonomous procurement workflows.
*   Potential Web-based/SaaS version.


## Contact
Created by [@ctylerallen](ctylerallen@protonmail.com) - feel free to contact me!



## License 
This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

