# BOM Analyzer - v1.1.0

**Streamline Electronic Component Sourcing, Optimize Costs, and Mitigate Supply Chain Risks during New Product Introduction (NPI).**

The NPI BOM Analyzer is a Python-based desktop application designed to automate the time-consuming process of analyzing Bill of Materials (BOMs) for new electronic products. It integrates directly with major supplier APIs to fetch real-time pricing, stock, and lead time data, calculates optimal purchasing strategies, assesses multi-factor risks, provides predictive insights, and generates AI-powered summaries to support faster, data-driven decision-making.


[Link to Full White Paper (PDF)] *(Replace with link to your hosted white paper PDF)*

## Key Features

*   **Automated Data Aggregation:** Fetches real-time data (pricing tiers, stock, lead time, COO, HTS, lifecycle) via official APIs (Digi-Key, Mouser, Octopart/Nexar).
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

*(Insert 1-2 key screenshots of the application interface here. E.g., the main analysis tab and the AI/Predictions tab.)*

**Example: Main Analysis View**