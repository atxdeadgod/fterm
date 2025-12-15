# WRDS Financial Terminal

A professional-grade financial terminal built with Python and Streamlit, leveraging the Wharton Research Data Services (WRDS) database for institutional-quality data.

## Features

*   **Real-Time Data Access**: Direct integration with WRDS (CRSP, Compustat, IBES, Fama-French).
*   **Price & Performance**: Interactive charts showing daily prices and cumulative excess returns vs. the market.
*   **Advanced Factor Analysis**:
    *   **Fama-French 6-Factor Model**: attribution to Market, Size (SMB), Value (HML), Profitability (RMW), Investment (CMA), and Momentum (MOM).
    *   **Rolling Beta**: Dynamic visualization of factor exposures over time.
    *   **Sector Beta**: Relative volatility analysis against an automatically constructed equal-weighted peer index.
*   **Deep Fundamentals**:
    *   **Dupont Analysis**: Decomposition of ROE into Net Margin, Asset Turnover, and Financial Leverage.
    *   **Valuation Model**: Historical P/E Fair Value Bands to identify over/undervalued conditions.
    *   **Estimates**: Analyst consensus estimates (EPS).
*   **Peer Group Ranking**: Automated industry peer identification (SIC) and relative ranking by ROE.
*   **Smart Caching**: Local parquet-based caching to minimize latency and WRDS query costs.

## Prerequisites

*   Python 3.12+
*   A valid [WRDS](https://wrds-www.wharton.upenn.edu/) account.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/bhft/fterm.git
    cd fterm
    ```

2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **WRDS Credentials**:
    *   Create a `.env` file or export environment variables:
    ```bash
    export WRDS_USERNAME="your_username"
    export WRDS_PASSWORD="your_password"
    ```
    *   Alternatively, set up a `.pgpass` file for password-less connection.

## Usage

Run the Streamlit dashboard:

```bash
streamlit run dashboard.py
```

The application will open in your default browser at `http://localhost:8501`.

## Project Structure

*   `dashboard.py`: Main Streamlit application entry point (UI Logic).
*   `backend/data_provider.py`: Core data engine handling WRDS connections, SQL queries, caching, and financial modeling.
*   `connect.py`: Connection utility.
