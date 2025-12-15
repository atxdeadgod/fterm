import os
import wrds
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta

# Import get_connection from the user's existing script if possible, 
# but for now we'll assume it's available or re-implement it carefully.
# We will try to import it from connect.py in the root if we run from root.
try:
    from connect import get_connection
except ImportError:
    # Fallback if running from backend dir or if connect.py is not in path
    def get_connection():
        return wrds.Connection(wrds_username=os.getenv("WRDS_USERNAME"), 
                               wrds_password=os.getenv("WRDS_PASSWORD"))

CACHE_DIR = Path("cache")

class DataManager:
    def __init__(self):
        self.conn = None
        self._ensure_cache_dir()

    def _ensure_cache_dir(self):
        if not CACHE_DIR.exists():
            CACHE_DIR.mkdir(parents=True)

    def get_ticker_list(self):
        """Fetch list of all tickers and names for autocomplete"""
        cache_path = CACHE_DIR / "tickers.parquet"
        
        if cache_path.exists():
            # Check if reasonably fresh (e.g. 7 days)
            if datetime.now().timestamp() - cache_path.stat().st_mtime < 7 * 86400:
                return pd.read_parquet(cache_path)
        
        print("Fetching master ticker list from WRDS...")
        db = self._get_conn()
        
        # Get active tickers (active within last 2 years) to keep list fast and relevant.
        # "nameenddt" is the end date of the name assignment. 
        # For active companies, this is usually null or a future date? 
        # In CRSP, '2099-12-31' or current date is often used for active.
        # We'll filter for nameenddt > '2023-01-01' OR NULL
        
        query = """
            select distinct ticker, comnam 
            from crsp.stocknames 
            where (nameenddt >= '2023-01-01' or nameenddt is null)
            and ticker is not null
            order by ticker
        """
        try:
            df = db.raw_sql(query)
            if df.empty:
                print("Warning: No tickers found with current filter.")
                return pd.DataFrame()
                
            # Dedup by ticker in pandas to be safe (keep first/last?)
            df = df.drop_duplicates(subset=['ticker'], keep='last')
            
            # Create a display label: "TICKER | Company Name"
            df['label'] = df['ticker'] + " | " + df['comnam'].fillna('')
            df.to_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Error fetching ticker list: {e}")
            return pd.DataFrame()

    def _get_conn(self):
        if self.conn is None:
            print("Connecting to WRDS...")
            self.conn = get_connection()
        return self.conn

    def _get_cache_path(self, ticker, data_type):
        ticker_dir = CACHE_DIR / ticker
        if not ticker_dir.exists():
            ticker_dir.mkdir(parents=True)
        return ticker_dir / f"{data_type}.parquet"

    def _get_metadata_path(self, ticker):
        return CACHE_DIR / ticker / "metadata.json"

    def _load_metadata(self, ticker):
        path = self._get_metadata_path(ticker)
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return {}

    def _save_metadata(self, ticker, metadata):
        path = self._get_metadata_path(ticker)
        with open(path, "w") as f:
            json.dump(metadata, f, indent=2)

    def _is_cache_valid(self, ticker, data_type):
        metadata = self._load_metadata(ticker)
        last_updated = metadata.get(data_type, {}).get("last_updated")
        if last_updated:
            # Simple policy: Cache is valid for 24 hours
            last_dt = datetime.fromisoformat(last_updated)
            if datetime.now() - last_dt < timedelta(hours=24):
                return True
        return False

    def _get_gvkey(self, ticker):
        """Helper to resolve ticker to GVKEY"""
        db = self._get_conn()
        try:
            query_gvkey = f"select distinct gvkey from comp.names where tic='{ticker}' limit 1"
            gvkey_df = db.raw_sql(query_gvkey)
            if not gvkey_df.empty:
                return gvkey_df.iloc[0]['gvkey']
        except Exception as e:
            print(f"Error resolving GVKEY: {e}")
            pass
            
        # Alternate: comp.company
        try:
            query_gvkey = f"select gvkey from comp.company where tic='{ticker}' limit 1"
            gvkey_df = db.raw_sql(query_gvkey)
            if not gvkey_df.empty:
                return gvkey_df.iloc[0]['gvkey']
        except Exception:
            pass
        return None

    def get_prices(self, ticker, start_date='2010-01-01'):
        """Fetch daily prices from Compustat (comp.secd) for better recency"""
        cache_path = self._get_cache_path(ticker, "prices")
        
        if self._is_cache_valid(ticker, "prices") and cache_path.exists():
            print(f"Loading {ticker} prices from cache...")
            df = pd.read_parquet(cache_path)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df

        print(f"Fetching {ticker} prices from WRDS (Compustat)...")
        db = self._get_conn()
        
        gvkey = self._get_gvkey(ticker)
        if not gvkey:
            print(f"No GVKEY found for {ticker}")
            return pd.DataFrame()

        # Fetch Price, Volume, and Adjustment Factors
        # prccd: Close Price
        # cshtrd: Volume
        # cshoc: Shares Outstanding
        # trfd: Total Return Factor Daily
        # ajexdi: Adjustment Factor (Cumulative)
        
        query_prices = f"""
            select datadate, prccd, cshtrd, cshoc, trfd, ajexdi
            from comp.secd 
            where gvkey='{gvkey}' 
            and datadate >= '{start_date}'
            and iid = (
                select iid from comp.secd 
                where gvkey='{gvkey}' 
                order by cshtrd desc limit 1
            )
            order by datadate asc
        """
        try:
            df = db.raw_sql(query_prices, date_cols=['datadate'])
            
            # Rename for compatibility
            df.rename(columns={
                'datadate': 'date',
                'prccd': 'prc',
                'cshtrd': 'vol',
                'cshoc': 'shrout' # Note: Compustat cshoc is Units, CRSP shrout is Thousands.
            }, inplace=True)
            
            df['date'] = pd.to_datetime(df['date'])
            cols = ['prc', 'vol', 'shrout', 'trfd', 'ajexdi']
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            
            # Handle Adjustments
            df['ajexdi'] = df['ajexdi'].fillna(1.0)
            df['trfd'] = df['trfd'].fillna(1.0)
            
            # Adjusted Price for Returns (Total Return)
            # prc_adj = prc / ajexdi * trfd
            # Standard Split-Adjusted Price (for Charting) = prc / ajexdi
            
            df = df[df['prc'] > 0].copy() # Filter invalid prices
            
            df['adj_close'] = df['prc'] / df['ajexdi']
            df['tot_ret_idx'] = df['prc'] / df['ajexdi'] * df['trfd'] # Total Return Index concept
            
            df['ret'] = df['tot_ret_idx'].pct_change()
            
            # Fill first ret with 0 or NaN
            df['ret'] = df['ret'].fillna(0.0)
            
            # Drop adjustment cols if not needed
            df = df[['date', 'prc', 'adj_close', 'vol', 'ret', 'shrout']]
            
            # Save to cache
            df.to_parquet(cache_path)
            
            # Update metadata
            metadata = self._load_metadata(ticker)
            metadata["prices"] = {
                "last_updated": datetime.now().isoformat(),
                "start_date": start_date,
                "max_date": df['date'].max().isoformat() if not df.empty else None
            }
            self._save_metadata(ticker, metadata)
            
            return df

        except Exception as e:
            print(f"Error fetching prices: {e}")
            return pd.DataFrame()

    def get_fundamentals(self, ticker):
        """Fetch quarterly fundamentals from Compustat"""
        cache_path = self._get_cache_path(ticker, "fundamentals")
        
        if self._is_cache_valid(ticker, "fundamentals") and cache_path.exists():
            print(f"Loading {ticker} fundamentals from cache...")
            return pd.read_parquet(cache_path)

        print(f"Fetching {ticker} fundamentals from WRDS...")
        db = self._get_conn()
        
        gvkey = self._get_gvkey(ticker)
        if not gvkey:
            return pd.DataFrame()
        
        # Step 2: Get Metrics (Quarterly)
        # We fetch last 15 years now
        start_date = (datetime.now() - timedelta(days=15*365)).strftime('%Y-%m-%d')
        query_fund = f"""
            select datadate, rdq, epspxq, saleq, niq, atq, ltq, cshoq
            from comp.fundq
            where gvkey='{gvkey}'
            and datadate >= '{start_date}'
            and indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'
            order by datadate desc
        """
        df = db.raw_sql(query_fund, date_cols=['datadate', 'rdq'])
        
        df.to_parquet(cache_path)
        
        metadata = self._load_metadata(ticker)
        # Find max date
        max_dt = df['datadate'].max() if not df.empty else None
        
        metadata["fundamentals"] = {
            "last_updated": datetime.now().isoformat(),
            "start_date": start_date,
            "max_date": max_dt.isoformat() if max_dt else None
        }
        self._save_metadata(ticker, metadata)
        
        return df

    def get_estimates(self, ticker):
        """Fetch analyst estimates from IBES"""
        cache_path = self._get_cache_path(ticker, "estimates")
        
        if self._is_cache_valid(ticker, "estimates") and cache_path.exists():
            print(f"Loading {ticker} estimates from cache...")
            return pd.read_parquet(cache_path)
            
        print(f"Fetching {ticker} estimates from WRDS...")
        db = self._get_conn()
        
        # IBES uses Ticker mostly
        # We focus on Summary Statistics (statsum_epsus)
        # measure='EPS', fiscal_period_typ='Q' (Quarterly)
        
        start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
        
        query_ibes = f"""
            select statpers, fpi, meanest, medest, numest, stdev
            from ibes.statsum_epsus
            where ticker='{ticker}'
            and measure='EPS'
            and statpers >= '{start_date}'
            order by statpers desc
        """
        # Note: IBES tickers might differ slightly (e.g. 'BRK.B' vs 'BRK/B'), but we try direct match first.
        try:
            df = db.raw_sql(query_ibes, date_cols=['statpers'])
        except Exception as e:
            print(f"Error fetching IBES data: {e}")
            return pd.DataFrame()
            
        df.to_parquet(cache_path)
        
        metadata = self._load_metadata(ticker)
        max_dt = df['statpers'].max() if not df.empty else None
        metadata["estimates"] = {
            "last_updated": datetime.now().isoformat(),
            "start_date": start_date,
            "max_date": max_dt.isoformat() if max_dt else None
        }
        self._save_metadata(ticker, metadata)
        
        return df

    def get_data_availability(self, ticker):
        """Return dict of latest available dates for each data source"""
        metadata = self._load_metadata(ticker)
        summary = {}
        for k, v in metadata.items():
            if 'max_date' in v:
                summary[k] = v['max_date']
        return summary

    def get_factors(self, start_date='2010-01-01'):
        """Fetch Fama-French Daily Factors"""
        # Global cache for factors (not per ticker)
        cache_path = CACHE_DIR / "ff_factors.parquet"
        
        # Check cache validity (allow slightly longer validity for factors, e.g., 24h)
        if cache_path.exists():
            # For simplicity, we just check existence and assume nightly update
            # Ideally check mod time
            if datetime.now().timestamp() - cache_path.stat().st_mtime < 86400: 
               print("Loading FF factors from cache...")
               return pd.read_parquet(cache_path)

        print("Fetching FF factors from WRDS...")
        db = self._get_conn()
        
        # Fama French 3 Factors (Daily)
        # Library: ff, Table: factors_daily
        # Columns: date, smb, hml, mktrf, rf
        
        query_ff = f"""
            select date, smb, hml, mktrf, rf
            from ff.factors_daily
            where date >= '{start_date}'
            order by date asc
        """
        try:
           df = db.raw_sql(query_ff, date_cols=['date'])
           # Ensure date is datetime
           df['date'] = pd.to_datetime(df['date'])
           # Ensure factors are numeric
           for col in ['smb', 'hml', 'mktrf', 'rf']:
               df[col] = pd.to_numeric(df[col], errors='coerce')
           
           df.to_parquet(cache_path)
           return df
        except Exception as e:
           print(f"Error fetching FF factors: {e}")
           return pd.DataFrame()

    def get_ratios(self, ticker):
        """Compute key fundamental ratios"""
        fund = self.get_fundamentals(ticker)
        prices = self.get_prices(ticker)
        
        if fund.empty or prices.empty:
            return pd.DataFrame()
            
        # Merge Price with Fundamentals (using 'asof' logic or simple merge on date)
        # For simplicity, we grab the latest price for current ratios, 
        # but for historical ratios we need to align dates.
        
        # We will return the augmented fundamentals dataframe
        df = fund.copy()
        
        # Ensure we have numeric types
        cols = ['niq', 'saleq', 'atq', 'ltq', 'cshoq']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Calculate TTM (Trailing Twelve Months) if possible, or just annualized quarter
        # Annualized EPS
        if 'epspxq' in df.columns:
            df['EPS_Ann'] = df['epspxq'] * 4 
        
        # Book Value per Share
        if 'atq' in df.columns and 'ltq' in df.columns and 'cshoq' in df.columns:
            df['BookValue'] = df['atq'] - df['ltq']
            df['BVPS'] = df['BookValue'] / df['cshoq']
        
        return df

    def get_sic_peers(self, ticker, limit=10):
        """Find peer tickers based on SIC code"""
        cache_path = self._get_cache_path(ticker, "peers")
        if self._is_cache_valid(ticker, "peers") and cache_path.exists():
            return pd.read_parquet(cache_path)['ticker'].tolist()
            
        print(f"Finding peers for {ticker}...")
        db = self._get_conn()
        
        try:
            # 1. Get SIC of target
            q_sic = f"""
                select siccd 
                from crsp.stocknames 
                where ticker='{ticker}' 
                and (nameenddt >= '2023-01-01' or nameenddt is null)
                order by nameenddt desc limit 1
            """
            sic_df = db.raw_sql(q_sic)
            if sic_df.empty:
                return []
            sic = int(sic_df.iloc[0]['siccd'])
            
            # 2. Find others with same SIC
            # We order by nothing specific for now, ideally market cap but that requires joining dsf.
            # Limiting to 50 to filter later
            q_peers = f"""
                select distinct ticker 
                from crsp.stocknames 
                where siccd={sic} 
                and (nameenddt >= '2023-01-01' or nameenddt is null)
                and ticker != '{ticker}'
                limit 50
            """
            peers_df = db.raw_sql(q_peers)
            peers = peers_df['ticker'].tolist()
            
            # Cache the list
            pd.DataFrame({'ticker': peers}).to_parquet(cache_path)
            
            # Update metadata
            metadata = self._load_metadata(ticker)
            metadata["peers"] = {"last_updated": datetime.now().isoformat()}
            self._save_metadata(ticker, metadata)
            
            return peers[:limit]
            
        except Exception as e:
            print(f"Error finding peers: {e}")
            return []

    def get_bulk_fundamentals(self, tickers):
        """Fetch fundamentals for multiple tickers (latest snapshot)"""
        # Note: Optimization - passing list to SQL IN clause
        # We only fetch LATEST year to save bandwidth for comparison
        if not tickers:
            return pd.DataFrame()
            
        t_list = "'" + "','".join(tickers) + "'"
        
        print(f"Fetching bulk fundamentals for {len(tickers)} peers...")
        db = self._get_conn()
        
        try:
            # First map tickers to GVKEYs
            q_map = f"select distinct gvkey, tic from comp.names where tic in ({t_list})"
            map_df = db.raw_sql(q_map)
            if map_df.empty:
                return pd.DataFrame()
            
            gvkeys = map_df['gvkey'].tolist()
            g_list = "'" + "','".join(gvkeys) + "'"
            
            # Get latest fundq for these gvkeys
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            q_fund = f"""
                select distinct on (gvkey) gvkey, datadate, rdq, epspxq, saleq, niq, atq, ltq, cshoq
                from comp.fundq
                where gvkey in ({g_list})
                and datadate >= '{start_date}'
                and indfmt='INDL' and datafmt='STD' and popsrc='D' and consol='C'
                order by gvkey, datadate desc
            """
            df = db.raw_sql(q_fund, date_cols=['datadate', 'rdq'])
            
            # Merge back ticker
            df = df.merge(map_df, on='gvkey', how='left')
            
            # Calculate metrics
            cols = ['niq', 'saleq', 'atq', 'ltq', 'cshoq']
            for c in cols:
                df[c] = pd.to_numeric(df[c], errors='coerce')

            if 'epspxq' in df.columns:
                df['EPS_Ann'] = df['epspxq'] * 4
            
            if 'atq' in df.columns and 'ltq' in df.columns and 'cshoq' in df.columns:
                df['BookValue'] = df['atq'] - df['ltq']
                df['BVPS'] = df['BookValue'] / df['cshoq']
                
            return df
            
        except Exception as e:
            print(f"Error bulk fetching: {e}")
            return pd.DataFrame()

    def get_valuation_metrics(self, ticker):
        """
        Compute daily valuation metrics (P/E, Fair Value Bands).
        Returns dataframe with real price and valuation bands.
        """
        prices = self.get_prices(ticker, start_date='2015-01-01') # Need longer history for avg P/E
        fund = self.get_fundamentals(ticker)
        
        if prices.empty or fund.empty:
            return pd.DataFrame()
            
        # Prepare Fundamentals: Calculate TTM EPS
        # Sort by date
        fund = fund.sort_values('datadate')
        
        # We need EPS TTM. sum last 4 quarters of epspxq
        if 'epspxq' in fund.columns:
            fund['EPS_TTM'] = fund['epspxq'].rolling(4).sum()
        else:
             return pd.DataFrame()
             
        # Merge Daily using forward fill (since earnings are reported quarterly)
        # We join on 'date' <= 'datadate'? No, we strictly use 'datadate' (report date) or 'rdq' (release date)
        # Using 'rdq' is more accurate for "point-in-time" analysis to avoid look-ahead bias.
        # If 'rdq' is missing, fallback to 'datadate' + 90 days?
        
        # Simplified: Use datadate for now (end of quarter).
        fund['date'] = pd.to_datetime(fund['datadate'])
        prices['date'] = pd.to_datetime(prices['date'])
        
        # Merge
        merged = pd.merge_asof(prices.sort_values('date'), 
                               fund[['date', 'EPS_TTM']].sort_values('date'), 
                               on='date', 
                               direction='backward')
                               
        merged = merged.dropna(subset=['EPS_TTM'])
        
        # Calculate P/E
        merged['PE'] = merged['prc'] / merged['EPS_TTM']
        
        # Filter outliers for P/E average (e.g. negative earnings or massive spikes)
        valid_pe = merged[(merged['PE'] > 0) & (merged['PE'] < 200)]
        
        if valid_pe.empty:
            return pd.DataFrame()
            
        avg_pe = valid_pe['PE'].mean()
        std_pe = valid_pe['PE'].std()
        
        merged['FairPrice'] = avg_pe * merged['EPS_TTM']
        merged['UpperBand'] = (avg_pe + 1.0 * std_pe) * merged['EPS_TTM']
        merged['LowerBand'] = (avg_pe - 1.0 * std_pe) * merged['EPS_TTM']
        merged['AvgPE'] = avg_pe
        merged['StdPE'] = std_pe
        
        return merged

    def get_dupont_data(self, ticker):
        """Compute Dupont Analysis constituents"""
        fund = self.get_fundamentals(ticker)
        if fund.empty:
            return pd.DataFrame()
        
        # Ensure numerics
        cols = ['niq', 'saleq', 'atq', 'ltq']
        for c in cols:
            if c in fund.columns:
                fund[c] = pd.to_numeric(fund[c], errors='coerce')
        
        df = fund.copy()
        
        # Calculate Equity
        if 'atq' in df.columns and 'ltq' in df.columns:
            df['Equity'] = df['atq'] - df['ltq']
        else:
            return pd.DataFrame()
            
        # 1. Net Profit Margin = Net Income / Sales
        df['NetMargin'] = df['niq'] / df['saleq']
        
        # 2. Asset Turnover = Sales / Total Assets
        df['AssetTurnover'] = df['saleq'] / df['atq']
        
        # 3. Financial Leverage (Equity Multiplier) = Total Assets / Equity
        df['Leverage'] = df['atq'] / df['Equity']
        
        # ROE (Quarterly) = Product
        df['ROE_Q'] = df['NetMargin'] * df['AssetTurnover'] * df['Leverage']
        
        return df[['datadate', 'NetMargin', 'AssetTurnover', 'Leverage', 'ROE_Q', 'niq', 'saleq', 'atq', 'Equity']].sort_values('datadate')

        return df[['datadate', 'NetMargin', 'AssetTurnover', 'Leverage', 'ROE_Q', 'niq', 'saleq', 'atq', 'Equity']].sort_values('datadate')

    def get_extended_factors(self, start_date='2010-01-01'):
        """Fetch FF5 Factors + Momentum (using fivefactors_daily which includes UMD)"""
        cache_path = CACHE_DIR / "ff6_factors.parquet"
        
        if cache_path.exists():
            if datetime.now().timestamp() - cache_path.stat().st_mtime < 86400: 
               print("Loading FF6 factors from cache...")
               return pd.read_parquet(cache_path)

        print("Fetching FF6 factors from WRDS...")
        db = self._get_conn()
        
        # 1. Fetch FF5 + Momentum (umd)
        # Table is 'ff.fivefactors_daily' and typically includes 'mktrf', 'smb', 'hml', 'rmw', 'cma', 'rf', 'umd'
        query_ff6 = f"""
            select date, mktrf, smb, hml, rmw, cma, umd as mom, rf
            from ff.fivefactors_daily
            where date >= '{start_date}'
            order by date asc
        """
        try:
            ff6 = db.raw_sql(query_ff6, date_cols=['date'])
            
            # If umd/mom is missing in some versions, handle it?
            # But we saw it in the describe_table output.
            
            if not ff6.empty:
                ff6.to_parquet(cache_path)
                return ff6
                
        except Exception as e:
            print(f"Error fetching FF6: {e}")
            return pd.DataFrame()
            
        return pd.DataFrame()

    def _get_cusip(self, ticker):
        """Helper to resolve ticker to CUSIP (Header/Historical)"""
        db = self._get_conn()
        # Try comp.secd (latest)
        try:
            query = f"select distinct cusip from comp.secd where tic='{ticker}' order by datadate desc limit 1"
            df = db.raw_sql(query)
            if not df.empty:
                return df.iloc[0]['cusip']
        except Exception:
            pass
            
        # Fallback to comp.names
        try:
            query = f"select distinct cusip from comp.names where tic='{ticker}' limit 1"
            df = db.raw_sql(query)
            if not df.empty:
                return df.iloc[0]['cusip']
        except Exception:
            pass
        return None

    def get_short_interest(self, ticker):
        """Fetch Short Interest from comp.sec_shortint"""
        cache_path = self._get_cache_path(ticker, "short_interest")
        if self._is_cache_valid(ticker, "short_interest") and cache_path.exists():
            return pd.read_parquet(cache_path)

        gvkey = self._get_gvkey(ticker)
        if not gvkey: 
            return pd.DataFrame()

        print(f"Fetching Short Interest for {ticker}...")
        db = self._get_conn()
        
        # shortint = Number of Shares Short
        # Note: avgdailyvol is NOT in sec_shortint.
        query = f"""
            select datadate, shortint
            from comp.sec_shortint
            where gvkey='{gvkey}'
            order by datadate asc
        """
        try:
            df = db.raw_sql(query, date_cols=['datadate'])
            if not df.empty:
                df['shortint'] = pd.to_numeric(df['shortint'], errors='coerce')
                # df['days_to_cover'] = ... (Need external volume data)
                
                df.to_parquet(cache_path)
                return df
        except Exception as e:
            print(f"Error fetching Short Int: {e}")
            
        return pd.DataFrame()

    def get_institutional_holdings(self, ticker):
        """Fetch Institutional Holdings from TFN S34 (13F)"""
        cache_path = self._get_cache_path(ticker, "institutional")
        if self._is_cache_valid(ticker, "institutional") and cache_path.exists():
            return pd.read_parquet(cache_path)

        cusip = self._get_cusip(ticker)
        if not cusip:
            # Try 8-digit CUSIP usually needed for TFN
            return pd.DataFrame()
            
        # TFN usually uses 8-char CUSIP. comp.secd returns 9-digit (inc checksum).
        # We might need to trim.
        cusip8 = cusip[:8]

        print(f"Fetching Institutional Holdings for {ticker} (CUSIP: {cusip8})...")
        db = self._get_conn()
        
        # s34 tables are huge. We query carefully.
        # datadate defined in rdate (Report Date) or fdate (File Date). usually fdate is used.
        # shares: number of shares held
        start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
        
        query = f"""
            select fdate, sum(shares) as total_shares, count(distinct mgrno) as num_institutions
            from tfn.s34
            where cusip='{cusip8}' 
            and fdate >= '{start_date}'
            group by fdate
            order by fdate asc
        """
        try:
            df = db.raw_sql(query, date_cols=['fdate'])
            if not df.empty:
                df.to_parquet(cache_path)
                return df
        except Exception as e:
            print(f"Error fetching Institutional: {e}")
        
        return pd.DataFrame()

    def get_insider_transactions(self, ticker):
        """Fetch Insider Transactions from TFN Table1"""
        cache_path = self._get_cache_path(ticker, "insider")
        if self._is_cache_valid(ticker, "insider") and cache_path.exists():
            return pd.read_parquet(cache_path)

        cusip = self._get_cusip(ticker)
        if not cusip: return pd.DataFrame()
        cusip8 = cusip[:8] # TFN uses 8 digit

        print(f"Fetching Insider Trades for {ticker}...")
        db = self._get_conn()
        
        start_date = (datetime.now() - timedelta(days=2*365)).strftime('%Y-%m-%d')
        
        # table1:
        # trandate: Transaction Date
        # shares: Number of shares
        # tprice: Price
        # acqdisp: A (Acquire), D (Dispose)
        
        
        query = f"""
            select trandate, shares, tprice, acqdisp, personid, owner, rolecode1
            from tfn.table1
            where ticker='{ticker}'
            and trandate >= '{start_date}'
            and acqdisp in ('A', 'D')
            order by trandate desc
        """
        try:
            df = db.raw_sql(query, date_cols=['trandate'])
            if not df.empty:
                df['shares'] = pd.to_numeric(df['shares'], errors='coerce')
                df['tprice'] = pd.to_numeric(df['tprice'], errors='coerce')
                
                # Calculate value
                df['value'] = df['shares'] * df['tprice']
                
                # Sign the value: A = Buy (+), D = Sell (-)
                df['signed_value'] = df.apply(lambda row: row['value'] if row['acqdisp'] == 'A' else -row['value'], axis=1)
                
                df.to_parquet(cache_path)
                return df
        except Exception as e:
            print(f"Error fetching Insiders: {e}")

        return pd.DataFrame()

    def get_analyst_revisions(self, ticker):
        """Fetch Analyst Revisions History (IBES Summary)"""
        cache_path = self._get_cache_path(ticker, "analyst_revisions")
        if self._is_cache_valid(ticker, "analyst_revisions") and cache_path.exists():
            return pd.read_parquet(cache_path)

        # IBES uses 'oftic' which is usually the ticker.
        # However, it might vary. We'll search by oftic first.
        
        print(f"Fetching Analyst Revisions for {ticker}...")
        db = self._get_conn()
        
        # statsum_epsus = US summary history
        # statpers = Statistical Period (Snapshot Date)
        # meanest = Mean Estimate
        # measure='EPS'
        # fpi='1' (Fiscal Period Indicator = 1 means Next Fiscal Year)
        
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        
        query = f"""
            select statpers, meanest, median, highest, lowest, numest, stdev
            from tr_ibes.statsum_epsus
            where oftic='{ticker}'
            and measure='EPS'
            and fpi='1'
            and statpers >= '{start_date}'
            order by statpers asc
        """
        try:
            df = db.raw_sql(query, date_cols=['statpers'])
            if not df.empty:
                for c in ['meanest', 'median', 'highest', 'lowest', 'stdev', 'numest']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                
                df.to_parquet(cache_path)
                return df
        except Exception as e:
            print(f"Error fetching Revisions: {e}")
            
        return pd.DataFrame()

    def get_analyst_revisions(self, ticker):
        """Fetch Analyst Revisions History (IBES Summary)"""
        cache_path = self._get_cache_path(ticker, "analyst_revisions")
        if self._is_cache_valid(ticker, "analyst_revisions") and cache_path.exists():
            return pd.read_parquet(cache_path)

        # IBES uses 'oftic' which is usually the ticker.
        print(f"Fetching Analyst Revisions for {ticker}...")
        db = self._get_conn()
        
        # statsum_epsus = US summary history
        # statpers = Statistical Period (Snapshot Date)
        # measure='EPS'
        # fpi='1' (Fiscal Period Indicator = 1 means Next Fiscal Year)
        
        start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y-%m-%d')
        
        query = f"""
            select statpers, meanest, medest, highest, lowest, numest, stdev
            from tr_ibes.statsum_epsus
            where oftic='{ticker}'
            and measure='EPS'
            and fpi='1'
            and statpers >= '{start_date}'
            order by statpers asc
        """
        try:
            df = db.raw_sql(query, date_cols=['statpers'])
            if not df.empty:
                # Rename medest to median for consistency if preferred, OR keep medest
                # In schema I saw 'medest'
                for c in ['meanest', 'medest', 'highest', 'lowest', 'stdev', 'numest']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                
                df.to_parquet(cache_path)
                return df
        except Exception as e:
            print(f"Error fetching Revisions: {e}")
            
        return pd.DataFrame()

    def get_corporate_loans(self, ticker):
        """Fetch Corporate Loan History from DealScan"""
        cache_path = self._get_cache_path(ticker, "dealscan_loans")
        if self._is_cache_valid(ticker, "dealscan_loans") and cache_path.exists():
            return pd.read_parquet(cache_path)

        print(f"Fetching DealScan Loans for {ticker}...")
        db = self._get_conn()
        
        # 1. Resolve Ticker to CompanyID
        # tr_dealscan.company table maps ticker to companyid
        
        q_id = f"select companyid from tr_dealscan.company where ticker='{ticker}'"
        
        try:
            ids = db.raw_sql(q_id)
            if ids.empty:
                return pd.DataFrame()
            
            # There might be multiple IDs (subsidiaries), we'll take all matches
            comp_ids = tuple(ids['companyid'].dropna().astype(str).tolist())
            if not comp_ids:
                return pd.DataFrame()
            
            # Format tuple for SQL IN clause
            if len(comp_ids) == 1:
                id_tuple = f"('{comp_ids[0]}')"
            else:
                id_tuple = str(comp_ids)
            
            # 2. Fetch Loan Packages
            # tr_dealscan.package
            
            query = f"""
                select dealactivedate, dealamount, currency, dealpurpose, dealstatus, company
                from tr_dealscan.package
                where borrowercompanyid in {id_tuple}
                order by dealactivedate desc
            """
            
            df = db.raw_sql(query, date_cols=['dealactivedate'])
            
            if not df.empty:
                 df['dealamount'] = pd.to_numeric(df['dealamount'], errors='coerce')
                 df.to_parquet(cache_path)
                 return df
                 
        except Exception as e:
            print(f"Error fetching DealScan: {e}")
            
        return pd.DataFrame()

    def get_news_sentiment(self, ticker):
        """Fetch News & Sentiment from RavenPack (2023)"""
        cache_path = self._get_cache_path(ticker, "ravenpack_news")
        if self._is_cache_valid(ticker, "ravenpack_news") and cache_path.exists():
            return pd.read_parquet(cache_path)

        print(f"Fetching RavenPack News for {ticker}...")
        db = self._get_conn()
        
        # 1. Resolve Ticker to RP Entity ID
        # ravenpack_common.rpa_entity_mappings
        # data_type='TICKER', data_value=ticker
        
        q_id = f"""
            select rp_entity_id 
            from ravenpack_common.rpa_entity_mappings 
            where data_type='TICKER' 
            and data_value='{ticker}'
            limit 1
        """
        
        try:
            ids = db.raw_sql(q_id)
            if ids.empty:
                print(f"No RavenPack Entity ID found for {ticker}")
                return pd.DataFrame()
            
            # Fetch all IDs
            rp_ids = tuple(ids['rp_entity_id'].unique().tolist())
            
            if len(rp_ids) == 1:
                id_tuple = f"('{rp_ids[0]}')"
            else:
                id_tuple = str(rp_ids)

            # 2. Fetch News from 2024 and 2025 Tables
            # rp_entity_id match
            # event_relevance > 50 (Reasonable relevance)
            # We union them to get the textstream
            
            query = f"""
                select timestamp_utc, headline, event_sentiment_score, event_relevance, topic, "group"
                from ravenpack_full.rpa_full_equities_2025
                where rp_entity_id in {id_tuple}
                and event_relevance >= 50
                union all
                select timestamp_utc, headline, event_sentiment_score, event_relevance, topic, "group"
                from ravenpack_full.rpa_full_equities_2024
                where rp_entity_id in {id_tuple}
                and event_relevance >= 50
                order by timestamp_utc desc
                limit 100
            """
            
            df = db.raw_sql(query, date_cols=['timestamp_utc'])
            
            if not df.empty:
                 df['event_sentiment_score'] = pd.to_numeric(df['event_sentiment_score'], errors='coerce')
                 df.to_parquet(cache_path)
                 return df
                 
        except Exception as e:
            print(f"Error fetching RavenPack: {e}")
            
        return pd.DataFrame()

    def get_governance_profile(self, ticker):
        """Fetch Governance Profile from RiskMetrics (ISS)"""
        cache_path = self._get_cache_path(ticker, "risk_governance")
        if self._is_cache_valid(ticker, "risk_governance") and cache_path.exists():
            return pd.read_parquet(cache_path)

        print(f"Fetching Governance Profile for {ticker}...")
        db = self._get_conn()
        
        # risk.rmgovernance
        # year, dualclass, cboard, ppill, gparachute, confvote, cumvote
        
        query = f"""
            select year, dualclass, cboard, ppill, gparachute, confvote, cumvote
            from risk.rmgovernance
            where ticker='{ticker}'
            order by year desc
            limit 1
        """
        
        try:
            df = db.raw_sql(query)
            
            if not df.empty:
                 df.to_parquet(cache_path)
                 return df
                 
        except Exception as e:
            print(f"Error fetching Governance: {e}")
            
        return pd.DataFrame()

    def get_volatility_surface(self, ticker):
        """Fetch Implied Volatility Surface from OptionMetrics (optionm.vsurfd)"""
        cache_path = self._get_cache_path(ticker, "option_vol_surf")
        if self._is_cache_valid(ticker, "option_vol_surf") and cache_path.exists():
             return pd.read_parquet(cache_path)

        print(f"Fetching Volatility Surface for {ticker}...")
        db = self._get_conn()
        
        # 1. Resolve Ticker to SECID (optionm.securd)
        try:
            q_secid = f"select secid from optionm.securd where ticker='{ticker}' limit 1"
            secid_df = db.raw_sql(q_secid)
            if secid_df.empty:
                print(f"No OptionMetrics SECID found for {ticker}")
                return pd.DataFrame()
            secid = secid_df.iloc[0]['secid']
        except Exception as e:
            print(f"Error resolving SECID: {e}")
            return pd.DataFrame()

        # 2. Fetch Volatility Surface (optionm.vsurfd2025)
        # Using 2025 table for latest data. Fallback logic could be added but 2025 is target.
        # Columns: date, days, delta, impl_volatility, cp_flag
        # cp_flag: C=Call, P=Put. Delta 50 Call ~ Delta -50 Put
        
        query = f"""
            select date, days, delta, impl_volatility, cp_flag
            from optionm.vsurfd2025
            where secid={secid}
            order by date desc
            limit 1000
        """
        
        try:
            df = db.raw_sql(query, date_cols=['date'])
            if not df.empty:
                # Filter for the single latest date
                latest_date = df['date'].max()
                df = df[df['date'] == latest_date].copy()
                
                # Numeric conversion
                df['impl_volatility'] = pd.to_numeric(df['impl_volatility'], errors='coerce')
                df['delta'] = pd.to_numeric(df['delta'], errors='coerce')
                df['days'] = pd.to_numeric(df['days'], errors='coerce')
                
                df.to_parquet(cache_path)
                return df
        except Exception as e:
            print(f"Error fetching Vol Surface: {e}")
            
        return pd.DataFrame()

        return pd.DataFrame()

    def get_bond_transactions(self, ticker):
        """Fetch Corporate Bond Transactions from TRACE"""
        cache_path = self._get_cache_path(ticker, "trace_bonds")
        if self._is_cache_valid(ticker, "trace_bonds") and cache_path.exists():
             return pd.read_parquet(cache_path)

        print(f"Fetching Bond Transactions for {ticker}...")
        db = self._get_conn()
        
        # trace.trace_enhanced
        # Fields: trd_exctn_dt, trd_exctn_tm, rptd_pr, yld_pt, entrd_vol_qt, bond_sym_id
        # Filter by company_symbol = ticker
        
        query = f"""
            select trd_exctn_dt, trd_exctn_tm, rptd_pr, yld_pt, entrd_vol_qt, bond_sym_id, cusip_id
            from trace.trace_enhanced
            where company_symbol='{ticker}'
            and trd_exctn_dt >= '2020-01-01'
            order by trd_exctn_dt desc
            limit 2000
        """
        
        try:
            df = db.raw_sql(query, date_cols=['trd_exctn_dt'])
            
            if not df.empty:
                # Numeric conversions
                df['rptd_pr'] = pd.to_numeric(df['rptd_pr'], errors='coerce')
                df['yld_pt'] = pd.to_numeric(df['yld_pt'], errors='coerce')
                df['entrd_vol_qt'] = pd.to_numeric(df['entrd_vol_qt'], errors='coerce')
                
                # Create a datetime column from date and time
                # Time is likely HH:MM:SS
                def combine_dt(row):
                    try:
                        t = row['trd_exctn_tm']
                        return pd.to_datetime(f"{row['trd_exctn_dt'].strftime('%Y-%m-%d')} {t}")
                    except:
                        return row['trd_exctn_dt']
                        
                df['datetime'] = df.apply(combine_dt, axis=1)
                
                df.to_parquet(cache_path)
                return df
                
        except Exception as e:
            print(f"Error fetching Bond Data: {e}")
            
        return pd.DataFrame()

        return pd.DataFrame()

    def get_macro_data(self, start_date='2015-01-01'):
        """Fetch Macroeconomic Data from Federal Reserve (frb.rates_daily)"""
        cache_path = self._get_cache_path("MACRO", "frb_rates")
        if self._is_cache_valid("MACRO", "frb_rates") and cache_path.exists():
             return pd.read_parquet(cache_path)

        print(f"Fetching Macro Data (FRB)...")
        db = self._get_conn()
        
        # dgs10: 10-Year Treasury Constant Maturity Rate
        # dgs2: 2-Year Treasury Constant Maturity Rate
        # dff: Effective Federal Funds Rate
        # t10y2y: 10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity
        # bamlh0a0hym2ey: ICE BofA US High Yield Index Effective Yield
        
        query = f"""
            select date, dgs10, dgs2, dff, t10y2y, bamlh0a0hym2ey
            from frb.rates_daily
            where date >= '{start_date}'
            order by date asc
        """
        
        try:
            df = db.raw_sql(query, date_cols=['date'])
            if not df.empty:
                cols = ['dgs10', 'dgs2', 'dff', 't10y2y', 'bamlh0a0hym2ey']
                for c in cols:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                
                df.to_parquet(cache_path)
                return df
        except Exception as e:
            print(f"Error fetching Low/Macro Data: {e}")
            
        return pd.DataFrame()



    def get_sector_index(self, ticker, start_date='2020-01-01'):
        """
        Construct an equal-weighted sector index from SIC peers.
        Returns dataframe with 'date' and 'sector_ret'.
        """
        cache_path = self._get_cache_path(ticker, "sector_index")
        if self._is_cache_valid(ticker, "sector_index") and cache_path.exists():
             return pd.read_parquet(cache_path)

        peers = self.get_sic_peers(ticker, limit=20) # Use top 20 peers
        if not peers:
            return pd.DataFrame()
            
        print(f"Constructing sector index for {ticker} from {len(peers)} peers...")
        db = self._get_conn()
        
        # Fetch returns for all peers (Optimized IN clause)
        p_list = "'" + "','".join(peers) + "'"
        
        # We need to map Ticker -> Permno to query dsf efficiently? 
        # Or just query dsf by ticker (might be slower if not indexed by ticker, but dsf usually has permno)
        # Actually dsf is indexed by permno. crsp.stocknames maps tic <-> permno.
        # But let's try querying dsf with a join on stocknames? 
        # Simpler: Get permnos first.
        
        q_perm = f"select distinct permno from crsp.stocknames where ticker in ({p_list})"
        try:
            perm_df = db.raw_sql(q_perm)
            permnos = perm_df['permno'].tolist()
            if not permnos:
                return pd.DataFrame()
            
            perm_str = ",".join([str(p) for p in permnos])
            
            q_ret = f"""
                select date, ret 
                from crsp.dsf 
                where permno in ({perm_str}) 
                and date >= '{start_date}'
            """
            returns_df = db.raw_sql(q_ret, date_cols=['date'])
            
            # Group by date and mean() to get equal-weighted index
            returns_df['ret'] = pd.to_numeric(returns_df['ret'], errors='coerce')
            sector_idx = returns_df.groupby('date')['ret'].mean().reset_index()
            sector_idx.rename(columns={'ret': 'sector_ret'}, inplace=True)
            
            sector_idx.to_parquet(cache_path)
            
            # Update metadata
            metadata = self._load_metadata(ticker)
            metadata["sector_index"] = {"last_updated": datetime.now().isoformat()}
            self._save_metadata(ticker, metadata)
            
            return sector_idx
            
        except Exception as e:
            print(f"Error constructing sector index: {e}")
            return pd.DataFrame()

if __name__ == "__main__":
    dm = DataManager()
    # print(dm.get_prices("AAPL").head())
    # print(dm.get_fundamentals("AAPL").head())
