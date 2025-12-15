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

    def get_prices(self, ticker, start_date='2010-01-01'):
        """Fetch daily prices from CRSP"""
        cache_path = self._get_cache_path(ticker, "prices")
        # For cache validity, if we request an earlier date than what's in cache, we should re-fetch.
        # But for simplicity, we just clear cache manually or rely on user.
        
        if self._is_cache_valid(ticker, "prices") and cache_path.exists():
            print(f"Loading {ticker} prices from cache...")
            df = pd.read_parquet(cache_path)
            # Ensure date is datetime even from cache
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            return df

        print(f"Fetching {ticker} prices from WRDS...")
        db = self._get_conn()
        
        # Step 1: Get PERMNO
        query_permno = f"select permno, comnam from crsp.stocknames where ticker='{ticker}'"
        try:
            permno_df = db.raw_sql(query_permno)
            if permno_df.empty:
                raise ValueError(f"Ticker {ticker} not found in CRSP")
            permno = permno_df.iloc[0]['permno']
        except Exception as e:
             # Fallback or error
             print(f"Error resolving ticker: {e}")
             return pd.DataFrame()

        # Step 2: Get Prices
        query_prices = f"""
            select date, prc, vol, ret, shrout 
            from crsp.dsf 
            where permno={permno} 
            and date >= '{start_date}'
        """
        try:
            df = db.raw_sql(query_prices, date_cols=['date'])
            # Ensure proper datetime type
            df['date'] = pd.to_datetime(df['date'])
            # Ensure numeric types
            for col in ['prc', 'vol', 'ret', 'shrout']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        except Exception as e:
            print(f"Error fetching prices: {e}")
            return pd.DataFrame()
        
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

    def get_fundamentals(self, ticker):
        """Fetch quarterly fundamentals from Compustat"""
        cache_path = self._get_cache_path(ticker, "fundamentals")
        
        if self._is_cache_valid(ticker, "fundamentals") and cache_path.exists():
            print(f"Loading {ticker} fundamentals from cache...")
            return pd.read_parquet(cache_path)

        print(f"Fetching {ticker} fundamentals from WRDS...")
        db = self._get_conn()
        
        # Step 1: Resolve Ticker to GVKEY (Compustat ID)
        # Use comp.names which maps tic -> gvkey. select distinct to avoid dupes.
        
        try:
            query_gvkey = f"select distinct gvkey from comp.names where tic='{ticker}' limit 1"
            gvkey_df = db.raw_sql(query_gvkey)
            if gvkey_df.empty:
                 print(f"No GVKEY found for {ticker}")
                 return pd.DataFrame()
            gvkey = gvkey_df.iloc[0]['gvkey']
        except Exception as e:
            print(f"Error resolving GVKEY: {e} - Trying alternate table")
            # Alternate: comp.security (tic)
            try:
                query_gvkey = f"select gvkey, conm from comp.company where tic='{ticker}' limit 1"
                gvkey_df = db.raw_sql(query_gvkey)
                if gvkey_df.empty:
                    return pd.DataFrame()
                gvkey = gvkey_df.iloc[0]['gvkey']
            except Exception as e2:
                print(f"Error resolving GVKEY (2nd attempt): {e2}")
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
