# File: src/data_loader.py

import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from src.config import config

class DataLoader:
    """Load and cache cleaned datasets with standardized column mapping"""
    
    def __init__(self):
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_sentiment(self) -> pd.DataFrame:
        """Load NLP-derived daily market sentiment scores."""
        path = config.RAW_DATA_DIR / "daily_market_sentiment.csv"
        if not path.exists():
            print("[WARN] daily_market_sentiment.csv not found. Generating now...")
            from src.models.sentiment import generate_mock_historical_sentiment
            generate_mock_historical_sentiment()
        return pd.read_csv(path, parse_dates=["Date"]).set_index("Date").sort_index()
    
    def load_bulk_deals(self, force_reload: bool = False) -> pd.DataFrame:
        """Load bulk deals data"""
        if 'bulk_deals' not in self._cache or force_reload:
            df = pd.read_csv(config.BULK_DEALS_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            self._cache['bulk_deals'] = df
        return self._cache['bulk_deals']
    
    def load_insider_trades(self, force_reload: bool = False) -> pd.DataFrame:
        """Load insider trading data with compatibility mapping"""
        if 'insider_trades' not in self._cache or force_reload:
            df = pd.read_csv(config.INSIDER_TRADING_FILE)
            
            # Fix Date mapping
            if 'Broadcast_Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Broadcast_Date'], errors='coerce')
            elif 'Acquisition_Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Acquisition_Date'], errors='coerce')
                
            # Compatibility Aliasing for Processor
            # Maps cleaned names to the raw names expected by existing logic
            column_map = {
                'Transaction_Type': 'ACQUISITION/DISPOSAL TRANSACTION TYPE',
                'Value': 'VALUE OF SECURITY (ACQUIRED/DISPLOSED)',
                'Insider_Name': 'NAME OF THE PERSON',
                'Category': 'CATEGORY OF PERSON'
            }
            
            for clean_col, raw_col in column_map.items():
                if clean_col in df.columns:
                    df[raw_col] = df[clean_col]

            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date')
            self._cache['insider_trades'] = df
        return self._cache['insider_trades']
    
    def load_corporate_filings(self, force_reload: bool = False) -> pd.DataFrame:
        """Load corporate announcements data"""
        if 'corporate_filings' not in self._cache or force_reload:
            df = pd.read_csv(config.CORPORATE_FILINGS_FILE)
            
            # Fix date column
            if 'EX-DATE' in df.columns:
                df['Date'] = pd.to_datetime(df['EX-DATE'], errors='coerce')
            
            # CRITICAL FIX: Ensure Symbol column exists (case-sensitive!)
            # The CSV has 'symbol' (lowercase) but processors expect 'Symbol' (uppercase)
            if 'symbol' in df.columns and 'Symbol' not in df.columns:
                df['Symbol'] = df['symbol']
            
            df = df.dropna(subset=['Date'])
            df = df.sort_values('Date')
            self._cache['corporate_filings'] = df
        return self._cache['corporate_filings']
    
    def load_prices(self, force_reload: bool = False) -> pd.DataFrame:
        """Load Nifty 50 prices"""
        if 'prices' not in self._cache or force_reload:
            df = pd.read_csv(config.NIFTY50_PRICES_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            self._cache['prices'] = df
        return self._cache['prices']
    
    def load_returns(self, force_reload: bool = False) -> pd.DataFrame:
        """Load Nifty 50 returns"""
        if 'returns' not in self._cache or force_reload:
            df = pd.read_csv(config.NIFTY50_RETURNS_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            self._cache['returns'] = df
        return self._cache['returns']
    
    def load_vix(self, force_reload: bool = False) -> pd.DataFrame:
        """Load India VIX data"""
        if 'vix' not in self._cache or force_reload:
            df = pd.read_csv(config.VIX_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date')
            self._cache['vix'] = df
        return self._cache['vix']
    
    def load_sector_mapping(self, force_reload: bool = False) -> pd.DataFrame:
        """Load sector classification with standardized column names"""
        if 'sectors' not in self._cache or force_reload:
            df = pd.read_csv(config.SECTOR_MAPPING_FILE)
            
            # CRITICAL FIX: Standardize column names (handle case variations)
            # The CSV has 'sector' (lowercase) but code expects 'Sector' (uppercase)
            column_map = {}
            for col in df.columns:
                if col.lower() == 'sector' and col != 'Sector':
                    column_map[col] = 'Sector'
                elif col.lower() == 'industry' and col != 'Industry':
                    column_map[col] = 'Industry'
                elif col.lower() == 'symbol' and col != 'Symbol':
                    column_map[col] = 'Symbol'
            
            if column_map:
                df = df.rename(columns=column_map)
            
            self._cache['sectors'] = df
        return self._cache['sectors']
    
    def get_stock_price_history(self, symbol: str, days: int = 90) -> Optional[pd.Series]:
        """Get price history for a specific stock"""
        prices = self.load_prices()
        symbol_clean = symbol.replace('.NS', '')
        if symbol_clean in prices.columns:
            return prices[symbol_clean].tail(days)
        elif f"{symbol_clean}.NS" in prices.columns:
            return prices[f"{symbol_clean}.NS"].tail(days)
        return None

    # File: src/data_loader.py  (additions to existing DataLoader class)

    def load_ohlcv_master(self, force_reload: bool = False) -> pd.DataFrame:
        """
        Load OHLCV master data with multi-level header parsing.
        
        The file has an unusual 3-row header:
        Row 0: Price labels (ADANIENT.NS_Close, ADANIENT.NS_High, ...)
        Row 1: Ticker labels (ADANIENT.NS, ADANIENT.NS, ...)
        Row 2: 'Date' label (then empty)
        
        Returns a flat DataFrame with columns like 'ADANIENT_Close', 'ADANIENT_Volume', ...
        """
        if 'ohlcv' not in self._cache or force_reload:
            # Read with multi-level header, skip the Ticker row
            raw = pd.read_csv(config.OHLCV_FILE, header=0, skiprows=[1, 2], index_col=0)
            raw.index = pd.to_datetime(raw.index, errors='coerce')
            raw = raw[~raw.index.isna()].sort_index()
            # Normalize column names: ADANIENT.NS_Close -> ADANIENT_Close
            raw.columns = [
                c.replace('.NS_', '_').replace('.NS', '').strip()
                for c in raw.columns
            ]
            self._cache['ohlcv'] = raw
        return self._cache['ohlcv'].copy()

    def load_ohlcv_close(self, force_reload: bool = False) -> pd.DataFrame:
        """Extract only Close prices from OHLCV master, aligned with nifty50_prices.csv"""
        ohlcv = self.load_ohlcv_master(force_reload=force_reload)
        close_cols = [c for c in ohlcv.columns if c.endswith('_Close')]
        df = ohlcv[close_cols].copy()
        df.columns = [c.replace('_Close', '') for c in df.columns]
        return df

    def load_nifty50_index(self, force_reload: bool = False) -> pd.DataFrame:
        """Load Nifty 50 benchmark index data"""
        if 'nifty50_index' not in self._cache or force_reload:
            df = pd.read_csv(config.NIFTY50_INDEX_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            self._cache['nifty50_index'] = df
        return self._cache['nifty50_index'].copy()

    def load_nifty500_index(self, force_reload: bool = False) -> pd.DataFrame:
        """Load Nifty 500 benchmark index data"""
        if 'nifty500_index' not in self._cache or force_reload:
            df = pd.read_csv(config.NIFTY500_INDEX_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.set_index('Date').sort_index()
            self._cache['nifty500_index'] = df
        return self._cache['nifty500_index'].copy()

    def load_gsec(self, force_reload: bool = False) -> pd.DataFrame:
        """Load 10-Year G-Sec yield as risk-free rate proxy"""
        if 'gsec' not in self._cache or force_reload:
            df = pd.read_csv(config.GSEC_FILE)
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).set_index('Date').sort_index()
            # Convert yield % to decimal
            df['yield_pct'] = df['Price'].astype(float)
            df['daily_rf'] = df['yield_pct'] / 100 / 252
            self._cache['gsec'] = df
        return self._cache['gsec'].copy()
    
    # Add to DataLoader class in src/data_loader.py

    def load_ohlcv(self, symbol: str, force_reload: bool = False) -> Optional[pd.DataFrame]:
        """
        Load OHLCV data for a single symbol from the master file.
        
        Returns DataFrame with columns: Open, High, Low, Close, Volume
        Index: DatetimeIndex
        """
        cache_key = f'ohlcv_{symbol}'
        
        if cache_key not in self._cache or force_reload:
            # Load master OHLCV (skip blank header rows)
            if 'ohlcv_master' not in self._cache or force_reload:
                df_master = pd.read_csv(
                    config.OHLCV_FILE,
                    header=0,
                    skiprows=[1, 2],
                    index_col=0
                )
                df_master.index = pd.to_datetime(df_master.index, errors='coerce')
                df_master = df_master[df_master.index.notna()].sort_index()
                self._cache['ohlcv_master'] = df_master
            
            df_master = self._cache['ohlcv_master']
            
            # Build column names for this symbol
            s = symbol.strip().upper()
            suffixes = ['', '.NS', '.BO']
            
            ohlcv = None
            for suffix in suffixes:
                sym = s + suffix
                open_col   = f'{sym}_Open'   if f'{sym}_Open'   in df_master.columns else None
                high_col   = f'{sym}_High'   if f'{sym}_High'   in df_master.columns else None
                low_col    = f'{sym}_Low'    if f'{sym}_Low'    in df_master.columns else None
                close_col  = f'{sym}_Close'  if f'{sym}_Close'  in df_master.columns else None
                volume_col = f'{sym}_Volume' if f'{sym}_Volume' in df_master.columns else None
                
                if all([open_col, high_col, low_col, close_col, volume_col]):
                    ohlcv = df_master[[open_col, high_col, low_col, close_col, volume_col]].copy()
                    ohlcv.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                    ohlcv = ohlcv.dropna(how='all')
                    break
            
            if ohlcv is None:
                # Fallback: try using prices-only (no volume)
                prices = self.load_prices()
                if s in prices.columns:
                    ohlcv = prices[[s]].rename(columns={s: 'Close'})
                    ohlcv['Open'] = ohlcv['High'] = ohlcv['Low'] = ohlcv['Close']
                    ohlcv['Volume'] = 0.0
                else:
                    return None
            
            self._cache[cache_key] = ohlcv
        
        return self._cache[cache_key].copy()
    
    def get_all_symbols(self) -> List[str]:
        """Return all available stock symbols from the prices file"""
        prices = self.load_prices()
        return [c.replace('.NS', '').strip().upper() for c in prices.columns]

    def get_risk_free_rate(self) -> float:
        """Get current annualized risk-free rate from latest G-Sec yield"""
        if config.PORTFOLIO_RISK_FREE_RATE is not None:
            return config.PORTFOLIO_RISK_FREE_RATE
        gsec = self.load_gsec()
        latest_yield = gsec['yield_pct'].iloc[-1]
        return float(latest_yield) / 100  # Annualized decimal
    
    def clear_cache(self):
        self._cache.clear()

data_loader = DataLoader()
