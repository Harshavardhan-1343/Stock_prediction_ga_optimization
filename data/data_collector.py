"""
Stock Data Collector using Yahoo Finance
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging
from typing import List, Dict
import time
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StockDataCollector:
    """
    Collects stock data from Yahoo Finance for specified tickers
    """
    
    def __init__(self, tickers: List[str] = None, start_date: str = None, end_date: str = None):
        """
        Initialize the data collector
        
        Args:
            tickers: List of stock tickers
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
        """
        self.tickers = tickers or config.BANK_TICKERS
        self.start_date = start_date or config.START_DATE
        self.end_date = end_date or config.END_DATE
        self.raw_data = {}
        
    def fetch_data(self, ticker: str, retry_count: int = 3) -> pd.DataFrame:
        """
        Fetch stock data for a single ticker with retry logic
        
        Args:
            ticker: Stock ticker symbol
            retry_count: Number of retries if fetch fails
            
        Returns:
            DataFrame with stock data
        """
        for attempt in range(retry_count):
            try:
                logger.info(f"Fetching data for {ticker} (Attempt {attempt + 1}/{retry_count})...")
                
                # Method 1: Using yfinance download
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                    threads=False
                )
                
                if df.empty:
                    # Method 2: Using Ticker object
                    logger.warning(f"Method 1 failed for {ticker}, trying alternative method...")
                    stock = yf.Ticker(ticker)
                    df = stock.history(
                        start=self.start_date,
                        end=self.end_date,
                        auto_adjust=True
                    )
                
                if df.empty:
                    logger.warning(f"No data found for {ticker} on attempt {attempt + 1}")
                    if attempt < retry_count - 1:
                        time.sleep(2)  # Wait before retry
                        continue
                    return None
                
                # Reset index to make Date a column
                df.reset_index(inplace=True)
                
                # Rename columns for consistency
                df.columns = [col.lower().replace(' ', '_') for col in df.columns]
                
                # Add ticker column
                df['ticker'] = ticker
                
                # Ensure required columns exist
                required_cols = ['date', 'open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    logger.error(f"Missing required columns for {ticker}: {missing_cols}")
                    return None
                
                logger.info(f"✅ Fetched {len(df)} records for {ticker}")
                return df
                
            except Exception as e:
                logger.error(f"❌ Error fetching data for {ticker} (Attempt {attempt + 1}): {str(e)}")
                if attempt < retry_count - 1:
                    time.sleep(2)  # Wait before retry
                    continue
                return None
        
        return None
    
    def fetch_all_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for all tickers
        
        Returns:
            Dictionary with ticker as key and DataFrame as value
        """
        logger.info(f"🚀 Starting data collection for {len(self.tickers)} banks...")
        logger.info(f"📅 Date range: {self.start_date} to {self.end_date}")
        
        successful_tickers = []
        failed_tickers = []
        
        for ticker in self.tickers:
            df = self.fetch_data(ticker)
            if df is not None and not df.empty:
                self.raw_data[ticker] = df
                successful_tickers.append(ticker)
            else:
                failed_tickers.append(ticker)
                logger.error(f"❌ Failed to fetch data for {ticker}")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"✅ Successfully fetched: {len(successful_tickers)}/{len(self.tickers)} banks")
        if successful_tickers:
            logger.info(f"   Success: {', '.join(successful_tickers)}")
        if failed_tickers:
            logger.warning(f"   Failed: {', '.join(failed_tickers)}")
        logger.info(f"{'='*60}\n")
        
        if not self.raw_data:
            logger.error("❌ No data collected! Check your internet connection and ticker symbols.")
            logger.info("\n💡 Troubleshooting Tips:")
            logger.info("   1. Check your internet connection")
            logger.info("   2. Try again after a few minutes (Yahoo Finance may be rate limiting)")
            logger.info("   3. Verify ticker symbols are correct")
            logger.info("   4. Try using a VPN if you're having connection issues")
            return None
        
        return self.raw_data
    
    def save_raw_data(self):
        """
        Save raw data to CSV files
        """
        if not self.raw_data:
            logger.warning("No data to save. Please fetch data first.")
            return
        
        for ticker, df in self.raw_data.items():
            filename = os.path.join(config.DATA_DIR, f"{ticker.replace('.NS', '')}_raw.csv")
            df.to_csv(filename, index=False)
            logger.info(f"💾 Saved {ticker} data to {filename}")
    
    def load_raw_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load raw data from CSV files
        
        Returns:
            Dictionary with ticker as key and DataFrame as value
        """
        logger.info("📂 Loading raw data from files...")
        
        for ticker in self.tickers:
            filename = os.path.join(config.DATA_DIR, f"{ticker.replace('.NS', '')}_raw.csv")
            
            if os.path.exists(filename):
                df = pd.read_csv(filename)
                df['date'] = pd.to_datetime(df['date'])
                self.raw_data[ticker] = df
                logger.info(f"✅ Loaded {ticker} data from {filename}")
            else:
                logger.warning(f"⚠️ File not found: {filename}")
        
        if not self.raw_data:
            logger.warning("No data files found. Will fetch fresh data.")
            return None
        
        return self.raw_data
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Get summary statistics of collected data
        
        Returns:
            DataFrame with summary statistics or None if no data
        """
        if not self.raw_data:
            logger.warning("No data available. Please fetch data first.")
            return None
        
        summary = []
        for ticker, df in self.raw_data.items():
            summary.append({
                'Ticker': ticker,
                'Records': len(df),
                'Start Date': df['date'].min(),
                'End Date': df['date'].max(),
                'Missing Values': df.isnull().sum().sum(),
                'Avg Close Price': df['close'].mean(),
                'Avg Volume': df['volume'].mean()
            })
        
        summary_df = pd.DataFrame(summary)
        return summary_df
    
    def combine_data(self) -> pd.DataFrame:
        """
        Combine all ticker data into a single DataFrame
        
        Returns:
            Combined DataFrame
        """
        if not self.raw_data:
            logger.warning("No data available. Please fetch data first.")
            return None
        
        combined_df = pd.concat(self.raw_data.values(), ignore_index=True)
        logger.info(f"✅ Combined data: {len(combined_df)} total records")
        
        return combined_df


if __name__ == "__main__":
    # Test the data collector
    collector = StockDataCollector()
    
    # Fetch data
    data = collector.fetch_all_data()
    
    if data:
        # Save data
        collector.save_raw_data()
        
        # Get summary
        summary = collector.get_data_summary()
        print("\n📊 Data Summary:")
        print(summary)
        
        # Combine data
        combined = collector.combine_data()
        print(f"\n📈 Combined dataset shape: {combined.shape}")
    else:
        print("\n❌ Data collection failed!")