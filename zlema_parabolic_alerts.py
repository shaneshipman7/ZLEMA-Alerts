#!/usr/bin/env python3
"""
ZLEMA Alerts: Free Tier Optimized (Massive / Polygon Engine)
Includes a 12-second call throttle and strict US Market Hours gating.
"""

import argparse
import os
import time
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import sys

# Standard configurations
DEFAULT_ZLEMA_PERIOD = 15
DEFAULT_ATR_PERIOD = 14
DEFAULT_MIN_STREAK = 3
DEFAULT_EXTENSION_PCT = 0.05

WATCHLIST_URL = "https://raw.githubusercontent.com/shaneshipman7/wild-swing-playbook/main/Playbook_Watchlist_Import_"


def is_market_open_now() -> bool:
    """
    Checks if the current time is within standard US Market Hours (9:30 AM - 4:00 PM Eastern).
    Returns False on weekends.
    """
    # Force UTC time, then offset to Eastern Time (ET handles daylight savings via native zone info if preferred, 
    # but a direct calculation or standard timestamp check keeps dependencies low)
    now_utc = datetime.utcnow()
    # Estimate Eastern Time (Standard: UTC-5, Daylight: UTC-4). 
    # For a simple robust check, we can safely approximate or use pandas timestamp tracking:
    et_time = pd.Timestamp(now_utc, tz='UTC').tz_convert('US/Eastern')
    
    # 1. Check Weekend
    if et_time.weekday() >= 5: # 5 = Saturday, 6 = Sunday
        print("🛑 Weekend detected. US Markets are closed.")
        return False
        
    # 2. Check Standard Trading Hours (09:30 to 16:00)
    market_start = et_time.replace(hour=9, minute=30, second=0, microsecond=0)
    market_end = et_time.replace(hour=16, minute=0, second=0, microsecond=0)
    
    if et_time < market_start or et_time > market_end:
        print(f"🛑 Outside Market Hours. Current ET: {et_time.strftime('%Y-%m-%d %H:%M:%S')}. Script terminating to save API calls.")
        return False
        
    print(f"🟢 Market is open. Current ET: {et_time.strftime('%H:%M:%S')}")
    return True


def get_latest_watchlist():
    """Fetch the most recent Playbook_Watchlist_Import file"""
    try:
        for days_ago in range(0, 8):
            date_str = (datetime.now() - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%d")
            url = f"{WATCHLIST_URL}{date_str}.txt"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                content = resp.text.strip()
                tickers = [t.strip().replace('$', '').upper() for t in content.split(',') if t.strip()]
                print(f"✅ Loaded {len(tickers)} tickers from latest watchlist ({date_str})")
                return tickers
        print("⚠️ Could not find recent watchlist, using fallback")
    except Exception as e:
        print(f"⚠️ Watchlist fetch failed: {e}")
    return ["HUBC", "ASTS", "WULF", "PIII", "IXHL"]


def fetch_massive_data(ticker: str, api_key: str) -> pd.DataFrame:
    """Fetch daily aggregated candlesticks while honoring endpoint configurations"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
    
    url = f"https://api.massive.com/v2/aggs/ticker/{ticker}/range/1/day/{start_date}/{end_date}?adjusted=true&sort=asc&limit=1000&apiKey={api_key}"
    
    try:
        resp = requests.get(url, timeout=15)
        if resp.status_code == 429:
            print(f"🚨 Rate Limit Hit (429)! The script will pause for 30 seconds to clear...")
            time.sleep(30)
            return pd.DataFrame()
        if resp.status_code != 200:
            print(f"⚠️ Massive error for {ticker}: Status {resp.status_code}")
            return pd.DataFrame()
            
        data = resp.json()
        if "results" not in data or not data["results"]:
            return pd.DataFrame()
            
        df = pd.DataFrame(data["results"])
        df = df.rename(columns={'o': 'Open', 'h': 'High', 'l': 'Low', 'c': 'Close', 'v': 'Volume', 't': 'Timestamp'})
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Timestamp', inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
    except Exception as e:
        print(f"⚠️ Request failed for {ticker}: {e}")
        return pd.DataFrame()


def calculate_zlema(series: pd.Series, period: int = 15) -> pd.Series:
    lag = (period - 1) // 2
    shifted = series.shift(lag)
    ema_input = series + (series - shifted.fillna(series.iloc[0]))
    return ema_input.ewm(span=period, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period, min_periods=period).mean()


def add_trend_streaks(df: pd.DataFrame) -> pd.DataFrame:
    up = (df['Close'] > df['ZLEMA'] * 0.999) & (df['ZLEMA'] > df['ZLEMA'].shift(1).fillna(df['ZLEMA'].iloc[0]) * 0.999)
    down = (df['Close'] < df['ZLEMA'] * 1.001) & (df['ZLEMA'] < df['ZLEMA'].shift(1).fillna(df['ZLEMA'].iloc[0]) * 1.001)
    
    df['uptrend_streak'] = 0
    df['downtrend_streak'] = 0
    
    up_streak, down_streak = 0, 0
    for i in range(len(df)):
        if up.iloc[i]:
            up_streak += 1
            down_streak = 0
        elif down.iloc[i]:
            down_streak += 1
            up_streak = 0
        else:
            up_streak, down_streak = 0, 0
            
        df.loc[df.index[i], 'uptrend_streak'] = up_streak
        df.loc[df.index[i], 'downtrend_streak'] = down_streak
    return df


def detect_zlema_flips(df: pd.DataFrame):
    prev_close, prev_zlema = df['Close'].shift(1), df['ZLEMA'].shift(1)
    curr_close, curr_zlema = df['Close'], df['ZLEMA']
    return (prev_close <= prev_zlema) & (curr_close > curr_zlema), (prev_close >= prev_zlema) & (curr_close < curr_zlema)


def scan_ticker(ticker: str, api_key: str, zlema_period: int, atr_period: int, min_streak: int, extension_pct: float):
    try:
        df = fetch_massive_data(ticker, api_key)
        if df.empty or len(df) < max(zlema_period, atr_period) + 20:
            return None, None

        df['ZLEMA'] = calculate_zlema(df['Close'], zlema_period)
        df['ATR'] = calculate_atr(df, atr_period)
        df = add_trend_streaks(df)
        
        bull_flip, bear_flip = detect_zlema_flips(df)
        df['bull_flip'], df['bear_flip'] = bull_flip, bear_flip

        recent, prev = df.iloc[-1], df.iloc[-2]
        flip_alert, parabolic_alert = None, None

        if df['bull_flip'].iloc[-1]:
            flip_alert = {'ticker': ticker.upper(), 'date': df.index[-1].date(), 'close': round(recent['Close'], 2), 'direction': 'BULLISH', 'trailing_sl': round(recent['Close'] - 1.5 * recent['ATR'], 2)}
        elif df['bear_flip'].iloc[-1]:
            flip_alert = {'ticker': ticker.upper(), 'date': df.index[-1].date(), 'close': round(recent['Close'], 2), 'direction': 'BEARISH', 'trailing_sl': round(recent['Close'] + 1.5 * recent['ATR'], 2)}

        up_streak, down_streak = int(recent['uptrend_streak']), int(recent['downtrend_streak'])
        zlema_slope = recent['ZLEMA'] - prev['ZLEMA']
        prev_slope = prev['ZLEMA'] - df['ZLEMA'].iloc[-3] if len(df) > 2 else 0

        if up_streak >= min_streak:
            extension = (recent['Close'] - recent['ZLEMA']) / recent['ZLEMA']
            if (extension >= extension_pct) or (zlema_slope > prev_slope and zlema_slope > 0):
                parabolic_alert = {'ticker': ticker.upper(), 'date': df.index[-1].date(), 'close': round(recent['Close'], 2), 'direction': 'BULLISH (RUN)', 'extension_pct': round(extension * 100, 1), 'streak': up_streak, 'trailing_sl': round(recent['Close'] - 1.5 * recent['ATR'], 2)}
        elif down_streak >= min_streak:
            extension = (recent['ZLEMA'] - recent['Close']) / recent['ZLEMA']
            if (extension >= extension_pct) or (zlema_slope < prev_slope and zlema_slope < 0):
                parabolic_alert = {'ticker': ticker.upper(), 'date': df.index[-1].date(), 'close': round(recent['Close'], 2), 'direction': 'BEARISH (DUMP)', 'extension_pct': round(extension * 100, 1), 'streak': down_streak, 'trailing_sl': round(recent['Close'] + 1.5 * recent['ATR'], 2)}

        return flip_alert, parabolic_alert
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None, None


def main():
    parser = argparse.ArgumentParser(description="Massive Engine: Free Plan Optimized")
    parser.add_argument('--tickers', type=str, default=None)
    parser.add_argument('--zlema-period', type=int, default=DEFAULT_ZLEMA_PERIOD)
    parser.add_argument('--atr-period', type=int, default=DEFAULT_ATR_PERIOD)
    parser.add_argument('--min-streak', type=int, default=DEFAULT_MIN_STREAK)
    parser.add_argument('--extension-threshold', type=float, default=DEFAULT_EXTENSION_PCT)
    args = parser.parse_args()

    # 1. RUN TIME PROTECTION
    if not is_market_open_now():
        sys.exit(0)

    api_key = os.getenv("MASSIVE_API_KEY")
    if not api_key:
        print("❌ Error: Missing MASSIVE_API_KEY environment variable.")
        return

    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()] if args.tickers else get_latest_watchlist()

    print(f"\n🔍 Scanning {len(tickers)} tickers. Free tier mode: Throttling 12s per asset to avoid API bans.")

    flips, parabolics = [], []

    for idx, ticker in enumerate(tickers):
        # Prevent trailing sleep on the final asset
        if idx > 0:
            print(f"⏱️ Sleeping 12 seconds to preserve free tier limit...")
            time.sleep(12)
            
        print(f"📡 Requesting data for: {ticker} ({idx + 1}/{len(tickers)})")
        flip, para = scan_ticker(ticker, api_key, args.zlema_period, args.atr_period, args.min_streak, args.extension_threshold)
        if flip: flips.append(flip)
        if para: parabolics.append(para)

    # Output Terminal summaries
    print("\n" + "="*50 + "\nSCAN RESULTS\n" + "="*50)
    for f in flips:
        print(f"Flip: {f['ticker']} -> {f['direction']} at ${f['close']}")
    for p in parabolics:
        print(f"Extension: {p['ticker']} -> {p['direction']} ({p['extension_pct']}% Overextended)")


if __name__ == "__main__":
    main()
