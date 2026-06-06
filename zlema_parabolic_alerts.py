#!/usr/bin/env python3
"""
ZLEMA Parabolic Run Alerts + Logical Trailing Stop Loss
For swing/momentum trading (daily timeframe recommended).

DISCLAIMER:
This is for educational and research purposes only. Not financial advice.
Trading involves substantial risk of loss. Past performance does not guarantee future results.
Always do your own due diligence and manage risk properly.
"""

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import yfinance as yf

DEFAULT_TICKERS = "HUBC,ASTS,WULF,PIII,IXHL"
DEFAULT_ZLEMA_PERIOD = 15
DEFAULT_ATR_PERIOD = 14
DEFAULT_MIN_STREAK = 3
DEFAULT_EXTENSION_PCT = 0.05

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
    atr = true_range.rolling(window=period, min_periods=period).mean()
    return atr

def add_uptrend_streak(df: pd.DataFrame) -> pd.DataFrame:
    up = (
        (df['Close'] > df['ZLEMA'] * 0.999) &
        (df['ZLEMA'] > df['ZLEMA'].shift(1).fillna(df['ZLEMA'].iloc[0]) * 0.999)
    )
    df['uptrend_streak'] = 0
    current_streak = 0
    for i in range(len(df)):
        if up.iloc[i]:
            current_streak += 1
            df.loc[df.index[i], 'uptrend_streak'] = current_streak
        else:
            current_streak = 0
    return df

def send_telegram_alert(message: str):
    """Placeholder - fill in your bot token and chat ID from your existing setups."""
    print("\n[TELEGRAM PLACEHOLDER] Would send:\n" + message)
    # import requests
    # bot_token = "YOUR_BOT_TOKEN"
    # chat_id = "YOUR_CHAT_ID"
    # url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    # requests.post(url, data={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"})

def scan_ticker(ticker: str, zlema_period: int, atr_period: int, min_streak: int, extension_pct: float):
    try:
        df = yf.download(ticker, period="1y", progress=False, auto_adjust=True)
        if df.empty or len(df) < max(zlema_period, atr_period) + 10:
            return None

        # Robust handling for modern yfinance (MultiIndex columns + alignment)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        df = df.dropna()

        df['ZLEMA'] = calculate_zlema(df['Close'], zlema_period)
        df['ATR'] = calculate_atr(df, atr_period)
        df = add_uptrend_streak(df)

        recent = df.iloc[-1]
        prev = df.iloc[-2]
        streak = int(recent['uptrend_streak'])
        if streak < min_streak:
            return None

        extension = (recent['Close'] - recent['ZLEMA']) / recent['ZLEMA']
        zlema_slope = recent['ZLEMA'] - prev['ZLEMA']
        prev_slope = prev['ZLEMA'] - df['ZLEMA'].iloc[-3]
        accelerating = zlema_slope > prev_slope and zlema_slope > 0
        is_parabolic = (extension >= extension_pct) or accelerating

        if not is_parabolic:
            return None

        trailing_sl = round(recent['ZLEMA'] - 1.5 * recent['ATR'], 2)
        extension_pct_display = round(extension * 100, 1)

        return {
            'ticker': ticker.upper(),
            'date': df.index[-1].date(),
            'close': round(recent['Close'], 2),
            'zlema': round(recent['ZLEMA'], 2),
            'extension_pct': extension_pct_display,
            'streak': streak,
            'atr': round(recent['ATR'], 2),
            'trailing_sl': trailing_sl,
            'zlema_slope': round(zlema_slope, 4),
        }
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="ZLEMA Parabolic Run Alerts + Trailing SL")
    parser.add_argument('--tickers', type=str, default=DEFAULT_TICKERS)
    parser.add_argument('--zlema-period', type=int, default=DEFAULT_ZLEMA_PERIOD)
    parser.add_argument('--atr-period', type=int, default=DEFAULT_ATR_PERIOD)
    parser.add_argument('--min-streak', type=int, default=DEFAULT_MIN_STREAK)
    parser.add_argument('--extension-threshold', type=float, default=DEFAULT_EXTENSION_PCT)
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
    print(f"\n🔍 Scanning {len(tickers)} tickers for ZLEMA parabolic runs...")

    parabolic_alerts = []
    for ticker in tickers:
        alert = scan_ticker(ticker, args.zlema_period, args.atr_period, args.min_streak, args.extension_threshold)
        if alert:
            parabolic_alerts.append(alert)

    if parabolic_alerts:
        print("\n" + "="*70)
        print("🚨 ZLEMA PARABOLIC RUN ALERTS — LOGICAL TRAILING STOP LOSS")
        print("="*70)
        for a in parabolic_alerts:
            print(f"\n{a['ticker']} | {a['date']}")
            print(f"  Close: ${a['close']:.2f}   |   ZLEMA: ${a['zlema']:.2f} (+{a['extension_pct']:.1f}%)")
            print(f"  Streak: {a['streak']} days   |   ZLEMA slope: {a['zlema_slope']:+.4f}")
            print(f"  ATR: ${a['atr']:.2f}")
            print(f"  >>> SUGGESTED TRAILING SL: ${a['trailing_sl']:.2f}  (ZLEMA - 1.5×ATR)")

        msg_lines = ["*ZLEMA Parabolic Run Alerts*"]
        for a in parabolic_alerts:
            msg_lines.append(f"*{a['ticker']}* | Close ${a['close']:.2f} (+{a['extension_pct']:.1f}% ext) | Streak {a['streak']}d | Trailing SL ~${a['trailing_sl']:.2f}")
        send_telegram_alert("\n".join(msg_lines))
    else:
        print("\nNo parabolic runs meeting criteria right now.")

    print("\n" + "="*70)
    print("Done. Risk responsibly.")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()