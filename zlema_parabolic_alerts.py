#!/usr/bin/env python3
"""
ZLEMA Multi-Timeframe Alerts: Bullish Flips + Parabolic Runs
Automatically detects best timeframe (1h/2h/4h/1d) for each ticker
"""

import argparse
import os
import pandas as pd
import requests
from datetime import datetime
import yfinance as yf

DEFAULT_ZLEMA_PERIOD = 15
DEFAULT_ATR_PERIOD = 14
DEFAULT_MIN_STREAK = 3
DEFAULT_EXTENSION_PCT = 0.05

WATCHLIST_URL_BASE = "https://raw.githubusercontent.com/shaneshipman7/wild-swing-playbook/main/Playbook_Watchlist_Import_"


def get_latest_watchlist():
    """Fetch most recent watchlist"""
    try:
        for days_ago in range(0, 8):
            date_str = (datetime.now() - pd.Timedelta(days=days_ago)).strftime("%Y-%m-%d")
            url = f"{WATCHLIST_URL_BASE}{date_str}.txt"
            resp = requests.get(url, timeout=10)
            if resp.status_code == 200:
                content = resp.text.strip()
                tickers = [t.strip().replace('$', '').upper() for t in content.split(',') if t.strip()]
                print(f"✅ Loaded {len(tickers)} tickers from {date_str}")
                return tickers
        print("⚠️ No recent watchlist — using fallback")
    except Exception as e:
        print(f"⚠️ Watchlist fetch failed: {e}")
    return ["HUBC", "ASTS", "WULF", "PIII", "IXHL", "TE", "EOSE", "GE", "SES", "TKO", "XPO"]


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


def add_uptrend_streak(df: pd.DataFrame) -> pd.DataFrame:
    up = (df['Close'] > df['ZLEMA'] * 0.999) & (df['ZLEMA'] > df['ZLEMA'].shift(1).fillna(df['ZLEMA'].iloc[0]) * 0.999)
    df['uptrend_streak'] = 0
    current_streak = 0
    for i in range(len(df)):
        if up.iloc[i]:
            current_streak += 1
            df.loc[df.index[i], 'uptrend_streak'] = current_streak
        else:
            current_streak = 0
    return df


def detect_zlema_flip(df: pd.DataFrame):
    prev_below = df['Close'].shift(1) <= df['ZLEMA'].shift(1)
    now_above = df['Close'] > df['ZLEMA']
    return prev_below & now_above


def get_best_timeframe(ticker: str, zlema_period=15, atr_period=14):
    """Test multiple timeframes and return best alignment"""
    timeframes = {
        '1h': {'interval': '60m', 'period': '5d'},
        '2h': {'interval': '60m', 'period': '10d'},   # resample later
        '4h': {'interval': '60m', 'period': '15d'},
        '1d': {'interval': '1d', 'period': '1y'}
    }
    
    best_score = -1
    best_tf = None
    best_data = None

    for tf_name, config in timeframes.items():
        try:
            df = yf.download(ticker, interval=config['interval'], period=config['period'], 
                           progress=False, auto_adjust=True)
            if df.empty or len(df) < zlema_period + 20:
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()

            # Resample for 2h/4h
            if tf_name in ['2h', '4h']:
                freq = '2H' if tf_name == '2h' else '4H'
                df = df.resample(freq).agg({'Open':'first', 'High':'max', 'Low':'min', 'Close':'last', 'Volume':'sum'}).dropna()

            df['ZLEMA'] = calculate_zlema(df['Close'], zlema_period)
            df['ATR'] = calculate_atr(df, atr_period)
            df = add_uptrend_streak(df)
            df['flip'] = detect_zlema_flip(df)

            recent = df.iloc[-1]
            streak = int(recent['uptrend_streak'])
            extension = (recent['Close'] - recent['ZLEMA']) / recent['ZLEMA'] if recent['ZLEMA'] != 0 else 0

            # Score: weighted streak + extension + recent flip
            score = (streak * 10) + (extension * 100) + (10 if df['flip'].iloc[-1] else 0)

            if score > best_score and streak >= DEFAULT_MIN_STREAK - 1:  # allow slightly lower streak on intraday
                best_score = score
                best_tf = tf_name
                best_data = {
                    'streak': streak,
                    'extension': round(extension * 100, 1),
                    'close': round(recent['Close'], 2),
                    'zlema': round(recent['ZLEMA'], 2),
                    'atr': round(recent['ATR'], 2),
                    'flip': bool(df['flip'].iloc[-1]),
                    'trailing_sl': round(recent['ZLEMA'] - 1.5 * recent['ATR'], 2)
                }
        except Exception as e:
            continue  # skip bad TF

    return best_tf, best_data


def send_telegram_alert(message: str):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print("\n[LOCAL MODE] Would have sent:\n" + message)
        return
    try:
        import requests
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=payload, timeout=10)
        print("✅ Telegram alert sent")
    except Exception as e:
        print(f"⚠️ Telegram failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Multi-TF ZLEMA Alerts")
    parser.add_argument('--tickers', type=str, default=None)
    args = parser.parse_args()

    tickers = get_latest_watchlist() if not args.tickers else [t.strip().upper() for t in args.tickers.split(',')]

    print(f"\n🔍 Scanning {len(tickers)} tickers across 1h/2h/4h/1d...")

    alerts = []
    for ticker in tickers:
        best_tf, data = get_best_timeframe(ticker)
        if best_tf and data:
            alerts.append({
                'ticker': ticker,
                'tf': best_tf,
                **data
            })

    if alerts:
        msg_lines = ["*🚨 MULTI-TF ZLEMA ALERTS*"]
        print("\n" + "="*80)
        print("🚨 BEST TIMEFRAME ZLEMA SIGNALS")
        print("="*80)

        for a in alerts:
            print(f"\n{a['ticker']} | Best TF: **{a['tf'].upper()}**")
            print(f"  Close: ${a['close']:.2f}   |   ZLEMA: ${a['zlema']:.2f} (+{a['extension']:.1f}%)")
            print(f"  Streak: {a['streak']} bars   |   Trailing SL: ${a['trailing_sl']:.2f}")
            if a['flip']:
                print("  → Fresh Bullish Flip!")
            
            msg_lines.append(
                f"*{a['ticker']}* ({a['tf'].upper()}) | Close ${a['close']:.2f} (+{a['extension']:.1f}%) | "
                f"Streak {a['streak']} | SL ~${a['trailing_sl']:.2f}"
            )

        send_telegram_alert("\n".join(msg_lines))
    else:
        print("\nNo strong ZLEMA alignments across timeframes right now.")

    print("\nDone. Risk responsibly.")


if __name__ == "__main__":
    main()
