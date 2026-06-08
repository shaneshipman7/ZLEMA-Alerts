#!/usr/bin/env python3
"""
ZLEMA Multi-Timeframe Alerts: Bullish + Bearish Flips + Parabolic Runs
Automatically detects best timeframe (1h/2h/4h/1d) for each ticker
Includes US Market Hours gate (9:30 AM - 4:00 PM ET, weekdays only)
"""

import argparse
import os
import sys
import pandas as pd
import requests
from datetime import datetime

DEFAULT_ZLEMA_PERIOD = 15
DEFAULT_ATR_PERIOD = 14
DEFAULT_MIN_STREAK = 3
DEFAULT_EXTENSION_PCT = 0.05

WATCHLIST_URL_BASE = "https://raw.githubusercontent.com/shaneshipman7/wild-swing-playbook/main/Playbook_Watchlist_Import_"


# ─────────────────────────────────────────────
# MARKET HOURS GATE
# ─────────────────────────────────────────────

def is_market_open_now() -> bool:
    """
    Returns True only during RTH (9:30–16:00 ET, weekdays).
    Uses America/New_York (IANA standard) instead of US/Eastern to avoid
    tzdata missing errors on GitHub Actions runners and clean Linux envs.
    """
    try:
        et_time = datetime.now(pd.UTC).tz_convert('America/New_York')
    except Exception:
        # Fallback for older pandas where pd.UTC isn't available
        et_time = pd.Timestamp(datetime.utcnow()).tz_localize('UTC').tz_convert('America/New_York')

    if et_time.weekday() >= 5:
        print("🛑 Weekend — markets closed. Exiting.")
        return False

    market_open  = et_time.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = et_time.replace(hour=16, minute=0,  second=0, microsecond=0)

    if et_time < market_open or et_time > market_close:
        print(f"🛑 Outside RTH. Current ET: {et_time.strftime('%Y-%m-%d %H:%M:%S')} — Exiting.")
        return False

    print(f"🟢 Market open. ET: {et_time.strftime('%H:%M:%S')}")
    return True


# ─────────────────────────────────────────────
# WATCHLIST
# ─────────────────────────────────────────────

def get_latest_watchlist():
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


# ─────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────

def calculate_zlema(series: pd.Series, period: int = 15) -> pd.Series:
    lag = (period - 1) // 2
    shifted = series.shift(lag)
    ema_input = series + (series - shifted.fillna(series.iloc[0]))
    return ema_input.ewm(span=period, adjust=False).mean()


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def add_trend_streaks(df: pd.DataFrame) -> pd.DataFrame:
    """Track both uptrend and downtrend streaks simultaneously."""
    up   = (df['Close'] > df['ZLEMA'] * 0.999) & (df['ZLEMA'] > df['ZLEMA'].shift(1).fillna(df['ZLEMA'].iloc[0]) * 0.999)
    down = (df['Close'] < df['ZLEMA'] * 1.001) & (df['ZLEMA'] < df['ZLEMA'].shift(1).fillna(df['ZLEMA'].iloc[0]) * 1.001)

    df['uptrend_streak']   = 0
    df['downtrend_streak'] = 0

    up_streak = down_streak = 0
    for i in range(len(df)):
        if up.iloc[i]:
            up_streak += 1
            down_streak = 0
        elif down.iloc[i]:
            down_streak += 1
            up_streak = 0
        else:
            up_streak = down_streak = 0

        df.loc[df.index[i], 'uptrend_streak']   = up_streak
        df.loc[df.index[i], 'downtrend_streak'] = down_streak

    return df


def detect_flips(df: pd.DataFrame):
    """Returns (bull_flip Series, bear_flip Series)"""
    prev_close = df['Close'].shift(1)
    prev_zlema = df['ZLEMA'].shift(1)
    curr_close = df['Close']
    curr_zlema = df['ZLEMA']

    bull_flip = (prev_close <= prev_zlema) & (curr_close > curr_zlema)
    bear_flip = (prev_close >= prev_zlema) & (curr_close < curr_zlema)
    return bull_flip, bear_flip


# ─────────────────────────────────────────────
# MULTI-TIMEFRAME SCAN
# ─────────────────────────────────────────────

def get_best_timeframe(ticker: str, zlema_period=15, atr_period=14):
    try:
        import yfinance as yf
    except ImportError:
        print("❌ yfinance not installed. Run: pip install yfinance")
        sys.exit(1)

    timeframes = {
        '1h': {'interval': '60m', 'period': '5d'},
        '2h': {'interval': '60m', 'period': '10d'},
        '4h': {'interval': '60m', 'period': '15d'},
        '1d': {'interval': '1d',  'period': '1y'},
    }

    best_score = -1
    best_tf    = None
    best_data  = None

    for tf_name, config in timeframes.items():
        try:
            df = yf.download(ticker, interval=config['interval'], period=config['period'],
                             progress=False, auto_adjust=True)
            if df.empty or len(df) < zlema_period + 20:
                continue

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna().copy()

            # Resample 1h data into 2h / 4h candles
            if tf_name in ['2h', '4h']:
                freq = '2h' if tf_name == '2h' else '4h'
                df = df.resample(freq).agg({
                    'Open': 'first', 'High': 'max', 'Low': 'min',
                    'Close': 'last', 'Volume': 'sum'
                }).dropna()

            if len(df) < zlema_period + 20:
                continue

            df['ZLEMA'] = calculate_zlema(df['Close'], zlema_period)
            df['ATR']   = calculate_atr(df, atr_period)
            df          = add_trend_streaks(df)

            bull_flip, bear_flip = detect_flips(df)
            df['bull_flip'] = bull_flip
            df['bear_flip'] = bear_flip

            recent        = df.iloc[-1]
            up_streak     = int(recent['uptrend_streak'])
            down_streak   = int(recent['downtrend_streak'])
            active_streak = max(up_streak, down_streak)

            zlema_val = recent['ZLEMA'] if recent['ZLEMA'] != 0 else 1
            bull_ext  = (recent['Close'] - zlema_val) / zlema_val
            bear_ext  = (zlema_val - recent['Close']) / zlema_val

            is_bull_flip = bool(df['bull_flip'].iloc[-1])
            is_bear_flip = bool(df['bear_flip'].iloc[-1])

            # Score: streak strength + extension + fresh flip bonus
            score = (active_streak * 10) + (max(bull_ext, bear_ext) * 100) + (10 if (is_bull_flip or is_bear_flip) else 0)

            if score > best_score and active_streak >= DEFAULT_MIN_STREAK - 1:
                best_score = score
                best_tf    = tf_name
                best_data  = {
                    'up_streak':    up_streak,
                    'down_streak':  down_streak,
                    'bull_ext':     round(bull_ext * 100, 1),
                    'bear_ext':     round(bear_ext * 100, 1),
                    'close':        round(float(recent['Close']), 2),
                    'zlema':        round(float(recent['ZLEMA']), 2),
                    'atr':          round(float(recent['ATR']), 2),
                    'bull_flip':    is_bull_flip,
                    'bear_flip':    is_bear_flip,
                    'bull_sl':      round(float(recent['ZLEMA'] - 1.5 * recent['ATR']), 2),
                    'bear_sl':      round(float(recent['ZLEMA'] + 1.5 * recent['ATR']), 2),
                }

        except Exception:
            continue

    return best_tf, best_data


# ─────────────────────────────────────────────
# TELEGRAM
# ─────────────────────────────────────────────

def send_telegram_alert(message: str):
    bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id   = os.getenv("TELEGRAM_CHAT_ID")
    if not bot_token or not chat_id:
        print("\n[LOCAL MODE] Alert:\n" + message)
        return
    try:
        url     = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        payload = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
        requests.post(url, data=payload, timeout=10)
        print("✅ Telegram alert sent")
    except Exception as e:
        print(f"⚠️ Telegram failed: {e}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    # ── Market hours gate ──────────────────────
    if not is_market_open_now():
        sys.exit(0)

    parser = argparse.ArgumentParser(description="Multi-TF ZLEMA Alerts")
    parser.add_argument('--tickers', type=str, default=None)
    args = parser.parse_args()

    tickers = (
        get_latest_watchlist() if not args.tickers
        else [t.strip().upper() for t in args.tickers.split(',')]
    )

    print(f"\n🔍 Scanning {len(tickers)} tickers across 1h/2h/4h/1d...")

    bull_alerts = []
    bear_alerts = []

    for ticker in tickers:
        best_tf, data = get_best_timeframe(ticker)
        if not best_tf or not data:
            continue

        entry = {'ticker': ticker, 'tf': best_tf, **data}

        # Route to correct bucket — flip takes priority, then parabolic run
        if data['bull_flip'] or data['up_streak'] >= DEFAULT_MIN_STREAK:
            bull_alerts.append(entry)
        if data['bear_flip'] or data['down_streak'] >= DEFAULT_MIN_STREAK:
            bear_alerts.append(entry)

    # ── Print + build Telegram message ─────────
    msg_lines = ["*🚨 MULTI-TF ZLEMA ALERTS*"]

    if bull_alerts:
        print("\n" + "="*70)
        print("🟢 BULLISH SIGNALS")
        print("="*70)
        msg_lines.append("\n*🟢 BULLISH*")
        for a in bull_alerts:
            flip_tag = " ← Fresh Flip!" if a['bull_flip'] else ""
            print(f"\n{a['ticker']} | TF: {a['tf'].upper()}{flip_tag}")
            print(f"  Close: ${a['close']:.2f}  ZLEMA: ${a['zlema']:.2f}  Ext: +{a['bull_ext']:.1f}%")
            print(f"  Streak: {a['up_streak']} bars  |  Trailing SL: ${a['bull_sl']:.2f}")
            msg_lines.append(
                f"*{a['ticker']}* ({a['tf'].upper()}){flip_tag} | ${a['close']:.2f} +{a['bull_ext']:.1f}% | "
                f"Streak {a['up_streak']} | SL ~${a['bull_sl']:.2f}"
            )

    if bear_alerts:
        print("\n" + "="*70)
        print("🔴 BEARISH SIGNALS")
        print("="*70)
        msg_lines.append("\n*🔴 BEARISH*")
        for a in bear_alerts:
            flip_tag = " ← Fresh Flip!" if a['bear_flip'] else ""
            print(f"\n{a['ticker']} | TF: {a['tf'].upper()}{flip_tag}")
            print(f"  Close: ${a['close']:.2f}  ZLEMA: ${a['zlema']:.2f}  Ext: -{a['bear_ext']:.1f}%")
            print(f"  Streak: {a['down_streak']} bars  |  Trailing SL: ${a['bear_sl']:.2f}")
            msg_lines.append(
                f"*{a['ticker']}* ({a['tf'].upper()}){flip_tag} | ${a['close']:.2f} -{a['bear_ext']:.1f}% | "
                f"Streak {a['down_streak']} | SL ~${a['bear_sl']:.2f}"
            )

    if not bull_alerts and not bear_alerts:
        print("\nNo strong ZLEMA alignments right now.")
    else:
        send_telegram_alert("\n".join(msg_lines))

    print("\nDone. Risk responsibly. 🎯")


if __name__ == "__main__":
    main()
