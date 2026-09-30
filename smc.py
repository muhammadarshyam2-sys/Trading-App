import numpy as np
import pandas as pd


def atr(df, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def find_fvg(df, min_gap_atr=0.0):
    """Fair Value Gaps. Returns list of dicts with formation index i and mitigation index mit_i."""
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    a = atr(df).values
    idx = df.index
    out = []
    for i in range(2, len(df)):
        if l[i] > h[i - 2]:
            gap = l[i] - h[i - 2]
            if np.isnan(a[i]) or gap >= min_gap_atr * a[i]:
                out.append(dict(kind="FVG", side="bull", i=i, bottom=h[i - 2], top=l[i], x0=idx[i - 2]))
        elif h[i] < l[i - 2]:
            gap = l[i - 2] - h[i]
            if np.isnan(a[i]) or gap >= min_gap_atr * a[i]:
                out.append(dict(kind="FVG", side="bear", i=i, bottom=h[i], top=l[i - 2], x0=idx[i - 2]))
    for z in out:
        z["mit_i"] = None
        for j in range(z["i"] + 1, len(df)):
            if z["side"] == "bull" and c[j] < z["bottom"]:
                z["mit_i"] = j
                break
            if z["side"] == "bear" and c[j] > z["top"]:
                z["mit_i"] = j
                break
        z["x1"] = idx[z["mit_i"]] if z["mit_i"] is not None else idx[-1]
    return out


def find_order_blocks(df, atr_mult=1.5, lookback=6):
    """Order block = last opposite candle before an impulsive move (body > atr_mult * ATR)."""
    o, h, l, c = df["Open"].values, df["High"].values, df["Low"].values, df["Close"].values
    a = atr(df).values
    idx = df.index
    out = []
    used = set()
    for i in range(lookback + 1, len(df)):
        if np.isnan(a[i]):
            continue
        body = c[i] - o[i]
        if abs(body) < atr_mult * a[i]:
            continue
        side = "bull" if body > 0 else "bear"
        for k in range(i - 1, max(i - lookback - 1, 0), -1):
            opposite = (c[k] < o[k]) if side == "bull" else (c[k] > o[k])
            if opposite:
                if k in used:
                    break
                used.add(k)
                out.append(dict(kind="OB", side=side, i=i, bottom=l[k], top=h[k], x0=idx[k]))
                break
    for z in out:
        z["mit_i"] = None
        for j in range(z["i"] + 1, len(df)):
            if z["side"] == "bull" and c[j] < z["bottom"]:
                z["mit_i"] = j
                break
            if z["side"] == "bear" and c[j] > z["top"]:
                z["mit_i"] = j
                break
        z["x1"] = idx[z["mit_i"]] if z["mit_i"] is not None else idx[-1]
    return out


def swings(df, n=5):
    """Pivot highs / lows."""
    h, l = df["High"], df["Low"]
    win = 2 * n + 1
    ph = h[h == h.rolling(win, center=True).max()]
    pl = l[l == l.rolling(win, center=True).min()]
    return ph, pl


def fib_levels(df, lookback=150):
    """Fib retracement between the highest high and lowest low of the last `lookback` bars.
    Direction follows which extreme came last. Returns dict with levels and premium/discount info."""
    d = df.iloc[-lookback:]
    hi, lo = d["High"].max(), d["Low"].min()
    hi_t, lo_t = d["High"].idxmax(), d["Low"].idxmin()
    up = lo_t < hi_t  # last leg is up
    ratios = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1]
    levels = {}
    for r in ratios:
        levels[r] = hi - (hi - lo) * r if up else lo + (hi - lo) * r
    return dict(levels=levels, hi=hi, lo=lo, eq=(hi + lo) / 2, up=up, x0=d.index[0], x1=d.index[-1])
