"""Smart-money helpers: Fair Value Gaps, Order Blocks, Fibonacci."""
import numpy as np
import pandas as pd


def _atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def fvg(df):
    """Unfilled Fair Value Gaps -> list of (start_time, bottom, top, 'bull'|'bear')."""
    h, l, idx = df.High.values, df.Low.values, df.index
    out = []
    for i in range(2, len(df)):
        if l[i] > h[i - 2]:
            kind, bot, top = "bull", h[i - 2], l[i]
        elif h[i] < l[i - 2]:
            kind, bot, top = "bear", h[i], l[i - 2]
        else:
            continue
        later = df.iloc[i + 1:]
        filled = (later.Low <= bot).any() if kind == "bull" else (later.High >= top).any()
        if not filled:
            out.append((idx[i - 2], bot, top, kind))
    return out


def order_blocks(df, k=1.5):
    """Unmitigated Order Blocks: last opposite candle before a strong displacement candle."""
    a = _atr(df).values
    o, c, h, l, idx = df.Open.values, df.Close.values, df.High.values, df.Low.values, df.index
    out = []
    for i in range(len(df) - 1):
        if np.isnan(a[i]):
            continue
        move = c[i + 1] - o[i + 1]
        if c[i] < o[i] and move > k * a[i]:
            kind = "bull"
        elif c[i] > o[i] and -move > k * a[i]:
            kind = "bear"
        else:
            continue
        later = df.iloc[i + 2:]
        mitigated = (later.Close < l[i]).any() if kind == "bull" else (later.Close > h[i]).any()
        if not mitigated:
            out.append((idx[i], l[i], h[i], kind))
    return out


def fib_levels(df):
    """Fibonacci retracement of the visible swing (auto direction)."""
    hi, lo = df.High.max(), df.Low.min()
    up = df.Low.idxmin() < df.High.idxmax()
    return {f"{r:g}": (hi - (hi - lo) * r if up else lo + (hi - lo) * r)
            for r in (0, .236, .382, .5, .618, .786, 1)}
