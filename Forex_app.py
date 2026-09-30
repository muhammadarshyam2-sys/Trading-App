import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf

from smc import find_fvg, find_order_blocks, fib_levels
from backtest import run_backtest, optimize

st.set_page_config(page_title="Forex Chart App", layout="wide")

SYMBOLS = {
    "Gold (XAUUSD)": "GC=F",
    "Silver (XAGUSD)": "SI=F",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "EURGBP": "EURGBP=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "AUDJPY": "AUDJPY=X",
    "BTCUSD": "BTC-USD",
}
INTERVALS = {"1m": "5d", "5m": "30d", "15m": "30d", "30m": "30d", "1h": "180d", "1d": "2y"}


@st.cache_data(ttl=60, show_spinner=False)
def load(ticker, interval, period):
    df = yf.download(ticker, interval=interval, period=period, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    if df.index.tz is not None:
        df.index = df.index.tz_convert("UTC").tz_localize(None)
    return df


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


# ---------------- Sidebar ----------------
st.sidebar.title("Settings")
name = st.sidebar.selectbox("Pair", list(SYMBOLS.keys()))
interval = st.sidebar.selectbox("Timeframe", list(INTERVALS.keys()), index=1)
bars = st.sidebar.slider("Candles to show", 50, 1000, 250, 50)

st.sidebar.subheader("Indicators")
ema_on = st.sidebar.checkbox("EMA 20 / 50 / 200", True)
sma_on = st.sidebar.checkbox("SMA 100", False)
bb_on = st.sidebar.checkbox("Bollinger Bands", False)
rsi_on = st.sidebar.checkbox("RSI", False)
macd_on = st.sidebar.checkbox("MACD", False)

st.sidebar.subheader("Smart Money")
fvg_on = st.sidebar.checkbox("Fair Value Gaps", True)
ob_on = st.sidebar.checkbox("Order Blocks", True)
fib_on = st.sidebar.checkbox("Fibonacci + Premium/Discount", True)
show_mitigated = st.sidebar.checkbox("Mitigated zones bhi dikhao", False)
fib_lb = st.sidebar.slider("Fib lookback (candles)", 30, 500, 150, 10)
ob_atr = st.sidebar.slider("OB impulse (x ATR)", 0.5, 3.0, 1.5, 0.1)

df = load(SYMBOLS[name], interval, INTERVALS[interval])

st.title(f"{name} - {interval}")
if df.empty:
    st.error("Data nahi mila. Thori dair baad refresh karein ya dusra pair/timeframe chunein.")
    st.stop()

tab_chart, tab_bt = st.tabs(["Chart", "Backtest"])

# ---------------- Chart ----------------
with tab_chart:
    d = df.copy()
    d["EMA20"] = d["Close"].ewm(span=20, adjust=False).mean()
    d["EMA50"] = d["Close"].ewm(span=50, adjust=False).mean()
    d["EMA200"] = d["Close"].ewm(span=200, adjust=False).mean()
    d["SMA100"] = d["Close"].rolling(100).mean()
    mid = d["Close"].rolling(20).mean()
    sd = d["Close"].rolling(20).std()
    d["BBU"], d["BBL"] = mid + 2 * sd, mid - 2 * sd
    d["RSI"] = rsi(d["Close"])
    macd = d["Close"].ewm(span=12, adjust=False).mean() - d["Close"].ewm(span=26, adjust=False).mean()
    sig = macd.ewm(span=9, adjust=False).mean()
    d["MACD"], d["SIG"], d["HIST"] = macd, sig, macd - sig

    rows = 1 + int(rsi_on) + int(macd_on)
    heights = [0.6] + [0.2] * (rows - 1) if rows > 1 else [1]
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=heights)
    v = d.iloc[-bars:]
    x0_view = v.index[0]

    fig.add_trace(go.Candlestick(x=v.index, open=v["Open"], high=v["High"], low=v["Low"], close=v["Close"], name="Price",
                                 increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), row=1, col=1)
    if ema_on:
        for col, color in [("EMA20", "#2962ff"), ("EMA50", "#ff9800"), ("EMA200", "#9c27b0")]:
            fig.add_trace(go.Scatter(x=v.index, y=v[col], name=col, line=dict(width=1.2, color=color)), row=1, col=1)
    if sma_on:
        fig.add_trace(go.Scatter(x=v.index, y=v["SMA100"], name="SMA100", line=dict(width=1.2, color="#795548")), row=1, col=1)
    if bb_on:
        fig.add_trace(go.Scatter(x=v.index, y=v["BBU"], name="BB Upper", line=dict(width=1, color="#607d8b")), row=1, col=1)
        fig.add_trace(go.Scatter(x=v.index, y=v["BBL"], name="BB Lower", line=dict(width=1, color="#607d8b"), fill="tonexty",
                                 fillcolor="rgba(96,125,139,0.08)"), row=1, col=1)

    # Smart money zones (calculated on full data, drawn only in view)
    zones = []
    if fvg_on:
        zones += find_fvg(df)
    if ob_on:
        zones += find_order_blocks(df, atr_mult=ob_atr)
    zones = [z for z in zones if z["x1"] >= x0_view and (show_mitigated or z["mit_i"] is None)]
    zones = sorted(zones, key=lambda z: z["i"])[-40:]
    for z in zones:
        bull = z["side"] == "bull"
        if z["kind"] == "FVG":
            fill = "rgba(38,166,154,0.25)" if bull else "rgba(239,83,80,0.25)"
        else:
            fill = "rgba(41,98,255,0.25)" if bull else "rgba(255,152,0,0.30)"
        fig.add_shape(type="rect", x0=max(z["x0"], x0_view), x1=z["x1"], y0=z["bottom"], y1=z["top"],
                      fillcolor=fill, line=dict(width=0), layer="below", row=1, col=1)
        fig.add_annotation(x=max(z["x0"], x0_view), y=z["top"], text=f'{z["kind"]}{"+" if bull else "-"}', showarrow=False,
                           font=dict(size=9), xanchor="left", yanchor="bottom", row=1, col=1)

    if fib_on:
        fb = fib_levels(df, fib_lb)
        cols = {0: "#787b86", 0.236: "#f44336", 0.382: "#ff9800", 0.5: "#4caf50", 0.618: "#00bcd4", 0.786: "#2196f3", 1: "#787b86"}
        for r, y in fb["levels"].items():
            fig.add_shape(type="line", x0=max(fb["x0"], x0_view), x1=v.index[-1], y0=y, y1=y,
                          line=dict(color=cols[r], width=1, dash="dot"), row=1, col=1)
            fig.add_annotation(x=v.index[-1], y=y, text=f"{r} ({y:.2f})", showarrow=False, xanchor="right",
                               yanchor="bottom", font=dict(size=9, color=cols[r]), row=1, col=1)
        eq, hi, lo = fb["eq"], fb["hi"], fb["lo"]
        xa, xb = max(fb["x0"], x0_view), v.index[-1]
        fig.add_shape(type="rect", x0=xa, x1=xb, y0=eq, y1=hi, fillcolor="rgba(239,83,80,0.05)", line=dict(width=0), layer="below", row=1, col=1)
        fig.add_shape(type="rect", x0=xa, x1=xb, y0=lo, y1=eq, fillcolor="rgba(38,166,154,0.05)", line=dict(width=0), layer="below", row=1, col=1)

    r_i = 2
    if rsi_on:
        fig.add_trace(go.Scatter(x=v.index, y=v["RSI"], name="RSI", line=dict(color="#7e57c2", width=1.2)), row=r_i, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="gray", row=r_i, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="gray", row=r_i, col=1)
        r_i += 1
    if macd_on:
        fig.add_trace(go.Bar(x=v.index, y=v["HIST"], name="Hist", marker_color=np.where(v["HIST"] >= 0, "#26a69a", "#ef5350")), row=r_i, col=1)
        fig.add_trace(go.Scatter(x=v.index, y=v["MACD"], name="MACD", line=dict(color="#2962ff", width=1)), row=r_i, col=1)
        fig.add_trace(go.Scatter(x=v.index, y=v["SIG"], name="Signal", line=dict(color="#ff9800", width=1)), row=r_i, col=1)

    fig.update_layout(height=650 if rows == 1 else 800, xaxis_rangeslider_visible=False, template="plotly_dark",
                      margin=dict(l=5, r=5, t=10, b=5), legend=dict(orientation="h", y=1.02), dragmode="pan")
    fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])] if interval != "1d" and "BTC" not in name else [])
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": True})

    last = df["Close"].iloc[-1]
    if fib_on:
        zone = "PREMIUM (sell zone)" if last > fb["eq"] else "DISCOUNT (buy zone)"
        st.info(f"Last price: {last:.4f} | Range midpoint (equilibrium): {fb['eq']:.4f} | Abhi price: {zone}")

# ---------------- Backtest ----------------
with tab_bt:
    st.subheader("Strategy backtest")
    c1, c2, c3 = st.columns(3)
    strategy = c1.selectbox("Strategy", ["OB+FVG", "OB only", "FVG only", "EMA Cross"])
    strategy = {"OB only": "OB", "FVG only": "FVG"}.get(strategy, strategy)
    rr = c2.number_input("Risk:Reward", 0.5, 10.0, 2.0, 0.5)
    risk_pct = c3.number_input("Risk per trade (%)", 0.1, 10.0, 1.0, 0.1)
    c4, c5, c6 = st.columns(3)
    ema_len = c4.number_input("Trend EMA", 5, 300, 50, 5)
    atr_mult = c5.number_input("OB impulse (x ATR)", 0.5, 3.0, 1.5, 0.1)
    spread = c6.number_input("Spread/cost (price units)", 0.0, 10.0, 0.0, 0.01)
    use_trend = st.checkbox("Trend filter (EMA)", True)

    if st.button("Run backtest", type="primary"):
        trades, stats, eq = run_backtest(df, strategy=strategy, rr=rr, ema_len=int(ema_len), atr_mult=atr_mult,
                                         use_trend=use_trend, risk_pct=risk_pct, spread=spread)
        m = st.columns(5)
        m[0].metric("Trades", stats["Trades"])
        m[1].metric("Win rate", f'{stats["WinRate"]}%')
        m[2].metric("Net R", stats["NetR"])
        m[3].metric("Profit factor", stats["ProfitFactor"])
        m[4].metric("Max DD", f'{stats["MaxDD"]}%')
        ef = go.Figure(go.Scatter(y=eq, mode="lines", line=dict(color="#26a69a")))
        ef.update_layout(height=300, template="plotly_dark", title="Equity curve (start = 100)", margin=dict(l=5, r=5, t=40, b=5))
        st.plotly_chart(ef, use_container_width=True)
        if trades:
            st.dataframe(pd.DataFrame(trades).round(4), use_container_width=True)
        else:
            st.warning("Is data par koi trade nahi bana. Settings badal kar dekhein.")

    st.divider()
    st.subheader("Parameter optimizer")
    o1, o2, o3 = st.columns(3)
    rr_txt = o1.text_input("RR values", "1.5,2,3")
    ema_txt = o2.text_input("EMA values", "20,50,100")
    atr_txt = o3.text_input("OB ATR values", "1,1.5,2")
    if st.button("Run optimizer"):
        try:
            rr_l = [float(x) for x in rr_txt.split(",")]
            ema_l = [int(x) for x in ema_txt.split(",")]
            atr_l = [float(x) for x in atr_txt.split(",")]
            with st.spinner("Optimizing..."):
                res = optimize(df, strategy, rr_l, ema_l, atr_l, risk_pct=risk_pct, spread=spread)
            if res.empty:
                st.warning("Koi result nahi (kam trades). Values badal kar dekhein.")
            else:
                st.dataframe(res, use_container_width=True)
                st.caption("Note: best result sirf isi data par hai (overfitting ka khatra). Dusre time-period par bhi test karein.")
        except ValueError:
            st.error("Values comma se alag likhein, jaise 1.5,2,3")

st.caption("Sirf education/analysis ke liye. Financial advice nahi. yfinance data delayed ho sakta hai.")
