"""
TradingView-style Forex Chart App
(Streamlit + yfinance + Plotly)
Install: pip install streamlit
yfinance plotly pandas numpy
Run: streamlit run forex_app.py
"""
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import
make_subplots
import backtest
import smc
st.set_page_config(page_title="Forex
Charts", layout="wide",
initial_sidebar_state="collapsed")
# ---------------- Markets MARKETS = {
"Majors": {"EUR/USD": "EURUSD=X",
"GBP/USD": "GBPUSD=X", "USD/JPY":
"JPY=X",
"USD/CHF": "CHF=X",
"AUD/USD": "AUDUSD=X", "USD/CAD":
"CAD=X",
"NZD/USD":
"NZDUSD=X"},
"Minors": {"EUR/GBP": "EURGBP=X",
"EUR/JPY": "EURJPY=X", "GBP/JPY":
"GBPJPY=X",
"AUD/JPY": "AUDJPY=X",
"EUR/CHF": "EURCHF=X", "EUR/AUD":
"EURAUD=X",
"GBP/CHF": "GBPCHF=X",
"CAD/JPY": "CADJPY=X", "NZD/JPY":
"NZDJPY=X"},
"Exotics": {"USD/PKR": "PKR=X",
"USD/INR": "INR=X", "USD/TRY":
"TRY=X",
"USD/ZAR": "ZAR=X",
"USD/MXN": "MXN=X", "USD/SGD":
"SGD=X"},
"Metals/Energy/Crypto": {"Gold":
"GC=F", "Silver": "SI=F", "Oil
(WTI)": "CL=F",
"Bitcoin": "BTC-USD", "Ethereum": "ETH-USD"},
}
PERIOD = {"1m": "7d", "5m": "60d",
"15m": "60d", "30m": "60d",
"1h": "730d", "1d": "5y",
"1wk": "10y"}
MT5_MAP = {"Gold": "XAUUSD",
"Silver": "XAGUSD", "Oil (WTI)":
"USOIL",
"Bitcoin": "BTCUSD",
"Ethereum": "ETHUSD"}
def load_mt5(sym, interval, n=5000):
"""Live data from MetaTrader 5
(Windows + MT5 terminal open + pip
install MetaTrader5)."""
import MetaTrader5 as mt5
tfm = {"1m": mt5.TIMEFRAME_M1,
"5m": mt5.TIMEFRAME_M5, "15m":
mt5.TIMEFRAME_M15,
"30m": mt5.TIMEFRAME_M30,
"1h": mt5.TIMEFRAME_H1, "1d":
mt5.TIMEFRAME_D1,
"1wk": mt5.TIMEFRAME_W1}
if not mt5.initialize():
raise RuntimeError(f"MT5 connect nahi hua:
{mt5.last_error()}")
mt5.symbol_select(sym, True)
rates =
mt5.copy_rates_from_pos(sym,
tfm[interval], 0, n)
mt5.shutdown()
if rates is None or len(rates) ==
0:
raise RuntimeError(f"MT5
symbol '{sym}' ka data nahi mila")
df = pd.DataFrame(rates)
df.index =
pd.to_datetime(df["time"], unit="s")
df = df.rename(columns={"open":
"Open", "high": "High", "low": "Low",
"close":
"Close", "tick_volume": "Volume"})
return df[["Open", "High", "Low",
"Close", "Volume"]]
@st.cache_data(ttl=20)
def load(symbol, interval,
source="Yahoo"):
if source == "MT5":
return load_mt5(symbol,
interval)
df = yf.download(symbol, period=PERIOD[interval],
interval=interval,
progress=False,
auto_adjust=False)
if isinstance(df.columns,
pd.MultiIndex):
df.columns =
df.columns.get_level_values(0)
return df.dropna()
# ---------------- Indicators -------
---------
def sma(s, n): return
s.rolling(n).mean()
def ema(s, n): return s.ewm(span=n,
adjust=False).mean()
def rsi(s, n=14):
d = s.diff()
up = d.clip(lower=0).ewm(alpha=1
/ n, adjust=False).mean()
dn = (-
d.clip(upper=0)).ewm(alpha=1 / n,
adjust=False).mean()
return 100 - 100 / (1 + up / dn) def macd(s, f=12, sl=26, sig=9):
m = ema(s, f) - ema(s, sl)
sg = ema(m, sig)
return m, sg, m - sg
def stoch(df, k=14, d=3):
lo, hi = df.Low.rolling(k).min(),
df.High.rolling(k).max()
kk = 100 * (df.Close - lo) / (hi
- lo)
return kk, kk.rolling(d).mean()
def atr(df, n=14):
pc = df.Close.shift()
tr = pd.concat([df.High - df.Low,
(df.High - pc).abs(), (df.Low -
pc).abs()], axis=1).max(axis=1)
return tr.ewm(alpha=1 / n,
adjust=False).mean()
def bollinger(s, n=20, k=2):
m, sd = sma(s, n),
s.rolling(n).std()
return m + k * sd, m, m - k * sd def supertrend(df, n=10, mult=3):
a = atr(df, n)
hl2 = (df.High + df.Low) / 2
ub, lb = (hl2 + mult * a).values,
(hl2 - mult * a).values
c = df.Close.values
st_, trend = np.full(len(df),
np.nan), True
for i in range(1, len(df)):
if c[i] > ub[i - 1]: trend =
True
elif c[i] < lb[i - 1]: trend
= False
if trend: lb[i] = max(lb[i],
lb[i - 1]) if c[i - 1] > lb[i - 1]
else lb[i]
else: ub[i] = min(ub[i], ub[i
- 1]) if c[i - 1] < ub[i - 1] else
ub[i]
st_[i] = lb[i] if trend else
ub[i]
return pd.Series(st_,
index=df.index)
def ichimoku(df):
t = (df.High.rolling(9).max() +
df.Low.rolling(9).min()) / 2
k = (df.High.rolling(26).max() + df.Low.rolling(26).min()) / 2
a = ((t + k) / 2).shift(26)
b = ((df.High.rolling(52).max() +
df.Low.rolling(52).min()) /
2).shift(26)
return t, k, a, b
def swings(df, w=5):
hi = df.High[(df.High ==
df.High.rolling(2 * w + 1,
center=True).max())]
lo = df.Low[(df.Low ==
df.Low.rolling(2 * w + 1,
center=True).min())]
return hi, lo
# ---------------- Sidebar ----------
------
with st.sidebar:
st.title("Forex Charts")
page = st.radio("Page", ["Chart",
"Backtest"], horizontal=True)
group = st.selectbox("Market",
list(MARKETS))
name = st.selectbox("Pair",
list(MARKETS[group]))
symbol = MARKETS[group][name] tf = st.selectbox("Timeframe",
list(PERIOD), index=1)
src = st.radio("Data source",
["Yahoo (free)", "MT5 (live)"],
horizontal=True)
if src.startswith("MT5"):
symbol = st.text_input("MT5
symbol", MT5_MAP.get(name,
name.replace("/", "")))
bars = st.slider("Candles", 50,
1000, 200)
ctype = st.radio("Chart",
["Candles", "Line"], horizontal=True)
st.subheader("Overlays")
o_sma = st.checkbox("SMA 20/50")
o_ema = st.checkbox("EMA 9/21")
o_bb = st.checkbox("Bollinger
Bands")
o_st = st.checkbox("Supertrend")
o_ich = st.checkbox("Ichimoku")
o_liq = st.checkbox("Liquidity
(Swing H/L)", True)
o_pd = st.checkbox("Premium /
Discount", True)
o_fvg = st.checkbox("Fair Value
Gaps (FVG)")
o_ob = st.checkbox("Order
Blocks") o_fib = st.checkbox("Fibonacci")
st.subheader("Panels")
p_rsi = st.checkbox("RSI", True)
p_macd = st.checkbox("MACD")
p_sto = st.checkbox("Stochastic")
p_atr = st.checkbox("ATR")
p_vol = st.checkbox("Volume")
live = st.checkbox("Auto refresh
(20s)")
# ---------------- Data + chart -----
-----------
try:
df = load(symbol, tf, "MT5" if
src.startswith("MT5") else "Yahoo")
except Exception as e:
st.error(f"Data error: {e}")
st.stop()
if df.empty:
st.error("Data nahi mila. Doosra
pair/timeframe try karo.")
st.stop()
if page == "Backtest":
backtest.ui(df, name, tf)
st.stop() full = df.copy()
df = df.tail(bars)
x = df.index
panels = [n for n, on in [("RSI",
p_rsi), ("MACD", p_macd), ("Stoch",
p_sto),
("ATR",
p_atr), ("Volume", p_vol)] if on]
rows = 1 + len(panels)
heights = [0.6] + [0.4 /
max(len(panels), 1)] * len(panels)
fig = make_subplots(rows=rows,
cols=1, shared_xaxes=True,
vertical_spacing=0.02,
row_heights=heights)
if ctype == "Candles":
fig.add_candlestick(x=x,
open=df.Open, high=df.High,
low=df.Low, close=df.Close,
increasing_line_color="#26a69a",
decreasing_line_color="#ef5350",
name=name,
row=1, col=1)
else:
fig.add_scatter(x=x, y=df.Close, name=name,
line=dict(color="#2962ff"), row=1,
col=1)
def line(s, label, color, row=1,
dash=None):
fig.add_scatter(x=x,
y=s.reindex(x), name=label,
mode="lines",
line=dict(color=color, width=1.3,
dash=dash), row=row, col=1)
c = full.Close
if o_sma: line(sma(c, 20), "SMA20",
"#f9a825"); line(sma(c, 50), "SMA50",
"#ab47bc")
if o_ema: line(ema(c, 9), "EMA9",
"#29b6f6"); line(ema(c, 21), "EMA21",
"#ff7043")
if o_bb:
u, m, l = bollinger(c)
line(u, "BB up", "#90a4ae");
line(m, "BB mid", "#607d8b",
dash="dot"); line(l, "BB low",
"#90a4ae")
if o_st: line(supertrend(full), "Supertrend", "#00e676")
if o_ich:
t, k, a, b = ichimoku(full)
line(t, "Tenkan", "#2962ff");
line(k, "Kijun", "#b71c1c")
line(a, "Span A", "#66bb6a",
dash="dot"); line(b, "Span B",
"#ef5350", dash="dot")
if o_liq:
hi, lo = swings(df)
for t_, p in hi.tail(6).items():
fig.add_shape(type="line",
x0=t_, x1=x[-1], y0=p, y1=p,
line=dict(color="#ef5350", width=1,
dash="dash"), row=1, col=1)
for t_, p in lo.tail(6).items():
fig.add_shape(type="line",
x0=t_, x1=x[-1], y0=p, y1=p,
line=dict(color="#26a69a", width=1,
dash="dash"), row=1, col=1)
fig.add_scatter(x=hi.index,
y=hi.values, mode="markers",
marker=dict(symbol="triangle-down",
color="#ef5350", size=7), name="Swing
High", row=1, col=1)
fig.add_scatter(x=lo.index,
y=lo.values, mode="markers",
marker=dict(symbol="triangle-up", color="#26a69a", size=7), name="Swing
Low", row=1, col=1)
if o_pd:
top, bot = df.High.max(),
df.Low.min()
mid, span = (top + bot) / 2, top
- bot
fig.add_hrect(y0=mid + span *
0.05, y1=top, fillcolor="red",
opacity=0.07, line_width=0, row=1,
col=1)
fig.add_hrect(y0=bot, y1=mid -
span * 0.05, fillcolor="green",
opacity=0.07, line_width=0, row=1,
col=1)
fig.add_hline(y=mid,
line=dict(color="gray", dash="dot"),
row=1, col=1)
if o_fvg:
for t_, lo_, hi_, k in
smc.fvg(df)[-12:]:
fig.add_shape(type="rect",
x0=t_, x1=x[-1], y0=lo_, y1=hi_,
line_width=0, opacity=0.3,
fillcolor="#26a69a" if k == "bull"
else "#ef5350", row=1, col=1) if o_ob:
for t_, lo_, hi_, k in
smc.order_blocks(df)[-8:]:
fig.add_shape(type="rect",
x0=t_, x1=x[-1], y0=lo_, y1=hi_,
opacity=0.35,
line=dict(color="#ab47bc" if k ==
"bull" else "#ff9800", width=1),
fillcolor="#ab47bc" if k == "bull"
else "#ff9800", row=1, col=1)
if o_fib:
for lv, pr in
smc.fib_levels(df).items():
fig.add_hline(y=pr,
line=dict(color="#ffd54f", width=1,
dash="dot"),
annotation_text=f"Fib {lv}",
annotation_position="right", row=1,
col=1)
r = 2
for p in panels:
if p == "RSI":
line(rsi(c), "RSI",
"#b388ff", r)
fig.add_hline(y=70, line=dict(color="gray", dash="dot"),
row=r, col=1)
fig.add_hline(y=30,
line=dict(color="gray", dash="dot"),
row=r, col=1)
elif p == "MACD":
m, sg, h = macd(c)
fig.add_bar(x=x,
y=h.reindex(x), name="Hist",
marker_color=np.where(h.reindex(x) >=
0, "#26a69a", "#ef5350"), row=r,
col=1)
line(m, "MACD", "#2962ff",
r); line(sg, "Signal", "#ff6d00", r)
elif p == "Stoch":
k, d = stoch(full)
line(k, "%K", "#2962ff", r);
line(d, "%D", "#ff6d00", r)
elif p == "ATR":
line(atr(full), "ATR",
"#ffab00", r)
elif p == "Volume":
fig.add_bar(x=x, y=df.Volume,
name="Volume",
marker_color="#546e7a", row=r, col=1)
r += 1
last, prev = df.Close.iloc[-1],
df.Close.iloc[-2] st.subheader(f"{name} {last:.5f}
({(last - prev) / prev *
100:+.3f}%)")
fig.update_layout(template="plotly_da
rk", height=520 + 130 * len(panels),
xaxis_rangeslider_visible=False,
margin=dict(l=5, r=5, t=10, b=5),
legend=dict(orientation="h", y=1.02),
dragmode="pan")
fig.update_xaxes(rangebreaks=
[dict(bounds=["sat", "mon"])] if tf
!= "1d" and group !=
"Metals/Energy/Crypto" else [])
st.plotly_chart(fig,
use_container_width=True, config=
{"scrollZoom": True,
"modeBarButtonsToAdd": [
"drawline",
"drawopenpath", "drawrect",
"drawcircle", "eraseshape"]})
if live:
time.sleep(20)
st.cache_data.clear()
st.rerun()
