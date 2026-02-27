import pandas as pd
import pandas_ta as ta
import yfinance as yf

df = yf.download("AAPL", period="1y", interval="1d")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

df.ta.bbands(length=20, std=2, append=True)
df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
try:
    df.ta.ichimoku(append=True)
except:
    pass
df.ta.adx(length=14, append=True)
df.ta.cci(length=14, append=True)
df.ta.willr(length=14, append=True)
df.ta.mfi(length=14, append=True)

print(df.columns[-20:])
