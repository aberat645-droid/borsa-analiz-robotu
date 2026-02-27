import re
import os

app_path = r"c:\Users\aydem\.gemini\antigravity\scratch\streamlit-borsa\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Replace calculate_technical_indicators
new_ta = """def calculate_technical_indicators(df):
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        # pandas_ta ile 10 strateji indikatör hesaplamaları
        df.ta.rsi(length=14, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        try:
            df.ta.ichimoku(append=True)
        except Exception:
            pass
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=5, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=22, append=True)
        df.ta.sma(length=50, append=True)
        df.ta.sma(length=200, append=True)
        df.ta.ema(length=50, append=True)
        df.ta.ema(length=200, append=True)
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        df.ta.cci(length=14, append=True)
        df.ta.willr(length=14, append=True)
        df.ta.supertrend(length=10, multiplier=3, append=True)

        # Eski kodla tam uyumluluk (Arayüz ve grafiklerin çökmemesi için)
        if 'RSI_14' in df.columns: df['RSI'] = df['RSI_14']
        if 'MACD_12_26_9' in df.columns: df['MACD'] = df['MACD_12_26_9']
        if 'MACDh_12_26_9' in df.columns: df['MACD_Hist'] = df['MACDh_12_26_9']
        if 'MACDs_12_26_9' in df.columns: df['MACD_Signal'] = df['MACDs_12_26_9']
        if 'BBU_20_2.0' in df.columns: df['Upper_Band'] = df['BBU_20_2.0']
        if 'BBL_20_2.0' in df.columns: df['Lower_Band'] = df['BBL_20_2.0']

        if 'ISA_9' in df.columns: df['Senkou_Span_A'] = df['ISA_9'] # Ichimoku Senkou Span A
        if 'ISB_26' in df.columns: df['Senkou_Span_B'] = df['ISB_26']
        if 'ITS_9' in df.columns: df['Tenkan_sen'] = df['ITS_9']
        if 'IKS_26' in df.columns: df['Kijun_sen'] = df['IKS_26']

        if 'ADX_14' in df.columns: df['ADX'] = df['ADX_14']
        if 'STOCHk_14_3_3' in df.columns: df['Stoch_K'] = df['STOCHk_14_3_3']
        if 'STOCHd_14_3_3' in df.columns: df['Stoch_D'] = df['STOCHd_14_3_3']
        if 'CCI_14_0.015' in df.columns: df['CCI'] = df['CCI_14_0.015']
        if 'WILLR_14' in df.columns: df['Williams_R'] = df['WILLR_14']
        
        if 'SUPERTd_10_3.0' in df.columns: df['Trend_Dir'] = df['SUPERTd_10_3.0'] # 1 (Bull), -1 (Bear)

        df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
    except Exception as e:
        pass
    return df
"""

# Find calculate_technical_indicators
pattern = re.compile(r"def calculate_technical_indicators\(df\):.*?return df\n", re.DOTALL)
content = re.sub(pattern, new_ta, content)

# 2. Add try-except wrapper to all backtest functions
def wrap_try_except(func_str, name):
    parts = func_str.split("\n", 1)
    indent = "    "
    wrapped_body = "\n".join([indent + line if line.strip() else "" for line in parts[1].split("\n")])
    return f"{parts[0]}\n    try:\n{wrapped_body}    except Exception:\n        return initial_balance, 0, 0.0\n"

# Replace backtest_rsi_macd_strategy
rsi_pattern = re.compile(r"def backtest_rsi_macd_strategy\(df, initial_balance=10000\):.*?return final_value, total_trades, win_rate\n", re.DOTALL)
match = rsi_pattern.search(content)
if match: content = content.replace(match.group(0), wrap_try_except(match.group(0), "rsi"))

st_pattern = re.compile(r"def backtest_supertrend_strategy\(df, initial_balance=10000\):.*?return final_value, total_trades, win_rate\n", re.DOTALL)
match = st_pattern.search(content)
if match: content = content.replace(match.group(0), wrap_try_except(match.group(0), "st"))

ma_pattern = re.compile(r"def backtest_ma_cross_strategy\(df, initial_balance=10000\):.*?return final_value, total_trades, win_rate\n", re.DOTALL)
match = ma_pattern.search(content)
if match: content = content.replace(match.group(0), wrap_try_except(match.group(0), "ma"))

sim_pattern = re.compile(r"def bt_simulator\(df, signal_logic, initial_balance=10000\):.*?return final_val, total_trades, win_rate\n", re.DOTALL)
match = sim_pattern.search(content)
if match: content = content.replace(match.group(0), wrap_try_except(match.group(0), "sim"))

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)

print("Injected pandas_ta with try-except fallback")
