import os
import re

app_path = r"c:\Users\aydem\.gemini\antigravity\scratch\streamlit-borsa\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update calculate_technical_indicators
pandas_indicators = """
    # --- YENİ TEKNİK İNDİKATÖRLER ---
    low_14 = low.rolling(window=14).min()
    high_14 = high.rolling(window=14).max()
    df['Stoch_K'] = 100 * ((close_series - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    tenkan = (high.rolling(window=9).max() + low.rolling(window=9).min()) / 2
    kijun = (high.rolling(window=26).max() + low.rolling(window=26).min()) / 2
    df['Senkou_Span_A'] = ((tenkan + kijun) / 2).shift(26)
    df['Senkou_Span_B'] = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(26)
    df['Tenkan_sen'] = tenkan
    df['Kijun_sen'] = kijun
    
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
    atr_14 = tr.rolling(window=14).sum()
    plus_di = 100 * (pd.Series(plus_dm, index=df.index).rolling(window=14).sum() / atr_14)
    minus_di = 100 * (pd.Series(minus_dm, index=df.index).rolling(window=14).sum() / atr_14)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    df['ADX'] = dx.rolling(window=14).mean()
    
    tp = (high + low + close_series) / 3
    sma_tp = tp.rolling(window=14).mean()
    mad = tp.rolling(window=14).apply(lambda x: np.abs(x - x.mean()).mean())
    df['CCI'] = (tp - sma_tp) / (0.015 * mad)
    
    df['Williams_R'] = -100 * ((high_14 - close_series) / (high_14 - low_14))
    
    raw_money_flow = tp * volume_series
    pos_flow = pd.Series(np.where(tp > tp.shift(1), raw_money_flow, 0.0), index=df.index)
    neg_flow = pd.Series(np.where(tp < tp.shift(1), raw_money_flow, 0.0), index=df.index)
    
    # 0'a bölme hatasını önlemek için +1e-5 eklendi
    mfr = pos_flow.rolling(window=14).sum() / (neg_flow.rolling(window=14).sum() + 1e-5)
    df['MFI'] = 100 - (100 / (1 + mfr))
"""

if "df['SMA_50']" in content and "df['Stoch_K']" not in content:
    content = content.replace("df['SMA_50'] = close_series.rolling(window=50).mean()", 
        pandas_indicators + "\n    df['SMA_50'] = close_series.rolling(window=50).mean()")

# 2. Add New Backtest Strategies
new_backtests = """
def bt_simulator(df, signal_logic, initial_balance=10000):
    if df.empty or len(df) < 50:
        return initial_balance, 0, 0.0
    balance, shares, total_trades, success = initial_balance, 0, 0, 0
    last_buy = 0
    for i in range(50, len(df)):
        signal = signal_logic(df, i, shares, last_buy)
        price = df['Close'].iloc[i]
        if signal == 1 and shares == 0:
            shares = balance / price
            balance = 0
            last_buy = price
            total_trades += 1
        elif signal == -1 and shares > 0:
            balance += shares * price
            if price > last_buy: success += 1
            shares = 0
            total_trades += 1
    final_val = balance + (shares * df['Close'].iloc[-1])
    win_rate = (success / (total_trades // 2) * 100) if (total_trades // 2) > 0 else 0
    return final_val, total_trades, win_rate

def bt_bbands(df, init_bal=10000):
    def logic(d, i, shares, buy_p):
        if d['Close'].iloc[i] < d['Lower_Band'].iloc[i]: return 1
        if shares > 0 and (d['Close'].iloc[i] > d['Upper_Band'].iloc[i] or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        return 0
    return bt_simulator(df, logic, init_bal)

def bt_stoch(df, init_bal=10000):
    def logic(d, i, shares, buy_p):
        if d['Stoch_K'].iloc[i] > d['Stoch_D'].iloc[i] and d['Stoch_K'].iloc[i] < 20: return 1
        if shares > 0 and (d['Stoch_K'].iloc[i] < d['Stoch_D'].iloc[i] and d['Stoch_K'].iloc[i] > 80 or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        return 0
    return bt_simulator(df, logic, init_bal)

def bt_ichimoku(df, init_bal=10000):
    def logic(d, i, shares, buy_p):
        price = d['Close'].iloc[i]
        span_a = d['Senkou_Span_A'].iloc[i]
        span_b = d['Senkou_Span_B'].iloc[i]
        if pd.isna(span_a) or pd.isna(span_b): return 0
        in_uptrend = price > max(span_a, span_b)
        tenkan_above_kijun = d['Tenkan_sen'].iloc[i] > d['Kijun_sen'].iloc[i]
        if in_uptrend and tenkan_above_kijun: return 1
        if shares > 0 and (price < min(span_a, span_b) or price <= buy_p * 0.93): return -1
        return 0
    return bt_simulator(df, logic, init_bal)

def bt_adx(df, init_bal=10000):
    def logic(d, i, shares, buy_p):
        if d['ADX'].iloc[i] > 25 and d['SMA_5'].iloc[i] > d['SMA_20'].iloc[i]: return 1
        if shares > 0 and (d['ADX'].iloc[i] < 20 or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        return 0
    return bt_simulator(df, logic, init_bal)

def bt_cci(df, init_bal=10000):
    def logic(d, i, shares, buy_p):
        if d['CCI'].iloc[i] > 100: return 1
        if shares > 0 and (d['CCI'].iloc[i] < -100 or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        return 0
    return bt_simulator(df, logic, init_bal)

def bt_willr(df, init_bal=10000):
    def logic(d, i, shares, buy_p):
        if d['Williams_R'].iloc[i] < -80 and d['Close'].iloc[i] > d['SMA_50'].iloc[i]: return 1
        if shares > 0 and (d['Williams_R'].iloc[i] > -20 or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        return 0
    return bt_simulator(df, logic, init_bal)

def bt_mfi(df, init_bal=10000):
    def logic(d, i, shares, buy_p):
        if d['MFI'].iloc[i] < 20: return 1
        if shares > 0 and (d['MFI'].iloc[i] > 80 or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        return 0
    return bt_simulator(df, logic, init_bal)
"""

if "def bt_simulator" not in content:
    content = content.replace("def get_current_signals(df):", new_backtests + "\ndef get_current_signals(df):")
    
# 3. Update execution block
new_strategies_dict = """    strategies = {
        "🚀 SuperTrend & Hacim": backtest_supertrend_strategy(df, 10000),
        "⚔️ Hareketli Ortalama Kesişimi (5/22)": backtest_ma_cross_strategy(df, 10000),
        "🛡️ RSI Dip Avcısı & MACD": backtest_rsi_macd_strategy(df, 10000),
        "🎡 Bollinger Bantları": bt_bbands(df, 10000),
        "🎢 Stochastic Oscillator": bt_stoch(df, 10000),
        "☁️ Ichimoku Bulutu": bt_ichimoku(df, 10000),
        "⚡ ADX (Trend Gücü)": bt_adx(df, 10000),
        "🎯 CCI": bt_cci(df, 10000),
        "📉 Williams %R": bt_willr(df, 10000),
        "💰 MFI (Para Akışı Endeksi)": bt_mfi(df, 10000)
    }"""
    
content = re.sub(
    r"    strategies = \{.*?\}", 
    new_strategies_dict, 
    content, 
    flags=re.DOTALL
)

# 4. Remove residual 'res_st = ...' if present
content = re.sub(r"    res_st = .*?\n", "", content)
content = re.sub(r"    res_ma = .*?\n", "", content)
content = re.sub(r"    res_rsi = .*?\n", "", content)

# 5. Fix logic on the UI for strategy mantığı to avoid key errors if they choose a new strategy
new_ui_logic = """    if "SuperTrend" in strategy_choice:
        st.markdown("**Strateji Mantığı:** SuperTrend (10, 3) Al sinyali ve Hacim Onayı ile işleme girer.")
    elif "Hareketli" in strategy_choice:
        st.markdown("**Strateji Mantığı:** 5 günlük ve 22 günlük Hareketli Ortalamaların kesişimini takip eder.")
    elif "RSI" in strategy_choice:
        st.markdown("**Strateji Mantığı:** Aşırı satılan yerlerde (RSI < 40) MACD'nin al verdiği güvenli yerlerde mal toplar.")
    else:
        st.markdown(f"**Strateji Mantığı:** {strategy_choice} stratejisi, ilgili indikatörlerin alım/satım kurallarına %7 stop-loss ile sadık kalır.")"""

content = re.sub(r"    if \"SuperTrend\" in strategy_choice:.*?else:.*?mal toplar\.\"\)", new_ui_logic, content, flags=re.DOTALL)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)
print("Injection Successful!")
