import re

app_path = r"c:\Users\aydem\.gemini\antigravity\scratch\streamlit-borsa\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

new_ta = """        df.ta.rsi(length=7, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        
        try:
            df.ta.ichimoku(append=True)
        except Exception:
            pass
            
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=5, append=True)
        df.ta.sma(length=22, append=True)
        df.ta.ema(length=9, append=True)
        df.ta.ema(length=21, append=True)
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        df.ta.cci(length=14, append=True)
        df.ta.willr(length=14, append=True)
        df.ta.supertrend(length=7, multiplier=2, append=True)
        
        # Sütun isimlerini arayüzde kolay kullanım için standartlaştıralım, varsa.
        ta_col_map = {
            'RSI_7': 'RSI',
            'MACD_12_26_9': 'MACD',
            'MACDh_12_26_9': 'MACD_Hist',
            'MACDs_12_26_9': 'MACD_Signal',
            'BBL_20_2.0': 'Lower_Band',
            'BBM_20_2.0': 'Middle_Band',
            'BBU_20_2.0': 'Upper_Band',
            'ADX_14': 'ADX',
            'STOCHk_14_3_3': 'Stoch_K',
            'STOCHd_14_3_3': 'Stoch_D',
            'CCI_14_0.015': 'CCI',
            'WILLR_14': 'Williams_R',
            'SMA_5': 'SMA_5',
            'SMA_22': 'SMA_22',
            'EMA_9': 'EMA_9',
            'EMA_21': 'EMA_21',
            'SUPERTd_7_2.0': 'Trend_Dir'
        }"""
        
content = re.sub(r"        df\.ta\.rsi\(length=14, append=True\).*?df\.ta\.ema\(length=200, append=True\)\n        df\.ta\.stoch\(k=14, d=3, smooth_k=3, append=True\)\n        df\.ta\.cci\(length=14, append=True\)\n        df\.ta\.willr\(length=14, append=True\)\n        \n        # Sütun isimlerini arayüzde kolay kullanım için standartlaştıralım, varsa\.\n        ta_col_map = \{.*?'EMA_200': 'EMA_200'\n        \}", new_ta, content, flags=re.DOTALL)

# Modify bt_simulator to return 4 things
simulator_repl = """            shares = 0
            total_trades += 1
            
    final_val = balance + (shares * df['Close'].iloc[-1])
    win_rate = (success / (total_trades // 2) * 100) if (total_trades // 2) > 0 else 0
    
    current_signal_code = signal_logic(df, len(df)-1, shares, last_buy)
    if current_signal_code == 1:
        sig_text = "AL"
    elif current_signal_code == -1:
        sig_text = "SAT"
    else:
        sig_text = "BEKLE"
        
    return final_val, total_trades, win_rate, sig_text"""
    
content = re.sub(r"            shares = 0\n            total_trades \+= 1\n            \n    final_val = balance \+ \(shares \* df\['Close'\]\.iloc\[-1\]\)\n    win_rate = \(success / \(total_trades // 2\) \* 100\) if \(total_trades // 2\) > 0 else 0\n    return final_val, total_trades, win_rate", simulator_repl, content)

# Modify ema cross
content = content.replace("e20 = d.get('EMA_20'", "e20 = d.get('EMA_9'")
content = content.replace("e50 = d.get('EMA_50'", "e50 = d.get('EMA_21'")
content = content.replace("pe20 = d.get('EMA_20'", "pe20 = d.get('EMA_9'")
content = content.replace("pe50 = d.get('EMA_50'", "pe50 = d.get('EMA_21'")
content = content.replace("EMA (20/50)", "EMA (9/21)")

# Add backtest_supertrend and inject to strategies
st_backtest = """def backtest_supertrend(df):
    def logic(d, i, shares, buy_p):
        try:
            td = d.get('Trend_Dir', pd.Series(dtype=float)).iloc[i]
            ptd = d.get('Trend_Dir', pd.Series(dtype=float)).iloc[i-1]
            if pd.isna(td) or pd.isna(ptd): return 0
            if ptd < 0 and td > 0: return 1
            if shares > 0 and (ptd > 0 and td < 0 or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        except: pass
        return 0
    return bt_simulator(df, logic)\n\n"""

content = content.replace("def run_all_strategies(df):", st_backtest + "def run_all_strategies(df):")
content = content.replace('"🔥 ADX (Trend Gücü)": backtest_adx(df)', '"🔥 ADX (Trend Gücü)": backtest_adx(df),\n        "🚀 SuperTrend (7/2)": backtest_supertrend(df)')

# Final block injection for current signal UI
ui_inject = """    best_strategy_name = max(strategies, key=lambda k: strategies[k][0])
    best_results = strategies[best_strategy_name]
    best_profit_pct = ((best_results[0] - 10000) / 10000) * 100
    current_signal = best_results[3]
    
    st.markdown("---")
    st.markdown(f"<h1 style='text-align: center; color: white;'>🎯 GÜNCEL SİNYAL: <span style='color: {'#00ff00' if current_signal == 'AL' else '#ff0000' if current_signal == 'SAT' else '#ffff00'};'>{current_signal}</span></h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'>🏆 Tavsiye Eden Şampiyon Taktik: {best_strategy_name}</h4>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.success(f"Eğer 1 yıl önce bu taktikle {display_symbol} hissesine 10.000 ₺ bağlansaydınız, getiri oranınız **%{best_profit_pct:.2f}** ile sonucunuz **{best_results[0]:,.2f} ₺** olurdu.")
    st.markdown("---")"""
    
content = re.sub(r"    best_strategy_name = max\(strategies, key=lambda k: strategies\[k\]\[0\]\)\n    best_results = strategies\[best_strategy_name\]\n    best_profit_pct = \(\(best_results\[0\] - 10000\) / 10000\) \* 100\n    \n    st\.markdown\(.*?st\.markdown\(\"---\"\)", ui_inject, content, flags=re.DOTALL)

content = content.replace("val, trades, win = strategies[strat_choice]", "val, trades, win, sig = strategies[strat_choice]")


with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)

print("success!")
