import re

app_path = r"c:\Users\aydem\.gemini\antigravity\scratch\streamlit-borsa\app.py"
with open(app_path, "r", encoding="utf-8") as f:
    content = f.read()

# 1. Update apply_ta to calculate new parameters and add df.dropna
new_ta = """        df.ta.rsi(length=3, append=True)
        df.ta.macd(fast=12, slow=26, signal=9, append=True)
        df.ta.bbands(length=20, std=2, append=True)
        
        try:
            df.ta.ichimoku(append=True)
        except Exception:
            pass
            
        df.ta.adx(length=14, append=True)
        df.ta.sma(length=5, append=True)
        df.ta.sma(length=22, append=True)
        df.ta.ema(length=3, append=True)
        df.ta.ema(length=8, append=True)
        df.ta.stoch(k=14, d=3, smooth_k=3, append=True)
        df.ta.cci(length=14, append=True)
        df.ta.willr(length=14, append=True)
        df.ta.supertrend(length=5, multiplier=1.5, append=True)
        
        # Sütun isimlerini arayüzde kolay kullanım için standartlaştıralım, varsa.
        ta_col_map = {
            'RSI_3': 'RSI',
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
            'EMA_3': 'EMA_3',
            'EMA_8': 'EMA_8',
            'SUPERTd_5_1.5': 'Trend_Dir'
        }
        
        for old_c, new_c in ta_col_map.items():
            if old_c in df.columns:
                df[new_c] = df[old_c]
                
        # Hataları engelle: Dropna eklendi.
        df.dropna(inplace=True)
        
    except Exception as e:
        st.warning(f"İndikatörler hesaplanırken bir sorun oluştu: {e}")
        
    return df"""
    
content = re.sub(r"        df\.ta\.rsi\(length=7, append=True\).*?return df", new_ta, content, flags=re.DOTALL)

# 2. Update bt_simulator to also return hold times
simulator_repl = """def bt_simulator(df, signal_logic, initial_balance=10000):
    if df.empty or len(df) < 20:
        return initial_balance, 0, 0.0, "BEKLE", 0
        
    balance = initial_balance
    shares = 0
    total_trades = 0
    success = 0
    last_buy = 0
    hold_days = []
    buy_date_index = 0
    
    for i in range(20, len(df)):
        signal = signal_logic(df, i, shares, last_buy)
        price = df['Close'].iloc[i]
        
        if signal == 1 and shares == 0:
            shares = balance / price
            balance = 0
            last_buy = price
            buy_date_index = i
            total_trades += 1
        elif signal == -1 and shares > 0:
            balance += shares * price
            if price > last_buy:
                success += 1
            shares = 0
            total_trades += 1
            hold_days.append(i - buy_date_index)
            
    final_val = balance + (shares * df['Close'].iloc[-1])
    win_rate = (success / (total_trades // 2) * 100) if (total_trades // 2) > 0 else 0
    avg_hold = sum(hold_days) / len(hold_days) if hold_days else 0
    
    current_signal_code = signal_logic(df, len(df)-1, shares, last_buy)
    if current_signal_code == 1:
        sig_text = "AL"
    elif current_signal_code == -1:
        sig_text = "SAT"
    else:
        sig_text = "BEKLE"
        
    return final_val, total_trades, win_rate, sig_text, avg_hold"""

content = re.sub(r"def bt_simulator\(df, signal_logic, initial_balance=10000\):.*?return final_val, total_trades, win_rate, sig_text", simulator_repl, content, flags=re.DOTALL)

# Default to new return value size
content = content.replace("return initial_balance, 0, 0.0", "return initial_balance, 0, 0.0, \"BEKLE\", 0")
content = content.replace("val, trades, win, sig = strategies[strat_choice]", "val, trades, win, sig, avg_hold = strategies[strat_choice]")

# 3. Update EMA Cross to 3/8
content = content.replace("e20 = d.get('EMA_9'", "e20 = d.get('EMA_3'")
content = content.replace("e50 = d.get('EMA_21'", "e50 = d.get('EMA_8'")
content = content.replace("pe20 = d.get('EMA_9'", "pe20 = d.get('EMA_3'")
content = content.replace("pe50 = d.get('EMA_21'", "pe50 = d.get('EMA_8'")
content = content.replace("EMA (9/21)", "EMA (3/8)")
content = content.replace("SuperTrend (7/2)", "SuperTrend (5/1.5)")

# 4. Integrate Advanced Al-Sat Tavsiyesi Geliştirme (Consensus signal)
signal_consensus = """    # Consensus logic
    rsi_sig = strategies["📉 RSI Dip Avcısı"][3]
    ema_sig = strategies["⚡ EMA (3/8) Kesişimi"][3]
    st_sig = strategies["🚀 SuperTrend (5/1.5)"][3]
    
    signals_list = [rsi_sig, ema_sig, st_sig]
    buy_count = signals_list.count("AL")
    sell_count = signals_list.count("SAT")
    
    if buy_count == 3:
        consensus = "GÜÇLÜ AL"
        color = "#00ff00"
    elif buy_count > 0 and sell_count == 0:
        consensus = "ZAYIF AL"
        color = "#aaffaa"
    elif sell_count == 3:
        consensus = "GÜÇLÜ SAT"
        color = "#ff0000"
    elif sell_count > 0 and buy_count == 0:
        consensus = "ZAYIF SAT"
        color = "#ffaaaa"
    else:
        consensus = current_signal # Default best strategy fallback
        color = '#ffff00' if consensus == 'BEKLE' else ('#00ff00' if consensus == 'AL' else '#ff0000')
        
    st.markdown("---")
    st.markdown(f"<h1 style='text-align: center; color: white;'>🎯 GÜNCEL DURUM: <span style='color: {color};'>{consensus}</span></h1>", unsafe_allow_html=True)
    st.markdown(f"<h4 style='text-align: center;'>🏆 Tavsiye Eden Şampiyon Taktik: {best_strategy_name}</h4>", unsafe_allow_html=True)
    st.markdown("---")"""

content = re.sub(r"    st\.markdown\(\"---\"\)\n    st\.markdown\(f\"<h1 style='text-align: center; color: white;'>🎯 GÜNCEL SİNYAL:.*?st\.markdown\(\"---\"\)", signal_consensus, content, flags=re.DOTALL)

# 5. Append missing Average Hold Time UI to backtest laboratuvar
ui_hold = """    c_lb1, c_lb2, c_lb3, c_lb4, c_lb5 = st.columns(5)
    c_lb1.metric("Başlangıç", "10,000.00 ₺")
    c_lb2.metric("Sonuç", f"{val:,.2f} ₺", f"{prof_pct:.2f}%", delta_color="normal" if prof_pct >= 0 else "inverse")
    c_lb3.metric("İşlem Sayısı", f"{trades}")
    c_lb4.metric("Win Rate", f"%{win:.1f}")
    c_lb5.metric("Ortalama Süre", f"{avg_hold:.1f} Gün")"""

content = re.sub(r"    c_lb1, c_lb2, c_lb3, c_lb4 = st\.columns\(4\)\n    c_lb1\.metric\(\"Başlangıç\".*?c_lb4\.metric\(\"Win Rate\".*?\)", ui_hold, content, flags=re.DOTALL)

with open(app_path, "w", encoding="utf-8") as f:
    f.write(content)
print("successful applied turbo!")
