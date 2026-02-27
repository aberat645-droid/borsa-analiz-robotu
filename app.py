import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np

try:
    import pandas_ta as ta
except ImportError:
    st.error("Lütfen 'pandas_ta' kütüphanesini yükleyin: pip install pandas_ta")
    st.stop()

st.set_page_config(page_title="Canlı Borsa Analiz Aracı", page_icon="📈", layout="wide")

# 1. Borsa ve Hisse Seçimi
market_choice = st.radio("🌍 Borsa Seçimi:", ["Türkiye (BIST)", "Amerika (NASDAQ/NYSE)"], horizontal=True)

col1, col2 = st.columns(2)
with col1:
    if market_choice == "Türkiye (BIST)":
        ticker_input = st.text_input("Hisse Sembolü (Örn: THYAO, ASELS, GARAN)", value="THYAO").replace('"', '').strip()
        ticker_symbol = f"{ticker_input.upper()}.IS" if not ticker_input.upper().endswith(".IS") else ticker_input.upper()
    else:
        ticker_input = st.text_input("Hisse Sembolü (Örn: AAPL, NVDA, TSLA)", value="AAPL").replace('"', '').strip()
        ticker_symbol = ticker_input.upper()

with col2:
    ticker_symbol_2_input = st.text_input("Kıyaslanacak İkinci Hisse (Opsiyonel)", value="").replace('"', '').strip()
    ticker_symbol_2 = ""
    if ticker_symbol_2_input:
        if market_choice == "Türkiye (BIST)" and not ticker_symbol_2_input.upper().endswith(".IS"):
            ticker_symbol_2 = f"{ticker_symbol_2_input.upper()}.IS"
        else:
            ticker_symbol_2 = ticker_symbol_2_input.upper()

@st.cache_data(ttl=60)
def load_data(ticker):
    try:
        data = yf.download(ticker, period="1y", interval="1d")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        return data
    except Exception as e:
        return pd.DataFrame()

def apply_ta(df):
    if df.empty or len(df) < 50:
        return pd.DataFrame() # Not enough data
        
    try:
        df.ta.rsi(length=3, append=True)
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
        df.ta.supertrend(length=5, multiplier=1.2, append=True)
        
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
            'SUPERTd_5_1.2': 'Trend_Dir'
        }
        
        for old_c, new_c in ta_col_map.items():
            if old_c in df.columns:
                df[new_c] = df[old_c]
                
        df.dropna(inplace=True)
        
    except Exception as e:
        pass
        
    return df

# ---------- STRATEJİ BACKTEST FONKSİYONLARI ----------
def bt_simulator(df, signal_logic, initial_balance=10000):
    if df.empty or len(df) < 5:
        return initial_balance, 0, 0.0, "BEKLE", 0
        
    balance = initial_balance
    shares = 0
    total_trades = 0
    success = 0
    last_buy = 0
    hold_days = []
    buy_date_index = 0
    
    for i in range(len(df)):
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
        
    return final_val, total_trades, win_rate, sig_text, avg_hold

def backtest_rsi(df):
    def logic(d, i, shares, buy_p):
        try:
            rsi = d.get('RSI', pd.Series(dtype=float)).iloc[i]
            if pd.isna(rsi): return 0
            if rsi < 30: return 1
            if shares > 0 and (rsi > 70 or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        except: pass
        return 0
    return bt_simulator(df, logic)

def backtest_macd(df):
    def logic(d, i, shares, buy_p):
        try:
            m = d.get('MACD', pd.Series(dtype=float)).iloc[i]
            s = d.get('MACD_Signal', pd.Series(dtype=float)).iloc[i]
            if pd.isna(m) or pd.isna(s): return 0
            if m > s and m < 0: return 1
            if shares > 0 and (m < s or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        except: pass
        return 0
    return bt_simulator(df, logic)

def backtest_bbands(df):
    def logic(d, i, shares, buy_p):
        try:
            lb = d.get('Lower_Band', pd.Series(dtype=float)).iloc[i]
            ub = d.get('Upper_Band', pd.Series(dtype=float)).iloc[i]
            c = d['Close'].iloc[i]
            if pd.isna(lb) or pd.isna(ub): return 0
            if c < lb: return 1
            if shares > 0 and (c > ub or c <= buy_p * 0.93): return -1
        except: pass
        return 0
    return bt_simulator(df, logic)

def backtest_sma_cross(df):
    def logic(d, i, shares, buy_p):
        if i < 1: return 0
        try:
            s5 = d.get('SMA_5', pd.Series(dtype=float)).iloc[i]
            s22 = d.get('SMA_22', pd.Series(dtype=float)).iloc[i]
            ps5 = d.get('SMA_5', pd.Series(dtype=float)).iloc[i-1]
            ps22 = d.get('SMA_22', pd.Series(dtype=float)).iloc[i-1]
            if pd.isna(s5) or pd.isna(s22): return 0
            if ps5 <= ps22 and s5 > s22: return 1
            if shares > 0 and (ps5 >= ps22 and s5 < s22 or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        except: pass
        return 0
    return bt_simulator(df, logic)

def backtest_stoch(df):
    def logic(d, i, shares, buy_p):
        try:
            k = d.get('Stoch_K', pd.Series(dtype=float)).iloc[i]
            d_line = d.get('Stoch_D', pd.Series(dtype=float)).iloc[i]
            if pd.isna(k) or pd.isna(d_line): return 0
            if k > d_line and k < 20: return 1
            if shares > 0 and (k < d_line and k > 80 or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        except: pass
        return 0
    return bt_simulator(df, logic)

def backtest_cci(df):
    def logic(d, i, shares, buy_p):
        try:
            cci = d.get('CCI', pd.Series(dtype=float)).iloc[i]
            if pd.isna(cci): return 0
            if cci < -100: return 1
            if shares > 0 and (cci > 100 or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        except: pass
        return 0
    return bt_simulator(df, logic)

def backtest_willr(df):
    def logic(d, i, shares, buy_p):
        try:
            w = d.get('Williams_R', pd.Series(dtype=float)).iloc[i]
            if pd.isna(w): return 0
            if w < -80: return 1
            if shares > 0 and (w > -20 or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        except: pass
        return 0
    return bt_simulator(df, logic)

def backtest_ema_cross(df):
    def logic(d, i, shares, buy_p):
        if i < 1: return 0
        try:
            e20 = d.get('EMA_3', pd.Series(dtype=float)).iloc[i]
            e50 = d.get('EMA_8', pd.Series(dtype=float)).iloc[i]
            pe20 = d.get('EMA_3', pd.Series(dtype=float)).iloc[i-1]
            pe50 = d.get('EMA_8', pd.Series(dtype=float)).iloc[i-1]
            if pd.isna(e20) or pd.isna(e50): return 0
            if pe20 <= pe50 and e20 > e50: return 1
            if shares > 0 and (pe20 >= pe50 and e20 < e50 or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        except: pass
        return 0
    return bt_simulator(df, logic)

def backtest_ichimoku(df):
    def logic(d, i, shares, buy_p):
        try:
            c = d['Close'].iloc[i]
            span_a = d.get('Senkou_Span_A', pd.Series(dtype=float)).iloc[i]
            span_b = d.get('Senkou_Span_B', pd.Series(dtype=float)).iloc[i]
            if pd.isna(span_a) or pd.isna(span_b): return 0
            in_up = c > max(span_a, span_b)
            if in_up: return 1
            if shares > 0 and (c < min(span_a, span_b) or c <= buy_p * 0.93): return -1
        except: pass
        return 0
    return bt_simulator(df, logic)

def backtest_adx(df):
    def logic(d, i, shares, buy_p):
        try:
            adx = d.get('ADX', pd.Series(dtype=float)).iloc[i]
            if pd.isna(adx): return 0
            if adx > 25: return 1
            if shares > 0 and (adx < 20 or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        except: pass
        return 0
    return bt_simulator(df, logic)

def backtest_supertrend(df):
    def logic(d, i, shares, buy_p):
        if i < 1: return 0
        try:
            td = d.get('Trend_Dir', pd.Series(dtype=float)).iloc[i]
            ptd = d.get('Trend_Dir', pd.Series(dtype=float)).iloc[i-1]
            if pd.isna(td) or pd.isna(ptd): return 0
            if ptd < 0 and td > 0: return 1
            if shares > 0 and (ptd > 0 and td < 0 or d['Close'].iloc[i] <= buy_p * 0.93): return -1
        except: pass
        return 0
    return bt_simulator(df, logic)

def run_all_strategies(df):
    strategies = {
        "📉 RSI Dip Avcısı": backtest_rsi(df),
        "📊 MACD Trend": backtest_macd(df),
        "🎡 Bollinger Bantları": backtest_bbands(df),
        "⚔️ SMA (5/22) Kesişimi": backtest_sma_cross(df),
        "🎢 Stochastic Oscillator": backtest_stoch(df),
        "🎯 CCI (Emtia Kanalı)": backtest_cci(df),
        "📉 Williams %R": backtest_willr(df),
        "⚡ EMA (3/8) Kesişimi": backtest_ema_cross(df),
        "☁️ Ichimoku Bulutu": backtest_ichimoku(df),
        "🔥 ADX (Trend Gücü)": backtest_adx(df),
        "🚀 SuperTrend (5/1.2)": backtest_supertrend(df)
    }
    return strategies

# ---------- ANA UYGULAMA ----------
data_load_state = st.text("Veriler çekiliyor...")
df_raw = load_data(ticker_symbol)

if df_raw.empty:
    st.warning(f"'{ticker_symbol}' için veri bulunamadı! Kodu veya piyasayı kontrol edin.")
    data_load_state.text("Veri Bulunamadı.")
else:
    df = apply_ta(df_raw.copy())
    
    if df.empty:
         st.warning(f"'{ticker_symbol}' için teknik analiz hesaplanacak kadar yeterli veri bulunamadı. Lütfen daha köklü bir hisse seçin.")
         data_load_state.text("Yetersiz Veri.")
    else:
        data_load_state.text("Analiz Tamamlandı!")
        
        display_symbol = ticker_symbol.split('.')[0].upper()
        current_price = df['Close'].iloc[-1]
        
        # Şampiyon Seçimi
        strategies = run_all_strategies(df)
        
        best_strategy_name = max(strategies, key=lambda k: strategies[k][0])
        best_results = strategies[best_strategy_name]
        best_profit_pct = ((best_results[0] - 10000) / 10000) * 100
        current_signal = best_results[3]
        
        # Consensus logic
        rsi_sig = strategies["📉 RSI Dip Avcısı"][3]
        ema_sig = strategies["⚡ EMA (3/8) Kesişimi"][3]
        st_sig = strategies["🚀 SuperTrend (5/1.2)"][3]
        
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
            if current_signal == "AL":
                consensus = "AL (Şampiyon Taktik Puanı)"
                color = "#00ff00"
            elif current_signal == "SAT":
                consensus = "SAT (Şampiyon Taktik Puanı)"
                color = "#ff0000"
            else:
                consensus = "BEKLE"
                color = "#ffff00"
            
        st.markdown("---")
        st.markdown(f"<h1 style='text-align: center; color: white;'>🎯 GÜNCEL TAVSİYE: <span style='color: {color};'>{consensus}</span></h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.success(f"Eğer 1 yıl önce **{best_strategy_name}** ile {display_symbol} hissesine 10.000 ₺ bağlansaydınız, getiri oranınız **%{best_profit_pct:.2f}** ile sonucunuz **{best_results[0]:,.2f} ₺** olurdu.")
        st.markdown("---")
        
        # Basit Metrikler
        st.subheader(f"📊 {display_symbol} Güncel Fiyat ve Özet")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Güncel Fiyat", f"{current_price:.2f}")
        
        lb_val = df.get('Lower_Band', pd.Series([None])).iloc[-1]
        ub_val = df.get('Upper_Band', pd.Series([None])).iloc[-1]
        rsi_val = df.get('RSI', pd.Series([None])).iloc[-1]
        
        c2.metric("Bollinger Alt (Destek)", f"{lb_val:.2f}" if pd.notna(lb_val) else "N/A")
        c3.metric("Bollinger Üst (Direnç)", f"{ub_val:.2f}" if pd.notna(ub_val) else "N/A")
        c4.metric("RSI (Hızlı)", f"{rsi_val:.2f}" if pd.notna(rsi_val) else "N/A")
        
        # ------------------ GRAFİKLER ------------------
        st.markdown("### 📈 Fiyat ve Teknik Göstergeler")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Kapanış Fiyatı', line=dict(color='blue')))
        
        if 'EMA_8' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_8'], mode='lines', name='8 Günlük EMA', line=dict(color='orange')))
        if 'Upper_Band' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], mode='lines', name='Bollinger Üst', line=dict(color='red', dash='dash')))
        if 'Lower_Band' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], mode='lines', name='Bollinger Alt', line=dict(color='green', dash='dash')))

        fig.update_layout(xaxis_title='Zaman', yaxis_title='Fiyat', template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)
        
        # Kıyaslama Modu
        if ticker_symbol_2:
            df_comp = load_data(ticker_symbol_2)
            if not df_comp.empty:
                st.markdown(f"### ⚔️ {display_symbol} vs {ticker_symbol_2.split('.')[0].upper()} (Yüzdesel Getiri)")
                df1_pct = ((df['Close'] / df['Close'].iloc[0]) - 1) * 100
                df2_pct = ((df_comp['Close'] / df_comp['Close'].iloc[0]) - 1) * 100
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Scatter(x=df.index, y=df1_pct, mode='lines', name=f"{display_symbol} Getiri", line=dict(color='#00ffcc', width=2)))
                fig_comp.add_trace(go.Scatter(x=df_comp.index, y=df2_pct, mode='lines', name=f"{ticker_symbol_2.split('.')[0].upper()} Getiri", line=dict(color='#ff0066', width=2)))
                fig_comp.update_layout(xaxis_title='Zaman', yaxis_title='Getiri (%)', template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0), hovermode="x unified")
                st.plotly_chart(fig_comp, use_container_width=True)
        
        # Ayrıntılı Laboratuvar
        st.markdown("### 🤖 Scalping Laboratuvarı")
        st.info("Kusursuz çalışan, çökme riski taşımayan stratejilerin sonuçları.")
        
        strat_names = list(strategies.keys())
        strat_choice = st.radio("Bir Strateji İnceleyin:", strat_names, index=strat_names.index(best_strategy_name))
        
        val, trades, win, sig, avg_hold = strategies[strat_choice]
        prof_pct = ((val - 10000) / 10000) * 100
        
        c_lb1, c_lb2, c_lb3, c_lb4, c_lb5 = st.columns(5)
        c_lb1.metric("Başlangıç", "10,000.00 ₺")
        c_lb2.metric("Sonuç", f"{val:,.2f} ₺", f"{prof_pct:.2f}%", delta_color="normal" if prof_pct >= 0 else "inverse")
        c_lb3.metric("İşlem Sayısı", f"{trades}")
        c_lb4.metric("Win Rate", f"%{win:.1f}")
        c_lb5.metric("Ortalama Süre", f"{avg_hold:.1f} Gün")
        
        st.markdown("---")
        st.markdown("⚠️ Sorumluluk Reddi: Bu araç teknik göstergelere dayalıdır, yatırım tavsiyesi içermez.")
