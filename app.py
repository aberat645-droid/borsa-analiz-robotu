import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import pandas_ta as ta

st.set_page_config(page_title="Canlı Borsa Analiz Aracı", page_icon="📈", layout="wide")

st.title("📈 Akıllı Borsa Analiz Aracı")

st.markdown("Bu araç, seçtiğiniz hissenin son 1 yıllık grafiğini analiz eder ve Bollinger Bantları / Hareketli Ortalamalar (SMA) gibi teknik göstergeleri kullanarak size tahmini bir **Alım Fiyatı** ve **Kar Al (Satış) Fiyatı** sunar.")

# Borsa Seçimi
market_choice = st.radio("🌍 Borsa Seçimi:", ["Türkiye (BIST)", "Amerika (NASDAQ/NYSE)"], horizontal=True)

# Hisse Arama Kutusu
col_search1, col_search2 = st.columns(2)
with col_search1:
    if market_choice == "Türkiye (BIST)":
        ticker_input = st.text_input("Hisse Sembolü (Örn: KBORU, GESAN, THYAO)", value="KBORU").replace('"', '').replace("'", "").strip()
        ticker_symbol = f"{ticker_input.upper()}.IS" if not ticker_input.upper().endswith(".IS") else ticker_input.upper()
    else:
        ticker_input = st.text_input("Hisse Sembolü (Örn: NVDA, TSLA, AAPL)", value="NVDA").replace('"', '').replace("'", "").strip()
        ticker_symbol = ticker_input.upper()
        
with col_search2:
    ticker_symbol_2_input = st.text_input("Kıyaslanacak İkinci Hisse (Opsiyonel)", value="").replace('"', '').replace("'", "").strip()
    if ticker_symbol_2_input:
        if market_choice == "Türkiye (BIST)" and not ticker_symbol_2_input.upper().endswith(".IS"):
            ticker_symbol_2 = f"{ticker_symbol_2_input.upper()}.IS"
        else:
            ticker_symbol_2 = ticker_symbol_2_input.upper()
    else:
        ticker_symbol_2 = ""

# Period sabit (1 yıllık), analiz için en az 1 yıllık veri genelde iyidir.
yf_period = "1y"
interval = "1d"

@st.cache_data(ttl=60)
def load_data(ticker):
    try:
        data = yf.download(ticker, period=yf_period, interval=interval)
        return data
    except Exception as e:
        return pd.DataFrame()

def calculate_technical_indicators(df):
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

def backtest_rsi_macd_strategy(df, initial_balance=10000):
    try:
        balance = initial_balance
        shares = 0
        total_trades = 0
        successful_trades = 0
        last_buy_price = 0

        for i in range(1, len(df)):
            price = df['Close'].iloc[i]
            rsi = df['RSI'].iloc[i]
            macd = df['MACD'].iloc[i]
            macd_signal = df['MACD_Signal'].iloc[i]
            sma_200 = df['SMA_200'].iloc[i]

            if pd.isna(rsi) or pd.isna(macd) or pd.isna(macd_signal) or pd.isna(sma_200):
                continue

            macd_buy_signal = macd > macd_signal
            trend_is_up = price > sma_200

            # Sinyal Durumu: Al(1), Sat(-1), Bekle(0)
            signal = 0
            if trend_is_up and rsi < 40 and macd_buy_signal:
                signal = 1
            elif shares > 0 and (price <= last_buy_price * 0.93 or rsi > 70):
                signal = -1

            if signal == 1 and shares == 0:
                shares = balance / price
                balance = 0
                last_buy_price = price
                total_trades += 1
            elif signal == -1 and shares > 0:
                balance += shares * price
                if price > last_buy_price:
                    successful_trades += 1
                shares = 0
                total_trades += 1

        final_value = balance + (shares * df['Close'].iloc[-1])
        win_rate = (successful_trades / (total_trades // 2) * 100) if (total_trades // 2) > 0 else 0

        return final_value, total_trades, win_rate
    except Exception:
        return initial_balance, 0, 0.0

def backtest_supertrend_strategy(df, initial_balance=10000):
    try:
        if df.empty or len(df) < 2:
            return initial_balance, 0, 0.0

        balance = initial_balance
        shares = 0
        total_trades = 0
        successful_trades = 0
        last_buy_price = 0

        for i in range(1, len(df)):
            price = df['Close'].iloc[i]

            if isinstance(df.columns, pd.MultiIndex):
                vol = df['Volume'].iloc[i, 0]
            else:
                vol = df['Volume'].iloc[i]

            vol_sma_10 = df['Volume_SMA_10'].iloc[i]
            trend_dir = df['Trend_Dir'].iloc[i]
            prev_trend_dir = df['Trend_Dir'].iloc[i-1]

            if pd.isna(trend_dir) or pd.isna(vol_sma_10):
                continue

            trend_just_turned_up = (prev_trend_dir == 1) and (trend_dir == -1)
            trend_just_turned_down = (prev_trend_dir == -1) and (trend_dir == 1)
            volume_confirm = vol > vol_sma_10

            signal = 0
            if trend_just_turned_up and volume_confirm:
                signal = 1
            elif trend_just_turned_down or (shares > 0 and price <= last_buy_price * 0.93):
                signal = -1

            if signal == 1 and shares == 0:
                shares = balance / price
                balance = 0
                last_buy_price = price
                total_trades += 1
            elif signal == -1 and shares > 0:
                balance += shares * price
                if price > last_buy_price:
                    successful_trades += 1
                shares = 0
                total_trades += 1

        final_value = balance + (shares * df['Close'].iloc[-1])
        win_rate = (successful_trades / (total_trades // 2) * 100) if (total_trades // 2) > 0 else 0

        return final_value, total_trades, win_rate
    except Exception:
        return initial_balance, 0, 0.0

def backtest_ma_cross_strategy(df, initial_balance=10000):
    try:
        if df.empty or len(df) < 2:
            return initial_balance, 0, 0.0

        balance = initial_balance
        shares = 0
        total_trades = 0
        successful_trades = 0
        last_buy_price = 0

        for i in range(1, len(df)):
            price = df['Close'].iloc[i]
            sma_5 = df['SMA_5'].iloc[i]
            sma_22 = df['SMA_22'].iloc[i]
            prev_sma_5 = df['SMA_5'].iloc[i-1]
            prev_sma_22 = df['SMA_22'].iloc[i-1]

            if pd.isna(sma_5) or pd.isna(sma_22) or pd.isna(prev_sma_5):
                continue

            golden_cross = (prev_sma_5 <= prev_sma_22) and (sma_5 > sma_22)
            death_cross = (prev_sma_5 >= prev_sma_22) and (sma_5 < sma_22)

            signal = 0
            if golden_cross:
                signal = 1
            elif death_cross or (shares > 0 and price <= last_buy_price * 0.93):
                signal = -1

            if signal == 1 and shares == 0:
                shares = balance / price
                balance = 0
                last_buy_price = price
                total_trades += 1
            elif signal == -1 and shares > 0:
                balance += shares * price
                if price > last_buy_price:
                    successful_trades += 1
                shares = 0
                total_trades += 1

        final_value = balance + (shares * df['Close'].iloc[-1])
        win_rate = (successful_trades / (total_trades // 2) * 100) if (total_trades // 2) > 0 else 0

        return final_value, total_trades, win_rate
    except Exception:
        return initial_balance, 0, 0.0


def bt_simulator(df, signal_logic, initial_balance=10000):
    try:
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
    except Exception:
        return initial_balance, 0, 0.0

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

def get_current_signals(df):
    if df.empty or len(df) < 2:
        return {"🚀 SuperTrend & Hacim": "BEKLE", "⚔️ Hareketli Ortalama Kesişimi (5/22)": "BEKLE", "🛡️ RSI Dip Avcısı & MACD": "BEKLE"}
        
    i = -1
    # ST
    trend_dir = df['Trend_Dir'].iloc[i]
    prev_trend_dir = df['Trend_Dir'].iloc[i-1]
    vol = df['Volume'].iloc[i, 0] if isinstance(df.columns, pd.MultiIndex) else df['Volume'].iloc[i]
    vol_sma_10 = df['Volume_SMA_10'].iloc[i]
    st_sig = "AL" if (prev_trend_dir == 1 and trend_dir == -1 and vol > vol_sma_10) else ("SAT" if prev_trend_dir == -1 and trend_dir == 1 else "BEKLE")
    # MA
    sma_5, sma_22, prev_sma_5, prev_sma_22 = df['SMA_5'].iloc[i], df['SMA_22'].iloc[i], df['SMA_5'].iloc[i-1], df['SMA_22'].iloc[i-1]
    ma_sig = "AL" if (prev_sma_5 <= prev_sma_22 and sma_5 > sma_22) else ("SAT" if prev_sma_5 >= prev_sma_22 and sma_5 < sma_22 else "BEKLE")
    # RSI
    rsi, macd, macd_sig, sma_200 = df['RSI'].iloc[i], df['MACD'].iloc[i], df['MACD_Signal'].iloc[i], df['SMA_200'].iloc[i]
    price = df['Close'].iloc[i]
    rsi_sig = "AL" if (price > sma_200 and rsi < 40 and macd > macd_sig) else ("SAT" if rsi > 70 else "BEKLE")
    
    return {
        "🚀 SuperTrend & Hacim": st_sig,
        "⚔️ Hareketli Ortalama Kesişimi (5/22)": ma_sig,
        "🛡️ RSI Dip Avcısı & MACD": rsi_sig
    }

data_load_state = st.text("Veriler çekiliyor ve analiz ediliyor...")
data = load_data(ticker_symbol)

if data.empty:
    data_load_state.text(f"'{ticker_symbol}' için veri bulunamadı. Lütfen sembolü kontrol edin.")
else:
    data_load_state.text("Analiz tamamlandı!")
    
    # Formatlama işlemi
    if isinstance(data.columns, pd.MultiIndex):
        df = data.copy()
        df.columns = df.columns.droplevel(1)
    else:
        df = data.copy()

    # Teknik Analizi Uygula
    df = calculate_technical_indicators(df)

    latest_data = df.iloc[-1]
    current_price = latest_data['Close']
    buy_price = latest_data.get('Lower_Band', current_price * 0.95)
    sell_price = latest_data.get('Upper_Band', current_price * 1.05)
    current_rsi = latest_data.get('RSI', 50.0)

    current_macd = latest_data.get('MACD', 0.0)
    current_macd_signal = latest_data.get('MACD_Signal', 0.0)

    if len(df) > 1:
        previous_data = df.iloc[-2]
        previous_rsi = previous_data.get('RSI', 50.0)
        previous_macd = previous_data.get('MACD', 0.0)
        previous_macd_signal = previous_data.get('MACD_Signal', 0.0)
    else:
        previous_rsi = current_rsi
        previous_macd = current_macd
        previous_macd_signal = current_macd_signal

    # Ekranda daha şık görünmesi için borsa uzantılarını (örn: .IS) atıp sadece hisse adını alalım
    display_symbol = ticker_symbol.split('.')[0].upper()

    # --- OTOMATİK OPTİMİZASYON (HANGİ STRATEJİ DAHA İYİ?) ---
    
    strategies = {
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
    }
    
    signals = get_current_signals(df)
    
    best_strategy_name = max(strategies, key=lambda k: strategies[k][0])
    best_results = strategies[best_strategy_name]
    best_profit_pct = ((best_results[0] - 10000) / 10000) * 100
    current_signal = signals.get(best_strategy_name, "BEKLE")
    
    # Şampiyon Strateji Kutusu
    st.markdown(f"## 🏆 {display_symbol} İçin En İyi Taktik: **{best_strategy_name}**")
    st.success(f"Bu hisseye 1 yıl önce en uygun taktikle 10.000₺ yatırsaydınız, **%{best_profit_pct:.2f} getiriyle** sermayeniz **{best_results[0]:,.2f}₺** olurdu.")
    
    st.markdown("---")

    # Özet Analiz Tablosunu Oluştur
    st.subheader(f"📊 {display_symbol} Güncel Fiyat Bilgileri")

    # Teknik Analiz Özeti
    st.markdown("### 📋 Teknik Analiz Özeti")
    summary_messages = []
    
    # RSI Yorumu
    if previous_rsi > 70 and current_rsi < previous_rsi:
        summary_messages.append("📉 **RSI:** Hisse aşırı alım bölgesinde yoruluyor, kâr satışı gelebilir.")
    elif previous_rsi < 30 and current_rsi > previous_rsi:
        summary_messages.append("📈 **RSI:** Dip seviyelerden tepki alımı geliyor.")
        
    # MACD Yorumu
    if current_macd > current_macd_signal and previous_macd <= previous_macd_signal:
        summary_messages.append("🚀 **MACD:** Trend güçleniyor, alıcılar iştahlı.")
        
    # Bollinger Yorumu
    if current_price >= sell_price:
        summary_messages.append("🧱 **Bollinger:** Direnç seviyesine gelindi.")
    elif current_price <= buy_price:
        summary_messages.append("🛡️ **Bollinger:** Destek seviyesinden dönüş beklenebilir.")

    if summary_messages:
        for msg in summary_messages:
            if "Dip" in msg or "güçleniyor" in msg or "Destek" in msg:
                st.success(msg)
            elif "yoruluyor" in msg or "Direnç" in msg:
                st.warning(msg)
            else:
                st.info(msg)
    else:
        st.info("Sakin bir piyasa, hissede belirgin bir sinyal veya kırılım görülmüyor.")
    
    st.markdown("---")

    # Metric kartları
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Güncel Fiyat", f"{current_price:.2f}")
    col2.metric("Önerilen Alım Fiyatı (Destek)", f"{buy_price:.2f}", delta="Bollinger Alt Bant", delta_color="off")
    col3.metric("Önerilen Satış Fiyatı (Direnç)", f"{sell_price:.2f}", delta="Bollinger Üst Bant", delta_color="off")
    col4.metric("RSI (14)", f"{current_rsi:.2f}", delta="Aşırı Alım/Satım Göstergesi", delta_color="off")
    
    # Sinyal Yorumu
    st.markdown("### 💡 Aksiyon Önerisi")
    if current_price <= buy_price or current_rsi < 30:
        st.success(f"**AL SİNYALİ!** {display_symbol} hissesi destek noktasına veya aşırı satım bölgesine yakın görünüyor. Mevcut fiyat {current_price:.2f}, hedeflenen tahmini dipten alım fiyatı {buy_price:.2f}.")
    elif current_price >= sell_price or (previous_rsi > 70 and current_rsi < previous_rsi):
        st.error(f"**SAT SİNYALİ!** {display_symbol} hissesinde fiyat dirence gelmiş olabilir veya RSI 70 üzerinde zirveden dönüş sinyali verdi. Karları realize etmeyi düşünebilirsiniz.")
    elif current_rsi > 70 and current_rsi >= previous_rsi:
        st.warning(f"**YÜKSELİŞ TRENDİ!** {display_symbol} hissesinde RSI 70'in üzerinde aşırı alım bölgesinde, ancak henüz zirveden dönüş sinyali yok. Trendi takip etmeye devam edebilirsiniz.")
    else:
        st.info(f"**BEKLE!** {display_symbol} hissesi şu an bant arasında normal seyrediyor. Yeni bir aksiyon almadan önce trendin netleşmesini bekleyebilirsiniz.")


    st.markdown("### 📈 Fiyat ve Bollinger Bantları Grafiği (Son 1 Yıl)")
    
    # Grafiği Çiz (Close, Upper Band, Lower Band, SMA)
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Kapanış Fiyatı', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='20 Günlük SMA', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Upper_Band'], mode='lines', name='Üst Bant (Kar Al)', line=dict(color='red', dash='dash')))
    fig.add_trace(go.Scatter(x=df.index, y=df['Lower_Band'], mode='lines', name='Alt Bant (Alım Yeri)', line=dict(color='green', dash='dash')))

    fig.update_layout(
        xaxis_title='Zaman',
        yaxis_title='Fiyat',
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # ------------------ KIYASLAMA MODU ------------------
    if ticker_symbol_2:
        data2 = load_data(ticker_symbol_2)
        if not data2.empty:
            st.markdown(f"### ⚔️ {display_symbol} vs {ticker_symbol_2.split('.')[0].upper()} Yüzdesel Kıyaslama")
            
            # Formatsız dataframe çıkartalım
            if isinstance(data2.columns, pd.MultiIndex):
                df2_close = data2['Close'].iloc[:, 0]
            else:
                df2_close = data2['Close']
                
            # İlk günün verisini 0 kabul edip genel yüzde değişimi bulalım
            df1_pct = ((df['Close'] / df['Close'].iloc[0]) - 1) * 100
            df2_pct = ((df2_close / df2_close.iloc[0]) - 1) * 100

            fig_comp = go.Figure()
            fig_comp.add_trace(go.Scatter(x=df.index, y=df1_pct, mode='lines', name=f"{display_symbol} Getirisi", line=dict(color='#00ffcc', width=2)))
            fig_comp.add_trace(go.Scatter(x=df2_pct.index, y=df2_pct, mode='lines', name=f"{ticker_symbol_2.split('.')[0].upper()} Getirisi", line=dict(color='#ff0066', width=2)))
            
            fig_comp.update_layout(
                xaxis_title='Zaman',
                yaxis_title='Getiri Yüzdesi (%)',
                template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified"
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning(f"'{ticker_symbol_2}' sembolü için veri alınamadı, kıyaslama yapılamıyor.")

    # ------------------ BACKTEST SİSTEMİ ------------------
    st.markdown("### 🤖 Borsa Stratejisi Test Laboratuvarı (Son 1 Yıl)")
    st.info("Her hissenin karakteri farklıdır. Ağır ilerleyen hisseler ile volatil yan tahtalar aynı stratejiye uymaz. Hissenin karakterine en uygun olan strateji sistem tarafından otomatik seçildi!")
    
    # Tüm strateji adlarının listesi
    strategy_names = list(strategies.keys())
    best_index = strategy_names.index(best_strategy_name)
    
    strategy_choice = st.radio(
        "📝 Strateji Seçimi (Otomatik olarak en iyisi seçili gelir):",
        strategy_names,
        index=best_index
    )
    
    # Seçilen stratejinin sonuçlarını dictionery'den çek (3 değer döner)
    final_val, trade_count, win_rate = strategies[strategy_choice]
    
    if "SuperTrend" in strategy_choice:
        st.markdown("**Strateji Mantığı:** SuperTrend (10, 3) Al sinyali ve Hacim Onayı ile işleme girer.")
    elif "Hareketli" in strategy_choice:
        st.markdown("**Strateji Mantığı:** 5 günlük ve 22 günlük Hareketli Ortalamaların kesişimini takip eder.")
    elif "RSI" in strategy_choice:
        st.markdown("**Strateji Mantığı:** Aşırı satılan yerlerde (RSI < 40) MACD'nin al verdiği güvenli yerlerde mal toplar.")
    else:
        st.markdown(f"**Strateji Mantığı:** {strategy_choice} stratejisi, ilgili indikatörlerin alım/satım kurallarına %7 stop-loss ile sadık kalır.")
    
    profit_loss = final_val - 10000
    profit_loss_pct = (profit_loss / 10000) * 100


    col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
    col_bt1.metric("Başlangıç Bakiyesi", "10,000.00 ₺")
    col_bt2.metric("Güncel Portföy Değeri", f"{final_val:,.2f} ₺", f"{profit_loss_pct:.2f}% Getiri", delta_color="normal" if profit_loss >= 0 else "inverse")
    col_bt3.metric("Toplam İşlem Sayısı", f"{trade_count} Adet", "Alış veya Satış")
    col_bt4.metric("Başarılı İşlem Oranı (Kârlı)", f"%{win_rate:.1f}", "Al-Sat Döngüsü Arasında")

    # Hacim Grafiği
    st.markdown("### 📊 İşlem Hacmi (Volume)")
    
    if isinstance(data.columns, pd.MultiIndex):
        volume_series = data['Volume'].iloc[:, 0]
    else:
        volume_series = data['Volume']

    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=df.index, y=volume_series, name='Hacim', marker_color='#1f77b4'))
    fig_vol.update_layout(
        xaxis_title='Zaman',
        yaxis_title='Hacim',
        template="plotly_dark",
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode="x unified"
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    # MACD Grafiği
    st.markdown("### 📉 MACD (12, 26, 9) Göstergesi")
    if 'MACD' in df.columns and 'MACD_Hist' in df.columns and 'MACD_Signal' in df.columns:
        fig_macd = go.Figure()
        
        # MACD Histogramı için renk belirleme (pozitif yeşil, negatif kırmızı)
        colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
        
        fig_macd.add_trace(go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', marker_color=colors))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color='blue')))
        fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Sinyal', line=dict(color='orange')))
        
        fig_macd.update_layout(
            xaxis_title='Zaman',
            yaxis_title='MACD',
            template="plotly_dark",
            margin=dict(l=0, r=0, t=30, b=0),
            hovermode="x unified"
        )
        st.plotly_chart(fig_macd, use_container_width=True)
    else:
        st.info("Bu hisse için yeterli MACD verisi hesaplanamadı.")

    # Haber Akışı
    st.markdown("---")
    st.markdown(f"### 📰 {display_symbol} Son Haberler")
    try:
        stock = yf.Ticker(ticker_symbol)
        news = stock.news
        if news:
            for n in news[:5]: # Son 5 haber
                title = n.get('title', 'Başlık Bulunamadı')
                link = n.get('link', '#')
                publisher = n.get('publisher', 'Bilinmeyen Kaynak')
                st.markdown(f"- [{title}]({link}) *(Kaynak: {publisher})*")
        else:
            st.info("Bu hisse için güncel haber bulunamadı.")
    except Exception as e:
        st.error("Haberler çekilirken bir hata oluştu.")

    st.warning("⚠️ Sorumluluk Reddi: Bu araç tamamen teknik göstergelere (Bollinger Bantları, RSI ve Hareketli Ortalamalar) dayalı matematiksel hesaplamalar sunar ve bir yatırım tavsiyesi (YTD) niteliği taşımaz. İşlem yapmadan önce kendi araştırmanızı yapınız.")
