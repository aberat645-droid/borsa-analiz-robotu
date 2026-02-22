import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(page_title="Canlı Borsa Analiz Aracı", page_icon="📈", layout="wide")

st.title("📈 Akıllı Borsa Analiz Aracı")
st.markdown("Bu araç, seçtiğiniz hissenin son 1 yıllık grafiğini analiz eder ve Bollinger Bantları / Hareketli Ortalamalar (SMA) gibi teknik göstergeleri kullanarak size tahmini bir **Alım Fiyatı** ve **Kar Al (Satış) Fiyatı** sunar.")

# Hisse Arama Kutusu
col_search1, col_search2 = st.columns(2)
with col_search1:
    ticker_symbol = st.text_input("Hisse Sembolü (Örn: KBORU.IS, GESAN.IS, THYAO.IS)", value="KBORU.IS")
with col_search2:
    ticker_symbol_2 = st.text_input("Kıyaslanacak İkinci Hisse (Opsiyonel)", value="")

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
    """
    20 günlük Basit Hareketli Ortalama (SMA) ve Bollinger Bantları
    kullanarak alım ve satım seviyeleri oluşturur.
    """
    # yfinance multi-index column döndürebiliyor, Close sütununu güvenle alalım
    if isinstance(df.columns, pd.MultiIndex):
        close_series = df['Close'].iloc[:, 0]
    else:
        close_series = df['Close']
        
    df['SMA_20'] = close_series.rolling(window=20).mean()
    df['StdDev'] = close_series.rolling(window=20).std()
    df['Upper_Band'] = df['SMA_20'] + (df['StdDev'] * 2)
    df['Lower_Band'] = df['SMA_20'] - (df['StdDev'] * 2)
    
    # RSI (Relative Strength Index) hesaplama
    delta = close_series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9) hesaplama
    df['EMA_12'] = close_series.ewm(span=12, adjust=False).mean()
    df['EMA_26'] = close_series.ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Kesişim için Hareketli Ortalamalar (5 ve 22 Günlük)
    df['SMA_5'] = close_series.rolling(window=5).mean()
    df['SMA_22'] = close_series.rolling(window=22).mean()
    
    # SuperTrend (10, 3) ve Hacim Ortalaması (10 Günlük)
    # Hacmi güvenle alalım
    if isinstance(df.columns, pd.MultiIndex):
        volume_series = df['Volume'].iloc[:, 0]
    else:
        volume_series = df['Volume']
        
    df['Volume_SMA_10'] = volume_series.rolling(window=10).mean()
    
    # ATR Hesaplama
    high = df['High'].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df['High']
    low = df['Low'].iloc[:, 0] if isinstance(df.columns, pd.MultiIndex) else df['Low']
    
    tr1 = high - low
    tr2 = (high - close_series.shift(1)).abs()
    tr3 = (low - close_series.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR'] = tr.rolling(window=10).mean()
    
    # SuperTrend Bantları (Multiplier: 3)
    hl2 = (high + low) / 2
    df['Basic_Upper_Band'] = hl2 + (3 * df['ATR'])
    df['Basic_Lower_Band'] = hl2 - (3 * df['ATR'])
    
    # SuperTrend sinyal çizgisi hesaplaması Pandas ile iterate edilmelidir
    # Basitlik açısından tam döngü yerine yaklaşık bir SuperTrend sütunu simülesi:
    supertrend = pd.Series(index=df.index, dtype=float)
    direction = pd.Series(index=df.index, dtype=int)
    
    for i in range(1, len(df)):
        if i == 1:
            supertrend.iloc[i] = df['Basic_Upper_Band'].iloc[i]
            direction.iloc[i] = 1
            continue
            
        if direction.iloc[i-1] == 1: # Trend Down
            if close_series.iloc[i] > supertrend.iloc[i-1]:
                direction.iloc[i] = -1 # Trend Up'a döndü
                supertrend.iloc[i] = df['Basic_Lower_Band'].iloc[i]
            else:
                supertrend.iloc[i] = min(df['Basic_Upper_Band'].iloc[i], supertrend.iloc[i-1])
                direction.iloc[i] = 1
        else: # Trend Up (-1)
            if close_series.iloc[i] < supertrend.iloc[i-1]:
                direction.iloc[i] = 1 # Trend Down'a döndü
                supertrend.iloc[i] = df['Basic_Upper_Band'].iloc[i]
            else:
                supertrend.iloc[i] = max(df['Basic_Lower_Band'].iloc[i], supertrend.iloc[i-1])
                direction.iloc[i] = -1
                
    # Direction: -1 ise Boğa (Trend Yukarı), 1 ise Ayı (Trend Aşağı)
    df['SuperTrend'] = supertrend
    df['Trend_Dir'] = direction

    # 20 Günlük ve Diğer Hareketli Ortalamalar
    df['SMA_50'] = close_series.rolling(window=50).mean()
    df['SMA_200'] = close_series.rolling(window=200).mean()

    return df

def backtest_supertrend_strategy(df, initial_balance=10000):
    balance = initial_balance
    shares = 0
    total_trades = 0
    successful_trades = 0
    last_buy_price = 0
    
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        
        # Hacim
        if isinstance(df.columns, pd.MultiIndex):
            vol = df['Volume'].iloc[i, 0]
        else:
            vol = df['Volume'].iloc[i]
            
        vol_sma_10 = df['Volume_SMA_10'].iloc[i]
        trend_dir = df['Trend_Dir'].iloc[i] # -1: Trend Yukarı, 1: Trend Aşağı
        prev_trend_dir = df['Trend_Dir'].iloc[i-1]
        
        # Göstergelerin tam hesabı için NaN kısımlarını atla
        if pd.isna(trend_dir) or pd.isna(vol_sma_10):
            continue
            
        trend_is_up = trend_dir == -1
        trend_just_turned_up = (prev_trend_dir == 1) and (trend_dir == -1)
        trend_just_turned_down = (prev_trend_dir == -1) and (trend_dir == 1)
        volume_confirm = vol > vol_sma_10
        
        if trend_just_turned_up and volume_confirm and shares == 0:
            # Alım sinyali: SuperTrend Al verdi + Hacim Onayı
            shares = balance / price
            balance = 0
            last_buy_price = price
            total_trades += 1
        elif shares > 0:
            # Satış sinyali (Zarar Kes %7 veya SuperTrend Sat sinyali verdiğinde)
            stop_loss = last_buy_price * 0.93
            
            if trend_just_turned_down or price <= stop_loss:
                balance += shares * price
                
                # Kâr ile kapandıysa istatistiğe ekle
                if price > last_buy_price:
                    successful_trades += 1
                    
                shares = 0
                total_trades += 1

    # Eğer son gün hala hissede kaldıysa güncel fiyattan bozdur
    final_value = balance + (shares * df['Close'].iloc[-1])
    
    # Başarı oranı hesabı (toplam satış işlemlerinin kârlı olanlara oranı)
    win_rate = (successful_trades / (total_trades // 2) * 100) if (total_trades // 2) > 0 else 0
    
    return final_value, total_trades, win_rate

def backtest_bist30_strategy(df, initial_balance=10000):
    balance = initial_balance
    shares = 0
    total_trades = 0
    successful_trades = 0
    last_buy_price = 0
    
    for i in range(1, len(df)):
        price = df['Close'].iloc[i]
        rsi = df['RSI'].iloc[i]
        sma_200 = df['SMA_200'].iloc[i]
        
        if pd.isna(rsi) or pd.isna(sma_200):
            continue
            
        # Çok düşmüş ama uzun vadeli ana trendi yukarı olan (BIST30 mantığı - Güvenli liman)
        if rsi < 35 and price > sma_200 and shares == 0:
            shares = balance / price
            balance = 0
            last_buy_price = price
            total_trades += 1
        elif shares > 0:
            # Satış: %5 Stop-Loss veya %10 Kâr Al veya RSI 70 zirvesi
            if price <= last_buy_price * 0.95 or price >= last_buy_price * 1.10 or rsi > 70:
                balance += shares * price
                if price > last_buy_price:
                    successful_trades += 1
                shares = 0
                total_trades += 1

    final_value = balance + (shares * df['Close'].iloc[-1])
    win_rate = (successful_trades / (total_trades // 2) * 100) if (total_trades // 2) > 0 else 0
    
    return final_value, total_trades, win_rate

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
    buy_price = latest_data['Lower_Band']
    sell_price = latest_data['Upper_Band']
    current_rsi = latest_data['RSI']

    current_macd = latest_data['MACD']
    current_macd_signal = latest_data['MACD_Signal']

    if len(df) > 1:
        previous_data = df.iloc[-2]
        previous_rsi = previous_data['RSI']
        previous_macd = previous_data['MACD']
        previous_macd_signal = previous_data['MACD_Signal']
    else:
        previous_rsi = current_rsi
        previous_macd = current_macd
        previous_macd_signal = current_macd_signal

    # Ekranda daha şık görünmesi için borsa uzantılarını (örn: .IS) atıp sadece hisse adını alalım
    display_symbol = ticker_symbol.split('.')[0].upper()

    # Özet Analiz Tablosunu Oluştur
    st.subheader(f"📊 {display_symbol} İçin Analiz Sonucu")

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
    st.info("Her hissenin karakteri farklıdır. Ağır ilerleyen BIST30 hisseleri ile volatil yan tahtalar aynı stratejiye uymaz. Hissenin karakterine en uygun olan stratejiyi seçip test edin!")
    
    strategy_choice = st.radio(
        "📝 Test Edilecek Stratejiyi Seçin:",
        ["1️⃣ Agresif SuperTrend (Yan Tahtalar ve Trend Hisseleri İçin)", 
         "2️⃣ BIST30 Kasa Katlama (Güvenli: Ana Trend + RSI Dipten Toplama)"]
    )
    
    if "SuperTrend" in strategy_choice:
        final_val, trade_count, win_rate = backtest_supertrend_strategy(df, 10000)
        st.markdown("**Strateji Mantığı:** SuperTrend (10, 3) Al sinyali ve Hacim Onayı ile işleme girer. %7 Stop-Loss uygular. Özellikle KBORU, GESAN gibi hızlı hisselerde (Trend Following) devasa kârlar üretirken, THYAO gibi yatay/ağır hisselerde çok fazla yanlış sinyal üretir.")
    else:
        final_val, trade_count, win_rate = backtest_bist30_strategy(df, 10000)
        st.markdown("**Strateji Mantığı:** 200 Günlük paranın (dev trendin) altında **ASLA** hisse almaz. Trendi yukarı olan hissenin aşırı satıldığı (**RSI < 35**) yani dip yaptığı güvenli yerlerde mal toplar. **%10 Kâr** gördüğünde veya **RSI 70**'te tepeyi satar. THYAO, TUPRS gibi ağır BIST30 hisseleri için biçilmiş kaftandır.")
    
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
