from binance.client import Client
import pandas as pd
from datetime import datetime, timedelta
import time

# Binance API anahtarlarınızı buraya ekleyin
api_key = 'qXcKqSvRoExKay13MPcQedAQKd1NiqJYKVSeOQdoIDztmq9MCO5FxiWJPRJ3pbCa'
api_secret = 'TnUJcH5jTvXcgZy1AG2l3RIvPerp1yrScdfcAkKVH3f8lxpXlQNIhoKhFsuOF8iw'

# Binance istemcisini başlat
client = Client(api_key, api_secret)

def get_historical_data(symbol='BTCUSDT', start_date='1 Jan 2025', end_date=None):
    try:
        # Bitiş tarihi belirtilmemişse şu anı kullan
        if end_date is None:
            end_date = datetime.now()
        else:
            end_date = datetime.strptime(end_date, '%d %b %Y')
        
        start_date = datetime.strptime(start_date, '%d %b %Y')
        
        # Verileri toplamak için boş liste
        all_klines = []
        
        print(f"Veriler toplanıyor: {start_date} - {end_date}")
        
        # Binance'den verileri al
        klines = client.get_historical_klines(
            symbol=symbol,
            interval=Client.KLINE_INTERVAL_1MINUTE,
            start_str=start_date.strftime('%d %b %Y'),
            end_str=end_date.strftime('%d %b %Y')
        )
        
        # Verileri DataFrame'e dönüştür
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 
            'volume', 'close_time', 'quote_asset_volume', 
            'number_of_trades', 'taker_buy_base_asset_volume',
            'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Sadece ihtiyacımız olan sütunları seçelim
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Fiyat verilerini z-score ile normalize et
        def z_score_normalize(data):
            mean = data.mean()
            std = data.std()
            return ((data - mean) / std).round(3)

        # Önce fiyat verilerini float'a çevir
        price_columns = ['open', 'high', 'low', 'close']
        df[price_columns] = df[price_columns].astype(float)
        
        # Otokorelasyon hesaplamaları
        for lag in [5, 10, 20]:
            df[f'Autocorr_{lag}'] = df['close'].astype(float).autocorr(lag)
        
        # Z-score normalizasyonu uygula (orijinal verileri tutarak)
        for col in price_columns:
            df[f'{col}_zscore'] = z_score_normalize(df[col])
        
        # Volume'ü normalize et
        df['volume'] = df['volume'].astype(float)
        df['volume_zscore'] = z_score_normalize(df['volume'])
        
        # Timestamp'i okunabilir formata dönüştür
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # MA hesaplamaları ve normalizasyon
        for period in [9, 21, 50, 200]:
            ma = df['close_zscore'].rolling(window=period).mean()
            df[f'MA_{period}_zscore'] = z_score_normalize(ma)
        
        # RSI hesaplama ve normalizasyon
        def calculate_rsi(data, periods=14):
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        rsi = calculate_rsi(df['close_zscore'])
        df['RSI_14_zscore'] = z_score_normalize(rsi)
        
        # Bollinger Bands hesaplama ve normalizasyon
        def calculate_bollinger_bands(data, window=20, num_std=2):
            ma = data.rolling(window=window).mean()
            std = data.rolling(window=window).std()
            upper = ma + (std * num_std)
            lower = ma - (std * num_std)
            return ma, upper, lower
        
        bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(df['close_zscore'])
        df['BB_middle_zscore'] = z_score_normalize(bb_middle)
        df['BB_upper_zscore'] = z_score_normalize(bb_upper)
        df['BB_lower_zscore'] = z_score_normalize(bb_lower)
        
        # EMA hesaplamaları ve normalizasyon
        for period in [9, 21, 50, 200]:
            ema = df['close_zscore'].ewm(span=period, adjust=False).mean()
            df[f'EMA_{period}_zscore'] = z_score_normalize(ema)
        
        # Sütunları istenen sıraya göre düzenle
        columns_order = ['timestamp',
                        'open', 'high', 'low', 'close', 'volume',
                        'open_zscore', 'high_zscore', 'low_zscore', 'close_zscore',
                        'volume_zscore',
                        'MA_9_zscore', 'MA_21_zscore', 'MA_50_zscore', 'MA_200_zscore',
                        'EMA_9_zscore', 'EMA_21_zscore', 'EMA_50_zscore', 'EMA_200_zscore',
                        'RSI_14_zscore',
                        'BB_middle_zscore', 'BB_upper_zscore', 'BB_lower_zscore',
                        'Autocorr_5', 'Autocorr_10', 'Autocorr_20']
        
        # Mevcut sütunları kontrol et ve sadece var olanları seç
        available_columns = [col for col in columns_order if col in df.columns]
        df = df[available_columns]
        
        # CSV'ye kaydet
        filename = f'{symbol}_1m_with_ema_autocorr_{start_date.strftime("%Y%m%d")}_{end_date.strftime("%Y%m%d")}.csv'
        df.to_csv(filename, index=False, float_format='%.3f', decimal='.')
        print(f"Veriler {filename} dosyasına kaydedildi.")
        
        return df
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        return None

if __name__ == "__main__":
    # Örnek kullanım: Son 6 aylık veri
    df = get_historical_data(
        symbol='BTCUSDT',
        start_date='28 Feb 2025',  # Başlangıç tarihi
        end_date='2 Mar 2025'  # Bitiş tarihi (None = şu an)
    )
