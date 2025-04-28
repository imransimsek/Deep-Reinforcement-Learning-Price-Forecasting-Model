# Deep-Reinforcement-Learning-Price-Forecasting-Model

## Türkçe

### Proje Hakkında
Bu proje, Bitcoin (BTC) fiyatını zaman serisi verileri üzerinde Temporal Fusion Transformer (TFT) kullanarak tahmin etmeyi amaçlar. PyTorch Lightning, PyTorch Forecasting ve MLflow ile deney takibi ve görselleştirme adımlarını içerir.

### Özellikler
- Binance API aracılığıyla dakika bazlı BTCUSDT verisi çekme ve işleme
- Verileri z-score normalizasyonu, hareketli ortalamalar, RSI, Bollinger Bantları ve otokorelasyon ile zenginleştirme
- PyTorch Lightning ve PyTorch Forecasting kullanarak TFT modeli oluşturma ve eğitme
- MLflow ile deney, metrik ve model kayıt takibi
- Gerçek zamanlı tahmin ve görselleştirme

### Kurulum
1. Depoyu klonlayın:
   ```bash
git clone <repo-url>
cd Deep-Reinforcement-Learning-Price-Forecasting-Model
```
2. Sanal ortam oluşturun ve etkinleştirin:
   ```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```
3. Bağımlılıkları yükleyin:
   ```bash
pip install -r requirements.txt
```

### Kullanım
- Veri oluşturma:
  ```bash
python data_creator.py
```
  Bu komut Binance API'den tanımladığınız tarih aralığına göre veri çeker ve `BTCUSDT_1m_with_ema_autocorr_YYYYMMDD_YYYYMMDD.csv` olarak kaydeder.

- Model eğitimi ve gerçek zamanlı tahmin:
  ```bash
python main.py
```
  `main.py` veri yükleme, model oluşturma, eğitim, gerçek zamanlı tahmin ve görselleştirme adımlarını otomatik olarak gerçekleştirir.

### Proje Yapısı
```
.
├── data_creator.py       # Veri toplama ve ön işleme
├── main.py               # Model tanımlama, eğitim ve tahmin
├── requirements.txt      # Proje bağımlılıkları
├── mlruns/               # MLflow deney kayıtları
├── README.md             # Proje açıklamaları
└── ...                   # Diğer dosyalar ve çıktı dosyaları
```

### Lisans
Bu proje MIT Lisansı ile lisanslanmıştır.

## English

### Project Overview
This project aims to forecast Bitcoin (BTC) prices using temporal time series data and the Temporal Fusion Transformer (TFT) model. It includes experiment tracking and visualization with PyTorch Lightning, PyTorch Forecasting, and MLflow.

### Features
- Fetch and process minute-level BTCUSDT data from Binance API
- Enrich data with z-score normalization, moving averages, RSI, Bollinger Bands, and autocorrelation features
- Build and train a TFT model with PyTorch Lightning and PyTorch Forecasting
- Track experiments, metrics, and models with MLflow
- Perform real-time forecasting and visualization

### Installation
1. Clone the repository:
   ```bash
git clone <repo-url>
cd Deep-Reinforcement-Learning-Price-Forecasting-Model
```
2. Create and activate a virtual environment:
   ```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
```
3. Install dependencies:
   ```bash
pip install -r requirements.txt
```

### Usage
- Create data:
  ```bash
python data_creator.py
```
  This command fetches data from the Binance API for a specified date range and saves it as `BTCUSDT_1m_with_ema_autocorr_YYYYMMDD_YYYYMMDD.csv`.

- Train model & real-time forecasting:
  ```bash
python main.py
```
  `main.py` automatically executes steps for data loading, model creation, training, real-time forecasting, and visualization.

### Project Structure
```
.
├── data_creator.py       
├── main.py               
├── requirements.txt      
├── mlruns/               
├── README.md             
└── ...                   
```

### License
This project is licensed under the MIT License.