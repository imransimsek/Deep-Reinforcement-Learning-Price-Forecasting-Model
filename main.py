import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
import mlflow
import mlflow.pytorch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class BTCPredictorModel(pl.LightningModule):
    def __init__(self, tft_model):
        super().__init__()
        self.model = tft_model
        self.save_hyperparameters(ignore=['loss', 'logging_metrics', 'tft_model'])
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        # Trainer'ı modele bağla
        if not hasattr(self.model, 'trainer'):
            self.model.trainer = self.trainer
        
        # Log fonksiyonunu geçici olarak devre dışı bırak
        original_log = self.model.log
        self.model.log = lambda *args, **kwargs: None
        
        try:
            output = self.model.training_step(batch, batch_idx)
            # Eğer output bir sözlük ise, loss değerini al
            loss = output['loss'] if isinstance(output, dict) else output
            self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        finally:
            # Log fonksiyonunu geri yükle
            self.model.log = original_log
    
    def validation_step(self, batch, batch_idx):
        # Trainer'ı modele bağla
        if not hasattr(self.model, 'trainer'):
            self.model.trainer = self.trainer
        
        # Log fonksiyonunu geçici olarak devre dışı bırak
        original_log = self.model.log
        self.model.log = lambda *args, **kwargs: None
        
        try:
            output = self.model.validation_step(batch, batch_idx)
            # Eğer output bir sözlük ise, loss değerini al
            loss = output['loss'] if isinstance(output, dict) else output
            self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
            return loss
        finally:
            # Log fonksiyonunu geri yükle
            self.model.log = original_log
    
    def configure_optimizers(self):
        return self.model.configure_optimizers()

    def on_fit_start(self):
        # Eğitim başlamadan önce trainer'ı bağla
        self.model.trainer = self.trainer

class BTCPredictor:
    def __init__(self, csv_path, sequence_length=60):
        self.csv_path = csv_path
        self.sequence_length = sequence_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        
        # MLflow deneyini başlat
        mlflow.set_experiment("BTC_Price_Prediction_TFT")
        
        # Özellik grupları
        self.static_features = []  # Statik özellikler
        self.time_varying_known_features = [  # Bilinen zamansal özellikler
            'MA_9_zscore', 'MA_21_zscore', 'MA_50_zscore', 'MA_200_zscore',
            'EMA_9_zscore', 'EMA_21_zscore', 'EMA_50_zscore', 'EMA_200_zscore'
        ]
        self.time_varying_unknown_features = [  # Bilinmeyen zamansal özellikler
            'close_zscore', 'RSI_14_zscore',
            'BB_upper_zscore', 'BB_lower_zscore', 'BB_middle_zscore',
            'Autocorr_5', 'Autocorr_20'
        ]
        
    def load_and_prepare_data(self):
        # Veriyi yükle
        df = pd.read_csv(self.csv_path)
        
        # Eksik değerleri doldur - yeni syntax kullanarak
        for feature in self.time_varying_known_features + self.time_varying_unknown_features:
            # İleri doğru doldurma
            df[feature] = df[feature].ffill()
            # Geri doğru doldurma
            df[feature] = df[feature].bfill()
            # Kalan NA değerler için 0 kullan
            df[feature] = df[feature].fillna(0)
            
            # Sonsuz değerleri temizle
            df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
            # Sonsuz değerler yerine en yakın değeri kullan
            df[feature] = df[feature].ffill().bfill()
        
        # Zaman indeksini ekle
        df['time_idx'] = np.arange(len(df))
        df['group'] = 0  # Tek bir zaman serisi olduğu için
        
        print("Veri şekli:", df.shape)
        print("Eksik değerler:")
        print(df[self.time_varying_known_features + self.time_varying_unknown_features].isnull().sum())
        
        # TimeSeriesDataSet için veriyi hazırla
        self.training = TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="close_zscore",  # Hedef değişken
            group_ids=["group"],
            min_encoder_length=self.sequence_length // 2,  # Minimum kodlayıcı uzunluğu
            max_encoder_length=self.sequence_length,  # Maksimum kodlayıcı uzunluğu
            min_prediction_length=1,  # Minimum tahmin uzunluğu
            max_prediction_length=7,  # Maksimum tahmin uzunluğu
            static_categoricals=self.static_features,
            time_varying_known_categoricals=[],
            time_varying_known_reals=self.time_varying_known_features,
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=self.time_varying_unknown_features,
            target_normalizer=GroupNormalizer(groups=["group"]),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        
        # Validation veri setini oluştur
        self.validation = TimeSeriesDataSet.from_dataset(
            self.training, df, min_prediction_idx=df.time_idx.max() - 7 * self.sequence_length
        )
        
        # Veri yükleyicileri oluştur
        self.train_dataloader = self.training.to_dataloader(
            train=True, batch_size=64, num_workers=0
        )
        self.val_dataloader = self.validation.to_dataloader(
            train=False, batch_size=64, num_workers=0
        )
        
        print("Veri yüklendi ve hazırlandı.")
        
    def create_model(self):
        # TFT modelini oluştur
        tft = TemporalFusionTransformer.from_dataset(
            self.training,
            learning_rate=0.001,
            hidden_size=64,
            attention_head_size=4,
            dropout=0.1,
            hidden_continuous_size=32,
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4
        )
        
        # LightningModule olarak wrap et
        self.tft = BTCPredictorModel(tft)
        
        print("Model oluşturuldu.")
        
    def train_model(self, max_epochs=100):
        with mlflow.start_run() as run:
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator='auto',
                devices='auto',
                gradient_clip_val=0.1,
                limit_train_batches=50,
                enable_model_summary=True,
                callbacks=[
                    pl.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        mode="min"
                    )
                ],
                # Bellek optimizasyonları
                accumulate_grad_batches=2,  # Gradient biriktirme
                precision='32-true',        # 32-bit hassasiyet kullan
                deterministic=True,         # Kararlı davranış
                # Bellek temizleme
                enable_checkpointing=False,  # Checkpoint'leri devre dışı bırak
                enable_progress_bar=True,    # İlerleme çubuğunu göster
                logger=True,                # Logging aktif
            )
            
            # Batch size'ı küçült
            self.train_dataloader = self.training.to_dataloader(
                train=True, 
                batch_size=32,  # 64'ten 32'ye düşür
                num_workers=0
            )
            self.val_dataloader = self.validation.to_dataloader(
                train=False, 
                batch_size=32,  # 64'ten 32'ye düşür
                num_workers=0
            )
            
            # Modeli eğit
            trainer.fit(
                model=self.tft,
                train_dataloaders=self.train_dataloader,
                val_dataloaders=self.val_dataloader
            )
            
            # Metrikleri kaydet
            mlflow.log_metrics({
                "train_loss": trainer.callback_metrics["train_loss"].item(),
                "val_loss": trainer.callback_metrics["val_loss"].item()
            })
            
            # Modeli kaydet
            mlflow.pytorch.log_model(self.tft, "model")
            
        print("Model eğitimi tamamlandı.")
        
    def predict_future(self, days_to_predict=7):
        # Son sequence_length kadar veriyi al
        encoder_data = self.training.get_last_encoder_data()
        
        # Tahminleri yap
        predictions = self.tft.model(encoder_data)
        
        # Tahminleri numpy dizisine dönüştür
        predictions = predictions.numpy()
        
        return predictions[:days_to_predict]
    
    def plot_predictions(self, predictions):
        # Gerçek verilerin son kısmını al
        last_known = self.training.y.numpy()[-self.sequence_length:]
        
        # Zaman ekseni
        time_known = np.arange(len(last_known))
        time_pred = np.arange(len(last_known)-1, len(last_known) + len(predictions)-1)
        
        # Grafik çizimi
        plt.figure(figsize=(15, 6))
        plt.plot(time_known, last_known, label='Gerçek Veri', color='blue')
        plt.plot(time_pred, predictions, label='Tahmin', color='red', linestyle='--')
        
        plt.title('BTC Fiyat Tahmini (TFT)')
        plt.xlabel('Zaman (dakika)')
        plt.ylabel('Fiyat (z-score)')
        plt.legend()
        plt.grid(True)
        
        # Grafiği kaydet
        plt.savefig('btc_prediction_tft.png')
        mlflow.log_artifact('btc_prediction_tft.png')
        plt.close()

    def plot_real_time_predictions(self, days_to_predict=7):
        """
        Gerçek zamanlı tahminleri görselleştir
        """
        # Son sequence_length kadar veriyi al
        encoder_data = self.training.get_last_encoder_data()
        
        # Tahminleri yap
        predictions = self.tft.model(encoder_data)
        predictions = predictions.numpy()[:days_to_predict]
        
        # Gerçek test verilerini al
        real_data = self.validation.y.numpy()[:days_to_predict]
        
        # Zaman ekseni
        time_points = np.arange(days_to_predict)
        
        # Grafik çizimi
        plt.figure(figsize=(15, 8))
        
        # Gerçek verileri çiz
        plt.plot(time_points, real_data, 
                label='Gerçek Fiyat', 
                color='blue', 
                linewidth=2, 
                marker='o')
        
        # Tahminleri çiz
        plt.plot(time_points, predictions, 
                label='Tahmin', 
                color='red', 
                linestyle='--', 
                linewidth=2, 
                marker='s')
        
        # Grafik detayları
        plt.title('BTC Fiyat Tahmini vs Gerçek Değerler', fontsize=16, pad=20)
        plt.xlabel('Zaman (Gün)', fontsize=12)
        plt.ylabel('Fiyat (Z-score)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # Tahmin noktalarını işaretle
        for i, (pred, real) in enumerate(zip(predictions, real_data)):
            plt.annotate(f'T: {pred:.2f}', 
                        (i, pred), 
                        textcoords="offset points", 
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8,
                        color='red')
            plt.annotate(f'G: {real:.2f}', 
                        (i, real), 
                        textcoords="offset points", 
                        xytext=(0,-15), 
                        ha='center',
                        fontsize=8,
                        color='blue')
        
        # Grafik düzeni
        plt.tight_layout()
        
        # Grafiği kaydet
        plt.savefig('btc_realtime_prediction.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('btc_realtime_prediction.png')
        plt.close()
        
        # Tahmin performansını hesapla
        mse = np.mean((predictions - real_data) ** 2)
        mae = np.mean(np.abs(predictions - real_data))
        
        print("\nTahmin Performansı:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        
        return predictions, real_data

def main():
    try:
        mlflow.set_tracking_uri("file:./mlruns")
        
        csv_path = "BTCUSDT_1m_with_ema_autocorr_20250228_20250302.csv"
        predictor = BTCPredictor(csv_path)
        
        print("Veri yükleniyor ve hazırlanıyor...")
        predictor.load_and_prepare_data()
        
        print("Model oluşturuluyor...")
        predictor.create_model()
        
        print("Model eğitiliyor...")
        predictor.train_model(max_epochs=100)
        
        print("Gerçek zamanlı tahminler yapılıyor ve görselleştiriliyor...")
        predictions, real_data = predictor.plot_real_time_predictions(days_to_predict=7)
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        raise e

if __name__ == "__main__":
    main() 