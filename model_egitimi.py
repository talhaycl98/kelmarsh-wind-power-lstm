import pandas as pd
import numpy as np
from io import StringIO
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ==========================================
# 1. SCADA VERİSİNİ OKUMA
# ==========================================
scada_yolu = "Turbine_Data_Kelmarsh_1_2021-01-01_-_2022-01-01_228.csv"
with open(scada_yolu, encoding='utf-8') as f:
    lines = [line.rstrip('\n') for line in f]
header_line = next(line for line in lines if line.lstrip().startswith('# Date and time'))
clean_lines = [header_line.lstrip('#').strip()] + [
    line.strip() for line in lines if line.strip() and not line.lstrip().startswith('#')
]
df = pd.read_csv(StringIO('\n'.join(clean_lines)), parse_dates=['Date and time'])
df.set_index('Date and time', inplace=True)
df.sort_index(inplace=True)

# ==========================================
# 2. STATUS VERİSİNİ OKUMA
# ==========================================
status_yolu = "Status_Kelmarsh_1_2021-01-01_-_2022-01-01_228.csv"

# Yorum satırlarını (# ile başlayanları) atlayarak doğrudan dosyayı okutuyoruz.
status_df = pd.read_csv(status_yolu, comment='#')

# Sadece 'Stop' (Durma) olaylarını filtreleme
stops = status_df[status_df['Status'] == 'Stop'].copy()

# Tarih formatına çevirme (Bitiş tarihi '-' olanları 'NaT' yani boş/hatalı değer yaparız)
stops['Timestamp start'] = pd.to_datetime(stops['Timestamp start'])
stops['Timestamp end'] = pd.to_datetime(stops['Timestamp end'], errors='coerce')

# ==========================================
# 3. ARIZALI ARALIKLARI SCADA'DAN ÇIKARMA
# ==========================================
temiz_scada = df.copy()
for _, row in stops.iterrows():
    start = row['Timestamp start']
    end = row['Timestamp end']
    
    # Eğer bir bitiş zamanı varsa, o aralıktaki tüm SCADA verilerini sil
    if pd.notna(end):
        mask = ~((temiz_scada.index >= start) & (temiz_scada.index <= end))
        temiz_scada = temiz_scada[mask]
    else:
        # Bitiş zamanı yoksa sadece olayın başladığı anı sil
        mask = ~(temiz_scada.index == start)
        temiz_scada = temiz_scada[mask]

# Model girdileri ve negatif güç filtrelemesi
features = ['Wind speed (m/s)', 'Wind direction (°)', 'Power (kW)']
data = temiz_scada[features].copy()
data.dropna(inplace=True)
data = data[data['Power (kW)'] >= 0]

print("Durma (Stop) durumları çıkarıldıktan sonra net veri boyutu:", data.shape)

# ==========================================
# 4. VERİ ÖLÇEKLENDİRME VE MODEL HAZIRLIĞI
# ==========================================
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

def create_dataset(dataset, time_steps=24):
    X, y = [], []
    for i in range(len(dataset) - time_steps):
        X.append(dataset[i:(i + time_steps), :])
        y.append(dataset[i + time_steps, 2]) 
    return np.array(X), np.array(y)

time_steps = 24
X, y = create_dataset(scaled_data, time_steps)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ==========================================
# 5. LSTM MODELİNİ YENİDEN EĞİTME
# ==========================================
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
print("\nYeni ve temiz veri setiyle model eğitimi başlıyor...")
history = model.fit(
    X_train, y_train,
    epochs=15,
    batch_size=64,
    validation_split=0.1,
    shuffle=False
)

# ==========================================
# 6. PERFORMANS TESTİ
# ==========================================
y_pred_scaled = model.predict(X_test)
dummy_pred = np.zeros((len(y_pred_scaled), 3))
dummy_pred[:, 2] = y_pred_scaled[:, 0]
y_pred_actual = scaler.inverse_transform(dummy_pred)[:, 2]

dummy_test = np.zeros((len(y_test), 3))
dummy_test[:, 2] = y_test
y_true_actual = scaler.inverse_transform(dummy_test)[:, 2]

mae = mean_absolute_error(y_true_actual, y_pred_actual)
rmse = np.sqrt(mean_squared_error(y_true_actual, y_pred_actual))
print(f"\n--- Filtrelenmiş Model Performansı ---")
print(f"Yeni MAE:  {mae:.2f} kW")
print(f"Yeni RMSE: {rmse:.2f} kW")

# ==========================================
# 7. GÖRSELLEŞTİRME
# ==========================================
plt.figure(figsize=(14, 6))
plt.plot(y_true_actual[:300], label='Gerçek Güç (kW)', color='blue')
plt.plot(y_pred_actual[:300], label='LSTM Tahmini (kW)', color='red', linestyle='--')
plt.title('Kelmarsh Türbin 1 - Filtrelenmiş Güç Tahmini (İlk 300 Adım)')
plt.xlabel('Zaman Adımı (10 Dakikalık Aralıklar)')
plt.ylabel('Güç (kW)')
plt.legend()
plt.grid(True)
plt.show()