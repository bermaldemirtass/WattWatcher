import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Veri setini oku
df = pd.read_csv("energydata_complete.csv")

# Sadece 'Appliances' sütununu kullanacağız
data = df[['Appliances']].values

# Normalize et
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(data)

print("Veri başarıyla yüklendi ve normalize edildi.")
import numpy as np

# LSTM için veriyi pencerelere bölelim
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 24  # Son 24 saatlik veriden tahmin
X, y = create_sequences(normalized_data, window_size)

print(f"Veri dizileri hazır: X shape = {X.shape}, y shape = {y.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split

# Eğitim ve doğrulama setlerine ayır
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

# LSTM modelini oluştur
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print("Model eğitiliyor...")
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20, batch_size=32)
print("Model eğitimi tamamlandı.")

import matplotlib.pyplot as plt

# Tahminleri al
predictions = model.predict(X_val)

# Görselleştir: İlk 100 örnek için
plt.figure(figsize=(10, 5))
plt.plot(y_val[:100], label="Gerçek", color='blue')
plt.plot(predictions[:100], label="Tahmin", color='red')
plt.title("Gerçek vs Tahmin (İlk 100 saat)")
plt.xlabel("Zaman (saat)")
plt.ylabel("Enerji Tüketimi (normalleştirilmiş)")
plt.legend()
plt.tight_layout()
plt.savefig("prediction_plot.png")
plt.savefig("prediction_plot.png")

plt.show()

