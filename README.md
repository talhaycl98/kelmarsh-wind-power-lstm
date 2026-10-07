@"
# Kelmarsh Rüzgar Santrali - LSTM Tabanlı Güç Tahmini

Bu projede, Kelmarsh rüzgar santralinde (UK) bulunan 2.05 MW Senvion MM92 tipi rüzgar türbinine ait 10 dakikalık SCADA ve türbin olay (Status) verileri kullanılarak LSTM tabanlı aktif güç tahmini gerçekleştirilmiştir.

## Veri Seti
- **Kaynak:** [Zenodo - Kelmarsh Wind Farm Data](https://doi.org/10.5281/zenodo.16807551) (Plumley & Takeuchi, 2025)
- **Kapsam:** 10 dakikalık SCADA ölçümleri ve türbin operasyon/durum günlükleri.

## Metodoloji
1. **Durum Tabanlı Temizleme (Status Filtering):** Türbinin arıza, bakım veya dış çevre faktörleri nedeniyle durdurulduğu (`Status == 'Stop'`) anlar SCADA verisinden ayıklanarak yapay zekanın yanıltıcı fiziksel verilerle eğitilmesi önlenmiştir.
2. **Kayan Pencere (Sliding Window):** Geçmiş 24 zaman adımı (4 saat) girdi olarak alınmış, sonraki 10 dakikalık aktif güç ($t+1$) hedef olarak belirlenmiştir.
3. **Model:** 2 Katmanlı LSTM (64, 32 nöron) + Dropout (%20) + Dense (16) mimarisi.

## Başarım Sonuçları
- **Filtresiz Temel Model MAE:** ~142.65 kW
- **Status Filtrelemeli Model MAE:** ~129.11 kW
- **Status Filtrelemeli Model RMSE:** ~174.86 kW
- Filtreleme sayesinde nominal kapasitesi 2050 kW olan türbinde hata oranı yaklaşık **%6.3** seviyesine indirilmiştir.

## Kullanım
Notebook'u çalıştırmak için:
\`\`\`bash
jupyter notebook kelmarsh_lstm_forecasting.ipynb
\`\`\`
"@ | Out-File -Encoding utf8 README.md