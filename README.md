# MNIST Derin Öğrenme Deneyleri

PyTorch ile MNIST üzerinde yapılmış karşılaştırmalı eğitim deneyleri. İki bölümden oluşur:

- **Görev 1** — Sıfırdan sinir ağı implementasyonu + MLP/CNN mimari ve dropout taraması
- **Görev 2** — Veri miktarı azaltma, sınıf dengesizliği ve accuracy dışındaki metrikler (precision / recall / F1 / confusion matrix)

Tüm deneylerde test seti **daima tam ve dengeli 10.000'lik MNIST test setidir**. Yalnızca eğitim seti değiştirilir — aksi halde azınlık sınıflarının recall'ü ölçülemezdi.

## Kurulum

Python 3.11 ve CUDA destekli PyTorch gerekir. Eğitim scriptleri GPU bulamazsa hata verip durur (`RuntimeError: GPU bulunamadi!`).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy
```

MNIST veri seti repoda tutulmaz; scriptler ilk çalıştırmada `kayitlar/data/` altına otomatik indirir (~64 MB).

## Çalıştırma

Scriptler konumlarına göre yol çözer, herhangi bir dizinden çalıştırılabilir.

```bash
# Görev 1
python gorev1/tek_noron.py            # tek katman ileri besleme, aktivasyon fonksiyonları
python gorev1/sinir_agi.py            # NumPy ile sıfırdan 2 katmanlı ağ, XOR problemi
python gorev1/egitim_no_cnn.py        # 15 MLP konfigürasyonu
python gorev1/egitim_cnn.py           # 15 CNN konfigürasyonu
python gorev1/egitim_cnn_uzun.py      # güçlü CNN, 50 epoch

# Görev 2
python gorev2/egitim_veri_azalt.py         # 20 deney  (~6 dk)
python gorev2/egitim_dengesiz_siniflar.py  # 12 deney  (~3 dk)
python gorev2/egitim_hazir_modeller.py     # 16 deney  (~16 dk)
```

Süreler RTX 4060 Laptop üzerinde ölçülmüştür.

Sonuçlar `kayitlar/gorev1/` ve `kayitlar/gorev2/` altına CSV olarak yazılır. Her deneyden sonra dosya güncellenir, yani koşu yarıda kesilse de o ana kadarki sonuçlar durur.

## Görev 1 — Mimari ve Dropout Taraması

| Script | İçerik |
|---|---|
| `tek_noron.py` | 2 girdi → 3 nöron, sigmoid / ReLU / tanh / softmax çıktılarının elle hesabı |
| `sinir_agi.py` | NumPy ile forward, BCE loss, backward, SGD; XOR üzerinde eğitim |
| `egitim_no_cnn.py` | MLP: 1/2/4/6 gizli katman × dropout 0 / 0.1 / 0.3 / 0.5 |
| `egitim_cnn.py` | CNN: kanal sayısı ve conv katman derinliği × dropout taraması |
| `egitim_cnn_uzun.py` | (64, 128, 256) + BatchNorm + augmentation + OneCycle, 50 epoch |

Güçlü CNN 50 epoch sonunda **%99.66–99.67** accuracy'ye ulaşıyor.

## Görev 2 — Veri Azaltma ve Sınıf Dengesizliği

Toplam **48 eğitim**, hepsi seed 27 / 10 epoch / batch 128 / Adam lr 0.001 ile.

| Script | Deney | Ne yapıyor |
|---|---|---|
| `egitim_veri_azalt.py` | 20 | Dengeli alt kümeler (1.000 / 3.000 / 5.000 / 10.000 / 60.000) × 4 konfigürasyon |
| `egitim_dengesiz_siniflar.py` | 12 | 3 dengesizlik profili × 4 konfigürasyon |
| `egitim_hazir_modeller.py` | 16 | ResNet18 ve MobileNetV2, aynı düzende (28×28 gri → 32×32×3, sıfırdan eğitim) |

Sınıf grupları: **çoğunluk** 0, 1 — **orta** 2, 3, 4, 7, 9 — **azınlık** 5, 6, 8.

Dengesizlik profilleri:

| Profil | Çoğunluk | Orta | Azınlık | Toplam | Oran |
|---|---|---|---|---|---|
| `P1_pdf` | 3.000 | 500 | 150 | 8.950 | 20:1 |
| `P2_agir` | 3.000 | 300 | 100 | 7.800 | 30:1 |
| `P3_5k` | 1.500 | 300 | 120 | 4.860 | 12.5:1 |

Her deney için accuracy, macro precision / recall / F1, azınlık recall, sınıf bazlı metrikler ve 10×10 confusion matrix kaydedilir.

## Öne Çıkan Bulgular

**Veri azaltma ayırt ediciliği artırıyor.** Konfigürasyonlar arası fark tam veride 1.28 puan, 3.000 örnekte 3.72 puan. Karşılaştırmalı deney için 3.000–5.000 aralığı öneriliyor.

**Accuracy dengesiz veride yanıltıcı — ölçüldü.** Aynı veri bütçesinde dengeden dengesizliğe geçildiğinde accuracy 2–4 puan düşerken azınlık recall'ü 6–13 puan düşüyor.

**F1 tek başına da yeterli değil.** Precision ve recall zıt yönlerde saptığında F1 ikisini ortalayıp sorunu gizliyor — sınıf 8'de precision %97.95, recall %73.51 iken F1 %83.99 görünüyor. Dengesiz veride ikisi ayrı ayrı raporlanmalı.

**Model seçimi veri miktarına bağlı.** 60.000 örnekte sıralama CNN_d03 > ResNet18 > MobileNetV2 > MLP iken, 1.000 örnekte ResNet18 > MLP > CNN > MobileNetV2'ye dönüyor. Tek bir veri boyutunda yapılan karşılaştırma yanıltıcı.

**Büyük model her zaman kazanmıyor.** Tam veri setinde 30 bin parametrelik basit CNN (%99.12), 11 milyon parametreli ResNet18'i (%98.91) geçiyor.

**GPU her modelde hızlandırmıyor.** MLP'de kazanç ≈1.0× (darboğaz hesaplama değil, veri yükleme); CNN'de 1.8×.

Ayrıntılı rapor: [`kayitlar/gorev2/rapor_gorev2.html`](kayitlar/gorev2/rapor_gorev2.html) — 48 deneyin tam tabloları, karşılaştırmalar ve confusion matrix analizleri. PDF sürümü aynı klasörde.

## Klasör Yapısı

```
gorev1/                     Görev 1 scriptleri
gorev2/                     Görev 2 scriptleri
kayitlar/
  data/                     MNIST (indirilir, repoda tutulmaz)
  gorev1/                   Görev 1 sonuçları
  gorev2/                   Görev 2 sonuçları
    confusion/              48 adet 10x10 confusion matrix
    rapor_gorev2.html       ayrıntılı rapor
csvler/                     eski koşuların sonuçları (arşiv)
```

## Notlar

- Tüm rastgelelik kaynakları seed 27 ile sabitlenmiştir; her model aynı başlangıç ağırlıklarıyla kurulur.
- `csvler/` klasörü scriptlerin daha eski bir sürümüyle alınmış, konfigürasyon adları farklı olan tam koşuları içerir; güncel scriptler bu klasöre yazmaz.
- Metrikler tek geçişte confusion matrix'ten türetilir (`metricler_hesapla`), bu yüzden her epoch sonunda tam test seti ölçümü almanın maliyeti düşüktür.
