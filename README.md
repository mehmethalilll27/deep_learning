# Derin Öğrenme Deneyleri

PyTorch ile yapılmış karşılaştırmalı eğitim deneyleri. Üç bölümden oluşur:

- **Görev 1** — Sıfırdan sinir ağı implementasyonu + MLP/CNN mimari ve dropout taraması (MNIST)
- **Görev 2** — Veri miktarı azaltma, sınıf dengesizliği ve accuracy dışındaki metrikler (precision / recall / F1 / confusion matrix) (MNIST)
- **Görev 3** — Kaggle veri setinde serbest deney: düşük bir baseline'dan başlayıp skoru adım adım yükseltme (Intel Image Classification)

Görev 1 ve 2'de test seti **daima tam ve dengeli 10.000'lik MNIST test setidir**. Yalnızca eğitim seti değiştirilir — aksi halde azınlık sınıflarının recall'ü ölçülemezdi.

Görev 3'te düzen farklıdır: eğitim seti train/validation olarak ikiye bölünür, **iyileştirme kararları validation setine bakarak verilir**, test seti her konfigürasyon için yalnızca bir kez ölçülür. Sebebi, o görevde tekrar tekrar ayar yapılması — her denemeyi test setine bakarak seçmek test set leakage olurdu.

## Kurulum

Python 3.11 ve CUDA destekli PyTorch gerekir. Eğitim scriptleri GPU bulamazsa hata verip durur (`RuntimeError: GPU bulunamadi!`).

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install numpy
```

MNIST veri seti repoda tutulmaz; scriptler ilk çalıştırmada `kayitlar/data/` altına otomatik indirir (~64 MB).

Görev 3 için ayrıca Kaggle veri seti gerekir (~346 MB, repoda tutulmaz):

```bash
pip install kaggle
# kaggle.com/settings -> Legacy API Credentials -> Create New Token
# inen kaggle.json dosyasi ~/.kaggle/kaggle.json konumuna konur

kaggle datasets download -d puneet6060/intel-image-classification -p kayitlar/data/intel
# zip kayitlar/data/intel/ icine acilir; seg_train/ seg_test/ seg_pred/ olusur
```

Görev 3'ün transfer learning scripti ilk çalıştırmada torchvision'ın ImageNet ağırlıklarını indirir (~50 MB, internet gerekir).

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

# Görev 3 — sırayla çalıştırılmalı
python gorev3/egitim_baseline.py       #  1 deney  (~3 dk)
python gorev3/egitim_augmentation.py   #  4 deney  (~20 dk)
python gorev3/egitim_mimari.py         #  5 deney  (~25 dk)
python gorev3/egitim_transfer.py       #  5 deney  (~35 dk)
```

Süreler RTX 4060 Laptop üzerinde ölçülmüştür.

Sonuçlar `kayitlar/gorev1/` ve `kayitlar/gorev2/` altına CSV olarak yazılır. Her deneyden sonra dosya güncellenir, yani koşu yarıda kesilse de o ana kadarki sonuçlar durur.

## Görev 1 — Mimari ve Dropout Taraması


| Script               | İçerik                                                                      |
| -------------------- | --------------------------------------------------------------------------- |
| `tek_noron.py`       | 2 girdi → 3 nöron, sigmoid / ReLU / tanh / softmax çıktılarının elle hesabı |
| `sinir_agi.py`       | NumPy ile forward, BCE loss, backward, SGD; XOR üzerinde eğitim             |
| `egitim_no_cnn.py`   | MLP: 1/2/4/6 gizli katman × dropout 0 / 0.1 / 0.3 / 0.5                     |
| `egitim_cnn.py`      | CNN: kanal sayısı ve conv katman derinliği × dropout taraması               |
| `egitim_cnn_uzun.py` | (64, 128, 256) + BatchNorm + augmentation + OneCycle, 50 epoch              |


Güçlü CNN 50 epoch sonunda **%99.66–99.67** accuracy'ye ulaşıyor.

## Görev 2 — Veri Azaltma ve Sınıf Dengesizliği

Toplam **48 eğitim**, hepsi seed 27 / 10 epoch / batch 128 / Adam lr 0.001 ile.


| Script                        | Deney | Ne yapıyor                                                                      |
| ----------------------------- | ----- | ------------------------------------------------------------------------------- |
| `egitim_veri_azalt.py`        | 20    | Dengeli alt kümeler (1.000 / 3.000 / 5.000 / 10.000 / 60.000) × 4 konfigürasyon |
| `egitim_dengesiz_siniflar.py` | 12    | 3 dengesizlik profili × 4 konfigürasyon                                         |
| `egitim_hazir_modeller.py`    | 16    | ResNet18 ve MobileNetV2, aynı düzende (28×28 gri → 32×32×3, sıfırdan eğitim)    |


Sınıf grupları: **çoğunluk** 0, 1 — **orta** 2, 3, 4, 7, 9 — **azınlık** 5, 6, 8.

Dengesizlik profilleri:


| Profil    | Çoğunluk | Orta | Azınlık | Toplam | Oran   |
| --------- | -------- | ---- | ------- | ------ | ------ |
| `P1_pdf`  | 3.000    | 500  | 150     | 8.950  | 20:1   |
| `P2_agir` | 3.000    | 300  | 100     | 7.800  | 30:1   |
| `P3_5k`   | 1.500    | 300  | 120     | 4.860  | 12.5:1 |


Her deney için accuracy, macro precision / recall / F1, azınlık recall, sınıf bazlı metrikler ve 10×10 confusion matrix kaydedilir.

## Görev 3 — Kaggle Veri Setinde Skor Yükseltme

Veri seti: [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) — 6 sınıf doğa/şehir manzarası (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`), 14.034 eğitim + 3.000 test görüntüsü, ~150×150 RGB. Sınıflar dengeli (en büyük/en küçük oranı 1.15:1), bu yüzden Görev 2'nin azınlık metrikleri buraya taşınmadı.

Görüntülerin %99.8'i 150×150 ama hepsi değil (150×113 gibi kareler var). Bu yüzden `transforms.Resize((128, 128))` ikili parametreyle çağrılır — tek sayı verilseydi en-boy oranı korunur, farklı boyutlar batch'e yığılamaz ve eğitim rastgele bir batch'te düşerdi.

**Deney zinciri.** Her aşama bir öncekinin üstüne tek bir şey ekler; model / epoch / seed sabit tutulur ki skordaki değişim başka bir şeye atfedilemesin.


| Script                   | Deney | Ne değişiyor                                                       |
| ------------------------ | ----- | ------------------------------------------------------------------ |
| `egitim_baseline.py`     | 1     | Basit CNN, augmentation / BatchNorm / dropout / scheduler **yok**  |
| `egitim_augmentation.py` | 4     | Augmentation profili: yok / hafif / orta / güçlü                   |
| `egitim_mimari.py`       | 5     | BatchNorm → dropout → Global Average Pooling → AdamW → OneCycle    |
| `egitim_transfer.py`     | 5     | ImageNet ön eğitimi: sıfırdan / donuk gövde / fine-tune × 3 mimari |


Baseline bilerek zayıf kurulmuştur. Amaç iyi bir ilk skor değil, sonraki adımların neyi düzelttiğini ölçebilmek. Sonuç beklendiği gibi: **test %81.00, train/val farkı +16.87 puan** — yani model ezberliyor.

Transfer learning deneylerinde learning rate konfigürasyona göre değişir: sıfırdan eğitim ve donuk gövdede `0.001`, fine-tune'da `0.0001`. Yüksek learning rate hazır ImageNet filtrelerini ilk epoch'ta bozar.

Tüm scriptler ortak `kayitlar/gorev3/sonuclar_gorev3.csv` dosyasına yazar; aynı adlı deney varsa üzerine yazılır, diğerleri korunur. Yani scriptler ayrı ayrı ve tekrar tekrar çalıştırılabilir, tablo tek parça kalır.

**Sonuç.** 15 deney, toplam 47.1 dakika. Aşama aşama ilerleme (her aşamanın kazananı validation'a göre seçilmiştir):


| Aşama           | Kazanan             | Test Accuracy | Overfit farkı | Baseline'a göre |
| --------------- | ------------------- | ------------- | ------------- | --------------- |
| Baseline        | `baseline_cnn`      | %81.00        | +16.87        | —               |
| 1. Augmentation | `aug_orta`          | %84.50        | +8.05         | +3.50           |
| 2. Mimari       | `mim_gap_onecycle`  | %86.03        | 0.00          | +5.03           |
| **3. Transfer** | `EfficientNetB0_ft` | **%94.33**    | +5.74         | **+13.33**      |


Test hatası 570 örnekten 170 örneğe indi (**%70.2 hata azalması**).

## Öne Çıkan Bulgular

**Veri azaltma ayırt ediciliği artırıyor.** Konfigürasyonlar arası fark tam veride 1.28 puan, 3.000 örnekte 3.72 puan. Karşılaştırmalı deney için 3.000–5.000 aralığı öneriliyor.

**Accuracy dengesiz veride yanıltıcı — ölçüldü.** Aynı veri bütçesinde dengeden dengesizliğe geçildiğinde accuracy 2–4 puan düşerken azınlık recall'ü 6–13 puan düşüyor.

**F1 tek başına da yeterli değil.** Precision ve recall zıt yönlerde saptığında F1 ikisini ortalayıp sorunu gizliyor — sınıf 8'de precision %97.95, recall %73.51 iken F1 %83.99 görünüyor. Dengesiz veride ikisi ayrı ayrı raporlanmalı.

**Model seçimi veri miktarına bağlı.** 60.000 örnekte sıralama CNN_d03 > ResNet18 > MobileNetV2 > MLP iken, 1.000 örnekte ResNet18 > MLP > CNN > MobileNetV2'ye dönüyor. Tek bir veri boyutunda yapılan karşılaştırma yanıltıcı.

**Büyük model her zaman kazanmıyor.** Tam veri setinde 206.922 parametrelik basit CNN (%99.12), 11.181.642 parametreli ResNet18'i (%98.91) geçiyor — 54 kat boyut farkına rağmen.

**GPU her modelde hızlandırmıyor.** MLP'de kazanç ≈1.0× (darboğaz hesaplama değil, veri yükleme); CNN'de 1.8×.

**Transfer learning diğer tüm çabaların toplamından fazla getiriyor.** Görev 3'teki 13.33 puanlık kazancın 8.30'u tek bir adımdan. Augmentation, BatchNorm, dropout, GAP, AdamW ve OneCycle'dan oluşan dokuz deneylik çalışma toplam 5.03 puan ekledi.

**3.078 parametre, sıfırdan eğitilen 11.18 milyonu 6.97 puan geçti.** Aynı ResNet18, aynı veri; fark yalnızca başlangıç ağırlıkları. Pretrained gövde dondurulup sadece son katman eğitildiğinde %86.17, sıfırdan eğitildiğinde %79.20.

**Parametrelerin %98.5'i işe yaramıyordu.** Flatten sonrası dev FC katmanı Global Average Pooling ile değiştirildiğinde parametre 66 kat azaldı (8.48M → 128K) ve skor yükseldi. "Büyük model her zaman kazanmıyor" bulgusunun daha keskin doğrulaması.

**Düzenlileştirme teknikleri toplanabilir değil.** Güçlü augmentation, BatchNorm ve AdamW — üçü de tek başına makul, ama zaten düzenlenmiş bir modele eklendiğinde skoru *düşürdüler*. Optimumu geçmek, hiç düzenlileştirmemek kadar zararlı olabiliyor.

**Sıfır overfit hedef değil.** `mim_gap_onecycle`'da train−val farkı tam 0.00'a indi ama model o noktada öğrenemiyordu. En iyi skor +5.74 overfit farkıyla geldi.

**Tek koşuluk deneylerde 1.2 puanlık gürültü var — ölçüldü.** Birebir aynı konfigürasyon (`baseline_cnn` / `aug_yok`) iki koşuda %81.00 ve %82.23 verdi. Sebep `cudnn.benchmark = True`. Bu eşiğin altındaki farklar üstünlük kanıtı sayılmamalı. Görev 1 ve 2'deki deneyler de tek koşuydu; aynı belirsizlik orada da vardı ama ölçülmemişti.

Ayrıntılı raporlar:

- `[kayitlar/gorev2/rapor_gorev2.html](kayitlar/gorev2/rapor_gorev2.html)` — 48 deneyin tam tabloları, karşılaştırmalar ve confusion matrix analizleri. PDF sürümü aynı klasörde.
- `[kayitlar/gorev3/rapor_gorev3.html](kayitlar/gorev3/rapor_gorev3.html)` — 15 deneyin tabloları, aşama aşama kazanç, confusion matrix ve sınıf bazlı analizler.



## Klasör Yapısı

```
gorev1/                     Görev 1 scriptleri
gorev2/                     Görev 2 scriptleri
gorev3/                     Görev 3 scriptleri
kayitlar/
  data/                     MNIST (indirilir, repoda tutulmaz)
    intel/                  Intel Image Classification (Kaggle, repoda tutulmaz)
  gorev1/                   Görev 1 sonuçları
  gorev2/                   Görev 2 sonuçları
    confusion/              48 adet 10x10 confusion matrix
    rapor_gorev2.html       ayrıntılı rapor
  gorev3/                   Görev 3 sonuçları
    confusion/              deney başına 6x6 confusion matrix
    sonuclar_gorev3.csv     tüm deneylerin ortak tablosu
    sinif_bazli_gorev3.csv  deney başına sınıf bazlı precision/recall/F1
    epoch_gecmisi_*.csv     deney başına epoch epoch seyir
    rapor_gorev3.html       ayrıntılı rapor
csvler/                     eski koşuların sonuçları (arşiv)
```



## Notlar

- Tüm rastgelelik kaynakları seed 27 ile sabitlenmiştir; her model aynı başlangıç ağırlıklarıyla kurulur.
- `csvler/` klasörü scriptlerin daha eski bir sürümüyle alınmış, konfigürasyon adları farklı olan tam koşuları içerir; güncel scriptler bu klasöre yazmaz.
- Metrikler tek geçişte confusion matrix'ten türetilir (`metricler_hesapla`), bu yüzden her epoch sonunda tam test seti ölçümü almanın maliyeti düşüktür.

