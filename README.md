# Derin Öğrenme Deneyleri

PyTorch ile yapılmış karşılaştırmalı eğitim deneyleri. Dört bölümden oluşur:

- **Görev 1** — Sıfırdan sinir ağı implementasyonu + MLP/CNN mimari ve dropout taraması (MNIST)
- **Görev 2** — Veri miktarı azaltma, sınıf dengesizliği ve accuracy dışındaki metrikler (precision / recall / F1 / confusion matrix) (MNIST)
- **Görev 3** — Kaggle veri setinde serbest deney: düşük bir baseline'dan başlayıp skoru adım adım yükseltme (Intel Image Classification)
- **Görev 4** — Semantik segmentasyon: maskeli veri setinde U-Net eğitimi, IoU/Dice ile değerlendirme ve skor yükseltme (Satellite Images of Water Bodies)

Görev 1 ve 2'de test seti **daima tam ve dengeli 10.000'lik MNIST test setidir**. Yalnızca eğitim seti değiştirilir — aksi halde azınlık sınıflarının recall'ü ölçülemezdi.

Görev 3 ve 4'te düzen farklıdır: veri train/validation/test olarak üçe bölünür, **iyileştirme kararları validation setine bakarak verilir**, test seti her konfigürasyon için yalnızca bir kez ölçülür. Sebebi, o görevlerde tekrar tekrar ayar yapılması — her denemeyi test setine bakarak seçmek test set leakage olurdu.

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

Görev 4 için ek paketler ve ayrı bir Kaggle veri seti gerekir (~150 MB, repoda tutulmaz):

```bash
pip install opencv-python matplotlib
pip install albumentations==1.4.18      # 2.x stringzilla istiyor, o da MSVC build tools istiyor
pip install segmentation-models-pytorch

kaggle datasets download -d franciscoescobar/satellite-images-of-water-bodies -p kayitlar/data/water
# zip kayitlar/data/water/ icine acilir; "Water Bodies Dataset/Images" ve ".../Masks" olusur
```

Görev 4'ün encoder scripti ilk çalıştırmada `segmentation_models_pytorch` üzerinden ImageNet ağırlıklarını indirir (~100 MB toplam).

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

# Görev 4 — sırayla çalıştırılmalı
python gorev4/egitim_baseline.py       #  1 deney  (~11 dk)
python gorev4/egitim_loss.py           #  4 deney  (~42 dk)
python gorev4/egitim_augmentation.py   #  4 deney  (~43 dk)
python gorev4/egitim_encoder.py        #  4 deney  (~24 dk)
python gorev4/egitim_mimari.py         #  4 deney  (~25 dk)
```

Süreler RTX 4060 Laptop üzerinde ölçülmüştür.

Sonuçlar ilgili `kayitlar/gorevN/` klasörüne CSV olarak yazılır. Her deneyden sonra dosya güncellenir, yani koşu yarıda kesilse de o ana kadarki sonuçlar durur.

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

## Görev 4 — Semantik Segmentasyon

Veri seti: [Satellite Images of Water Bodies](https://www.kaggle.com/datasets/franciscoescobar/satellite-images-of-water-bodies) — 2.841 Sentinel-2 görüntüsü + birebir eşleşen ikili su maskesi. Görüntü boyutları değişken (300 örnekte 298 farklı boyut), hepsi 256×256'ya getirilir. Ortalama su oranı **%32.89**; hiç su içermeyen görüntü yok, tamamen su olan 123 tane.

Görev 3'ün veri seti kullanılamadı: `seg_train` / `seg_test` / `seg_pred` klasör adları yanıltıcı, 24.335 dosyanın tamamı `.jpg` görüntü, **maske yok**.

**Segmentasyona özgü üç zorunluluk:**

- **Maskeler eşiklenmeli.** `.jpg` saklandıkları için ikili değiller — `water_body_1.jpg` maskesinde 88 farklı piksel değeri var. Hepsi 127 eşiğiyle ikiliye çevrilir.
- **Maske nearest ile yeniden boyutlandırılmalı.** Bilinear kullanılsaydı kenarlarda 0.37 gibi ara değerler oluşur, etiketler sessizce bozulurdu. `albumentations.Resize` bunu otomatik ayırır.
- **Augmentation'da maske de dönmeli.** Görüntü ve maske tek çağrıya verilir, aynı rastgele parametrelerle dönüştürülür.

Ortak yardımcı kod `gorev4/ortak.py` içinde toplandı — Görev 2 ve 3'teki "her script kendi içinde çalışsın" düzeninden bilinçli sapma. Segmentasyon yardımcıları (veri okuma, maske eşikleme, IoU/Dice, görsel üretme) beş scripte kopyalanamayacak kadar büyük.


| Script                   | Deney | Ne değişiyor                                                            |
| ------------------------ | ----- | ----------------------------------------------------------------------- |
| `egitim_baseline.py`     | 1     | Sıfırdan U-Net (7.763.041 param), BCE, augmentation **yok**             |
| `egitim_loss.py`         | 4     | BCE / Dice / BCE+Dice / Focal                                           |
| `egitim_augmentation.py` | 4     | yok / hafif / orta / güçlü                                              |
| `egitim_encoder.py`      | 4     | resnet34 sıfırdan, resnet34 / efficientnet-b0 / mobilenet_v2 pretrained |
| `egitim_mimari.py`       | 4     | Unet / UnetPlusPlus / DeepLabV3Plus / FPN (encoder sabit)               |


**Sonuç.** 17 deney, toplam 145.8 dakika.


| Aşama           | Kazanan                          | Test IoU   | Test Dice  | Baseline'a göre  |
| --------------- | -------------------------------- | ---------- | ---------- | ---------------- |
| Baseline        | `baseline_unet`                  | %71.28     | %83.23     | —                |
| A1 Loss         | `loss_BCE_Dice`                  | %74.25     | %85.22     | +2.97            |
| A2 Augmentation | `aug_hafif`                      | %72.07     | %83.77     | *gürültü içinde* |
| **A3 Encoder**  | `enc_efficientnet_b0_pretrained` | **%81.80** | **%89.99** | **+10.52**       |
| A4 Mimari       | `mim_Unet`                       | **%82.75** | **%90.56** | **+11.47**       |


Nihai model: `Unet + efficientnet-b0 (ImageNet pretrained) + BCE+Dice + hafif augmentation`, 6.251.469 parametre.

Kazancın kaynağı recall: precision %89.82 → %91.48 (+1.66) iken recall %77.54 → %89.66 (**+12.12**). Model kaçırdığı suyu bulmayı öğrendi, su uydurmayı değil.

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

**Gürültü ölçülmeli, varsayılmamalı — Görev 4'te üç kez ölçüldü.** Aynı konfigürasyonun üç tekrar çifti **0.14 / 0.95 / 2.33** puan fark verdi. İlk çift görülüp "gürültü 0.14" denseydi raporun neredeyse tüm sonuçları yanlış yorumlanırdı. Görev 4'te 2.33 puandan (test IoU) ve 3.24 puandan (validation IoU) küçük hiçbir fark iyileştirme sayılmadı.

**Piksel doğruluğu segmentasyondaki iyileştirmenin üçte ikisini gizliyor.** Aynı iyileştirme IoU'da 11.47, Dice'ta 7.33, piksel doğruluğunda yalnızca 4.13 puan görünüyor. Piksellerin %67'si arka plan olduğu için doğruluk büyük ölçüde kolay kısmı ölçüyor.

**Segmentasyonda dört iyileştirme kaldıracından yalnızca biri ölçülebilir etki üretti.** Pretrained encoder +11.61 puan (gürültünün 5 katı); loss, augmentation ve mimari seçimlerinin etkisi gürültünün içinde veya sınırında kaldı. Dördüne de "kazanan" atamak mümkündü, ölçüm bunu desteklemiyordu.

**Aynı reçete farklı hastalığa uygulanınca işe yaramıyor.** Görev 3'te augmentation 16.87 puanlık overfit'i 8 puana indirmişti. Görev 4'ün baseline'ında overfit yoktu (+0.15; train IoU test IoU'nun *altında*) — sorun ezber değil yetersiz öğrenmeydi, augmentation da hiçbir şey değiştirmedi.

**Bir adımın etkisi kendisinden sonraki adıma bağlı olabilir.** Görev 4'te augmentation A2'de test edildi (overfit +0.02, etki yok), ama A3'te pretrained encoder gelince overfit +6.81'e çıktı — ezber *sonradan* oluştu. "Her adımı sırayla dene" yaklaşımının kör noktası bu.

**Focal loss veri setine uymadı ve nedeni önceden ölçülmüştü.** Focal ağır sınıf dengesizliği varsayar; bu veri setinde su oranı %32.89. Veri setinin özelliğini varsaymak yerine ölçmek yanlış araç seçmekten korudu — Focal en kötü sonucu verdi (%63.52 IoU, görüntü başına %55.64).

**Ortalama skor loss'un ne yaptığını göstermiyor, hata profili gösteriyor.** BCE+Dice ile BCE arasındaki toplam IoU farkı gürültü sınırındayken recall farkı iki bağımsız koşuda da sağlam: BCE %76.59 / %77.54'e karşı BCE+Dice %83.63 / %84.07, aralıklar hiç örtüşmüyor.

**Pretrained ağırlık olmadan büyük mimari dezavantaj.** Sıfırdan resnet34 (24.4M param) %69.94 IoU; el yazması U-Net (7.8M param) aynı koşulda %72.07. Görev 2'nin "büyük model her zaman kazanmıyor" bulgusunun çok daha net versiyonu.

**Transfer learning yalnızca tavanı değil hızı da değiştiriyor.** Pretrained model **1. epoch sonunda** %68.99 val IoU aldı — sıfırdan eğitilen modellerin **20 epoch sonunda** ulaştığı seviye. Sıfırdan modeller hiçbir epoch'ta %78'e ulaşamadı.

**Skor eşitken maliyet karar verir.** efficientnet-b0 ile resnet34 arasındaki 0.25 puanlık fark anlamsızdı; 3.9 kat parametre farkı (6.25M / 24.44M) anlamlıydı.

**Öngörü ölçümün yerini tutmuyor.** Görev 4'te su kütleleri büyük ve bütünlüklü olduğu için DeepLabV3+ ve FPN'in öne çıkacağı öngörülmüştü; ikisi de sonuncu oldu. Gözden kaçan şey DeepLabV3+'ın ince skip bağlantısının olmaması — sınırlar kaba kalıyor, bu veri setinde ise sınır hassasiyeti kritik.

Ayrıntılı raporlar:

- `[kayitlar/gorev2/rapor_gorev2.html](kayitlar/gorev2/rapor_gorev2.html)` — 48 deneyin tam tabloları, karşılaştırmalar ve confusion matrix analizleri. PDF sürümü aynı klasörde.
- `[kayitlar/gorev3/rapor_gorev3.html](kayitlar/gorev3/rapor_gorev3.html)` — 15 deneyin tabloları, aşama aşama kazanç, confusion matrix ve sınıf bazlı analizler.
- `[kayitlar/gorev4/rapor_gorev4.html](kayitlar/gorev4/rapor_gorev4.html)` — 17 deneyin tabloları, IoU/Dice metodolojisi, gürültü eşiği analizi, bölge bazlı görsel karşılaştırma.



## Klasör Yapısı

```
gorev1/                     Görev 1 scriptleri
gorev2/                     Görev 2 scriptleri
gorev3/                     Görev 3 scriptleri
gorev4/                     Görev 4 scriptleri
  ortak.py                  paylaşılan katman (veri, metrik, loss, eğitim, görsel)
  unet.py                   sıfırdan U-Net tanımı
kayitlar/
  data/                     MNIST (indirilir, repoda tutulmaz)
    intel/                  Intel Image Classification (Kaggle, repoda tutulmaz)
    water/                  Satellite Images of Water Bodies (Kaggle, repoda tutulmaz)
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
  gorev4/                   Görev 4 sonuçları
    gorseller/              deney başına 6 örneklik görsel karşılaştırma (png)
      baseline/             aşamaya göre ayrılmış: baseline, a1_loss,
      a1_loss/              a2_augmentation, a3_encoder, a4_mimari
      a2_augmentation/
      a3_encoder/
      a4_mimari/
    sonuclar_gorev4.csv     tüm deneylerin ortak tablosu
    su_oranlari.csv         maske başına su piksel oranı (önbellek)
    epoch_gecmisi_*.csv     deney başına epoch epoch seyir
    rapor_gorev4.html       ayrıntılı rapor
csvler/                     eski koşuların sonuçları (arşiv)
```



## Notlar

- Tüm rastgelelik kaynakları seed 27 ile sabitlenmiştir; her model aynı başlangıç ağırlıklarıyla kurulur.
- `csvler/` klasörü scriptlerin daha eski bir sürümüyle alınmış, konfigürasyon adları farklı olan tam koşuları içerir; güncel scriptler bu klasöre yazmaz.
- Metrikler tek geçişte confusion matrix'ten türetilir (`metricler_hesapla`), bu yüzden her epoch sonunda tam test seti ölçümü almanın maliyeti düşüktür.
- Görev 4'te `cv2.imread` kullanılmaz. Windows'ta yolu ANSI olarak işler ve Türkçe karakter içeren yolu açamaz — **sessizce** `None` **döner**. Proje yolu `nöron` içerdiği için tüm okumalar başarısız oluyordu. Yerine `goruntu_oku()` (`np.fromfile` + `cv2.imdecode`) kullanılır ve okunamama durumunda istisna fırlatır.
- Görev 4'te eğitim loader'ında `drop_last=True` zorunludur. Son batch tek örnek kalırsa DeepLabV3+'ın ASPP katmanı 1×1 uzamsal çıktı üretir ve BatchNorm `Expected more than 1 value per channel` hatasıyla düşer.
- `albumentations` her import'ta sürüm kontrolü yapıp uyarı basar; `NUM_WORKERS=4` olduğu için her işçi süreci ayrı basar. `ortak.py` bunu `NO_ALBUMENTATIONS_UPDATE=1` ile susturur (import'tan **önce** ayarlanmalı).

