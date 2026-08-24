# Raporun Anlatımı — `rapor_gorev4.html`

Bu dosya, [kayitlar/gorev4/rapor_gorev4.html](kayitlar/gorev4/rapor_gorev4.html) raporunu sıfırdan anlatır: rapor nasıl kurgulanmış, her bölüm neden orada, hangi sayıya neden bakılıyor, hangi cümle neyi iddia ediyor.

---

## Önce: raporun ana fikri

Rapor bir "skorumuz şu kadar yükseldi" belgesi değil. Öyle başlıyor ama üçte birinden sonra **başka bir şeye** dönüşüyor:

> Ölçtüğüm 17 deneyin çoğu bir şey değiştirmiş *gibi görünüyor*. Ölçüm gürültüsünü ölçtüm; gürültüyü aşan tek bir tanesi var. Geri kalanına "kazandı" demeyeceğim.

Raporun bel kemiği bu. Bölümlerin sırası da bunu kurmak için tasarlanmış: önce metrik neden IoU (Bölüm 2), sonra gürültü eşiği kaç (Bölüm 3), *ondan sonra* deney sonuçları geliyor. Yani okuyucuya sonuçlar gösterilmeden önce **o sonuçları okuma kuralı** veriliyor.

---

## Raporun haritası

| # | Bölüm | Ne yapıyor | Neden orada |
| --- | --- | --- | --- |
| — | Yönetici özeti (mavi kutu) | 17 deney, +11,47 puan, 145,8 dk | Tek paragrafta cevap |
| 1 | Veri Seti ve Deney Kurulumu | Veri, bölme, sabitler | Sonuçların geçerlilik zemini |
| 2 | Neden Piksel Doğruluğu Yetmez | IoU / Dice tanımı | **Metrik kuralı** |
| 3 | Ölçüm Gürültüsü: 2,33 Puanlık Eşik | Tekrar koşuların farkı | **Yorumlama kuralı** |
| 4 | Adım 1 — Baseline | Sıfır noktası | Karşılaştırma referansı |
| 5 | Adım A1 — Loss | 4 loss | 1. kaldıraç |
| 6 | Adım A2 — Augmentation | 4 profil | 2. kaldıraç |
| 7 | Adım A3 — Pretrained Encoder | 4 encoder | 3. kaldıraç — **tek işe yarayan** |
| 8 | Adım A4 — Mimari | 4 decoder | 4. kaldıraç |
| 9 | Nihai Sonuç | Baseline vs nihai | Manşet |
| 10 | Hangi Bölgeler İyi/Kötü | 6 örnek üzerinden | Skorun *nerede* kazanıldığı |
| 11 | Tüm Deneyler | 17 satırlık ham tablo | Denetlenebilirlik |
| 12 | Bulgular | 10 madde | Çıkarımlar |
| 13 | Sınırlılıklar | 5 madde | Neyin kanıtlanmadığı |

### Rapordaki renk kodları

Raporu tarayıcıda açtığınızda renkler bilgi taşıyor:

- **Mavi kutu** → bilgi / bağlam
- **Turuncu kutu** → uyarı: burada bir tuzak var ya da "bu sonuç kanıtlanmadı"
- **Yeşil kutu** → bulgu: kanıtlanmış çıkarım
- Tablo satırlarında **yeşil** = kazanan, **kırmızı** = en kötü, **sarı** = dikkat

Turuncu kutular raporun en dürüst kısmı — çoğu "bu adım işe yaramadı" diyor.

---

## Bölüm 1 — Veri Seti ve Deney Kurulumu

### Neden bu veri seti

Görevin iki şartı vardı: veri seti **maske içersin** ve **daha önce çalışılmamış** olsun.

Rapor burada bir turuncu kutuyla Görev 3'ün veri setinin neden elenmiş olduğunu açıklıyor: Intel Image Classification'ın klasörleri `seg_train` / `seg_test` / `seg_pred` — segmentasyon veri seti sanılıyor. Kontrol edilince **24.335 dosyanın hepsi `.jpg` görüntü, tek maske yok**; "seg" öneki *segment* değil *Kaggle yarışma bölümü* anlamında.

Bu kutunun raporda olması önemli: "neden yeni veri seti aldın" sorusuna hazır cevap.

### Veri setinin künyesi

| Özellik | Değer |
| --- | --- |
| Görüntü–maske çifti | 2.841 |
| Görüntü boyutları | değişken (300 örnekte 298 farklı boyut) |
| Eğitim boyutu | 256 × 256 |
| **Ortalama su oranı** | **%32,89** |
| Su oranı aralığı | %2,00 – %100,00 |
| Tamamen su olan görüntü | 123 |
| Hiç su içermeyen görüntü | 0 |

**%32,89 sayısını aklınızda tutun.** Rapor burada onu ölçüp bırakıyor, ama A1 bölümünde geri dönüp Focal loss'un neden battığını bu sayıyla açıklıyor. Yani rapor bir sayıyı önce koyup sonra kullanıyor — "varsaymak yerine ölçmek" iddiasının kurulduğu yer burası.

### İki hazırlık zorunluluğu

**Maskeler eşiklenmeli.** Maskeler `.jpg` olarak saklanmış, JPEG kayıplı sıkıştırma yaptığı için **ikili değiller** — `water_body_1.jpg` maskesinde **88 farklı piksel değeri** var. Doğrudan hedef olarak kullanılsa model "0,37 oranında su" gibi anlamsız hedeflere öğrenmeye çalışırdı. Hepsi **127 eşiğiyle** ikiliye çevrildi.

**Maske nearest ile boyutlandırılmalı.** Görüntü bilinear, maske nearest. Bilinear kullanılsaydı kenarlarda ara değerler oluşur ve etiketler bozulurdu — **hata mesajı vermeden**, sadece skoru düşürerek. Aynı ilke augmentation'da da geçerli: görüntü dönerse maske de aynı rastgele parametrelerle dönmek zorunda.

### Raporun en öğretici turuncu kutusu: OpenCV hatası

Proje yolu `…\nöron\…` içerdiği için `cv2.imread` **tüm dosyalar için sessizce `None` döndürdü** — hata fırlatmadı. Sonuç zinciri:

1. 2.841 maskenin hepsi "su oranı %0" olarak hesaplandı
2. Bu değerler **önbelleğe yazıldı**
3. Katmanlı bölme bozuldu

Çözüm: dosyayı Python seviyesinde açıp (`np.fromfile`) baytları çözmek (`cv2.imdecode`), ve **okunamazsa istisna fırlatmak**.

Raporun bu kutudaki cümlesi tek başına akılda kalıcı: *"Sessiz başarısızlık, gürültülü başarısızlıktan çok daha tehlikelidir."* Hata verse 1 dakikada bulunurdu; sessizce yanlış sonuç verdiği için tüm bölme sessizce bozuldu.

### Katmanlı bölme — ve doğrulaması

Görev 3'te bölme sınıf etiketine göre katmanlanmıştı. Segmentasyonda sınıf etiketi yok; onun yerine **her görüntünün su oranı** sürekli bir değişken. Görüntüler su oranına göre 5 kovaya ayrıldı, her kova kendi içinde %70/15/15 bölündü:

| Su oranı kovası | Toplam | Train | Val | Test |
| --- | --- | --- | --- | --- |
| < %10 | 414 | 290 | 62 | 62 |
| %10 – %25 | 854 | 598 | 128 | 128 |
| %25 – %40 | 687 | 481 | 103 | 103 |
| %40 – %60 | 551 | 385 | 83 | 83 |
| > %60 | 335 | 235 | 50 | 50 |
| **Toplam** | **2.841** | **1.989** | **426** | **426** |

Rapor bununla yetinmiyor, **bölmenin işe yaradığını da kanıtlıyor**:

| Küme | Görüntü | Ortalama su oranı | Medyan |
| --- | --- | --- | --- |
| train | 1.989 | %32,90 | %27,84 |
| validation | 426 | %32,74 | %27,50 |
| test | 426 | %32,94 | %28,42 |

Üç kümenin ortalaması **0,20 puan içinde** örtüşüyor. "Bölme başarılı" cümlesi böyle destekleniyor — iddia değil, ölçüm.

### Sabitler ve gerekçeleri

Tabloda dikkat çeken şey: her sabitin yanında **neden o değer olduğu** yazıyor.

| Parametre | Değer | Gerekçe |
| --- | --- | --- |
| Epoch | 20 | süre / yakınsama dengesi |
| Batch | 16 | 256×256'da 8 GB VRAM sınırı |
| Optimizer | Adam | tek değişken kalsın diye sabit |
| Eşik | 0,50 | sigmoid çıkışını ikiliye çevirme |
| Seed | 27 | her deney başında yeniden kurulur |
| Normalizasyon | ImageNet ort./std | A3'te transfer learning verimli olsun diye **baştan** |

Son satır ileriye dönük bir karar: A3'te pretrained encoder kullanılacağı biliniyordu, o yüzden normalizasyon **baştan** ImageNet istatistikleriyle yapıldı. Sonradan değiştirilseydi A3 öncesi/sonrası karşılaştırılamazdı.

---

## Bölüm 2 — Neden Piksel Doğruluğu Yetmez: IoU ve Dice

Bu bölüm raporun **metrik kuralını** koyuyor.

Piksel doğruluğu her pikseli tek tek doğru/yanlış sayar. Bu veri setinde piksellerin **%67'si arka plan** — model hiç su tahmin etmese bile %67 doğruluk alır. Model gerçekten iyi olduğunda bile doğruluk büyük ölçüde arka planın kolay kısmını ölçer.

```
IoU  = kesişim / birleşim
Dice = 2 × kesişim / (tahmin alanı + gerçek alan)
```

İkisi de yalnızca **hedef bölgeye** bakar; arka planı doğru bilmekten puan gelmez.

**Dice her zaman IoU'dan büyüktür** (aynı örtüşmede kesişimi iki kez saydığı için). Bu yüzden iki çalışma karşılaştırılırken hangisinin raporlandığı mutlaka belirtilmeli — aksi halde daha iyi metrik seçen daha iyi görünür.

### Bölümün kilit tablosu

Rapor "accuracy yeterli değil" uyarısını **sayısallaştırıyor**. Nihai model baseline'a göre:

| Metrik | Kazanç |
| --- | --- |
| IoU | **+11,47** |
| Dice | +7,33 |
| Piksel doğruluğu | **+4,13** |

Sadece piksel doğruluğuna bakılsaydı **iyileştirmenin üçte ikisi görünmeyecekti.** Bu, raporun en çok tekrarladığı bulgu — Bölüm 9 ve Bulgular'da tekrar geliyor.

### IoU'nun iki hesabı

| Hesap biçimi | Nasıl | Ne ölçer | Baseline | Nihai |
| --- | --- | --- | --- | --- |
| **Veri seti geneli** | tüm kesişim ve birleşimler toplanıp bölünür | büyük su kütleleri ağırlıklı, daha kararlı | %71,28 | %82,75 |
| **Görüntü başına** | her görüntü için ayrı hesaplanıp ortalanır | küçük su kütleli görüntüler eşit ağırlıkta, daha sert | %68,47 | %79,25 |

Aradaki 3–4 puanlık fark tesadüf değil: görüntü başına hesap, küçük su kütlelerinde yapılan hataları cezalandırıyor. Literatürde ikisi de "IoU" adıyla raporlanır; hangisinin kullanıldığını belirtmemek yaygın bir karşılaştırma hatası. Rapor **ikisini de** kaydediyor.

> Bu ayrım A1'de işe yarıyor: Focal loss'un genel IoU'su %63,52 ama görüntü başına IoU'su %55,64. Aradaki 8 puanlık uçurum, Focal'in **küçük su kütlelerini tamamen ıskaladığını** gösteriyor — tek metrikle görülemezdi.

---

## Bölüm 3 — Ölçüm Gürültüsü: 2,33 Puanlık Eşik

**Raporun en kritik bölümü.** Buradan sonraki her tablo bu eşiğe göre okunuyor.

### Gürültü nasıl ölçüldü

Deney tasarımı gereği bazı konfigürasyonlar iki farklı script'te birebir tekrarlanıyor (her aşama bir öncekinin kazananını referans satırı olarak yeniden koşuyor). Bu tekrarlar **bedava bir gürültü ölçümü** sağladı:

| Birebir aynı konfigürasyon | Koşu A | Koşu B | Test IoU farkı | Val IoU farkı |
| --- | --- | --- | --- | --- |
| UNet + BCE + aug yok | `baseline_unet` | `loss_BCE` | 0,14 | 0,33 |
| UNet + effnet-b0 pretrained | `enc_efficientnet_b0_pretrained` | `mim_Unet` | 0,95 | 0,32 |
| UNet + BCE+Dice + aug yok | `loss_BCE_Dice` | `aug_yok` | **2,33** | **3,24** |

### Neden farklı çıkıyor

Kaynak `cudnn.benchmark = True` ayarı ve CUDA çekirdeklerinin **deterministik olmayan toplama sırası**. Her epoch'ta biriken küçük farklar 20 epoch sonunda birkaç puana ulaşabiliyor. Ayarı kapatmak eğitimi belirgin yavaşlatacağı için tercih edilmemiş; bunun yerine gürültü **ölçülüp raporlanmış**.

### Raporun temel kuralı

> Test IoU'da **2,33 puandan**, validation IoU'da **3,24 puandan** küçük hiçbir fark "iyileştirme" olarak yorumlanmamıştır.

Ve devamı, raporun en dürüst cümlesi:

> **Kazanan uydurulmamış, ölçüm karar veremediğinde karar verilemediği yazılmıştır.**

### Bu bölümün neden bu kadar önemli olduğu

Üç ölçüm **0,14 / 0,95 / 2,33** çıktı. Yani gürültü tek bir sayı değil, bir dağılım — ve en kötü hâli 2,33.

İlk çift görülüp "gürültü 0,14 puan" denseydi, **raporun neredeyse tüm sonuçları yanlış yorumlanacaktı**: A1'in 2,83 puanlık farkı "20 kat gürültü, kesin sonuç" sanılırdı, A2'nin 1,24'ü "gürültünün 9 katı, augmentation işe yaradı" olurdu. Tek ölçümle yetinmek, olmayan iyileştirmelere inanmanın en kolay yolu.

---

## Bölüm 4 — Adım 1: Baseline

Baseline'ın işlevi iyi olmak değil, **ölçüm sıfır noktası** olmak. Kasıtlı olarak sade: sıfırdan yazılmış U-Net, BCE loss, augmentation yok, scheduler yok.

| Bileşen | Yapı |
| --- | --- |
| Encoder | 4 seviye, kanallar 32 → 64 → 128 → 256 |
| Bottleneck | 512 kanal |
| Decoder | 4 seviye, `ConvTranspose2d` ile büyütme |
| Skip bağlantı | her seviyede, **havuzlamadan önce** saklanır |
| Blok | conv → BatchNorm → ReLU (×2) |
| **Parametre** | **7.763.041** |

### Sonuçlar

| Metrik | Değer | Metrik | Değer |
| --- | --- | --- | --- |
| Test IoU | **%71,28** | Precision | %89,82 |
| Test Dice | **%83,23** | Recall | **%77,54** |
| Görüntü başına IoU | %68,47 | Train IoU | %69,49 |
| Piksel doğruluğu | %89,71 | Overfit farkı | **+0,15** |
| En iyi epoch | 18 / 20 | Süre | 645,9 sn |

### İki teşhis — raporun sonraki bölümlerini bunlar belirliyor

**1) Overfit yok, tam tersi.** Train IoU (%69,49) test IoU'dan (%71,28) *düşük*. Görev 3'te baseline +16,87 puan overfit etmişti; burada +0,15. Sorun ezberleme değil, **yetersiz öğrenme**. En iyi epoch'un 18/20 olması da modelin hâlâ yükselirken durdurulduğunu gösteriyor.

Bu ölçüm **A2'nin sonucunu baştan haber veriyor**: kırılacak ezber yoksa augmentation'ın kıracak bir şeyi de yok. Rapor bunu A2 sonuçlarını göstermeden önce söylüyor — yani augmentation'ın başarısızlığı sürpriz değil, öngörülmüş.

**2) Model çekingen.** Precision %89,82 iken recall %77,54. "Su" dediğine güvenilebilir ama **gerçek suyun %22'sini kaçırıyor**.

Bu asimetri, sonraki adımlarda kazanılacak puanın nereden geleceğini belirliyor: nihai modelde precision 1,66, recall **12,12** puan artıyor. Yani rapor baseline'da bir zaaf teşhis edip, sonunda o zaafın kapandığını gösteriyor — hikâye kapanıyor.

### Görsel

`gorseller/baseline/baseline_unet.png` — test setinden su oranı %2,1'den %100'e sıralanmış altı örnek. Sütunlar: **görüntü | gerçek maske | tahmin | üst üste bindirme**. Bindirmede yeşil = doğru su, kırmızı = kaçırılan su, mavi = fazladan su.

**Kırmızının maviye baskın olması, "model çekingen" teşhisinin görsel karşılığı.** Sayı ile görsel aynı şeyi söylüyor.

---

## Bölüm 5 — Adım A1: Loss Fonksiyonu

Segmentasyonun sınıflandırmadan en çok ayrıldığı yer. Görev 1–3'te CrossEntropy dışında pratik seçenek yoktu; segmentasyonda loss seçimi ayrı bir tasarım kararı.

Model, epoch, seed, lr ve augmentation baseline ile aynı — **yalnızca loss değişiyor**.

| Loss | Val IoU | Test IoU | Test Dice | Görüntü-IoU | Precision | Recall | Overfit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **BCE+Dice** | **73,37** | **74,25** | **85,22** | **72,98** | 86,88 | 83,63 | +0,02 |
| BCE | 69,68 | 71,42 | 83,33 | 67,74 | 91,36 | **76,59** | +0,14 |
| Dice | 67,34 | 68,59 | 81,37 | 67,76 | 81,93 | 80,82 | −0,18 |
| Focal | 63,42 | 63,52 | 77,69 | **55,64** | 94,82 | **65,80** | −0,86 |

### Raporun okuma açısı: dört "cesaret ayarı"

Bu tabloyu okumanın püf noktası **precision/recall sütununa** bakmak. Rapor dört loss'u aynı modelin dört farklı cesaret ayarı gibi okuyor:

- **Focal en çekingen** (%94,82 / %65,80). Focal kolay örnekleri bastırıp zorlara odaklanmak için tasarlandı ve **ağır sınıf dengesizliği varsayar**. Bu veri setinde su ortalama %32,89 — dengesizlik ılımlı. Yanlış araç, en kötü sonuç. Görüntü başına IoU'nun %55,64'e düşmesi küçük su kütlelerini tamamen ıskaladığını gösteriyor.
- **BCE de çekingen** (%91,36 / %76,59). Her pikseli eşit sayıyor; arka plan çoğunlukta olduğu için model "kara de, güvende ol" davranışına kayıyor.
- **Dice cesur ama kaba** (%81,93 / %80,82). Doğrudan örtüşmeyi optimize ettiği için recall yükseliyor, ama piksel düzeyinde denetim vermediği için genel skor düşüyor.
- **BCE+Dice ikisini dengeliyor** (%86,88 / %83,63). BCE piksel disiplinini, Dice bölge bütünlüğünü sağlıyor.

Bu okuma tarzı önemli: ortalama skora bakmak loss'un *ne yaptığını* göstermiyor, **hata profiline** bakmak gösteriyor.

### Rapor burada kendi sonucunu sınırlıyor

Turuncu kutu, kazananı ilan edip geçmiyor; hangi kısmının kanıtlandığını ayırıyor:

- BCE+Dice ile BCE arasındaki toplam IoU farkı **+2,83 puan** — eşik 2,33. Yani **sınırda**.
- Sağlam olan iki bulgu:
  1. **Focal açık farkla en kötü** (−7,90 puan test IoU) ve gerekçesi ölçülmüş.
  2. **BCE+Dice'ın recall avantajı iki bağımsız koşuda da tutarlı**: BCE %76,59 / %77,54'e karşı BCE+Dice %83,63 / %84,07. Aralıklar hiç örtüşmüyor, ortalama fark +6,78 puan.

BCE+Dice bu iki gerekçeyle sabitlendi — toplam skoru en yüksek olduğu için değil, **etkisi tekrarlanabilir olduğu için**.

> **Raporu savunurken dikkat:** kutuda "Dice'ın BCE'ye göre dezavantajı (−2,83 test / −2,34 val) eşiğin altındadır" yazıyor. Bu **validation için** doğru (2,34 < 3,24 eşiği), test için değil (2,83 > 2,33 eşiği). Karar validation'a göre verildiği için sonuç değişmiyor, ama sorulursa "validation eşiğine göre" demek gerekiyor.

---

## Bölüm 6 — Adım A2: Augmentation

### Segmentasyondaki fark

**Görüntü dönerken maske de aynı rastgele parametrelerle dönmek zorunda.** `albumentations` bunu görüntü ve maskeyi tek çağrıya alarak otomatik yapıyor; elle yazılsaydı en sık hata burada olurdu.

### Augmentation seçimi veri setine bağlı

Rapordaki mavi kutu bunu üç örnekle kuruyor:

| Veri seti | Dikey çevirme | Neden |
| --- | --- | --- |
| **Uydu (bu görev)** | ✔ geçerli | Dikey çekilen görüntüde "yukarı" diye bir yön yok |
| Görev 3 manzara | ✘ yanlış | Baş aşağı dağ yoktur |
| MNIST | ✘ felaket | Aynalanmış 2 artık 2 değildir |

Augmentation listesi kopyala-yapıştır edilecek bir şey değil; veri setinin fiziğine bağlı.

### Sonuçlar

| Profil | İçerik | Val IoU | Test IoU | Precision | Recall | Overfit |
| --- | --- | --- | --- | --- | --- | --- |
| hafif | yatay/dikey çevirme + 90° döndürme | 70,35 | 72,07 | 86,33 | 81,36 | −0,61 |
| yok | — | 70,13 | 71,92 | 83,26 | 84,07 | +0,70 |
| orta | hafif + kaydırma/ölçekleme/döndürme + parlaklık-kontrast | 69,87 | 71,52 | 84,52 | 82,30 | −0,32 |
| guclu | orta + elastik deformasyon + ızgara bozulması | 69,27 | 70,83 | 86,84 | 79,35 | −0,53 |

**Val aralığı 1,08 puan (eşik 3,24), test aralığı 1,24 puan (eşik 2,33).** Her iki ölçüde de eşiğin çok altında. Bu tablo hiçbir profilin diğerinden iyi olduğunu göstermiyor — **ölçüm karar veremedi.**

### Neden karar veremedi

Şaşırtıcı değil ve mekanizması baştan belli: augmentation'ın klasik işi **ezberi kırmak**. Bu modelde kırılacak ezber yoktu — baseline'da overfit +0,15, BCE+Dice'ta +0,02.

Görev 3'te augmentation 16,87 puanlık overfit'i 8 puana indirmişti çünkü orada **gerçek bir ezber vardı**. Burada var olmayan hastalığa ilaç verildi.

### Seçim skora değil ilkeye dayandırıldı

`hafif` seçildi, çünkü:
- Çevirme ve 90° döndürme uydu görüntüsünde **fiziksel olarak geçerli**
- Maliyeti sıfıra yakın (647,8 sn / 642,6 sn)
- Sonraki adımda model gerçekten overfit etmeye başlarsa **hazır**

`guclu` profilin hem validation hem test'te sonuncu olması, elastik deformasyonun su kıyılarını fiziksel olarak imkânsız şekillere büktüğü hipotezini destekliyor — ama rapor bunu iddia etmiyor, "tek koşuyla iddia edilemez" diyor.

### Bölümün en değerli kısmı: metodolojik itiraf

> **Bu adım yanlış anda yapıldı.** A3'te pretrained encoder gelince overfit farkı +0,02'den **+6,81**'e çıktı; yani ezber *sonradan* ortaya çıktı. "Her adımı sırayla dene" yaklaşımının kör noktası budur: bir adımın etkisi kendisinden *sonraki* adıma bağlı olabilir. Doğru sıralama, augmentation taramasını pretrained encoder sabitlendikten sonra yapmak olurdu.

Rapor kendi tasarım hatasını buluyor, açıklıyor ve doğrusunu söylüyor. Sınırlılıklar bölümünde tekrar geçiyor.

---

## Bölüm 7 — Adım A3: Pretrained Encoder

Başlığın yanında yeşil bir not var: **"(tek ölçülebilir iyileştirme)"**. Rapor okuyucuyu buraya yönlendiriyor.

### Fikir

U-Net'in encoder'ı aslında bir **sınıflandırma ağından farksız** — görüntüyü küçültüp özellik çıkarıyor. O hâlde ImageNet'te eğitilmiş bir ağ doğrudan encoder olarak kullanılabilir; decoder rastgele başlar ve sıfırdan öğrenir.

### Tablonun ilk satırı kasıtlı

`enc_resnet34_scratch` — **aynı mimari** (Unet + resnet34), tek farkı başlangıç ağırlıklarının rastgele olması. Bu satır olmasaydı "kazandıran pretrained mi, yoksa ResNet mimarisi mi" ayrılamazdı. Görev 3'teki ResNet18 scratch/donuk/finetune üçlüsünün aynı mantığı.

| Encoder | Pretrained | Parametre | Val IoU | Test IoU | Test Dice | Recall | Overfit | Süre |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **efficientnet-b0** | ✔ | 6.251.469 | **80,23** | **81,80** | **89,99** | 88,41 | +6,81 | 361,8 sn |
| resnet34 | ✔ | 24.436.369 | 79,79 | 81,55 | 89,84 | 90,18 | +5,57 | 378,4 sn |
| mobilenet_v2 | ✔ | 6.628.945 | 77,86 | 80,22 | 89,02 | 88,12 | +6,78 | **316,2 sn** |
| resnet34 | ✘ | 24.436.369 | **68,84** | **69,94** | 82,31 | 82,95 | −0,81 | 399,4 sn |

### Kontrollü karşılaştırma — raporun manşeti

Aynı mimari, tek fark başlangıç ağırlıkları:

> **%69,94 → %81,55 = +11,61 puan IoU**

Gürültü eşiği 2,33 puan; bu fark onun **beş katı**. Çalışmadaki dört kaldıraçtan **ölçülebilir etki üreten tek kaldıraç bu.**

### İkinci bulgu: sıfırdan resnet34 başarısız oldu

24,4 milyon parametreyle %69,94 aldı. El yazması **7,8 milyonluk** U-Net aynı koşulda %72,07 almıştı.

**3,1 kat parametre, daha düşük skor.** Görev 2'nin "büyük model her zaman kazanmıyor" bulgusunun çok daha net versiyonu: pretrained ağırlıklar olmadan büyük mimari, yalnızca **daha zor optimize edilen** bir modeldir.

### Üçüncü bulgu: yakınsama hızı

| Deney | Epoch 1 | Epoch 5 | Epoch 10 | Epoch 20 | %70'e ulaşma | %78'e ulaşma |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_unet | 54,72 | 62,39 | 65,66 | 69,35 | 18. epoch | ulaşamadı |
| aug_hafif (sıfırdan) | 55,27 | 60,00 | 65,48 | 70,35 | 18. epoch | ulaşamadı |
| enc_resnet34_scratch | 47,29 | 63,34 | 68,48 | 68,84 | 15. epoch | ulaşamadı |
| **enc_efficientnet_b0_pretrained** | **68,99** | 77,27 | 78,48 | 80,23 | **2. epoch** | **7. epoch** |

Pretrained model **1. epoch sonunda %68,99** aldı — sıfırdan eğitilen modellerin **20 epoch sonunda** ulaştığı seviye. Sıfırdan modeller hiçbir epoch'ta %78'e ulaşamadı.

Transfer learning yalnızca daha yüksek tavan değil, **çok daha hızlı yakınsama** da sağlıyor. Bu tablo raporun en çarpıcı görselleştirmesi: aynı satırda 20 epoch'luk bir emek ile 1 epoch'luk bir emek eşitleniyor.

### Seçim: efficientnet-b0

resnet34 ile arasındaki fark (test 0,25 / val 0,44 puan) **gürültünün içinde** — skor karar veremiyor. Aynı skoru **3,9 kat az parametreyle** (6,25M / 24,44M) ve daha kısa sürede vermesi kararı belirledi.

> *Skor eşitken maliyet karar verir.*

---

## Bölüm 8 — Adım A4: Segmentasyon Mimarisi

Loss, augmentation ve encoder sabitlendi; son değişken **decoder mimarisi**. Dördü de aynı efficientnet-b0 encoder'ını kullanıyor.

| Mimari | Fikir | Parametre | Val IoU | Test IoU | Görüntü-IoU | Süre |
| --- | --- | --- | --- | --- | --- | --- |
| **Unet** | klasik skip bağlantı | 6.251.469 | **79,91** | 82,75 | 79,25 | 375,9 sn |
| UnetPlusPlus | yoğunlaştırılmış skip, ara evrişimler | 6.569.581 | 78,69 | **82,92** | 79,26 | 542,7 sn |
| FPN | özellik piramidi, her ölçekten ayrı tahmin | 5.759.485 | 79,58 | 81,33 | 77,38 | **285,5 sn** |
| DeepLabV3Plus | atrous konvolüsyon, skip yerine geniş bağlam | 4.907.469 | 78,59 | 80,38 | **76,98** | 301,7 sn |

**Val IoU aralığı 1,32 puan; eşik 3,24.** Dört mimari istatistiksel olarak eşit. Decoder seçimi bu veri setinde ölçülebilir bir fark yaratmadı.

### Öngörü tutmadı — ve nedeni anlaşıldı

Çalışma öncesi tahmin: su kütleleri büyük ve bütünlüklü olduğu için DeepLabV3+'ın atrous konvolüsyonu ve FPN'in çok ölçekli yapısı avantajlı olacak.

**İkisi de sonuncu oldu.**

Gözden kaçan nokta: **DeepLabV3+'ın ince skip bağlantısı yok** — sonda 4× tek seferde büyütüyor, bu yüzden sınırlar kaba kalıyor. Bu veri setinde ise sınır hassasiyeti kritik (Bölüm 10'daki kıyı halkası örneği). Görüntü başına IoU'da en düşük olması (%76,98) tam bunu gösteriyor.

> Öngörü, ölçümün yerini tutmuyor.

### Nihai seçim: Unet — ve neden UnetPlusPlus değil

UnetPlusPlus test IoU'da **0,17 puan önde**, ama validation'da **1,22 puan geride**.

Seçimi test skoruna bakarak yapmak, Görev 3'ten beri özellikle kaçınılan **test set leakage** olurdu. Üstelik Unet **%44 daha hızlı** (375,9 sn / 542,7 sn).

Bu, raporun metodoloji tutarlılığının en görünür yeri: daha yüksek test skoru elin altındayken, kurala uyup almıyor.

---

## Bölüm 9 — Nihai Sonuç

```
Unet + efficientnet-b0 (ImageNet pretrained) + BCE+Dice loss + hafif augmentation
6.251.469 parametre | lr 0,0001 | 20 epoch | 375,9 sn
```

| Metrik | Baseline | Nihai | Kazanç |
| --- | --- | --- | --- |
| **Test IoU** | %71,28 | **%82,75** | **+11,47** |
| **Test Dice** | %83,23 | **%90,56** | **+7,33** |
| Görüntü başına IoU | %68,47 | %79,25 | +10,78 |
| Görüntü başına Dice | %78,51 | %87,04 | +8,53 |
| **Recall** | %77,54 | **%89,66** | **+12,12** |
| Precision | %89,82 | %91,48 | +1,66 |
| Piksel doğruluğu | %89,71 | %93,84 | +4,13 |

### Bu tablo iki şeyi aynı anda kanıtlıyor

**1) Kazancın kaynağı recall.** Precision 1,66 puan artarken recall **12,12** puan arttı. Model **kaçırdığı suyu bulmayı** öğrendi, su uydurmayı değil. Baseline'da teşhis edilen zaaf (çekingenlik) doğrudan kapandı.

**2) Piksel doğruluğu iyileştirmenin üçte ikisini gizliyor.** Aynı iyileştirme IoU'da 11,47, piksel doğruluğunda yalnızca **4,13** puan. Sadece doğruluk raporlansaydı iyileşmenin çoğu görünmezdi.

Yani rapor Bölüm 2'de koyduğu metrik kuralını, son tabloda **kendi verisiyle** doğruluyor. Baştaki iddia ile sondaki kanıt aynı sayfada kapanıyor.

---

## Bölüm 10 — Hangi Bölgeler İyi, Hangileri Kötü Segmentleniyor

Görevin ayrıca istediği bir madde. Skorun *ne kadar* yükseldiğini değil, **nerede** yükseldiğini gösteriyor.

Değerlendirme test setinden su oranı en düşükten en yükseğe sıralanan **altı sabit örnek** üzerinden. Aynı indeksler her deneyde kullanıldığı için deneyler arası karşılaştırma anlamlı — farklı görüntülere bakıp yanılma riski yok.

| Örnek | Gerçek su | Zorluk tipi | Baseline | A3 | Nihai | Değişim |
| --- | --- | --- | --- | --- | --- | --- |
| water_body_7891 | %2,1 | çok küçük, dağınık su lekeleri | %38,2 | %41,7 | **%48,2** | +10,0 |
| water_body_8672 | %12,9 | ince, dallanan nehir kolu | **%36,3** | %74,3 | %71,3 | **+35,0** |
| water_body_811 | %22,4 | kompakt göl, bulanık kıyı | %67,0 | %69,5 | %78,2 | +11,2 |
| water_body_890 | %34,5 | büyük, tek parça, kontrastlı | %87,1 | %89,5 | %91,0 | +3,9 |
| water_body_1093 | %47,9 | sığ kıyı halkası, kademeli sınır | %67,6 | %83,8 | %85,4 | **+17,8** |
| water_body_7868 | %100,0 | tamamı su (sığ lagün) | %62,0 | %88,4 | %99,9 | **+37,9** |

**A3 sütunu ayrıca konmuş** — kazancın büyük kısmının orada gerçekleştiği satır satır görülüyor. Örneğin nehir kolunda 36,3 → 74,3 sıçraması tamamen A3'te oluyor.

### İyi segmentlenen bölgeler

- **Büyük, tek parça, kontrastlı su kütleleri.** Baseline bile %87 alıyordu, nihai model %91. Bu kategori **zaten çözülmüştü**; iyileştirmeye yer azdı.
- **Homojen sahneler.** Tamamı su olan görüntüde nihai model %99,9. Baseline'ın %62'de kalması, bu tür sahneleri *"fazla iyi görünüyor, olamaz"* diye reddetmesindendi — çekingenliğin en uç örneği.

### İyileşen ama hâlâ zayıf bölgeler

- **İnce, dallanan nehir kolları.** En büyük kazanç burada: %36,3 → %71,3 (**+35,0 puan**). Baseline yalnızca nehrin en geniş yerindeki lekeyi buluyordu; nihai model kolları da izliyor. Yine de dar uçlar kaybediliyor.
- **Sığ kıyı halkaları ve kademeli sınırlar.** %67,6 → %85,4. Suyun karaya kademeli geçtiği bantlar maskede **keskin sınır** olarak etiketlenmiş; model ise geçişi görüyor. Rapor burada dürüst bir ayrım yapıyor: bu **kısmen etiketin kendisindeki belirsizlik**, tamamen modelin hatası değil.

### Çözülmeyen bölge

**Çok küçük ve dağınık su lekeleri (su oranı < %5) hâlâ bulunamıyor.** water_body_7891'de gerçek su %2,1 iken nihai model %1,3 tahmin ediyor; IoU %48,2. **Dört iyileştirme adımının hiçbiri bu kategoriyi çözmedi.**

En olası neden **çözünürlük**: görüntü 256×256'ya küçültüldüğünde o lekeler birkaç piksele iniyor ve encoder'ın ilk indirgeme adımlarında tamamen kayboluyor. Bu, mimari değişikliğiyle değil **daha yüksek girdi çözünürlüğüyle** ele alınabilecek bir sınırlılık — ve çalışmanın kapsamı dışında bırakılmış.

### Nihai modelin görseli

`gorseller/a4_mimari/mim_Unet.png` — baseline ile **aynı altı örnek**. Baseline görselindeki kırmızı baskınlığının büyük ölçüde kaybolduğu, özellikle 2. satırdaki nehir kolunun ve 5. satırdaki kıyı halkasının artık yakalandığı görülüyor. 1. satırdaki küçük su lekeleri hâlâ bulunamıyor.

İki görseli yan yana açmak, tablodaki +12,12 recall kazancının ne demek olduğunu tek bakışta gösteriyor.

---

## Bölüm 11 — Tüm Deneyler

17 satırlık ham tablo: her deney için parametre, val/test IoU, Dice, görüntü-IoU, piksel accuracy, precision, recall, overfit, süre.

Bu tablonun işlevi **denetlenebilirlik**. Rapordaki her iddia bu tablodan doğrulanabilir; okuyucu yalnızca yazarın seçtiği sayılara bakmak zorunda değil. Toplam: 8.751 saniye = **145,8 dakika** GPU süresi.

Tabloyu okurken işe yarayan üç dikey tarama:

| Bakış | Ne görülüyor |
| --- | --- |
| **Overfit sütunu** | İlk 9 satır ~0 civarı, son 8 satır +5,57 … +6,94. Ezberin **A3'te başladığı** tek bakışta görünüyor |
| **Parametre sütunu** | 24,4M olan iki satırdan biri en kötülerden (scratch), biri en iyilerden (pretrained). Boyut değil, **ağırlık** belirliyor |
| **Süre sütunu** | Sıfırdan U-Net'ler ~640 sn, pretrained encoder'lar ~360 sn. Hazır encoder hem daha iyi hem **daha hızlı** |

Ham veri: `sonuclar_gorev4.csv`, epoch bazlı geçmişler `epoch_gecmisi_*.csv`, görseller `gorseller/`.

---

## Bölüm 12 — Bulgular (10 madde)

| # | Bulgu | Dayanak |
| --- | --- | --- |
| 1 | **Dört kaldıraçtan yalnızca biri ölçülebilir etki üretti** | Pretrained encoder +11,61 (gürültünün 5 katı); loss/aug/mimari eşik içinde |
| 2 | Piksel doğruluğu iyileştirmenin üçte ikisini gizler | IoU +11,47 / Dice +7,33 / acc +4,13 |
| 3 | Baseline'ın hastalığı overfit değil, **yetersiz öğrenme**ydi | Train IoU %69,49 < test IoU %71,28 |
| 4 | Augmentation **yanlış anda** test edildi | A2'de overfit +0,02; A3'te +6,81 |
| 5 | Loss seçiminin etkisi **asimetrik** | Toplam IoU sınırda, recall farkı iki koşuda da sağlam |
| 6 | Focal veri setine uymadı, **nedeni önceden ölçülmüştü** | Su oranı %32,89 — ağır dengesizlik yok |
| 7 | Pretrained olmadan büyük mimari **dezavantaj** | resnet34 24,4M %69,94 vs U-Net 7,8M %72,07 |
| 8 | Transfer learning tavanı **ve hızı** değiştirir | 1. epoch %68,99 = sıfırdan modelin 20. epoch'u |
| 9 | Skor eşitken **maliyet** karar verir | 0,25 puan anlamsız, 3,9 kat parametre anlamlı |
| 10 | Gürültü **ölçülmeli**, varsayılmamalı | Üç tekrar: 0,14 / 0,95 / 2,33 |

**1. madde raporun tezi.** Rapor kendisi de bunu söylüyor: *"Bu, çalışmanın en önemli ve en dürüst sonucudur — dört adımın dördüne de 'kazanan' atamak mümkündü, ancak ölçüm bunu desteklemiyordu."*

Bir "skor yükseltme" raporunun kendi dört adımından üçünün ölçülemediğini yazması alışıldık değil. Raporun asıl değeri de burada.

---

## Bölüm 13 — Sınırlılıklar (5 madde)

Rapor neyi kanıtlamadığını da yazıyor. Sunumda "peki ya şu?" sorularının çoğu buradan geliyor, o yüzden bu beş madde ezberlenmeye değer.

**1) Her deney tek seed ile koşuldu.** Gürültü tabanı üç tekrar çiftinden kestirilmiş olsa da, **adım içi sıralamalar** (örneğin dört mimari arasındaki sıra) çoklu seed ile doğrulanmadı.
→ *Savunma:* manşet iddia (+11,47) gürültünün beş katı olduğu için bundan etkilenmiyor; adım içi sıralamalar zaten "karar verilemedi" olarak raporlandı.

**2) Epoch bütçesi bağlayıcıydı.** Birçok deneyde en iyi validation skoru 18–20. epoch'ta gerçekleşti — modeller **hâlâ yükselirken durduruldu**.
→ *Savunma:* daha uzun eğitim mutlak skorları yükseltebilir, ama **adımlar arası karşılaştırma etkilenmez** çünkü bütçe tüm deneylerde aynı.

**3) Girdi çözünürlüğü 256×256 ile sabit.** Çözülmeyen tek bölge kategorisi (< %5 su oranı) muhtemelen bu sınırdan kaynaklanıyor ve **test edilmedi**.

**4) Maske etiketlerinde belirsizlik var.** Sığ kıyı bantları maskede keskin sınırla etiketlenmiş; model kademeli geçişi görüyor. Bu bölgelerdeki hatanın ne kadarının model, ne kadarının etiket kaynaklı olduğu **ayrıştırılmadı**.

**5) Adım sırası sonucu etkiledi.** Augmentation taraması pretrained encoder'dan önce yapıldığı için etkisi ölçülemedi. Doğru sıralama **A3 → A2** olurdu; bu, tasarımın **kabul edilmiş bir zaafı**.

---

## Raporu tek bakışta okumak isteyene

Sırayla üç şeye bakın, gerisi detay:

1. **Bölüm 3'teki gürültü tablosu** — 2,33 puanlık eşik. Bu sayı olmadan hiçbir tablo doğru okunamaz.
2. **Bölüm 7'deki kontrollü karşılaştırma** — %69,94 → %81,55. Tüm çalışmanın tek kanıtlanmış kazancı.
3. **Bölüm 9'daki son tablo** — recall +12,12 vs precision +1,66. Kazancın karakteri.

Geri kalan her şey bu üçünü kuruyor ya da sınırlıyor.

---

## Raporun asıl söylediği

Rapor iki katmanlı. Yüzeyde: *"baseline %71,28'den %82,75'e çıktı, +11,47 puan."*

Altında: *"bu 11,47 puanın hemen hepsi tek bir adımdan geldi. Diğer üç adım işe yaramış gibi göründü ama gürültüyü ölçünce göründüğü kadar olmadıkları anlaşıldı. İkisini birbirinden ayıran şey, gürültüyü ölçmüş olmak."*

Bir "skoru yükselttim" raporunun kendi adımlarının dörtte üçünü ölçülemez ilan etmesi, raporu daha zayıf değil **daha güvenilir** yapıyor. Kalan tek iddia (+11,61 puan, gürültünün 5 katı) artık gerçekten duruyor.

---

**İlgili dosyalar:**

- Rapor: [kayitlar/gorev4/rapor_gorev4.html](kayitlar/gorev4/rapor_gorev4.html)
- Ham sonuçlar: [kayitlar/gorev4/sonuclar_gorev4.csv](kayitlar/gorev4/sonuclar_gorev4.csv)
- Epoch geçmişleri: [kayitlar/gorev4/](kayitlar/gorev4/) → `epoch_gecmisi_*.csv`
- Görseller: [kayitlar/gorev4/gorseller/](kayitlar/gorev4/gorseller/)
- Kod: [gorev4/](gorev4/)
