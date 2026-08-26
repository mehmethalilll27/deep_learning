"""
GOREV 5 - ORTAK KATMAN (her iki gorev icin)

Bu dosya PASCAL VOC 2012 veri setinin okunmasi, bolunmesi ve sonuclarin
kaydedilmesi gibi HEM tespit HEM bolutleme tarafinin ihtiyac duydugu seyleri
tutar. Goreve ozgu kod ayri iki modulde:

    tespit.py      -> nesne tespiti (Faster R-CNN, mAP/IoU)
    bolutleme.py   -> semantik bolutleme (DeepLabV3, mIoU/Dice)

Gorev 4'te ortak katman tek dosyaydi; burada iki farkli problem var ve tek
dosyada toplamak 1500 satirlik bir modul demekti. Ortak olan sey veri setinin
kendisi, o yuzden bolme cizgisi buradan gecti.

VERI SETI: PASCAL VOC 2012 (Kaggle: gopalbhattrai/pascal-voc-2012-dataset)
  17.125 JPEG goruntu, 20 nesne kategorisi
  Detection : Annotations/*.xml  -> train 5.717 / val 5.823
  Segmentation: SegmentationClass/*.png -> train 1.464 / val 1.449
  Segmentation etiketleri detection'in ALT KUMESI; ayni goruntunun hem kutusu
  hem maskesi olabilir ama maskeli goruntu sayisi cok daha az.

TEST SETI NEDEN val'DEN BOLUNUYOR
  VOC 2012'nin resmi test setinin etiketleri yayinlanmaz (skor ancak VOC
  degerlendirme sunucusuna gonderilerek alinir). Elimizde etiketli iki kume
  var: train ve val. Gorev 3 ve 4'teki duzeni korumak icin resmi val listesi
  seed 27 ile ikiye bolunur:
      val  -> iyilestirme kararlari buna bakilarak verilir
      test -> her konfigurasyon icin yalnizca bir kez olculur
  Boylece test set leakage olmadan "karar seti / rapor seti" ayrimi korunur.
"""

import csv
import os
import random
import xml.etree.ElementTree as ET

import numpy as np
import torch
from PIL import Image

# 1) YOLLAR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJE_DIR = os.path.dirname(BASE_DIR)
KAYITLAR_DIR = os.path.join(PROJE_DIR, "kayitlar")
SONUC_DIR = os.path.join(KAYITLAR_DIR, "gorev5")
GORSEL_DIR = os.path.join(SONUC_DIR, "gorseller")
DATA_KOK = os.path.join(KAYITLAR_DIR, "data", "voc")

os.makedirs(SONUC_DIR, exist_ok=True)
os.makedirs(GORSEL_DIR, exist_ok=True)

TESPIT_SONUC = os.path.join(SONUC_DIR, "sonuclar_tespit.csv")
BOLUTLEME_SONUC = os.path.join(SONUC_DIR, "sonuclar_bolutleme.csv")
TESPIT_SINIF_SONUC = os.path.join(SONUC_DIR, "sinif_bazli_tespit.csv")
BOLUTLEME_SINIF_SONUC = os.path.join(SONUC_DIR, "sinif_bazli_bolutleme.csv")


def gorsel_klasoru(problem, asama):
    """Gorselleri once probleme, sonra asamaya gore alt klasore ayirir.

    Gorev 4'te tek seviye yetiyordu; burada iki problem AYNI asama adlarini
    kullaniyor ("Referans", "Baseline"), o yuzden ust seviyede problem adi
    gerekiyor — yoksa tespitin ve bolutlemenin referanslari ayni klasora
    duserdi. Asama adi konfigden turetilir ("A1 Transfer" -> "a1_transfer"),
    boylece yeni asama eklendiginde burasi degismez.
    """
    yol = os.path.join(GORSEL_DIR, problem, asama.lower().replace(" ", "_"))
    os.makedirs(yol, exist_ok=True)
    return yol


_voc_kok_onbellek = None


def voc_kok():
    """Etiketli VOC klasorunu arayarak bulur.

    Kaggle arsivi ic ice klasorler halinde aciliyor (VOC2012_train_val/
    VOC2012_train_val/...), surumden surume derinlik degisiyor. Sabit yol
    yazmak yerine klasor yapisindan taniyoruz.

    SegmentationClass ARANMASI ZORUNLU: arsivde ayrica VOC2012_test/ var ve
    onun da Annotations/ + JPEGImages/ klasorleri mevcut — ama etiketleri bos,
    ImageSets/Main altinda yalnizca test.txt duruyor. Alfabetik sirada "test"
    once geldigi icin iki klasorlu arama once ONU bulur ve tum kume listeleri
    sessizce bos doner.
    """
    global _voc_kok_onbellek
    if _voc_kok_onbellek is not None:
        return _voc_kok_onbellek

    if not os.path.isdir(DATA_KOK):
        raise RuntimeError(
            f"VOC klasoru yok: {DATA_KOK}\n"
            "  kaggle datasets download -d gopalbhattrai/pascal-voc-2012-dataset "
            f"-p {DATA_KOK} --unzip")

    gerekli = {"Annotations", "JPEGImages", "ImageSets", "SegmentationClass"}
    for kok, dizinler, _ in os.walk(DATA_KOK):
        if gerekli.issubset(set(dizinler)):
            _voc_kok_onbellek = kok
            return kok

    raise RuntimeError(
        f"{', '.join(sorted(gerekli))} klasorlerini birlikte iceren dizin "
        f"bulunamadi: {DATA_KOK}")


def jpeg_yolu(ad):
    return os.path.join(voc_kok(), "JPEGImages", f"{ad}.jpg")


def annot_yolu(ad):
    return os.path.join(voc_kok(), "Annotations", f"{ad}.xml")


def maske_yolu(ad):
    return os.path.join(voc_kok(), "SegmentationClass", f"{ad}.png")


# 2) SINIFLAR
# Sira VOC'un resmi sirasidir. Bolutlemede piksel degeri = bu listedeki
# indeks + 1 (0 arka plan, 255 "void" = sinir pikselleri, yok sayilir).
# Tespitte de etiket = indeks + 1 (0 arka plan, Faster R-CNN'in bekledigi duzen).
VOC_SINIFLAR = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
SINIF_SAYISI = len(VOC_SINIFLAR)                 # 20
SINIF_INDEKS = {ad: i + 1 for i, ad in enumerate(VOC_SINIFLAR)}
BOLUT_SINIF_SAYISI = SINIF_SAYISI + 1            # + arka plan = 21
VOID = 255

# COCO ile egitilmis bir modeli hic egitmeden VOC'ta olcebilmek icin gereken
# esleme. 20 VOC sinifinin hepsi COCO'da var, dordunun adi farkli.
VOC_COCO_ADI = {
    "aeroplane": "airplane",
    "motorbike": "motorcycle",
    "pottedplant": "potted plant",
    "sofa": "couch",
    "diningtable": "dining table",
    "tvmonitor": "tv",
}


# 3) AYARLAR (tum deneylerde sabit)
SEED = 27
CIHAZ = "GPU"
NUM_WORKERS = 4

# ImageNet istatistikleri. Hem Faster R-CNN'in ic transform'u hem DeepLabV3
# bu degerlerle egitilmis agirliklari kullaniyor.
NORM_ORTALAMA = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU bulunamadi! CUDA destekli PyTorch gerekli.")
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


def seed_ayarla(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 4) KUME LISTELERI
def kume_listesi(ad, alt="Main"):
    """ImageSets/<alt>/<ad>.txt icindeki goruntu adlarini dondurur.

    alt = "Main"          -> detection listeleri (train 5.717 / val 5.823)
    alt = "Segmentation"  -> segmentation listeleri (train 1.464 / val 1.449)
    """
    yol = os.path.join(voc_kok(), "ImageSets", alt, f"{ad}.txt")
    if not os.path.exists(yol):
        raise RuntimeError(f"Kume listesi yok: {yol}")
    with open(yol, encoding="utf-8") as f:
        return [s.strip().split()[0] for s in f if s.strip()]


def val_bol(adlar, seed=SEED):
    """Resmi val listesini val (karar) ve test (rapor) olarak ikiye boler.

    Bolme seed'e bagli ve deterministik; her script ayni bolmeyi elde eder.
    """
    idx = np.arange(len(adlar))
    np.random.default_rng(seed).shuffle(idx)
    orta = len(adlar) // 2
    val = sorted(adlar[i] for i in idx[:orta])
    test = sorted(adlar[i] for i in idx[orta:])
    return val, test


def alt_kume(adlar, n, seed=SEED):
    """Deterministik alt kume. n None ya da liste boyundan buyukse aynen dondurur.

    Tespit tarafinda egitim ve olcum kumeleri bilerek kucultuluyor: tam VOC
    detection egitimi (5.717 goruntu x 6 epoch) tek GPU'da deney basina ~1 saat
    surer, dort deney bir gunu bulurdu. Alt kume tum deneylerde AYNI oldugu icin
    karsilastirma gecerliligini korur; mutlak skorlar literaturun altinda kalir.
    """
    if n is None or n >= len(adlar):
        return list(adlar)
    idx = np.random.default_rng(seed).choice(len(adlar), size=n, replace=False)
    return sorted(adlar[i] for i in sorted(idx))


# 5) ANNOTATION OKUMA
def annot_oku(ad):
    """Bir VOC XML dosyasini okur.

    Doner: {"boxes": (N,4) float32, "labels": (N,) int64, "difficult": (N,) bool,
            "genislik": int, "yukseklik": int}

    'difficult' bayragi onemli: VOC'ta zor/belirsiz olarak isaretlenmis nesneler
    egitimde kullanilmaz ve degerlendirmede YOK SAYILIR (bir tahmin difficult bir
    nesneyi bulursa ne dogru ne yanlis sayilir). Bu bayragi gormezden gelmek
    mAP'i yapay olarak dusuren yaygin bir hatadir.
    """
    kok = ET.parse(annot_yolu(ad)).getroot()
    boyut = kok.find("size")
    genislik = int(boyut.find("width").text)
    yukseklik = int(boyut.find("height").text)

    kutular, etiketler, zorlar = [], [], []
    for nesne in kok.findall("object"):
        sinif = nesne.find("name").text.strip().lower()
        if sinif not in SINIF_INDEKS:
            continue
        zor_dugum = nesne.find("difficult")
        zor = zor_dugum is not None and zor_dugum.text.strip() == "1"

        kutu = nesne.find("bndbox")
        x1 = float(kutu.find("xmin").text)
        y1 = float(kutu.find("ymin").text)
        x2 = float(kutu.find("xmax").text)
        y2 = float(kutu.find("ymax").text)
        # Bozuk kutu (genislik veya yukseklik 0) modeli NaN'a dusuruyor;
        # VOC'ta birkac tane var, atiliyorlar.
        if x2 <= x1 or y2 <= y1:
            continue

        kutular.append([x1, y1, x2, y2])
        etiketler.append(SINIF_INDEKS[sinif])
        zorlar.append(zor)

    return {
        "boxes": np.array(kutular, dtype=np.float32).reshape(-1, 4),
        "labels": np.array(etiketler, dtype=np.int64),
        "difficult": np.array(zorlar, dtype=bool),
        "genislik": genislik,
        "yukseklik": yukseklik,
    }


def goruntu_oku(ad):
    """JPEG'i RGB PIL goruntusu olarak acar.

    Gorev 4'te cv2.imread Windows'ta Turkce karakterli yolu sessizce None
    donduruyordu; PIL yolu Unicode isler, ayni sorun burada yok.
    """
    return Image.open(jpeg_yolu(ad)).convert("RGB")


# 6) VOC RENK PALETI (gorsellestirme icin)
def voc_paleti():
    """VOC'un resmi sinif renk paleti (21 x 3, uint8).

    Maske PNG'leri bu paletle kaydedilmis; ayni renkleri kullanmak tahmin ile
    gercek maskeyi yan yana koyarken gozle karsilastirmayi mumkun kiliyor.
    """
    palet = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        r = g = b = 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palet[i] = [r, g, b]
    return palet


PALET = voc_paleti()


def maskeyi_renklendir(maske):
    """(H, W) sinif indeksi -> (H, W, 3) float RGB [0, 1]. 255 (void) beyaz."""
    renkli = PALET[np.clip(maske, 0, 255)].astype(np.float32) / 255.0
    renkli[maske == VOID] = 1.0
    return renkli


# 7) CSV KAYIT
def _mevcut_oku(dosya):
    if not os.path.exists(dosya):
        return []
    with open(dosya, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sonuc_ekle(yeni, dosya, alanlar, anahtar="deney"):
    """Ayni adli deneyi gunceller, digerlerini korur.

    Gorev 3 ve 4'teki duzenin aynisi: scriptler ayri ayri ve tekrar tekrar
    calistirilabilir, tablo tek parca kalir.
    """
    adlar = {s[anahtar] for s in yeni}
    kalan = [s for s in _mevcut_oku(dosya) if s.get(anahtar) not in adlar]
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=alanlar, extrasaction="ignore")
        w.writeheader()
        w.writerows(kalan + yeni)


def sinif_bazli_kaydet(satirlar, dosya, alanlar):
    """Sinif bazli tablo; ayni deneyin eski satirlarini siler, yenilerini yazar."""
    adlar = {s["deney"] for s in satirlar}
    kalan = [s for s in _mevcut_oku(dosya) if s.get("deney") not in adlar]
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=alanlar, extrasaction="ignore")
        w.writeheader()
        w.writerows(kalan + satirlar)


def gecmis_kaydet(gecmis, deney_adi):
    dosya = os.path.join(SONUC_DIR, f"epoch_gecmisi_{deney_adi}.csv")
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gecmis[0].keys()))
        w.writeheader()
        w.writerows(gecmis)
    return dosya


# 8) TABLO YAZDIRMA
def ozet_yazdir(sonuclar, baslik, sutunlar, siralama_alani):
    """sutunlar: [(baslik, alan_adi, genislik), ...]"""
    toplam = sum(g for _, _, g in sutunlar)
    print("=" * toplam)
    print(baslik)
    print("=" * toplam)
    print("".join(f"{b:<{g}}" if i == 0 else f"{b:>{g}}"
                  for i, (b, _, g) in enumerate(sutunlar)))
    print("-" * toplam)
    for s in sonuclar:
        print("".join(f"{str(s.get(a, '')):<{g}}" if i == 0
                      else f"{str(s.get(a, '')):>{g}}"
                      for i, (_, a, g) in enumerate(sutunlar)))
    print("-" * toplam)
    if sonuclar:
        en_iyi = max(sonuclar, key=lambda s: float(s[siralama_alani]))
        print(f"Validation'a gore en iyi: {en_iyi['deney']} "
              f"({siralama_alani} %{en_iyi[siralama_alani]})")


# 9) EGITIM YARDIMCILARI
def poly_lr(optimizer, taban_lr, adim, toplam_adim, guc=0.9):
    """Bolutlemede standart 'poly' learning rate cizelgesi.

    lr = taban_lr * (1 - adim/toplam)^0.9 — DeepLab makalesinin kullandigi
    cizelge. Step LR'e gore daha yumusak azalir, kisa egitimlerde farki gorunur.
    """
    lr = taban_lr * (1 - adim / max(1, toplam_adim)) ** guc
    for grup in optimizer.param_groups:
        grup["lr"] = lr * grup.get("lr_carpani", 1.0)
    return lr


def isinma_lr(optimizer, taban_lr, adim, isinma_adim):
    """Ilk N adimda lr'yi kademeli yukseltir.

    Faster R-CNN'de gerekli: SGD lr=0.005 ile ilk adimlarda RPN kayiplari
    patlayip loss NaN olabiliyor. torchvision'in referans egitim scripti de
    ayni isinmayi kullanir.
    """
    carpan = min(1.0, (adim + 1) / isinma_adim)
    for grup in optimizer.param_groups:
        grup["lr"] = taban_lr * carpan * grup.get("lr_carpani", 1.0)
    return taban_lr * carpan
