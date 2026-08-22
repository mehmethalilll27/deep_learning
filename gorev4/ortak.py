"""
GOREV 4 - ORTAK KATMAN

Gorev 2 ve 3'te her script kendi icinde calisiyordu ve yardimci fonksiyonlar
kopyalaniyordu. Segmentasyonda yardimci kod cok daha buyuk (veri okuma,
maske esikleme, IoU/Dice, gorsel uretme) ve bes scripte kopyalamak hem
okunmaz hem de bir hata bulundugunda bes yerde duzeltmek gerekirdi.

Bu yuzden Gorev 4'te ortak kod tek dosyada. Egitim scriptleri bunu import eder
ve yalnizca kendi degistirdikleri seyi tanimlar.

Veri seti: Satellite Images of Water Bodies (Kaggle)
  2.841 goruntu + maske cifti, isimler birebir eslesiyor
  Goruntu boyutlari degisken (300 ornekte 298 farkli boyut)
  Maskeler JPEG -> ikili degil, esiklenmesi gerekiyor
"""

import csv
import os
import random

# albumentations her import'ta surum kontrolu yapip uyari basiyor. NUM_WORKERS=4
# oldugu icin her isci sureci ayri ayri basiyor, cikti kirleniyor. Bu satir
# albumentations import'undan ONCE gelmeli.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

# 1) YOLLAR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJE_DIR = os.path.dirname(BASE_DIR)
KAYITLAR_DIR = os.path.join(PROJE_DIR, "kayitlar")
SONUC_DIR = os.path.join(KAYITLAR_DIR, "gorev4")
GORSEL_DIR = os.path.join(SONUC_DIR, "gorseller")
DATA_DIR = os.path.join(KAYITLAR_DIR, "data", "water", "Water Bodies Dataset")

IMG_DIR = os.path.join(DATA_DIR, "Images")
MSK_DIR = os.path.join(DATA_DIR, "Masks")

os.makedirs(SONUC_DIR, exist_ok=True)
os.makedirs(GORSEL_DIR, exist_ok=True)

SONUC_DOSYASI = os.path.join(SONUC_DIR, "sonuclar_gorev4.csv")
ORAN_ONBELLEK = os.path.join(SONUC_DIR, "su_oranlari.csv")


# 2) AYARLAR (tum deneylerde sabit)
SEED = 27
EPOCHS = 20
BATCH_SIZE = 16
LEARNING_RATE = 0.001
IMG_SIZE = 256
NUM_WORKERS = 4
CIHAZ = "GPU"
ESIK = 0.5              # sigmoid ciktisini ikiliye cevirme esigi

VAL_ORAN = 0.15
TEST_ORAN = 0.15

# Pretrained encoder'lar bu degerlerle egitildi; bastan kullanirsak A3'te
# transfer learning verimli olur (Gorev 3'teki ayni gerekce).
NORM_ORTALAMA = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

ALAN_ADLARI = [
    "deney", "asama", "model", "encoder", "pretrained", "loss", "augmentation",
    "seed", "epochs", "img_size", "batch", "lr", "parametre",
    "val_iou", "val_dice", "test_iou", "test_dice",
    "test_iou_goruntu", "test_dice_goruntu",
    "test_piksel_acc", "test_precision", "test_recall",
    "train_iou", "overfit_farki", "test_loss",
    "en_iyi_val_iou", "en_iyi_epoch", "sure_sn", "cihaz", "notlar",
]


# 3) CIHAZ
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


# 4) VERI
def goruntu_oku(yol, gri=False):
    """cv2.imread yerine kullanilir.

    NEDEN: Windows'ta cv2.imread yolu ANSI olarak isler ve Turkce karakter
    iceren yollari acamaz — sessizce None doner, hata firlatmaz. Proje yolu
    "...\\nöron\\..." icerdigi icin tum okumalar basarisiz oluyordu.

    np.fromfile Python seviyesinde dosya acar (Unicode sorunsuz), cv2.imdecode
    ise bellekteki baytlari cozer. Islevsel olarak imread ile ayni.
    """
    baytlar = np.fromfile(yol, dtype=np.uint8)
    bayrak = cv2.IMREAD_GRAYSCALE if gri else cv2.IMREAD_COLOR
    goruntu = cv2.imdecode(baytlar, bayrak)
    if goruntu is None:
        raise RuntimeError(f"Goruntu okunamadi: {yol}")
    return goruntu


def dosya_listesi():
    """Images/ ve Masks/ altindaki isimler birebir ayni; ortak olanlari dondurur."""
    imgs = set(os.listdir(IMG_DIR))
    msks = set(os.listdir(MSK_DIR))
    ortak = sorted(imgs & msks)
    if not ortak:
        raise RuntimeError(f"Eslesen dosya yok. {IMG_DIR} ve {MSK_DIR} kontrol edin.")
    return ortak


def su_oranlari_hesapla(adlar):
    """Her maskedeki su piksel oranini olcer. Bolme islemi buna gore katmanlanir.

    2.841 maskeyi okumak birkac dakika suruyor, bu yuzden sonuc CSV'ye
    onbelleklenir; sonraki kosular dosyadan okur.
    """
    if os.path.exists(ORAN_ONBELLEK):
        with open(ORAN_ONBELLEK, newline="", encoding="utf-8") as f:
            kayit = {s["dosya"]: float(s["su_orani"]) for s in csv.DictReader(f)}
        if all(ad in kayit for ad in adlar):
            return np.array([kayit[ad] for ad in adlar])

    print("  Su oranlari hesaplaniyor (ilk kosuda bir kez)...")
    oranlar = []
    for ad in adlar:
        m = goruntu_oku(os.path.join(MSK_DIR, ad), gri=True)
        oranlar.append(float((m > 127).mean()))

    with open(ORAN_ONBELLEK, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dosya", "su_orani"])
        w.writerows(zip(adlar, oranlar))
    print(f"  Onbelleklendi: {ORAN_ONBELLEK}")
    return np.array(oranlar)


def katmanli_bol(oranlar, seed=SEED):
    """Su oranina gore 5 kovaya ayirip her kovayi ayni oranlarda boler.

    Duz rastgele bolmede sans eseri az sulu goruntulerin cogu test'e dusebilir;
    o zaman skor performansi degil bolmenin sansini olcer. Gorev 3'teki
    katmanli bolmenin surekli degisken icin uyarlanmis hali.
    """
    kovalar = np.digitize(oranlar, [0.10, 0.25, 0.40, 0.60])
    rng = np.random.default_rng(seed)

    train_idx, val_idx, test_idx = [], [], []
    for kova in np.unique(kovalar):
        idx = np.where(kovalar == kova)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_val = int(round(n * VAL_ORAN))
        n_test = int(round(n * TEST_ORAN))
        val_idx.extend(idx[:n_val].tolist())
        test_idx.extend(idx[n_val:n_val + n_test].tolist())
        train_idx.extend(idx[n_val + n_test:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


class SuVeriSeti(Dataset):
    """Goruntu ve maskeyi birlikte okur, ayni donusumden gecirir.

    Maske isleme iki adim:
      1) Gri tonlamaya cevir  - maskeler RGB ama uc kanal birebir ayni
      2) 127'de esikle        - maskeler JPEG oldugu icin ikili degil;
                                orn. water_body_1.jpg'de 88 farkli deger var
                                (0 ve 255 baskin, gerisi kenar artefakti)
    """

    def __init__(self, adlar, indeksler, transform):
        self.adlar = [adlar[i] for i in indeksler]
        self.transform = transform

    def __len__(self):
        return len(self.adlar)

    def __getitem__(self, i):
        ad = self.adlar[i]
        img = cv2.cvtColor(goruntu_oku(os.path.join(IMG_DIR, ad)), cv2.COLOR_BGR2RGB)
        msk = (goruntu_oku(os.path.join(MSK_DIR, ad), gri=True) > 127).astype(np.float32)

        cikti = self.transform(image=img, mask=msk)
        # maske (H, W) geliyor -> (1, H, W)
        return cikti["image"], cikti["mask"].unsqueeze(0).float()


def transform_olustur(augment_adimlari=None):
    """Resize + (varsa augmentation) + normalize + tensor.

    ONEMLI: A.Resize goruntuye bilinear, MASKEYE NEAREST uygular. Bu otomatik
    ve zorunlu — maske bilinear ile yeniden boyutlandirilsaydi kenarlarda 0.37
    gibi ara degerler olusur, etiketler bozulurdu. Hata vermez, sessizce
    skoru dusururdu.
    """
    adimlar = [A.Resize(IMG_SIZE, IMG_SIZE)]
    if augment_adimlari:
        adimlar += augment_adimlari
    adimlar += [
        A.Normalize(mean=NORM_ORTALAMA, std=NORM_STD),
        ToTensorV2(),
    ]
    return A.Compose(adimlar)


def veri_hazirla(augment_adimlari=None):
    """train/val/test setlerini ve loader'lari kurar.

    train'e augmentation uygulanir, val ve test'e UYGULANMAZ. Aksi halde
    validation skoru her olcumde degisir, hangi denemenin daha iyi oldugu
    soylenemezdi (Gorev 3'teki ayni gerekce).
    """
    adlar = dosya_listesi()
    oranlar = su_oranlari_hesapla(adlar)
    train_idx, val_idx, test_idx = katmanli_bol(oranlar)

    egitim_tf = transform_olustur(augment_adimlari)
    olcum_tf = transform_olustur(None)

    setler = {
        "train": SuVeriSeti(adlar, train_idx, egitim_tf),
        "val": SuVeriSeti(adlar, val_idx, olcum_tf),
        "test": SuVeriSeti(adlar, test_idx, olcum_tf),
        # overfit olcumu icin: ayni egitim goruntuleri ama augmentation'siz
        "train_olcum": SuVeriSeti(adlar, train_idx, olcum_tf),
    }

    ortak = {"num_workers": NUM_WORKERS, "pin_memory": True,
             "persistent_workers": NUM_WORKERS > 0}
    loaderlar = {
        # drop_last=True ZORUNLU: son batch tek ornek kalirsa DeepLabV3+'in ASPP
        # katmani 1x1 uzamsal cikti uretir ve BatchNorm "Expected more than 1
        # value per channel" hatasiyla duser. Egitimde tek ornegi atmak zararsiz.
        "train": DataLoader(setler["train"], batch_size=BATCH_SIZE, shuffle=True,
                            drop_last=True, **ortak),
        "val": DataLoader(setler["val"], batch_size=BATCH_SIZE, shuffle=False, **ortak),
        "test": DataLoader(setler["test"], batch_size=BATCH_SIZE, shuffle=False, **ortak),
        "train_olcum": DataLoader(setler["train_olcum"], batch_size=BATCH_SIZE,
                                  shuffle=False, **ortak),
    }
    return setler, loaderlar, oranlar, (train_idx, val_idx, test_idx)


# 5) METRIKLER
def metrikler_hesapla(model, loader, criterion, device, esik=ESIK, eps=1e-7):
    """IoU ve Dice'i IKI ayri sekilde hesaplar.

    veri seti geneli (aggregate): tum kesisim ve birlesimler toplanip bolunur.
        Daha kararli, buyuk su kutleleri agirlikli.
    goruntu bazli (per-image): her goruntu icin ayri hesaplanip ortalanir.
        Daha sert; kucuk su kutlesi olan goruntuler esit agirlikta sayilir.

    Ikisi farkli sayilar verir ve literaturde ikisi de kullanilir. Hangisinin
    raporlandigini belirtmemek yaygin bir hata, bu yuzden ikisini de kaydediyoruz.
    """
    model.eval()
    toplam_loss = 0.0
    kesisim_top = birlesim_top = tahmin_top = gercek_top = 0.0
    dogru_top = piksel_top = 0.0
    iou_liste, dice_liste = [], []

    with torch.no_grad():
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            toplam_loss += criterion(logits, masks).item()

            tahmin = (torch.sigmoid(logits) > esik).float()
            boyut = (1, 2, 3)

            kesisim = (tahmin * masks).sum(dim=boyut)
            tahmin_alan = tahmin.sum(dim=boyut)
            gercek_alan = masks.sum(dim=boyut)
            birlesim = tahmin_alan + gercek_alan - kesisim

            iou_liste.append(((kesisim + eps) / (birlesim + eps)).cpu())
            dice_liste.append(
                ((2 * kesisim + eps) / (tahmin_alan + gercek_alan + eps)).cpu())

            kesisim_top += kesisim.sum().item()
            birlesim_top += birlesim.sum().item()
            tahmin_top += tahmin_alan.sum().item()
            gercek_top += gercek_alan.sum().item()
            dogru_top += (tahmin == masks).sum().item()
            piksel_top += masks.numel()

    model.train()

    iou_g = torch.cat(iou_liste).mean().item()
    dice_g = torch.cat(dice_liste).mean().item()

    return {
        "loss": toplam_loss / len(loader),
        "iou": kesisim_top / (birlesim_top + eps),
        "dice": 2 * kesisim_top / (tahmin_top + gercek_top + eps),
        "iou_goruntu": iou_g,
        "dice_goruntu": dice_g,
        "piksel_acc": dogru_top / piksel_top,
        "precision": kesisim_top / (tahmin_top + eps),
        "recall": kesisim_top / (gercek_top + eps),
    }


# 6) LOSS FONKSIYONLARI
class DiceLoss(nn.Module):
    """1 - Dice. Dogrudan metrigin kendisini optimize eder.

    BCE her pikseli esit agirlikta sayar; arka plan cok oldugunda modeli
    'hepsi arka plan' demeye iter. Dice loss ise kesisim/alan oranina baktigi
    icin bu tuzaga dusmez.
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, hedef):
        olasilik = torch.sigmoid(logits)
        boyut = (1, 2, 3)
        kesisim = (olasilik * hedef).sum(dim=boyut)
        toplam = olasilik.sum(dim=boyut) + hedef.sum(dim=boyut)
        dice = (2 * kesisim + self.eps) / (toplam + self.eps)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """BCE ve Dice'in agirlikli toplami. Pratikte en yaygin tercih.

    BCE piksel bazli keskinlik saglar, Dice bolge butunlugunu korur.
    """

    def __init__(self, bce_agirlik=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.w = bce_agirlik

    def forward(self, logits, hedef):
        return self.w * self.bce(logits, hedef) + (1 - self.w) * self.dice(logits, hedef)


class FocalLoss(nn.Module):
    """Kolay orneklerin katkisini bastirip zor orneklere odaklanir."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits, hedef):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, hedef, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * hedef + (1 - p) * (1 - hedef)
        at = self.alpha * hedef + (1 - self.alpha) * (1 - hedef)
        return (at * (1 - pt) ** self.gamma * bce).mean()


def loss_kur(ad):
    if ad == "BCE":
        return nn.BCEWithLogitsLoss()
    if ad == "Dice":
        return DiceLoss()
    if ad == "BCE+Dice":
        return BCEDiceLoss()
    if ad == "Focal":
        return FocalLoss()
    raise ValueError(f"Bilinmeyen loss: {ad}")


# 7) EGITIM
def bir_epoch_egit(model, loader, criterion, optimizer, device):
    toplam_loss = 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), masks)
        loss.backward()
        optimizer.step()
        toplam_loss += loss.item()
    return toplam_loss / len(loader)


def train_model(model, loaderlar, criterion, device, epochs=EPOCHS, lr=LEARNING_RATE):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gecmis = []
    en_iyi_val, en_iyi_epoch = 0.0, 0
    import time
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loss = bir_epoch_egit(model, loaderlar["train"], criterion, optimizer, device)
        va = metrikler_hesapla(model, loaderlar["val"], criterion, device)

        if va["iou"] > en_iyi_val:
            en_iyi_val, en_iyi_epoch = va["iou"], epoch

        gecmis.append({
            "epoch": epoch,
            "train_loss": round(loss, 4),
            "val_iou": round(va["iou"] * 100, 2),
            "val_dice": round(va["dice"] * 100, 2),
            "val_loss": round(va["loss"], 4),
        })

        print(f"  Epoch {epoch:>2}/{epochs} | Loss: {loss:.4f} | "
              f"Val IoU: {va['iou'] * 100:.2f}% | Val Dice: {va['dice'] * 100:.2f}%")

    return gecmis, time.perf_counter() - start, en_iyi_val, en_iyi_epoch


# 8) GORSEL KARSILASTIRMA (belge madde 2: "gorsel olarak karsilastirin")
def gorsel_ornekler_sec(oranlar, test_idx, adet=6):
    """Test setinden su orani dusukten yuksege siralanmis ornekler secer.

    Ayni indeksler her deneyde kullanilir; boylece deneyler arasi gorsel
    karsilastirma anlamli olur (farkli goruntulere bakip yanilmayiz).
    """
    test_oranlari = [(i, oranlar[i]) for i in test_idx]
    test_oranlari.sort(key=lambda x: x[1])
    n = len(test_oranlari)
    konumlar = [int(round(k * (n - 1) / (adet - 1))) for k in range(adet)]
    return [test_oranlari[k][0] for k in konumlar]


def gorsel_kaydet(model, adlar, indeksler, device, dosya, baslik):
    """Her ornek icin 4 sutun: goruntu | gercek maske | tahmin | ustuste bindirme."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    olcum_tf = transform_olustur(None)
    model.eval()
    n = len(indeksler)
    fig, eksenler = plt.subplots(n, 4, figsize=(13, 3.1 * n))
    if n == 1:
        eksenler = eksenler[None, :]

    ortalama = np.array(NORM_ORTALAMA)
    std = np.array(NORM_STD)

    with torch.no_grad():
        for satir, idx in enumerate(indeksler):
            ad = adlar[idx]
            img = cv2.cvtColor(goruntu_oku(os.path.join(IMG_DIR, ad)), cv2.COLOR_BGR2RGB)
            msk = (goruntu_oku(os.path.join(MSK_DIR, ad), gri=True) > 127
                   ).astype(np.float32)
            cikti = olcum_tf(image=img, mask=msk)
            x = cikti["image"].unsqueeze(0).to(device)
            gercek = cikti["mask"].numpy()

            tahmin = (torch.sigmoid(model(x))[0, 0] > ESIK).float().cpu().numpy()

            # normalizasyonu geri al
            gosterim = cikti["image"].permute(1, 2, 0).numpy() * std + ortalama
            gosterim = np.clip(gosterim, 0, 1)

            kesisim = (tahmin * gercek).sum()
            birlesim = tahmin.sum() + gercek.sum() - kesisim
            iou = kesisim / (birlesim + 1e-7)

            # ustuste bindirme: yesil = dogru, kirmizi = kacirilan, mavi = fazladan
            bindirme = gosterim.copy()
            bindirme[(tahmin > 0) & (gercek > 0)] = [0.0, 0.9, 0.2]
            bindirme[(tahmin == 0) & (gercek > 0)] = [0.9, 0.1, 0.1]
            bindirme[(tahmin > 0) & (gercek == 0)] = [0.1, 0.3, 0.95]

            for sutun, (veri, alt_baslik, cmap) in enumerate([
                (gosterim, f"{ad}", None),
                (gercek, f"gercek maske  (su %{100 * gercek.mean():.1f})", "gray"),
                (tahmin, f"tahmin  (su %{100 * tahmin.mean():.1f})", "gray"),
                (bindirme, f"IoU %{100 * iou:.1f}", None),
            ]):
                ax = eksenler[satir, sutun]
                ax.imshow(veri, cmap=cmap, vmin=0 if cmap else None,
                          vmax=1 if cmap else None)
                ax.set_title(alt_baslik, fontsize=8)
                ax.axis("off")

    model.train()
    fig.suptitle(f"{baslik}\nyesil = dogru su  ·  kirmizi = kacirilan su  ·  "
                 f"mavi = fazladan su", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(dosya, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return dosya


# 9) CSV KAYIT
def _mevcut_oku(dosya):
    if not os.path.exists(dosya):
        return []
    with open(dosya, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sonuc_ekle(yeni, dosya=SONUC_DOSYASI, alanlar=ALAN_ADLARI):
    """Ayni adli deneyi gunceller, digerlerini korur.

    Bes script de bu tek dosyaya yazar; ayri ayri ve tekrar tekrar
    calistirilabilir, tablo tek parca kalir.
    """
    adlar = {s["deney"] for s in yeni}
    kalan = [s for s in _mevcut_oku(dosya) if s.get("deney") not in adlar]
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=alanlar, extrasaction="ignore")
        w.writeheader()
        w.writerows(kalan + yeni)


def gecmis_kaydet(gecmis, deney_adi):
    dosya = os.path.join(SONUC_DIR, f"epoch_gecmisi_{deney_adi}.csv")
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gecmis[0].keys()))
        w.writeheader()
        w.writerows(gecmis)
    return dosya


# 10) DENEY KOSTURUCU (bes scriptin ortak dongusu)
def deney_calistir(konfig, model_kurucu, loaderlar, adlar, gorsel_idx, device):
    """Tek bir deneyi bastan sona calistirir: egit, olc, kaydet, gorsel uret."""
    seed_ayarla(SEED)
    model = model_kurucu().to(device)
    parametre = sum(p.numel() for p in model.parameters())
    criterion = loss_kur(konfig["loss"])

    print(f"[{konfig['ad']}] {konfig['not']}")
    print(f"      Model: {konfig['model']} | Encoder: {konfig['encoder']} | "
          f"Loss: {konfig['loss']} | Aug: {konfig['augmentation']} | "
          f"Parametre: {parametre:,}".replace(",", "."))

    gecmis, sure, en_iyi_val, en_iyi_epoch = train_model(
        model, loaderlar, criterion, device, lr=konfig.get("lr", LEARNING_RATE))

    tr = metrikler_hesapla(model, loaderlar["train_olcum"], criterion, device)
    va = metrikler_hesapla(model, loaderlar["val"], criterion, device)
    te = metrikler_hesapla(model, loaderlar["test"], criterion, device)

    satir = {
        "deney": konfig["ad"],
        "asama": konfig["asama"],
        "model": konfig["model"],
        "encoder": konfig["encoder"],
        "pretrained": konfig["pretrained"],
        "loss": konfig["loss"],
        "augmentation": konfig["augmentation"],
        "seed": SEED, "epochs": EPOCHS, "img_size": IMG_SIZE,
        "batch": BATCH_SIZE, "lr": konfig.get("lr", LEARNING_RATE),
        "parametre": parametre,
        "val_iou": round(va["iou"] * 100, 2),
        "val_dice": round(va["dice"] * 100, 2),
        "test_iou": round(te["iou"] * 100, 2),
        "test_dice": round(te["dice"] * 100, 2),
        "test_iou_goruntu": round(te["iou_goruntu"] * 100, 2),
        "test_dice_goruntu": round(te["dice_goruntu"] * 100, 2),
        "test_piksel_acc": round(te["piksel_acc"] * 100, 2),
        "test_precision": round(te["precision"] * 100, 2),
        "test_recall": round(te["recall"] * 100, 2),
        "train_iou": round(tr["iou"] * 100, 2),
        "overfit_farki": round((tr["iou"] - va["iou"]) * 100, 2),
        "test_loss": round(te["loss"], 4),
        "en_iyi_val_iou": round(en_iyi_val * 100, 2),
        "en_iyi_epoch": en_iyi_epoch,
        "sure_sn": round(sure, 2),
        "cihaz": CIHAZ,
        "notlar": konfig["not"],
    }

    sonuc_ekle([satir])
    gecmis_kaydet(gecmis, konfig["ad"])
    gorsel = gorsel_kaydet(
        model, adlar, gorsel_idx, device,
        os.path.join(GORSEL_DIR, f"{konfig['ad']}.png"),
        f"{konfig['ad']}  —  Test IoU %{satir['test_iou']:.2f} / "
        f"Dice %{satir['test_dice']:.2f}")

    print(f"  -> SONUC | Test IoU: {satir['test_iou']:.2f}% | "
          f"Dice: {satir['test_dice']:.2f}% | "
          f"Piksel Acc: {satir['test_piksel_acc']:.2f}% | "
          f"Overfit: {satir['overfit_farki']:+.2f} | Sure: {sure:.1f} sn")
    print(f"     Gorsel: {gorsel}\n")
    return satir


def ozet_yazdir(sonuclar, baslik):
    print("=" * 104)
    print(baslik)
    print("=" * 104)
    print(f"{'Deney':<26}{'Param':>11}{'TrainIoU':>10}{'ValIoU':>9}{'TestIoU':>9}"
          f"{'TestDice':>10}{'PikselAcc':>11}{'Overfit':>9}{'Sure':>9}")
    print("-" * 104)
    for s in sonuclar:
        print(f"{s['deney']:<26}{s['parametre']:>11}{s['train_iou']:>10}"
              f"{s['val_iou']:>9}{s['test_iou']:>9}{s['test_dice']:>10}"
              f"{s['test_piksel_acc']:>11}{s['overfit_farki']:>9}{s['sure_sn']:>9}")
    print("-" * 104)
    if sonuclar:
        en_iyi = max(sonuclar, key=lambda s: s["val_iou"])
        print(f"Validation'a gore en iyi: {en_iyi['deney']} "
              f"(val IoU %{en_iyi['val_iou']}, test IoU %{en_iyi['test_iou']})")
