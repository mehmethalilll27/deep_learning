"""
GOREV 3 - ADIM 3 / A1: AUGMENTATION TARAMASI

Baseline ezberliyordu (train accuracy val'den belirgin yuksek). Ilk aksiyon:
veri artirma. Model her epoch'ta ayni goruntuyu biraz farkli gorurse ezberlemesi
zorlasir.

Model, epoch, seed, learning rate BASELINE ILE AYNI birakildi. Tek degisen
augmentation profili - boylece skordaki degisim baska bir seye atfedilemez.

4 profil:
  yok    : baseline tekrari (referans satiri)
  hafif  : yatay cevirme + 15 derece dondurme
  orta   : hafif + renk oynatma (ColorJitter)
  guclu  : rastgele kirpma + daha agresif dondurme + renk oynatma

Beklenti: overfit farki kapanacak. "guclu" profilde fazla bozulma olursa
skor DUSEBILIR - augmentation'in da bir optimumu vardir, bunu gormek de sonuc.

Sure: ~20 dk (RTX 4060 Laptop)
"""

import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# 1) YOLLAR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJE_DIR = os.path.dirname(BASE_DIR)
KAYITLAR_DIR = os.path.join(PROJE_DIR, "kayitlar")
SONUC_DIR = os.path.join(KAYITLAR_DIR, "gorev3")
CONFUSION_DIR = os.path.join(SONUC_DIR, "confusion")
DATA_DIR = os.path.join(KAYITLAR_DIR, "data", "intel")

TRAIN_DIR = os.path.join(DATA_DIR, "seg_train", "seg_train")
TEST_DIR = os.path.join(DATA_DIR, "seg_test", "seg_test")

os.makedirs(SONUC_DIR, exist_ok=True)
os.makedirs(CONFUSION_DIR, exist_ok=True)


# 2) AYARLAR (baseline ile birebir ayni)
SEED = 27
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001
IMG_SIZE = 128
VAL_ORAN = 0.20
NUM_WORKERS = 4
CIHAZ = "GPU"

NORM_ORTALAMA = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Tum scriptler bu tek dosyaya yazar; ayni ada sahip deney varsa uzerine yazilir.
SONUC_DOSYASI = os.path.join(SONUC_DIR, "sonuclar_gorev3.csv")
SINIF_DOSYASI = os.path.join(SONUC_DIR, "sinif_bazli_gorev3.csv")

ALAN_ADLARI = [
    "deney", "model", "augmentation", "optimizer", "scheduler", "pretrained",
    "seed", "epochs", "img_size", "batch", "lr", "parametre",
    "train_acc", "val_acc", "val_f1", "en_iyi_val_acc", "en_iyi_epoch",
    "test_acc", "test_f1", "test_precision", "test_recall",
    "overfit_farki", "test_loss", "sure_sn", "cihaz", "notlar",
]
SINIF_ALANLARI = ["deney", "sinif", "test_destek", "precision", "recall", "f1"]

AUGMENT_PROFILLER = ["yok", "hafif", "orta", "guclu"]


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


# 4) MODEL (baseline ile ayni - degistirmiyoruz ki fark augmentation'dan gelsin)
class BasitCNN(nn.Module):
    def __init__(self, sinif_sayisi=6, img_size=IMG_SIZE):
        super().__init__()
        self.ozellik = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        kenar = img_size // 8
        self.siniflandirici = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * kenar * kenar, 256), nn.ReLU(),
            nn.Linear(256, sinif_sayisi),
        )

    def forward(self, x):
        return self.siniflandirici(self.ozellik(x))


def model_kur(sinif_sayisi, device):
    seed_ayarla(SEED)
    return BasitCNN(sinif_sayisi).to(device)


# 5) VERI
def augment_adimlari(profil):
    """Profil adina gore augmentation katmanlarini dondurur."""
    if profil == "yok":
        return []
    if profil == "hafif":
        return [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
        ]
    if profil == "orta":
        return [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        ]
    if profil == "guclu":
        return [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        ]
    raise ValueError(f"Bilinmeyen profil: {profil}")


def transform_olustur(profil="yok"):
    """Resize'a ikili verilmesi zorunlu: veri setinde 150x113 gibi kareler var,
    tek sayi verilirse en-boy orani korunur ve batch olusturma patlar."""
    adimlar = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    adimlar += augment_adimlari(profil)
    adimlar += [
        transforms.ToTensor(),
        transforms.Normalize(NORM_ORTALAMA, NORM_STD),
    ]
    return transforms.Compose(adimlar)


def katmanli_bol(dataset, val_oran=VAL_ORAN, seed=SEED):
    """Her siniftan ayni oranda ayirir. ImageFolder'da .targets duz liste,
    tensor degil - Gorev 2'deki .targets.numpy() burada calismaz."""
    etiketler = np.array(dataset.targets)
    rng = np.random.default_rng(seed)

    train_idx, val_idx = [], []
    for sinif in np.unique(etiketler):
        idx = np.where(etiketler == sinif)[0]
        rng.shuffle(idx)
        kesme = int(round(len(idx) * val_oran))
        val_idx.extend(idx[:kesme].tolist())
        train_idx.extend(idx[kesme:].tolist())

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return train_idx, val_idx


def veri_yukle(profil):
    """Ayni klasoru iki ImageFolder ile acar: train augmentation'li,
    validation augmentation'siz. Tek dataset'i Subset ile bolsek ikisi de
    ayni transform'u paylasir ve validation skoru her olcumde degisirdi."""
    egitim_tf = transform_olustur(profil)
    olcum_tf = transform_olustur("yok")

    tam_egitim = datasets.ImageFolder(TRAIN_DIR, transform=egitim_tf)
    tam_olcum = datasets.ImageFolder(TRAIN_DIR, transform=olcum_tf)
    test_data = datasets.ImageFolder(TEST_DIR, transform=olcum_tf)

    train_idx, val_idx = katmanli_bol(tam_olcum)

    train_set = Subset(tam_egitim, train_idx)
    val_set = Subset(tam_olcum, val_idx)
    train_olcum_set = Subset(tam_olcum, train_idx)   # overfit olcumu (augmentation'siz)

    return train_set, val_set, test_data, train_olcum_set, tam_olcum.classes


def loader_hazirla(train_set, val_set, test_data, train_olcum_set):
    ortak = {"num_workers": NUM_WORKERS, "pin_memory": True,
             "persistent_workers": NUM_WORKERS > 0}
    return (
        DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, **ortak),
        DataLoader(val_set, batch_size=128, shuffle=False, **ortak),
        DataLoader(test_data, batch_size=128, shuffle=False, **ortak),
        DataLoader(train_olcum_set, batch_size=128, shuffle=False, **ortak),
    )


# 6) METRIKLER
def metricler_hesapla(model, loader, criterion, device, sinif_sayisi=6, eps=1e-12):
    model.eval()
    toplam_loss = 0.0
    confusion = np.zeros((sinif_sayisi, sinif_sayisi), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            toplam_loss += criterion(logits, labels).item()
            preds = logits.argmax(dim=1)
            np.add.at(confusion,
                      (labels.detach().cpu().numpy(), preds.detach().cpu().numpy()), 1)

    model.train()

    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    destek = confusion.sum(axis=1).astype(np.int64)

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return {
        "accuracy": tp.sum() / confusion.sum(),
        "loss": toplam_loss / len(loader),
        "precision_macro": precision.mean(),
        "recall_macro": recall.mean(),
        "f1_macro": f1.mean(),
        "precision": precision, "recall": recall, "f1": f1,
        "destek": destek, "confusion": confusion,
    }


def confusion_kaydet(confusion, deney_adi, siniflar):
    dosya = os.path.join(CONFUSION_DIR, f"{deney_adi}.csv")
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gercek\\tahmin"] + list(siniflar))
        for i, satir in enumerate(confusion):
            writer.writerow([siniflar[i]] + satir.tolist())
    return dosya


def matris_yazdir(confusion, siniflar):
    print("\n  CONFUSION MATRIX (test)")
    print("            " + "".join(f"{s[:7]:>9}" for s in siniflar) + "    <- TAHMIN")
    print("           +" + "-" * (9 * len(siniflar)))
    for i, satir in enumerate(confusion):
        hucreler = []
        for j, deger in enumerate(satir):
            if i == j:
                hucreler.append(f"{deger:>8} ")
            elif deger >= 10:
                hucreler.append(f"{deger:>8}*")
            elif deger > 0:
                hucreler.append(f"{deger:>8} ")
            else:
                hucreler.append(f"{'.':>8} ")
        print(f"  {siniflar[i][:9]:<9}|" + "".join(hucreler))
    print("  ^ GERCEK       ( * = 10+ yanlis tahmin,  . = 0 )")


def sinif_tablosu_yazdir(m, siniflar):
    print(f"\n  {'Sinif':<12}{'Destek':>8}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    print("  " + "-" * 52)
    for s, ad in enumerate(siniflar):
        print(f"  {ad:<12}{m['destek'][s]:>8}"
              f"{m['precision'][s] * 100:>11.2f}%{m['recall'][s] * 100:>9.2f}%"
              f"{m['f1'][s] * 100:>9.2f}%")
    print("  " + "-" * 52)
    print(f"  {'MACRO':<12}{m['destek'].sum():>8}"
          f"{m['precision_macro'] * 100:>11.2f}%{m['recall_macro'] * 100:>9.2f}%"
          f"{m['f1_macro'] * 100:>9.2f}%")


def karisan_ciftler_yazdir(confusion, siniflar, ilk=5):
    satir_toplam = confusion.sum(axis=1)
    ciftler = []
    for i in range(len(siniflar)):
        for j in range(len(siniflar)):
            if i != j and confusion[i, j] > 0:
                ciftler.append((i, j, int(confusion[i, j]),
                                100.0 * confusion[i, j] / max(satir_toplam[i], 1)))
    ciftler.sort(key=lambda x: x[2], reverse=True)
    print("\n  EN COK KARISAN CIFTLER")
    for gercek, tahmin, adet, oran in ciftler[:ilk]:
        print(f"    {siniflar[gercek]:>10} -> {siniflar[tahmin]:<10} "
              f"{adet:>4} ornek (%{oran:.1f})")


# 7) EGITIM
def bir_epoch_egit(model, loader, criterion, optimizer, device):
    toplam_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        toplam_loss += loss.item()
    return toplam_loss / len(loader)


def train_model(model, loaders, device, sinif_sayisi, epochs=EPOCHS):
    train_loader, val_loader, _, train_olcum_loader = loaders
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    gecmis = []
    en_iyi_val, en_iyi_epoch = 0.0, 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loss = bir_epoch_egit(model, train_loader, criterion, optimizer, device)
        tr = metricler_hesapla(model, train_olcum_loader, criterion, device, sinif_sayisi)
        va = metricler_hesapla(model, val_loader, criterion, device, sinif_sayisi)
        fark = (tr["accuracy"] - va["accuracy"]) * 100

        if va["accuracy"] > en_iyi_val:
            en_iyi_val, en_iyi_epoch = va["accuracy"], epoch

        gecmis.append({
            "epoch": epoch,
            "train_loss": round(loss, 4),
            "train_acc": round(tr["accuracy"] * 100, 2),
            "val_acc": round(va["accuracy"] * 100, 2),
            "val_f1": round(va["f1_macro"] * 100, 2),
            "overfit_farki": round(fark, 2),
        })

        print(f"  Epoch {epoch:>2}/{epochs} | Loss: {loss:.4f} | "
              f"Train Acc: {tr['accuracy'] * 100:.2f}% | "
              f"Val Acc: {va['accuracy'] * 100:.2f}% | "
              f"Val F1: {va['f1_macro'] * 100:.2f}% | Fark: {fark:+.2f} puan")

    return gecmis, time.perf_counter() - start, en_iyi_val, en_iyi_epoch


# 8) CSV KAYIT (ortak dosyaya birlestirerek yazar)
def _mevcut_oku(dosya, alanlar):
    if not os.path.exists(dosya):
        return []
    with open(dosya, newline="", encoding="utf-8") as f:
        return [satir for satir in csv.DictReader(f)]


def sonuc_ekle(yeni_satirlar, dosya=SONUC_DOSYASI, alanlar=ALAN_ADLARI):
    """Ayni ada sahip deneyi gunceller, digerlerini korur.

    Boylece baseline / augmentation / mimari / transfer scriptleri ayri ayri
    calistirilsa da tek bir karsilastirma tablosu olusur.
    """
    adlar = {s["deney"] for s in yeni_satirlar}
    kalan = [s for s in _mevcut_oku(dosya, alanlar) if s.get("deney") not in adlar]
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=alanlar, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kalan + yeni_satirlar)


def sinif_bazli_ekle(yeni_satirlar, dosya=SINIF_DOSYASI, alanlar=SINIF_ALANLARI):
    adlar = {s["deney"] for s in yeni_satirlar}
    kalan = [s for s in _mevcut_oku(dosya, alanlar) if s.get("deney") not in adlar]
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=alanlar, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kalan + yeni_satirlar)


def gecmis_kaydet(gecmis, deney_adi):
    dosya = os.path.join(SONUC_DIR, f"epoch_gecmisi_{deney_adi}.csv")
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(gecmis[0].keys()))
        writer.writeheader()
        writer.writerows(gecmis)
    return dosya


# 9) CALISTIR
def main():
    device = get_device()
    seed_ayarla(SEED)

    sonuclar = []
    genel_baslangic = time.perf_counter()

    print("=" * 78)
    print(f"GOREV 3 / A1 - AUGMENTATION TARAMASI | {len(AUGMENT_PROFILLER)} deney")
    print(f"Model, epoch, seed, lr baseline ile AYNI. Tek degisen: augmentation.")
    print(f"Cihaz: {CIHAZ} ({torch.cuda.get_device_name(0)})")
    print("=" * 78 + "\n")

    for i, profil in enumerate(AUGMENT_PROFILLER, 1):
        deney_adi = f"aug_{profil}"

        train_set, val_set, test_data, train_olcum_set, siniflar = veri_yukle(profil)
        loaders = loader_hazirla(train_set, val_set, test_data, train_olcum_set)

        print(f"[{i}/{len(AUGMENT_PROFILLER)}] {deney_adi} | "
              f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_data)}")

        model = model_kur(len(siniflar), device)
        parametre = sum(p.numel() for p in model.parameters())

        gecmis, sure, en_iyi_val, en_iyi_epoch = train_model(
            model, loaders, device, len(siniflar))

        criterion = nn.CrossEntropyLoss()
        test_m = metricler_hesapla(model, loaders[2], criterion, device, len(siniflar))
        son = gecmis[-1]

        sonuclar.append({
            "deney": deney_adi,
            "model": "BasitCNN",
            "augmentation": profil,
            "optimizer": "Adam",
            "scheduler": "yok",
            "pretrained": False,
            "seed": SEED, "epochs": EPOCHS, "img_size": IMG_SIZE,
            "batch": BATCH_SIZE, "lr": LEARNING_RATE, "parametre": parametre,
            "train_acc": son["train_acc"],
            "val_acc": son["val_acc"],
            "val_f1": son["val_f1"],
            "en_iyi_val_acc": round(en_iyi_val * 100, 2),
            "en_iyi_epoch": en_iyi_epoch,
            "test_acc": round(test_m["accuracy"] * 100, 2),
            "test_f1": round(test_m["f1_macro"] * 100, 2),
            "test_precision": round(test_m["precision_macro"] * 100, 2),
            "test_recall": round(test_m["recall_macro"] * 100, 2),
            "overfit_farki": son["overfit_farki"],
            "test_loss": round(test_m["loss"], 4),
            "sure_sn": round(sure, 2),
            "cihaz": CIHAZ,
            "notlar": f"A1 augmentation profili: {profil}",
        })

        sinif_satirlari = [{
            "deney": deney_adi, "sinif": ad,
            "test_destek": int(test_m["destek"][s]),
            "precision": round(test_m["precision"][s] * 100, 2),
            "recall": round(test_m["recall"][s] * 100, 2),
            "f1": round(test_m["f1"][s] * 100, 2),
        } for s, ad in enumerate(siniflar)]

        # her deneyden sonra yaz - kosu yarida kesilse de sonuclar durur
        sonuc_ekle(sonuclar)
        sinif_bazli_ekle(sinif_satirlari)
        confusion_kaydet(test_m["confusion"], deney_adi, siniflar)
        gecmis_kaydet(gecmis, deney_adi)

        print(f"  -> SONUC | Test Acc: {test_m['accuracy'] * 100:.2f}% | "
              f"Test F1: {test_m['f1_macro'] * 100:.2f}% | "
              f"Overfit farki: {son['overfit_farki']:+.2f} puan | Sure: {sure:.1f} sn")
        matris_yazdir(test_m["confusion"], siniflar)
        sinif_tablosu_yazdir(test_m, siniflar)
        karisan_ciftler_yazdir(test_m["confusion"], siniflar)
        print()

    toplam = time.perf_counter() - genel_baslangic
    print("=" * 78)
    print("A1 AUGMENTATION TARAMASI BITTI")
    print(f"Toplam sure: {toplam:.1f} sn ({toplam / 60:.1f} dk)")
    print("=" * 78)
    print(f"{'Deney':<16}{'Train%':>9}{'Val%':>9}{'Test%':>9}{'TestF1%':>10}"
          f"{'Overfit':>10}{'Sure':>9}")
    print("-" * 78)
    for s in sonuclar:
        print(f"{s['deney']:<16}{s['train_acc']:>9}{s['val_acc']:>9}"
              f"{s['test_acc']:>9}{s['test_f1']:>10}{s['overfit_farki']:>10}"
              f"{s['sure_sn']:>9}")
    print("-" * 78)

    en_iyi = max(sonuclar, key=lambda s: s["val_acc"])
    print(f"\nValidation'a gore en iyi profil: {en_iyi['augmentation']} "
          f"(val {en_iyi['val_acc']}%, test {en_iyi['test_acc']}%)")
    print("Bir sonraki adimda (A2 mimari) bu profil sabitlenecek.")


if __name__ == "__main__":
    main()
