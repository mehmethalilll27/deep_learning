"""
GOREV 3 - ADIM 3 / A2: MIMARI VE OPTIMIZASYON

A1'de augmentation ile ezberlemeyi kirdik. Simdi modelin kendisine ve
optimizasyon duzenine dokunuyoruz. Augmentation profili SABIT tutuluyor
(AUGMENT_PROFIL) ki fark mimariden gelsin.

Deneyler ust uste birikir - her satir bir oncekine tek bir sey ekler:
  mim_bn              BatchNorm eklenir
  mim_bn_dropout      + dropout 0.3
  mim_gap             + FC yerine Global Average Pooling
  mim_gap_adamw       + Adam yerine AdamW (agirlik cezasi ayrilir)
  mim_gap_onecycle    + OneCycle learning rate scheduler

Global Average Pooling neden onemli: baseline'da parametrelerin neredeyse
tamami tek bir FC katmanindaydi (128*16*16 -> 256 = 8.4 milyon). GAP bunu
128 -> 256'ya indirir, model ~35 bin parametreye duser. Skor dusebilir de
artabilir de - Gorev 2'deki "buyuk model her zaman kazanmiyor" bulgusunun
bu veri setindeki karsiligini olcuyoruz.

Sure: ~25 dk (RTX 4060 Laptop)
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


# 2) AYARLAR
SEED = 27
EPOCHS = 15
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_LR = 0.003          # OneCycle tepe degeri
IMG_SIZE = 128
VAL_ORAN = 0.20
NUM_WORKERS = 4
CIHAZ = "GPU"

# A1 kazananina gore guncellenmeli. Varsayilan "orta".
AUGMENT_PROFIL = "orta"

NORM_ORTALAMA = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

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

# Her konfig bir oncekine tek bir sey ekler
KONFIGLER = [
    {"ad": "mim_bn",           "bn": True, "dropout": 0.0, "gap": False,
     "opt": "Adam",  "sched": "yok",      "not": "A2 - BatchNorm eklendi"},
    {"ad": "mim_bn_dropout",   "bn": True, "dropout": 0.3, "gap": False,
     "opt": "Adam",  "sched": "yok",      "not": "A2 - BatchNorm + dropout 0.3"},
    {"ad": "mim_gap",          "bn": True, "dropout": 0.3, "gap": True,
     "opt": "Adam",  "sched": "yok",      "not": "A2 - FC yerine GlobalAvgPool"},
    {"ad": "mim_gap_adamw",    "bn": True, "dropout": 0.3, "gap": True,
     "opt": "AdamW", "sched": "yok",      "not": "A2 - GAP + AdamW"},
    {"ad": "mim_gap_onecycle", "bn": True, "dropout": 0.3, "gap": True,
     "opt": "AdamW", "sched": "OneCycle", "not": "A2 - GAP + AdamW + OneCycle"},
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


# 4) MODEL
class GelismisCNN(nn.Module):
    """Baseline'in ayni iskeleti; BatchNorm / dropout / GAP acilip kapanabilir."""

    def __init__(self, sinif_sayisi=6, img_size=IMG_SIZE,
                 bn=False, dropout=0.0, gap=False):
        super().__init__()
        self.gap = gap

        def blok(girdi, cikti):
            katmanlar = [nn.Conv2d(girdi, cikti, kernel_size=3, padding=1)]
            if bn:
                katmanlar.append(nn.BatchNorm2d(cikti))
            katmanlar += [nn.ReLU(), nn.MaxPool2d(2)]
            return katmanlar

        self.ozellik = nn.Sequential(
            *blok(3, 32),
            *blok(32, 64),
            *blok(64, 128),
        )

        if gap:
            # 128 kanalin her birinin uzamsal ortalamasi -> 128 uzunlugunda vektor
            self.havuz = nn.AdaptiveAvgPool2d(1)
            girdi_boyutu = 128
        else:
            self.havuz = nn.Identity()
            kenar = img_size // 8
            girdi_boyutu = 128 * kenar * kenar

        katmanlar = [nn.Flatten(), nn.Linear(girdi_boyutu, 256), nn.ReLU()]
        if dropout > 0:
            katmanlar.append(nn.Dropout(dropout))
        katmanlar.append(nn.Linear(256, sinif_sayisi))
        self.siniflandirici = nn.Sequential(*katmanlar)

    def forward(self, x):
        return self.siniflandirici(self.havuz(self.ozellik(x)))


def model_kur(konfig, sinif_sayisi, device):
    seed_ayarla(SEED)
    model = GelismisCNN(
        sinif_sayisi=sinif_sayisi,
        bn=konfig["bn"],
        dropout=konfig["dropout"],
        gap=konfig["gap"],
    )
    return model.to(device)


def optimizer_kur(model, konfig, adim_sayisi):
    if konfig["opt"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                      weight_decay=0.01)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = None
    if konfig["sched"] == "OneCycle":
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=MAX_LR, epochs=EPOCHS, steps_per_epoch=adim_sayisi)
    return optimizer, scheduler


# 5) VERI
def augment_adimlari(profil):
    if profil == "yok":
        return []
    if profil == "hafif":
        return [transforms.RandomHorizontalFlip(), transforms.RandomRotation(15)]
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
    adimlar = [transforms.Resize((IMG_SIZE, IMG_SIZE))]
    adimlar += augment_adimlari(profil)
    adimlar += [transforms.ToTensor(), transforms.Normalize(NORM_ORTALAMA, NORM_STD)]
    return transforms.Compose(adimlar)


def katmanli_bol(dataset, val_oran=VAL_ORAN, seed=SEED):
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
    egitim_tf = transform_olustur(profil)
    olcum_tf = transform_olustur("yok")

    tam_egitim = datasets.ImageFolder(TRAIN_DIR, transform=egitim_tf)
    tam_olcum = datasets.ImageFolder(TRAIN_DIR, transform=olcum_tf)
    test_data = datasets.ImageFolder(TEST_DIR, transform=olcum_tf)

    train_idx, val_idx = katmanli_bol(tam_olcum)
    return (Subset(tam_egitim, train_idx), Subset(tam_olcum, val_idx),
            test_data, Subset(tam_olcum, train_idx), tam_olcum.classes)


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
def bir_epoch_egit(model, loader, criterion, optimizer, device, scheduler=None):
    toplam_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()
        # OneCycle her BATCH'te adim atar, epoch sonunda degil
        if scheduler is not None:
            scheduler.step()
        toplam_loss += loss.item()
    return toplam_loss / len(loader)


def train_model(model, loaders, device, konfig, sinif_sayisi, epochs=EPOCHS):
    train_loader, val_loader, _, train_olcum_loader = loaders
    criterion = nn.CrossEntropyLoss()
    optimizer, scheduler = optimizer_kur(model, konfig, len(train_loader))

    gecmis = []
    en_iyi_val, en_iyi_epoch = 0.0, 0
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loss = bir_epoch_egit(model, train_loader, criterion, optimizer,
                              device, scheduler)
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
            "lr": round(optimizer.param_groups[0]["lr"], 6),
        })

        print(f"  Epoch {epoch:>2}/{epochs} | Loss: {loss:.4f} | "
              f"Train Acc: {tr['accuracy'] * 100:.2f}% | "
              f"Val Acc: {va['accuracy'] * 100:.2f}% | "
              f"Val F1: {va['f1_macro'] * 100:.2f}% | Fark: {fark:+.2f} puan")

    return gecmis, time.perf_counter() - start, en_iyi_val, en_iyi_epoch


# 8) CSV KAYIT
def _mevcut_oku(dosya):
    if not os.path.exists(dosya):
        return []
    with open(dosya, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def sonuc_ekle(yeni_satirlar, dosya=SONUC_DOSYASI, alanlar=ALAN_ADLARI):
    adlar = {s["deney"] for s in yeni_satirlar}
    kalan = [s for s in _mevcut_oku(dosya) if s.get("deney") not in adlar]
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=alanlar, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kalan + yeni_satirlar)


def sinif_bazli_ekle(yeni_satirlar, dosya=SINIF_DOSYASI, alanlar=SINIF_ALANLARI):
    adlar = {s["deney"] for s in yeni_satirlar}
    kalan = [s for s in _mevcut_oku(dosya) if s.get("deney") not in adlar]
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

    train_set, val_set, test_data, train_olcum_set, siniflar = veri_yukle(AUGMENT_PROFIL)
    loaders = loader_hazirla(train_set, val_set, test_data, train_olcum_set)

    print("=" * 78)
    print(f"GOREV 3 / A2 - MIMARI VE OPTIMIZASYON | {len(KONFIGLER)} deney")
    print(f"Augmentation SABIT: {AUGMENT_PROFIL} | Tek degisen: mimari / optimizasyon")
    print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_data)}")
    print(f"Cihaz: {CIHAZ} ({torch.cuda.get_device_name(0)})")
    print("=" * 78 + "\n")

    sonuclar = []
    genel_baslangic = time.perf_counter()

    for i, konfig in enumerate(KONFIGLER, 1):
        model = model_kur(konfig, len(siniflar), device)
        parametre = sum(p.numel() for p in model.parameters())

        print(f"[{i}/{len(KONFIGLER)}] {konfig['ad']} | BN: {konfig['bn']} | "
              f"Dropout: {konfig['dropout']} | GAP: {konfig['gap']} | "
              f"Opt: {konfig['opt']} | Sched: {konfig['sched']} | "
              f"Parametre: {parametre:,}".replace(",", "."))

        gecmis, sure, en_iyi_val, en_iyi_epoch = train_model(
            model, loaders, device, konfig, len(siniflar))

        criterion = nn.CrossEntropyLoss()
        test_m = metricler_hesapla(model, loaders[2], criterion, device, len(siniflar))
        son = gecmis[-1]

        sonuclar.append({
            "deney": konfig["ad"],
            "model": f"CNN(bn={konfig['bn']},do={konfig['dropout']},gap={konfig['gap']})",
            "augmentation": AUGMENT_PROFIL,
            "optimizer": konfig["opt"],
            "scheduler": konfig["sched"],
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
            "notlar": konfig["not"],
        })

        sinif_satirlari = [{
            "deney": konfig["ad"], "sinif": ad,
            "test_destek": int(test_m["destek"][s]),
            "precision": round(test_m["precision"][s] * 100, 2),
            "recall": round(test_m["recall"][s] * 100, 2),
            "f1": round(test_m["f1"][s] * 100, 2),
        } for s, ad in enumerate(siniflar)]

        sonuc_ekle(sonuclar)
        sinif_bazli_ekle(sinif_satirlari)
        confusion_kaydet(test_m["confusion"], konfig["ad"], siniflar)
        gecmis_kaydet(gecmis, konfig["ad"])

        print(f"  -> SONUC | Test Acc: {test_m['accuracy'] * 100:.2f}% | "
              f"Test F1: {test_m['f1_macro'] * 100:.2f}% | "
              f"Overfit farki: {son['overfit_farki']:+.2f} puan | Sure: {sure:.1f} sn")
        matris_yazdir(test_m["confusion"], siniflar)
        sinif_tablosu_yazdir(test_m, siniflar)
        karisan_ciftler_yazdir(test_m["confusion"], siniflar)
        print()

    toplam = time.perf_counter() - genel_baslangic
    print("=" * 90)
    print("A2 MIMARI DENEYLERI BITTI")
    print(f"Toplam sure: {toplam:.1f} sn ({toplam / 60:.1f} dk)")
    print("=" * 90)
    print(f"{'Deney':<20}{'Parametre':>12}{'Train%':>9}{'Val%':>9}{'Test%':>9}"
          f"{'TestF1%':>10}{'Overfit':>10}{'Sure':>9}")
    print("-" * 90)
    for s in sonuclar:
        print(f"{s['deney']:<20}{s['parametre']:>12}{s['train_acc']:>9}"
              f"{s['val_acc']:>9}{s['test_acc']:>9}{s['test_f1']:>10}"
              f"{s['overfit_farki']:>10}{s['sure_sn']:>9}")
    print("-" * 90)

    en_iyi = max(sonuclar, key=lambda s: s["val_acc"])
    print(f"\nValidation'a gore en iyi: {en_iyi['deney']} "
          f"(val {en_iyi['val_acc']}%, test {en_iyi['test_acc']}%)")


if __name__ == "__main__":
    main()
