"""
GOREV 3 - ADIM 1: VERI SAYISINI AZALT

Sorun: Tam veri setinde (60.000 ornek) tum modeller ~%98-99 accuracy veriyor,
konfigurasyonlar arasindaki fark gozlemlenemiyor.

Cozum: MNIST'ten rastgele DENGELI alt kumeler olustur (1.000 / 3.000 / 5.000 /
10.000) ve her veri miktari icin modeli yeniden egit. Test seti her zaman tam
10.000'lik dengeli test setidir (yoksa metrikler karsilastirilamaz).
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
KAYITLAR_DIR = os.path.join(PROJE_DIR, "kayıtlar")
SONUC_DIR = os.path.join(KAYITLAR_DIR, "gorev3")
CONFUSION_DIR = os.path.join(SONUC_DIR, "confusion")
DATA_DIR = os.path.join(KAYITLAR_DIR, "data")

os.makedirs(SONUC_DIR, exist_ok=True)
os.makedirs(CONFUSION_DIR, exist_ok=True)


# 2) AYARLAR
SEED = 27
EPOCHS = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CIHAZ = "GPU"

VERI_BOYUTLARI = [1000, 3000, 5000, 10000, None]


AZINLIK_SINIFLARI = [5, 6, 8]
COGUNLUK_SINIFLARI = [0, 1]

SONUC_DOSYASI = os.path.join(SONUC_DIR, "sonuclar_f1_veri_azalt.csv")
SINIF_DOSYASI = os.path.join(SONUC_DIR, "sinif_bazli_veri_azalt.csv")


# 3) CIHAZ
def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU bulunamadi! CUDA destekli PyTorch gerekli.")
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


def seed_ayarla(seed=SEED):
    """Ayni sonucu tekrar uretebilmek icin tum rastgelelik kaynaklarini sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 4) MODELLER
class MLP(nn.Module):
    def __init__(self, hidden_layers, dropout=0.0):
        super().__init__()
        layers = [nn.Flatten()]  
        prev = 28 * 28

        for size in hidden_layers:
            layers.append(nn.Linear(prev, size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = size

        layers.append(nn.Linear(prev, 10))  # cikis: 10 rakam
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """conv_channels uzunlugu = conv+pool katman sayisi (or: (16,32) -> 2 katman)."""

    def __init__(self, conv_channels=(16, 32), dropout=0.0, fc_size=128):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in conv_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            ])
            in_ch = out_ch

        self.features = nn.Sequential(*layers)
        spatial = 28 // (2 ** len(conv_channels))
        flat = conv_channels[-1] * spatial * spatial

        classifier = [nn.Flatten(), nn.Linear(flat, fc_size), nn.ReLU()]
        if dropout > 0:
            classifier.append(nn.Dropout(dropout))
        classifier.append(nn.Linear(fc_size, 10))
        self.classifier = nn.Sequential(*classifier)

    def forward(self, x):
        return self.classifier(self.features(x))


def model_kur(konfig, device):
    seed_ayarla(SEED)  # her model ayni baslangic agirliklariyla kurulsun
    if konfig["tur"] == "MLP":
        model = MLP(konfig["hidden"], dropout=konfig["dropout"])
    else:
        model = CNN(konfig["conv"], dropout=konfig["dropout"])
    return model.to(device)


# 5) VERI: DENGELI ALT KUME
def veri_yukle():
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    return train_data, test_data


def dengeli_alt_kume(train_data, toplam, seed=SEED):
    """Her siniftan (toplam/10) ornek rastgele secer."""
    if toplam is None:
        return train_data, len(train_data)

    hedef = toplam // 10
    etiketler = train_data.targets.numpy()
    rng = np.random.default_rng(seed)

    secilen = []
    for sinif in range(10):
        idx = np.where(etiketler == sinif)[0]
        n = min(hedef, len(idx))
        secilen.extend(rng.choice(idx, size=n, replace=False).tolist())

    rng.shuffle(secilen)
    return Subset(train_data, secilen), len(secilen)


def loader_hazirla(train_subset, test_data):
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, pin_memory=True)
    return train_loader, test_loader


# 6) METRIKLER
def metricler_hesapla(model, loader, criterion, device, num_classes=10, eps=1e-12):
    model.eval()
    toplam_loss = 0.0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            toplam_loss += criterion(logits, labels).item()

            preds = logits.argmax(dim=1)
            np.add.at(
                confusion,
                (labels.detach().cpu().numpy(), preds.detach().cpu().numpy()),
                1,
            )

    model.train()

    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp
    destek = confusion.sum(axis=1).astype(np.int64)

    precision = tp / (tp + fp + eps)   # tahminlerin ne kadari dogru
    recall = tp / (tp + fn + eps)      # gercek pozitiflerin ne kadari yakalandi
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return {
        "accuracy": tp.sum() / confusion.sum(),
        "loss": toplam_loss / len(loader),
        "precision_macro": precision.mean(),
        "recall_macro": recall.mean(),
        "f1_macro": f1.mean(),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "destek": destek,
        "confusion": confusion,
        "azinlik_recall": recall[AZINLIK_SINIFLARI].mean(),
        "azinlik_f1": f1[AZINLIK_SINIFLARI].mean(),
        "cogunluk_recall": recall[COGUNLUK_SINIFLARI].mean(),
    }


def confusion_kaydet(confusion, deney_adi):
    """10x10 matrisi CSV'ye yazar (satir = gercek sinif, sutun = tahmin)."""
    dosya = os.path.join(CONFUSION_DIR, f"{deney_adi}.csv")
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["gercek\\tahmin"] + list(range(10)))
        for i, satir in enumerate(confusion):
            writer.writerow([i] + satir.tolist())
    return dosya


def matris_yazdir(confusion):
    """Terminale okunakli 10x10 matris basar (satir = gercek, sutun = tahmin)."""
    print("\n  CONFUSION MATRIX")
    print("        " + "".join(f"{j:>7}" for j in range(10)) + "    <- TAHMIN")
    print("      +" + "-" * 70)
    for i, satir in enumerate(confusion):
        hucreler = []
        for j, deger in enumerate(satir):
            if i == j:
                hucreler.append(f"{deger:>6} ")     
            elif deger >= 10:
                hucreler.append(f"{deger:>6}*")     
            elif deger > 0:
                hucreler.append(f"{deger:>6} ")
            else:
                hucreler.append(f"{'.':>6} ")      
        print(f"  {i}   |" + "".join(hucreler))
    print("  ^ GERCEK       ( * = 10+ yanlis tahmin,  . = 0 )")


def sinif_tablosu_yazdir(m):
    """Sinif bazli Precision / Recall / F1 tablosu."""
    print(f"\n  {'Sinif':<7}{'Tip':<10}{'Destek':>8}{'Precision':>12}{'Recall':>10}{'F1':>10}")
    print("  " + "-" * 57)
    for s in range(10):
        if s in COGUNLUK_SINIFLARI:
            tip = "cogunluk"
        elif s in AZINLIK_SINIFLARI:
            tip = "AZINLIK"
        else:
            tip = "orta"
        print(
            f"  {s:<7}{tip:<10}{m['destek'][s]:>8}"
            f"{m['precision'][s] * 100:>11.2f}%{m['recall'][s] * 100:>9.2f}%"
            f"{m['f1'][s] * 100:>9.2f}%"
        )
    print("  " + "-" * 57)
    print(
        f"  {'MACRO':<17}{m['destek'].sum():>8}"
        f"{m['precision_macro'] * 100:>11.2f}%{m['recall_macro'] * 100:>9.2f}%"
        f"{m['f1_macro'] * 100:>9.2f}%"
    )


def karisan_ciftler_yazdir(confusion, ilk=5):
    """Kosegen disindaki en buyuk hucreler: hangi sinif hangisiyle karisiyor."""
    satir_toplam = confusion.sum(axis=1)
    ciftler = []
    for i in range(10):
        for j in range(10):
            if i != j and confusion[i, j] > 0:
                ciftler.append((i, j, int(confusion[i, j]),
                                100.0 * confusion[i, j] / max(satir_toplam[i], 1)))
    ciftler.sort(key=lambda x: x[2], reverse=True)

    print("\n  EN COK KARISAN CIFTLER")
    for gercek, tahmin, adet, oran in ciftler[:ilk]:
        print(f"    Sinif {gercek} -> {tahmin} : {adet:>4} ornek (%{oran:.1f})")


# 7) EGITIM

def bir_epoch_egit(model, loader, criterion, optimizer, device):
    toplam_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                    # 1) eski gradyanlari sifirla
        loss = criterion(model(images), labels)  # 2) forward + loss
        loss.backward()                          # 3) backward
        optimizer.step()                         # 4) w = w - lr * dw

        toplam_loss += loss.item()
    return toplam_loss / len(loader)


def train_model(model, train_loader, test_loader, device, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loss = bir_epoch_egit(model, train_loader, criterion, optimizer, device)
        # her epoch'ta olc ve yazdir (metricler_hesapla tek geciste hepsini cikarir)
        ara = metricler_hesapla(model, test_loader, criterion, device)
        print(
            f"  Epoch {epoch:>2}/{epochs} | Loss: {loss:.4f} | "
            f"Acc: {ara['accuracy'] * 100:.2f}% | F1: {ara['f1_macro'] * 100:.2f}% | "
            f"Prec: {ara['precision_macro'] * 100:.2f}% | Rec: {ara['recall_macro'] * 100:.2f}%"
        )

    sure = time.perf_counter() - start
    metrics = metricler_hesapla(model, test_loader, criterion, device)
    return metrics, sure


# 8) CSV KAYIT
def sonuclari_kaydet(sonuclar, dosya=SONUC_DOSYASI):
    fieldnames = [
        "deney", "model", "konfig", "veri_sayisi", "gercek_ornek", "seed", "epochs",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "azinlik_recall",
        "cogunluk_recall",
        "loss",
        "sure_sn",
        "cihaz",
        "notlar",
    ]
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sonuclar)


def sinif_bazli_kaydet(satirlar, dosya=SINIF_DOSYASI):
    fieldnames = ["deney", "veri_sayisi", "sinif", "destek", "precision", "recall", "f1"]
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(satirlar)


# 9) DENEY LISTESI
KONFIGLER = [
    {"ad": "MLP_d0", "tur": "MLP", "hidden": [256, 128], "dropout": 0.0,
     "not": "MLP 2 katman, dropout yok (eski MLP_2_katman ile ayni)"},
    {"ad": "MLP_d03", "tur": "MLP", "hidden": [256, 128], "dropout": 0.3,
     "not": "MLP 2 katman + dropout 0.3"},
    {"ad": "CNN_d0", "tur": "CNN", "conv": (16, 32), "dropout": 0.0,
     "not": "CNN 2 conv katman, dropout yok (eski CNN_dropout_0 ile ayni)"},
    {"ad": "CNN_d03", "tur": "CNN", "conv": (16, 32), "dropout": 0.3,
     "not": "CNN 2 conv katman + dropout 0.3"},
]


def veri_etiketi(toplam):
    return "60000" if toplam is None else str(toplam)


def tum_deneyler():
    deneyler = []
    for konfig in KONFIGLER:
        for toplam in VERI_BOYUTLARI:
            deneyler.append({
                "ad": f"{konfig['ad']}_{veri_etiketi(toplam)}",
                "konfig": konfig,
                "veri": toplam,
            })
    return deneyler


# 10) CALISTIR
def main():
    device = get_device()
    seed_ayarla(SEED)
    train_data, test_data = veri_yukle()
    deneyler = tum_deneyler()
    sonuclar = []
    sinif_satirlari = []
    genel_baslangic = time.perf_counter()

    print(f"=== ADIM 1: VERI SAYISINI AZALT | Toplam deney: {len(deneyler)} | "
          f"Seed: {SEED} | Epoch: {EPOCHS} | Cihaz: {CIHAZ} ({device}) ===\n")

    for i, deney in enumerate(deneyler, 1):
        konfig = deney["konfig"]
        train_subset, gercek_ornek = dengeli_alt_kume(train_data, deney["veri"])
        train_loader, test_loader = loader_hazirla(train_subset, test_data)

        print(
            f"[{i}/{len(deneyler)}] {deney['ad']} | "
            f"Egitim ornegi: {gercek_ornek} | Dropout: {konfig['dropout']}"
        )

        model = model_kur(konfig, device)
        metrics, sure = train_model(model, train_loader, test_loader, device)

        sonuclar.append({
            "deney": deney["ad"],
            "model": konfig["tur"],
            "konfig": konfig["ad"],
            "veri_sayisi": veri_etiketi(deney["veri"]),
            "gercek_ornek": gercek_ornek,
            "seed": SEED,
            "epochs": EPOCHS,
            "accuracy": round(metrics["accuracy"] * 100, 2),
            "precision_macro": round(metrics["precision_macro"] * 100, 2),
            "recall_macro": round(metrics["recall_macro"] * 100, 2),
            "f1_macro": round(metrics["f1_macro"] * 100, 2),
            "azinlik_recall": round(metrics["azinlik_recall"] * 100, 2),
            "cogunluk_recall": round(metrics["cogunluk_recall"] * 100, 2),
            "loss": round(metrics["loss"], 4),
            "sure_sn": round(sure, 2),
            "cihaz": CIHAZ,
            "notlar": konfig["not"],
        })

        for sinif in range(10):
            sinif_satirlari.append({
                "deney": deney["ad"],
                "veri_sayisi": veri_etiketi(deney["veri"]),
                "sinif": sinif,
                "destek": int(metrics["destek"][sinif]),
                "precision": round(metrics["precision"][sinif] * 100, 2),
                "recall": round(metrics["recall"][sinif] * 100, 2),
                "f1": round(metrics["f1"][sinif] * 100, 2),
            })

        confusion_kaydet(metrics["confusion"], deney["ad"])
        sonuclari_kaydet(sonuclar)      # her deneyden sonra kaydet
        sinif_bazli_kaydet(sinif_satirlari)

        print(
            f"  -> SONUC | Acc: {metrics['accuracy'] * 100:.2f}% | "
            f"F1(macro): {metrics['f1_macro'] * 100:.2f}% | "
            f"Precision: {metrics['precision_macro'] * 100:.2f}% | "
            f"Recall: {metrics['recall_macro'] * 100:.2f}% | "
            f"Azinlik Recall: {metrics['azinlik_recall'] * 100:.2f}% | "
            f"Sure: {sure:.1f} sn"
        )
        # PDF: "Confusion Matrix'i her deneyde mutlaka cikti olarak alin"
        matris_yazdir(metrics["confusion"])
        sinif_tablosu_yazdir(metrics)
        karisan_ciftler_yazdir(metrics["confusion"])
        print()

    toplam_sure = time.perf_counter() - genel_baslangic
    print("=" * 78)
    print("ADIM 1 BITTI - VERI SAYISINI AZALT")
    print(f"Toplam sure: {toplam_sure:.1f} sn ({toplam_sure / 60:.1f} dk)")
    print(f"Sonuclar        : {SONUC_DOSYASI}")
    print(f"Sinif bazli     : {SINIF_DOSYASI}")
    print(f"Confusion matris: {CONFUSION_DIR}")
    print("=" * 78)
    print(f"{'Deney':<20} {'Veri':>7} {'Acc%':>7} {'F1%':>7} {'Prec%':>7} "
          f"{'Rec%':>7} {'AzRec%':>8} {'Loss':>8}")
    print("-" * 78)
    for s in sonuclar:
        print(
            f"{s['deney']:<20} {s['veri_sayisi']:>7} {s['accuracy']:>7} "
            f"{s['f1_macro']:>7} {s['precision_macro']:>7} {s['recall_macro']:>7} "
            f"{s['azinlik_recall']:>8} {s['loss']:>8}"
        )

    # PDF'teki tablo formati: her konfig icin veri sayisina gore karsilastirma
    print()
    for konfig in KONFIGLER:
        kendi = [s for s in sonuclar if s["konfig"] == konfig["ad"]]
        if not kendi:
            continue
        print("=" * 62)
        print(f"TABLO - {konfig['ad']} ({konfig['not']})")
        print("=" * 62)
        print(f"{'Veri Sayisi':<16}{'Accuracy':>12}{'F1 (macro)':>14}{'Recall Azinlik':>18}")
        print("-" * 62)
        for anahtar in ["60000", "10000", "5000", "3000", "1000"]:
            s = next((x for x in kendi if x["veri_sayisi"] == anahtar), None)
            if not s:
                continue
            etiket = "60.000 (tam)" if anahtar == "60000" else f"{int(anahtar):,}".replace(",", ".")
            print(
                f"{etiket:<16}{'%' + str(s['accuracy']):>12}"
                f"{'%' + str(s['f1_macro']):>14}{'%' + str(s['azinlik_recall']):>18}"
            )
        print("-" * 62)


if __name__ == "__main__":
    main()
