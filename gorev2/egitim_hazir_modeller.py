"""
GOREV 2 - ADIM 4 (EK): HAZIR MIMARILER — ResNet18 / MobileNetV2

PDF adim 4'te "her model icin ayri karsilastirma yap (ResNet18, MobileNetV2, ...)"
deniyor. Bu dosya torchvision'daki hazir mimarileri ayni deney duzeninde calistirir:
  - Ayni dengeli alt kumeler (1.000 / 3.000 / 5.000 / 10.000 / 60.000)
  - Ayni dengesiz profiller (P1_pdf / P2_agir / P3_5k)
  - Ayni metrikler (accuracy, precision, recall, F1 macro, azinlik recall, confusion)

MNIST uyarlamasi:
  - 28x28 gri goruntu -> IMG_SIZE'a (32) buyutulur, 1 kanal 3 kanala kopyalanir
  - Mimariler degistirilmez, sadece cikis katmani 10 sinifa ayarlanir
  - PRETRAINED = False (sifirdan egitim, internet gerektirmez).
    True yapilirsa ImageNet agirliklari indirilir ve sadece cikis katmani yenilenir.
"""

import csv
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

# 1) YOLLAR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJE_DIR = os.path.dirname(BASE_DIR)
KAYITLAR_DIR = os.path.join(PROJE_DIR, "kayitlar")
SONUC_DIR = os.path.join(KAYITLAR_DIR, "gorev2")
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
IMG_SIZE = 32          # 28x28 -> 32x32 (ResNet/MobileNet stride'lari icin daha uygun)
PRETRAINED = False     # True -> ImageNet agirliklari (internet gerekir)

MODEL_ADLARI = ["ResNet18", "MobileNetV2"]
VERI_BOYUTLARI = [1000, 3000, 5000, 10000, None]   # None = tam veri (60.000)

COGUNLUK_SINIFLARI = [0, 1]
AZINLIK_SINIFLARI = [5, 6, 8]
ORTA_SINIFLAR = [2, 3, 4, 7, 9]

DENGESIZ_PROFILLER = {
    "P1_pdf":  {"cogunluk": 3000, "orta": 500, "azinlik": 150},
    "P2_agir": {"cogunluk": 3000, "orta": 300, "azinlik": 100},
    "P3_5k":   {"cogunluk": 1500, "orta": 300, "azinlik": 120},
}

SONUC_DOSYASI = os.path.join(SONUC_DIR, "sonuclar_f1_hazir_modeller.csv")
SINIF_DOSYASI = os.path.join(SONUC_DIR, "sinif_bazli_hazir_modeller.csv")


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


# 4) MODELLER (torchvision hazir mimarileri)
def model_kur(ad, device):
    seed_ayarla(SEED)

    if ad == "ResNet18":
        if PRETRAINED:
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            model.fc = nn.Linear(model.fc.in_features, 10)
        else:
            model = models.resnet18(weights=None, num_classes=10)

    elif ad == "MobileNetV2":
        if PRETRAINED:
            model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
            model.classifier[1] = nn.Linear(model.last_channel, 10)
        else:
            model = models.mobilenet_v2(weights=None, num_classes=10)

    else:
        raise ValueError(f"Bilinmeyen model: {ad}")

    return model.to(device)


# 5) VERI
def gri_to_rgb(x):
    return x.repeat(3, 1, 1)


def veri_yukle():
    transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Lambda(gri_to_rgb),
        transforms.Normalize([0.1307] * 3, [0.3081] * 3),
    ])
    train_data = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    return train_data, test_data


def dengeli_alt_kume(train_data, toplam, seed=SEED):
    """Her siniftan (toplam/10) ornek; toplam=None ise tam veri seti."""
    if toplam is None:
        return train_data, len(train_data)

    hedef = toplam // 10
    etiketler = train_data.targets.numpy()
    rng = np.random.default_rng(seed)

    secilen = []
    for sinif in range(10):
        idx = np.where(etiketler == sinif)[0]
        secilen.extend(rng.choice(idx, size=min(hedef, len(idx)), replace=False).tolist())

    rng.shuffle(secilen)
    return Subset(train_data, secilen), len(secilen)


def dengesiz_alt_kume(train_data, profil, seed=SEED):
    """Profildeki sayilara gore her siniftan farkli miktarda ornek secer."""
    p = DENGESIZ_PROFILLER[profil]
    dagilim = {}
    for sinif in range(10):
        if sinif in COGUNLUK_SINIFLARI:
            dagilim[sinif] = p["cogunluk"]
        elif sinif in AZINLIK_SINIFLARI:
            dagilim[sinif] = p["azinlik"]
        else:
            dagilim[sinif] = p["orta"]

    etiketler = train_data.targets.numpy()
    rng = np.random.default_rng(seed)

    secilen = []
    gercek = {}
    for sinif in range(10):
        idx = np.where(etiketler == sinif)[0]
        n = min(dagilim[sinif], len(idx))
        secilen.extend(rng.choice(idx, size=n, replace=False).tolist())
        gercek[sinif] = n

    rng.shuffle(secilen)
    return Subset(train_data, secilen), gercek


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

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
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
        "azinlik_precision": precision[AZINLIK_SINIFLARI].mean(),
        "azinlik_f1": f1[AZINLIK_SINIFLARI].mean(),
        "cogunluk_recall": recall[COGUNLUK_SINIFLARI].mean(),
        "cogunluk_f1": f1[COGUNLUK_SINIFLARI].mean(),
    }


def confusion_kaydet(confusion, deney_adi):
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

        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

        toplam_loss += loss.item()
    return toplam_loss / len(loader)


def train_model(model, train_loader, test_loader, device, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loss = bir_epoch_egit(model, train_loader, criterion, optimizer, device)
        # her epoch'ta olc ve yazdir
        ara = metricler_hesapla(model, test_loader, criterion, device)
        print(
            f"  Epoch {epoch:>2}/{epochs} | Loss: {loss:.4f} | "
            f"Acc: {ara['accuracy'] * 100:.2f}% | F1: {ara['f1_macro'] * 100:.2f}% | "
            f"Azinlik Recall: {ara['azinlik_recall'] * 100:.2f}%"
        )

    sure = time.perf_counter() - start
    metrics = metricler_hesapla(model, test_loader, criterion, device)
    return metrics, sure


# 8) CSV KAYIT

def sonuclari_kaydet(sonuclar, dosya=SONUC_DOSYASI):
    fieldnames = [
        "deney", "model", "senaryo", "veri_sayisi", "gercek_ornek", "profil",
        "seed", "epochs", "img_size", "pretrained",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "azinlik_recall",
        "cogunluk_recall",
        "acc_azinlik_farki",
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
    fieldnames = ["deney", "model", "senaryo", "sinif", "test_destek",
                  "precision", "recall", "f1"]
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(satirlar)


# 9) DENEY LISTESI
# 2 model x (5 dengeli boyut + 3 dengesiz profil) = 16 kombinasyon
def veri_etiketi(toplam):
    return "60000" if toplam is None else str(toplam)


def tum_deneyler():
    deneyler = []
    for model_adi in MODEL_ADLARI:
        for toplam in VERI_BOYUTLARI:
            deneyler.append({
                "ad": f"{model_adi}_{veri_etiketi(toplam)}",
                "model": model_adi,
                "senaryo": "dengeli",
                "veri": toplam,
                "profil": "",
                "not": f"{model_adi} - dengeli {veri_etiketi(toplam)} ornek",
            })
        for profil in DENGESIZ_PROFILLER:
            deneyler.append({
                "ad": f"{model_adi}_dengesiz_{profil}",
                "model": model_adi,
                "senaryo": "dengesiz",
                "veri": None,
                "profil": profil,
                "not": f"{model_adi} - dengesiz profil {profil}",
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

    print(f"=== HAZIR MIMARILER: {', '.join(MODEL_ADLARI)} | "
          f"Toplam deney: {len(deneyler)} | Seed: {SEED} | Epoch: {EPOCHS} | "
          f"Girdi: {IMG_SIZE}x{IMG_SIZE}x3 | Pretrained: {PRETRAINED} | "
          f"Cihaz: {CIHAZ} ({device}) ===\n")

    for i, deney in enumerate(deneyler, 1):
        if deney["senaryo"] == "dengeli":
            train_subset, gercek_ornek = dengeli_alt_kume(train_data, deney["veri"])
            veri_etk = veri_etiketi(deney["veri"])
        else:
            train_subset, dagilim = dengesiz_alt_kume(train_data, deney["profil"])
            gercek_ornek = sum(dagilim.values())
            veri_etk = f"{gercek_ornek} (dengesiz)"

        train_loader, test_loader = loader_hazirla(train_subset, test_data)

        print(
            f"[{i}/{len(deneyler)}] {deney['ad']} | Senaryo: {deney['senaryo']} | "
            f"Egitim ornegi: {gercek_ornek}"
        )

        model = model_kur(deney["model"], device)
        metrics, sure = train_model(model, train_loader, test_loader, device)

        acc = metrics["accuracy"] * 100
        azinlik_rec = metrics["azinlik_recall"] * 100

        sonuclar.append({
            "deney": deney["ad"],
            "model": deney["model"],
            "senaryo": deney["senaryo"],
            "veri_sayisi": veri_etk,
            "gercek_ornek": gercek_ornek,
            "profil": deney["profil"],
            "seed": SEED,
            "epochs": EPOCHS,
            "img_size": IMG_SIZE,
            "pretrained": PRETRAINED,
            "accuracy": round(acc, 2),
            "precision_macro": round(metrics["precision_macro"] * 100, 2),
            "recall_macro": round(metrics["recall_macro"] * 100, 2),
            "f1_macro": round(metrics["f1_macro"] * 100, 2),
            "azinlik_recall": round(azinlik_rec, 2),
            "cogunluk_recall": round(metrics["cogunluk_recall"] * 100, 2),
            "acc_azinlik_farki": round(acc - azinlik_rec, 2),
            "loss": round(metrics["loss"], 4),
            "sure_sn": round(sure, 2),
            "cihaz": CIHAZ,
            "notlar": deney["not"],
        })

        for sinif in range(10):
            sinif_satirlari.append({
                "deney": deney["ad"],
                "model": deney["model"],
                "senaryo": deney["senaryo"],
                "sinif": sinif,
                "test_destek": int(metrics["destek"][sinif]),
                "precision": round(metrics["precision"][sinif] * 100, 2),
                "recall": round(metrics["recall"][sinif] * 100, 2),
                "f1": round(metrics["f1"][sinif] * 100, 2),
            })

        confusion_kaydet(metrics["confusion"], deney["ad"])
        sonuclari_kaydet(sonuclar)
        sinif_bazli_kaydet(sinif_satirlari)

        print(
            f"  -> SONUC | Acc: {acc:.2f}% | F1(macro): {metrics['f1_macro'] * 100:.2f}% | "
            f"Precision: {metrics['precision_macro'] * 100:.2f}% | "
            f"Recall: {metrics['recall_macro'] * 100:.2f}% | "
            f"Azinlik Recall: {azinlik_rec:.2f}% | Sure: {sure:.1f} sn"
        )
        # PDF: "Confusion Matrix'i her deneyde mutlaka cikti olarak alin"
        matris_yazdir(metrics["confusion"])
        sinif_tablosu_yazdir(metrics)
        karisan_ciftler_yazdir(metrics["confusion"])
        print()

    toplam_sure = time.perf_counter() - genel_baslangic
    print("=" * 88)
    print("HAZIR MIMARI DENEYLERI BITTI")
    print(f"Toplam sure: {toplam_sure:.1f} sn ({toplam_sure / 60:.1f} dk)")
    print(f"Sonuclar        : {SONUC_DOSYASI}")
    print(f"Sinif bazli     : {SINIF_DOSYASI}")
    print(f"Confusion matris: {CONFUSION_DIR}")
    print("=" * 88)
    print(f"{'Deney':<28} {'Veri':>16} {'Acc%':>7} {'F1%':>7} {'AzRec%':>8} {'Sure':>8}")
    print("-" * 88)
    for s in sonuclar:
        print(
            f"{s['deney']:<28} {s['veri_sayisi']:>16} {s['accuracy']:>7} "
            f"{s['f1_macro']:>7} {s['azinlik_recall']:>8} {s['sure_sn']:>8}"
        )

    # PDF'teki tablo formati: her mimari icin ayri tablo
    print()
    for model_adi in MODEL_ADLARI:
        kendi = [s for s in sonuclar if s["model"] == model_adi]
        if not kendi:
            continue
        print("=" * 74)
        print(f"TABLO - {model_adi}")
        print("=" * 74)
        print(f"{'Veri Sayisi':<28}{'Accuracy':>12}{'F1 (macro)':>14}{'Recall Azinlik':>18}")
        print("-" * 74)
        for anahtar in ["60000", "10000", "5000", "3000", "1000"]:
            s = next((x for x in kendi
                      if x["senaryo"] == "dengeli" and x["veri_sayisi"] == anahtar), None)
            if not s:
                continue
            etiket = "60.000 (tam)" if anahtar == "60000" else f"{int(anahtar):,}".replace(",", ".")
            print(
                f"{etiket:<28}{'%' + str(s['accuracy']):>12}"
                f"{'%' + str(s['f1_macro']):>14}{'%' + str(s['azinlik_recall']):>18}"
            )
        for s in [x for x in kendi if x["senaryo"] == "dengesiz"]:
            etiket = f"{s['gercek_ornek']} + dengesiz ({s['profil']})"
            print(
                f"{etiket:<28}{'%' + str(s['accuracy']):>12}"
                f"{'%' + str(s['f1_macro']):>14}{'%' + str(s['azinlik_recall']):>18}"
            )
        print("-" * 74)


if __name__ == "__main__":
    main()
