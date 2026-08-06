"""
GOREV 3 - ADIM 2: SINIFLARI DENGESIZ HALE GETIR

Amac: Bazi siniflara cok az ornek birakarak (class imbalance) modelin azinlik
siniflarina karsi nasil davrandigini gozlemlemek.

  Cogunluk siniflari (0, 1)   -> 1.500-3.000 ornek
  Orta siniflar (2,3,4,7,9)   -> 300-500 ornek
  Azinlik siniflari (5, 6, 8) -> 100-150 ornek

Test seti DAIMA tam ve dengeli 10.000'lik MNIST test setidir. Egitim setini
dengesizlestirip testi dengeli birakmak sart: yoksa azinlik recall'u olculemez.

Beklenen sonuc (PDF): Accuracy hala yuksek gorunur ama azinlik sinifi recall'u
cok duser -> Accuracy'nin tek basina yaniltici oldugunun kaniti.
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

COGUNLUK_SINIFLARI = [0, 1]
AZINLIK_SINIFLARI = [5, 6, 8]
ORTA_SINIFLAR = [2, 3, 4, 7, 9]

# 3 dengesizlik profili: siddet arttikca azinlik recall'unun nasil coktugunu gosterir
DENGESIZ_PROFILLER = {
    "P1_pdf":  {"cogunluk": 3000, "orta": 500, "azinlik": 150},
    "P2_agir": {"cogunluk": 3000, "orta": 300, "azinlik": 100},
    "P3_5k":   {"cogunluk": 1500, "orta": 300, "azinlik": 120},
}

SONUC_DOSYASI = os.path.join(SONUC_DIR, "sonuclar_f1_dengesiz.csv")
SINIF_DOSYASI = os.path.join(SONUC_DIR, "sinif_bazli_dengesiz.csv")


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


# 4) MODELLER (adim 1 ile birebir ayni konfigler)

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

        layers.append(nn.Linear(prev, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
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
    seed_ayarla(SEED)
    if konfig["tur"] == "MLP":
        model = MLP(konfig["hidden"], dropout=konfig["dropout"])
    else:
        model = CNN(konfig["conv"], dropout=konfig["dropout"])
    return model.to(device)


# 5) VERI: DENGESIZ ALT KUME
def veri_yukle():
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    return train_data, test_data


def sinif_dagilimi(profil):
    """Profil adindan sinif -> ornek sayisi sozlugu uretir."""
    p = DENGESIZ_PROFILLER[profil]
    dagilim = {}
    for sinif in range(10):
        if sinif in COGUNLUK_SINIFLARI:
            dagilim[sinif] = p["cogunluk"]
        elif sinif in AZINLIK_SINIFLARI:
            dagilim[sinif] = p["azinlik"]
        else:
            dagilim[sinif] = p["orta"]
    return dagilim


def dengesiz_alt_kume(train_data, profil, seed=SEED):
    """Profildeki sayilara gore her siniftan rastgele ornek secer."""
    dagilim = sinif_dagilimi(profil)
    etiketler = train_data.targets.numpy()
    rng = np.random.default_rng(seed)

    secilen = []
    gercek_dagilim = {}
    for sinif in range(10):
        idx = np.where(etiketler == sinif)[0]
        n = min(dagilim[sinif], len(idx))
        secilen.extend(rng.choice(idx, size=n, replace=False).tolist())
        gercek_dagilim[sinif] = n

    rng.shuffle(secilen)
    return Subset(train_data, secilen), gercek_dagilim


def loader_hazirla(train_subset, test_data):
    train_loader = DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
    )
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, pin_memory=True)
    return train_loader, test_loader


# 6) METRIKLER (tek gecislik confusion matrix'ten turetilir)
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
        etiket = ""
        if i in AZINLIK_SINIFLARI:
            etiket = "  <- AZINLIK"
        elif i in COGUNLUK_SINIFLARI:
            etiket = "  <- cogunluk"
        print(f"  {i}   |" + "".join(hucreler) + etiket)
    print("  ^ GERCEK       ( * = 10+ yanlis tahmin,  . = 0 )")


def sinif_tablosu_yazdir(m, dagilim):
    """Sinif bazli Precision / Recall / F1 + o sinifin egitim ornegi sayisi."""
    print(f"\n  {'Sinif':<7}{'Tip':<10}{'Egitim':>8}{'Test':>7}"
          f"{'Precision':>12}{'Recall':>10}{'F1':>10}")
    print("  " + "-" * 64)
    for s in range(10):
        if s in COGUNLUK_SINIFLARI:
            tip = "cogunluk"
        elif s in AZINLIK_SINIFLARI:
            tip = "AZINLIK"
        else:
            tip = "orta"
        print(
            f"  {s:<7}{tip:<10}{dagilim[s]:>8}{m['destek'][s]:>7}"
            f"{m['precision'][s] * 100:>11.2f}%{m['recall'][s] * 100:>9.2f}%"
            f"{m['f1'][s] * 100:>9.2f}%"
        )
    print("  " + "-" * 64)
    print(
        f"  {'MACRO':<17}{sum(dagilim.values()):>8}{m['destek'].sum():>7}"
        f"{m['precision_macro'] * 100:>11.2f}%{m['recall_macro'] * 100:>9.2f}%"
        f"{m['f1_macro'] * 100:>9.2f}%"
    )


def azinlik_karisimi_yazdir(confusion):
    """Azinlik siniflarinin en cok hangi sinifla karistigini ozetler."""
    print("\n  AZINLIK SINIFLARI NEREYE KACIYOR")
    for sinif in AZINLIK_SINIFLARI:
        satir = confusion[sinif]
        toplam = satir.sum()
        yanlis = toplam - satir[sinif]
        # kendi disindaki en buyuk 2 hedef
        sirali = [(j, int(satir[j])) for j in range(10) if j != sinif]
        sirali.sort(key=lambda x: x[1], reverse=True)
        hedefler = ", ".join(f"'{j}'({adet})" for j, adet in sirali[:2] if adet > 0)
        print(
            f"    Sinif {sinif}: {yanlis}/{toplam} tahmin yanlis sinifa gidiyor"
            + (f" -> en cok {hedefler} ile karisiyor" if hedefler else "")
        )


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
        # her epoch'ta olc ve yazdir - azinlik recall'un epoch epoch nasil
        # geride kaldigi burada dogrudan gorulur
        ara = metricler_hesapla(model, test_loader, criterion, device)
        print(
            f"  Epoch {epoch:>2}/{epochs} | Loss: {loss:.4f} | "
            f"Acc: {ara['accuracy'] * 100:.2f}% | "
            f"F1: {ara['f1_macro'] * 100:.2f}% | "
            f"Azinlik Recall: {ara['azinlik_recall'] * 100:.2f}% | "
            f"Cogunluk Recall: {ara['cogunluk_recall'] * 100:.2f}%"
        )

    sure = time.perf_counter() - start
    metrics = metricler_hesapla(model, test_loader, criterion, device)
    return metrics, sure


# 8) CSV KAYIT
def sonuclari_kaydet(sonuclar, dosya=SONUC_DOSYASI):
    fieldnames = [
        "deney", "model", "konfig", "profil", "toplam_ornek",
        "cogunluk_ornek", "orta_ornek", "azinlik_ornek", "dengesizlik_orani",
        "seed", "epochs",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "azinlik_precision",
        "azinlik_recall",
        "azinlik_f1",
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
    fieldnames = [
        "deney", "profil", "sinif", "sinif_tipi", "egitim_ornegi",
        "test_destek", "precision", "recall", "f1",
    ]
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(satirlar)


# 9) DENEY LISTESI
# 4 konfig x 3 profil = 12 kombinasyon
KONFIGLER = [
    {"ad": "MLP_d0", "tur": "MLP", "hidden": [256, 128], "dropout": 0.0,
     "not": "MLP 2 katman, dropout yok"},
    {"ad": "MLP_d03", "tur": "MLP", "hidden": [256, 128], "dropout": 0.3,
     "not": "MLP 2 katman + dropout 0.3"},
    {"ad": "CNN_d0", "tur": "CNN", "conv": (16, 32), "dropout": 0.0,
     "not": "CNN 2 conv katman, dropout yok"},
    {"ad": "CNN_d03", "tur": "CNN", "conv": (16, 32), "dropout": 0.3,
     "not": "CNN 2 conv katman + dropout 0.3"},
]


def tum_deneyler():
    deneyler = []
    for konfig in KONFIGLER:
        for profil in DENGESIZ_PROFILLER:
            deneyler.append({
                "ad": f"{konfig['ad']}_dengesiz_{profil}",
                "konfig": konfig,
                "profil": profil,
            })
    return deneyler


def sinif_tipi(sinif):
    if sinif in COGUNLUK_SINIFLARI:
        return "cogunluk"
    if sinif in AZINLIK_SINIFLARI:
        return "azinlik"
    return "orta"


# 10) CALISTIR
def main():
    device = get_device()
    seed_ayarla(SEED)
    train_data, test_data = veri_yukle()
    deneyler = tum_deneyler()
    sonuclar = []
    sinif_satirlari = []
    genel_baslangic = time.perf_counter()

    print(f"=== ADIM 2: SINIFLARI DENGESIZ HALE GETIR | Toplam deney: {len(deneyler)} | "
          f"Seed: {SEED} | Epoch: {EPOCHS} | Cihaz: {CIHAZ} ({device}) ===")
    print("Profiller:")
    for ad, p in DENGESIZ_PROFILLER.items():
        toplam = 2 * p["cogunluk"] + 5 * p["orta"] + 3 * p["azinlik"]
        print(
            f"  {ad:<8} | 0-1: {p['cogunluk']:>5} | 2,3,4,7,9: {p['orta']:>4} | "
            f"5,6,8: {p['azinlik']:>4} | toplam: {toplam:>6} | "
            f"oran: {p['cogunluk'] / p['azinlik']:.1f}:1"
        )
    print()

    for i, deney in enumerate(deneyler, 1):
        konfig = deney["konfig"]
        profil = deney["profil"]
        train_subset, dagilim = dengesiz_alt_kume(train_data, profil)
        train_loader, test_loader = loader_hazirla(train_subset, test_data)

        toplam_ornek = sum(dagilim.values())
        cogunluk_ornek = dagilim[COGUNLUK_SINIFLARI[0]]
        orta_ornek = dagilim[ORTA_SINIFLAR[0]]
        azinlik_ornek = dagilim[AZINLIK_SINIFLARI[0]]

        print(
            f"[{i}/{len(deneyler)}] {deney['ad']} | Profil: {profil} | "
            f"Toplam ornek: {toplam_ornek} | Dropout: {konfig['dropout']}"
        )

        model = model_kur(konfig, device)
        metrics, sure = train_model(model, train_loader, test_loader, device)

        acc = metrics["accuracy"] * 100
        azinlik_rec = metrics["azinlik_recall"] * 100

        sonuclar.append({
            "deney": deney["ad"],
            "model": konfig["tur"],
            "konfig": konfig["ad"],
            "profil": profil,
            "toplam_ornek": toplam_ornek,
            "cogunluk_ornek": cogunluk_ornek,
            "orta_ornek": orta_ornek,
            "azinlik_ornek": azinlik_ornek,
            "dengesizlik_orani": round(cogunluk_ornek / azinlik_ornek, 1),
            "seed": SEED,
            "epochs": EPOCHS,
            "accuracy": round(acc, 2),
            "precision_macro": round(metrics["precision_macro"] * 100, 2),
            "recall_macro": round(metrics["recall_macro"] * 100, 2),
            "f1_macro": round(metrics["f1_macro"] * 100, 2),
            "azinlik_precision": round(metrics["azinlik_precision"] * 100, 2),
            "azinlik_recall": round(azinlik_rec, 2),
            "azinlik_f1": round(metrics["azinlik_f1"] * 100, 2),
            "cogunluk_recall": round(metrics["cogunluk_recall"] * 100, 2),
            # Accuracy ile azinlik recall arasindaki ucurum = yaniltici accuracy'nin olcusu
            "acc_azinlik_farki": round(acc - azinlik_rec, 2),
            "loss": round(metrics["loss"], 4),
            "sure_sn": round(sure, 2),
            "cihaz": CIHAZ,
            "notlar": konfig["not"],
        })

        for sinif in range(10):
            sinif_satirlari.append({
                "deney": deney["ad"],
                "profil": profil,
                "sinif": sinif,
                "sinif_tipi": sinif_tipi(sinif),
                "egitim_ornegi": dagilim[sinif],
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
            f"Azinlik Recall: {azinlik_rec:.2f}% | "
            f"Cogunluk Recall: {metrics['cogunluk_recall'] * 100:.2f}% | "
            f"Sure: {sure:.1f} sn"
        )
        print(f"     Accuracy - Azinlik Recall farki: {acc - azinlik_rec:.2f} puan "
              f"(buyudukce accuracy o kadar yaniltici)")
        # PDF: "Confusion Matrix'i her deneyde mutlaka cikti olarak alin"
        matris_yazdir(metrics["confusion"])
        sinif_tablosu_yazdir(metrics, dagilim)
        azinlik_karisimi_yazdir(metrics["confusion"])
        print()

    toplam_sure = time.perf_counter() - genel_baslangic
    print("=" * 92)
    print("ADIM 2 BITTI - DENGESIZ SINIF DENEYLERI")
    print(f"Toplam sure: {toplam_sure:.1f} sn ({toplam_sure / 60:.1f} dk)")
    print(f"Sonuclar        : {SONUC_DOSYASI}")
    print(f"Sinif bazli     : {SINIF_DOSYASI}")
    print(f"Confusion matris: {CONFUSION_DIR}")
    print("=" * 92)
    print(
        f"{'Deney':<26} {'Profil':>8} {'Acc%':>7} {'F1%':>7} "
        f"{'AzRec%':>8} {'CogRec%':>8} {'Fark':>7}"
    )
    print("-" * 92)
    for s in sonuclar:
        print(
            f"{s['deney']:<26} {s['profil']:>8} {s['accuracy']:>7} {s['f1_macro']:>7} "
            f"{s['azinlik_recall']:>8} {s['cogunluk_recall']:>8} {s['acc_azinlik_farki']:>7}"
        )
    print("-" * 92)
    print("Fark = Accuracy - Azinlik Recall. Buyudukce accuracy o kadar yaniltici demektir.")

    # PDF'teki tablo formati: her konfig icin profil profil karsilastirma
    print()
    for konfig in KONFIGLER:
        kendi = [s for s in sonuclar if s["konfig"] == konfig["ad"]]
        if not kendi:
            continue
        print("=" * 74)
        print(f"TABLO - {konfig['ad']} ({konfig['not']})")
        print("=" * 74)
        print(f"{'Veri Sayisi':<28}{'Accuracy':>12}{'F1 (macro)':>14}{'Recall Azinlik':>18}")
        print("-" * 74)
        for s in kendi:
            etiket = f"{s['toplam_ornek']} + dengesiz ({s['profil']})"
            print(
                f"{etiket:<28}{'%' + str(s['accuracy']):>12}"
                f"{'%' + str(s['f1_macro']):>14}{'%' + str(s['azinlik_recall']):>18}"
            )
        print("-" * 74)


if __name__ == "__main__":
    main()
