import csv
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Tum kayitlar bu klasore yazilir
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJE_DIR = os.path.dirname(BASE_DIR)
KAYITLAR_DIR = os.path.join(PROJE_DIR, "kayitlar")
SONUC_DIR = os.path.join(KAYITLAR_DIR, "gorev1")
DATA_DIR = os.path.join(KAYITLAR_DIR, "data")
os.makedirs(SONUC_DIR, exist_ok=True)

# 1) AYARLAR
# EPOCHS        -> kac tur egitecegiz
# BATCH_SIZE    -> her adimda kac goruntu
# LEARNING_RATE -> guncelleme adim buyuklugu
# SONUC_DOSYASI -> sonuclari kaydetmek icin dosya adi
# CIHAZ -> cihaz adi (GPU)
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SONUC_DOSYASI = os.path.join(SONUC_DIR, "sonuclar_no_cnn_gpu.csv")
CIHAZ = "GPU"



# 2) CIHAZ SECIMI (GPU)

def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU bulunamadi! CUDA destekli PyTorch gerekli.")
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


# 3) MODEL (MLP = CNN'siz sinir agi)
# Flatten: 28x28 resmi duzlestir -> 784 sayi
# Linear : x @ w + b
# ReLU   : aktivasyon
# Dropout: egitimde rastgele noron kapat
# Cikis  : 10 sinif (rakam 0-9)

class MLP(nn.Module):
    def __init__(self, hidden_layers, dropout=0.0):
        super().__init__()
        layers = [nn.Flatten()]  # [batch, 1, 28, 28] -> [batch, 784]
        prev = 28 * 28

        # gizli katmanlari sirayla ekle
        for size in hidden_layers:
            layers.append(nn.Linear(prev, size))  # x @ w + b
            layers.append(nn.ReLU())  # aktivasyon
            if dropout > 0:
                layers.append(nn.Dropout(dropout))  # egitimde rastgele noron kapat
            prev = size

        # cikis katmani: 10 rakam
        layers.append(nn.Linear(prev, 10))
        self.net = nn.Sequential(*layers)  # sinir agi olustur yani katmanları sirayla dizer

    # ileri besleme (forward_pass)
    def forward(self, x):
        return self.net(x)  # sinir agi uzerinden ileri besleme yapar


# 4) VERI YUKLEME (MNIST)
# XOR yerine gercek rakam goruntuleri

def veri_yukle():
    transform = transforms.ToTensor()  # piksel -> 0-1 tensor

    train_data = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)

    # DataLoader: veriyi batch batch verir
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, pin_memory=True)
    # pin_memory: GPU'ya veri transferi yaparken daha hizli olmasi icin kullanilir
    return train_loader, test_loader


# 5) DOGRULUK (accuracy)
# argmax: 10 skor icinden en yuksek sinifi sec
# XOR'da 0.5 esigi vardi; burada 10 sinif oldugu icin argmax kullanilir

def dogruluk_hesapla(model, loader, device):
    model.eval()  # test modu (dropout kapali)
    dogru, toplam = 0, 0

    with torch.no_grad():  # gradyan hesaplama (sadece olcum)
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            tahminler = model(images).argmax(dim=1)
            dogru += (tahminler == labels).sum().item()
            toplam += labels.size(0)

    model.train()  # tekrar egitim moduna don
    return dogru / toplam


# 6) LOSS OLCME (test seti)

def loss_hesapla(model, loader, criterion, device):
    model.eval()
    toplam_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)  # GPU'ya tasi
            toplam_loss += criterion(model(images), labels).item()  # CrossEntropyLoss: 10 sinif icin loss

    model.train()
    return toplam_loss / len(loader)  # test seti icin ortalama loss


def metricler_hesapla(
    model, loader, criterion, device, num_classes=10, topk=5, eps=1e-12
):
    """
    MNIST (10 sinif) icin:
    - accuracy
    - precision/recall/F1 (macro)
    - top-k accuracy (varsayilan top-5)
    """
    model.eval()
    toplam_loss = 0.0
    dogru = 0
    toplam = 0
    topk_dogru = 0

    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)  # [batch, num_classes]

            toplam_loss += criterion(logits, labels).item()

            preds = logits.argmax(dim=1)
            dogru += (preds == labels).sum().item()
            toplam += labels.size(0)

            # Confusion matrix'i tek geciste topla
            labels_np = labels.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            np.add.at(confusion, (labels_np, preds_np), 1)

            if topk is not None and topk > 1:
                topk_indices = logits.topk(k=topk, dim=1).indices  # [batch, topk]
                topk_dogru += (topk_indices == labels.unsqueeze(1)).any(dim=1).sum().item()

    model.train()

    acc = dogru / toplam
    loss_ort = toplam_loss / len(loader)

    # Macro precision/recall/F1
    tp = np.diag(confusion).astype(np.float64)
    fp = confusion.sum(axis=0).astype(np.float64) - tp
    fn = confusion.sum(axis=1).astype(np.float64) - tp

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    return {
        "accuracy": acc,
        "loss": loss_ort,
        "precision_macro": precision.mean(),
        "recall_macro": recall.mean(),
        "f1_macro": f1.mean(),
        "top5_accuracy": (topk_dogru / toplam) if topk is not None and topk > 1 else acc,
    }


# 7) 1 EPOCH EGITIM
# forward -> loss -> backward -> update
#  paket al → tahmin et → hatayı hesapla → geri yay → ağırlığı güncelle.
def bir_epoch_egit(model, loader, criterion, optimizer, device):  # 1 tur egitim
    toplam_loss = 0.0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                    # 1) eski gradyanlari sifirla
        loss = criterion(model(images), labels)  # 2) forward + loss
        loss.backward()                          # 3) backward (otomatik)
        optimizer.step()                         # 4) w = w - lr * dw

        toplam_loss += loss.item()

    return toplam_loss / len(loader)


# 8) EGITIM DONGUSU
# criterion = loss_function
# optimizer = update_parameters (Adam)

def train_model(model, train_loader, test_loader, device, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()  # 10 sinif icin loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE) # ağırlık güncelleyici
    start = time.perf_counter() # başlangıç zamanı

    for epoch in range(1, epochs + 1):
        # 1) egit (paket al → tahmin et → hatayı hesapla → geri yay → ağırlığı güncelle.)
        loss = bir_epoch_egit(model, train_loader, criterion, optimizer, device)
        # 2) test accuracy olc (test seti icin accuracy olc)
        acc = dogruluk_hesapla(model, test_loader, device)
        print(f"  Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}%")

    sure = time.perf_counter() - start
    metrics = metricler_hesapla(model, test_loader, criterion, device)
    return metrics["accuracy"], metrics["loss"], sure, metrics



# 9) SONUCLARI CSV'YE KAYDET

def sonuclari_kaydet(sonuclar, dosya=SONUC_DOSYASI):
    os.makedirs(os.path.dirname(dosya), exist_ok=True)
    fieldnames = [
        "deney", "katmanlar", "katman_sayisi", "dropout",
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "top5_accuracy",
        "loss",
        "sure_sn",
        "cihaz",
        "notlar",
    ]
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sonuclar)


# 10) DENEY LISTESI
# hidden = gizli katman boyutlari
# ornek: [256, 128] -> 2 gizli katman

def tum_deneyler():
    h1 = [128]
    h2 = [256, 128]
    h4 = [512, 256, 128, 64]
    h6 = [512, 256, 256, 128, 64, 32]

    return [
        # Grup 1: 1 katman + dropout etkisi
        {"ad": "MLP_1_katman_d0", "hidden": h1, "dropout": 0.0, "not": "Grup1 - 1 katman, dropout yok"},
        {"ad": "MLP_1_katman_d0.1", "hidden": h1, "dropout": 0.1, "not": "Grup1 - 1 katman, dropout 0.1"},
        {"ad": "MLP_1_katman_d0.3", "hidden": h1, "dropout": 0.3, "not": "Grup1 - 1 katman, dropout 0.3"},
        {"ad": "MLP_1_katman_d0.5", "hidden": h1, "dropout": 0.5, "not": "Grup1 - 1 katman, dropout 0.5"},

        # Grup 2: dropout yokken katman sayisi etkisi
        {"ad": "MLP_2_katman_d0", "hidden": h2, "dropout": 0.0, "not": "Grup2 - 2 katman, dropout yok"},
        {"ad": "MLP_4_katman_d0", "hidden": h4, "dropout": 0.0, "not": "Grup2 - 4 katman, dropout yok"},
        {"ad": "MLP_6_katman_d0", "hidden": h6, "dropout": 0.0, "not": "Grup2 - 6 katman, dropout yok"},

        # Grup 3: 2 katman + dropout taramasi
        {"ad": "MLP_2_katman_d0.1", "hidden": h2, "dropout": 0.1, "not": "Grup3 - 2 katman, dropout 0.1"},
        {"ad": "MLP_2_katman_d0.3", "hidden": h2, "dropout": 0.3, "not": "Grup3 - 2 katman, dropout 0.3"},
        {"ad": "MLP_2_katman_d0.5", "hidden": h2, "dropout": 0.5, "not": "Grup3 - 2 katman, dropout 0.5"},

        # Grup 4: 4 katman + dropout taramasi
        {"ad": "MLP_4_katman_d0.1", "hidden": h4, "dropout": 0.1, "not": "Grup4 - 4 katman, dropout 0.1"},
        {"ad": "MLP_4_katman_d0.3", "hidden": h4, "dropout": 0.3, "not": "Grup4 - 4 katman, dropout 0.3"},

        # Grup 5: 6 katman + dropout taramasi
        {"ad": "MLP_6_katman_d0.1", "hidden": h6, "dropout": 0.1, "not": "Grup5 - 6 katman, dropout 0.1"},
        {"ad": "MLP_6_katman_d0.3", "hidden": h6, "dropout": 0.3, "not": "Grup5 - 6 katman, dropout 0.3"},
        {"ad": "MLP_6_katman_d0.5", "hidden": h6, "dropout": 0.5, "not": "Grup5 - 6 katman, dropout 0.5"},
    ]



# 11) CALISTIR
# sirayla: cihaz al -> veri yukle -> her deneyi egit -> kaydet

def main():
    device = get_device()
    train_loader, test_loader = veri_yukle()
    deneyler = tum_deneyler()
    sonuclar = []
    genel_baslangic = time.perf_counter()

    print(f"=== MNIST MLP Deneyleri | Toplam: {len(deneyler)} | Cihaz: {CIHAZ} ({device}) ===\n")

    for i, deney in enumerate(deneyler, 1):
        print(
            f"[{i}/{len(deneyler)}] {deney['ad']} | "
            f"Katmanlar: {deney['hidden']} | Dropout: {deney['dropout']}"
        )

        # modeli kur (GPU'ya tasi)
        model = MLP(deney["hidden"], dropout=deney["dropout"]).to(device)

        # egit
        acc, loss, sure, metrics = train_model(model, train_loader, test_loader, device)

        # sonuclari topla
        sonuc = {
            "deney": deney["ad"],
            "katmanlar": str(deney["hidden"]),
            "katman_sayisi": len(deney["hidden"]),
            "dropout": deney["dropout"],
            "accuracy": round(acc * 100, 2),
            "precision_macro": round(metrics["precision_macro"] * 100, 2),
            "recall_macro": round(metrics["recall_macro"] * 100, 2),
            "f1_macro": round(metrics["f1_macro"] * 100, 2),
            "top5_accuracy": round(metrics["top5_accuracy"] * 100, 2),
            "loss": round(loss, 4),
            "sure_sn": round(sure, 2),
            "cihaz": CIHAZ,
            "notlar": deney["not"],
        }
        sonuclar.append(sonuc)
        sonuclari_kaydet(sonuclar)

        print(
            f"  -> Accuracy: {acc * 100:.2f}% | Loss: {loss:.4f} | "
            f"Sure: {sure:.1f} sn\n"
        )

    toplam_sure = time.perf_counter() - genel_baslangic
    print("=" * 55)
    print("TUM MLP DENEYLERI BITTI")
    print(f"Toplam sure: {toplam_sure:.1f} sn ({toplam_sure / 60:.1f} dk)")
    print(f"Sonuclar: {SONUC_DOSYASI}")
    print("=" * 55)
    print(f"{'Deney':<24} {'Acc%':>7} {'Loss':>8} {'Sure':>7}")
    print("-" * 55)
    for s in sonuclar:
        print(f"{s['deney']:<24} {s['accuracy']:>7} {s['loss']:>8} {s['sure_sn']:>7}")


if __name__ == "__main__":
    main()
