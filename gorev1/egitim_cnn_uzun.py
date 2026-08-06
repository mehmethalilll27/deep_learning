import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJE_DIR = os.path.dirname(BASE_DIR)
KAYITLAR_DIR = os.path.join(PROJE_DIR, "kayitlar")
SONUC_DIR = os.path.join(KAYITLAR_DIR, "gorev1")
DATA_DIR = os.path.join(KAYITLAR_DIR, "data")
os.makedirs(SONUC_DIR, exist_ok=True)

BATCH_SIZE = 256
LEARNING_RATE = 0.001
SONUC_DOSYASI = os.path.join(SONUC_DIR, "sonuclar_cnn_uzun.csv")
CIHAZ = "GPU"
CONV_CHANNELS = (32, 64, 128)
FC_SIZE = 256


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU bulunamadi! CUDA destekli PyTorch gerekli.")
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


class CNN_Guclu(nn.Module):
    """Buyuk CNN: (32,64,128) conv, BatchNorm yok, dropout istege bagli."""

    def __init__(self, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        layers = [
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, FC_SIZE),
            nn.ReLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(FC_SIZE, 10))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(self.features(x))


def veri_yukle():
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=512, shuffle=False, pin_memory=True)
    return train_loader, test_loader


def dogruluk_hesapla(model, loader, device):
    model.eval()
    dogru, toplam = 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            dogru += (model(images).argmax(dim=1) == labels).sum().item()
            toplam += labels.size(0)
    model.train()
    return dogru / toplam


def loss_hesapla(model, loader, criterion, device):
    model.eval()
    toplam_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            toplam_loss += criterion(model(images), labels).item()
    model.train()
    return toplam_loss / len(loader)


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


def train_model(model, train_loader, test_loader, device, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    start = time.perf_counter()
    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        train_loss = bir_epoch_egit(model, train_loader, criterion, optimizer, device)
        test_loss = loss_hesapla(model, test_loader, criterion, device)
        acc = dogruluk_hesapla(model, test_loader, device)
        scheduler.step(test_loss)
        best_acc = max(best_acc, acc)
        print(
            f"  Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | "
            f"Accuracy: {acc * 100:.2f}%"
        )

    sure = time.perf_counter() - start
    final_acc = dogruluk_hesapla(model, test_loader, device)
    final_loss = loss_hesapla(model, test_loader, criterion, device)
    return final_acc, final_loss, sure, best_acc


def sonuclari_kaydet(sonuclar, dosya=SONUC_DOSYASI):
    os.makedirs(os.path.dirname(dosya), exist_ok=True)
    fieldnames = [
        "deney", "conv_channels", "epochs", "dropout", "batch_size",
        "accuracy",
        "best_accuracy",
        "loss",
        "sure_sn",
        "cihaz",
        "notlar",
    ]
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sonuclar)


def tum_deneyler():
    return [
        {
            "ad": "CNN_guclu_d0.3_ep30",
            "dropout": 0.3,
            "epochs": 50,
            "not": "Guclu CNN + dropout 0.3 + 50 epoch",
        },
        {
            "ad": "CNN_guclu_d0_ep30",
            "dropout": 0.0,
            "epochs": 50,
            "not": "Guclu CNN + dropout yok + 50 epoch",
        },
        {
            "ad": "CNN_guclu_d0.3_ep50",
            "dropout": 0.3,
            "epochs": 50,
            "not": "Guclu CNN + dropout 0.3 + 50 epoch",
        },
    ]


def main():
    device = get_device()
    train_loader, test_loader = veri_yukle()
    deneyler = tum_deneyler()
    sonuclar = []
    genel_baslangic = time.perf_counter()

    print(f"=== MNIST Guclu CNN | Toplam: {len(deneyler)} | Cihaz: {CIHAZ} ({device}) ===")
    print(f"Conv: {CONV_CHANNELS} | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}\n")

    for i, deney in enumerate(deneyler, 1):
        print(
            f"[{i}/{len(deneyler)}] {deney['ad']} | "
            f"Dropout: {deney['dropout']} | Epoch: {deney['epochs']}"
        )
        model = CNN_Guclu(dropout=deney["dropout"]).to(device)
        acc, loss, sure, best_acc = train_model(
            model, train_loader, test_loader, device, epochs=deney["epochs"]
        )

        sonuc = {
            "deney": deney["ad"],
            "conv_channels": str(CONV_CHANNELS),
            "epochs": deney["epochs"],
            "dropout": deney["dropout"],
            "batch_size": BATCH_SIZE,
            "accuracy": round(acc * 100, 2),
            "best_accuracy": round(best_acc * 100, 2),
            "loss": round(loss, 4),
            "sure_sn": round(sure, 2),
            "cihaz": CIHAZ,
            "notlar": deney["not"],
        }
        sonuclar.append(sonuc)
        sonuclari_kaydet(sonuclar)

        print(
            f"  -> Final Acc: {acc * 100:.2f}% | Best Acc: {best_acc * 100:.2f}% | "
            f"Loss: {loss:.4f} | Sure: {sure:.1f} sn\n"
        )

    toplam_sure = time.perf_counter() - genel_baslangic
    print("=" * 60)
    print("TUM GUCLU CNN DENEYLERI BITTI")
    print(f"Toplam sure: {toplam_sure:.1f} sn ({toplam_sure / 60:.1f} dk)")
    print(f"Sonuclar: {SONUC_DOSYASI}")
    print("=" * 60)
    print(f"{'Deney':<24} {'Acc%':>7} {'Best%':>7} {'Loss':>8} {'Sure':>7}")
    print("-" * 60)
    for s in sonuclar:
        print(
            f"{s['deney']:<24} {s['accuracy']:>7} {s['best_accuracy']:>7} "
            f"{s['loss']:>8} {s['sure_sn']:>7}"
        )


if __name__ == "__main__":
    main()
