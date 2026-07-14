import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SONUC_DOSYASI = "sonuclar_cnn_gpu.csv"
CIHAZ = "GPU"


def get_device():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU bulunamadi! CUDA destekli PyTorch gerekli.")
    torch.backends.cudnn.benchmark = True
    return torch.device("cuda")


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


def veri_yukle():
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False, pin_memory=True)
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


def train_model(model, train_loader, test_loader, device, epochs=EPOCHS):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        loss = bir_epoch_egit(model, train_loader, criterion, optimizer, device)
        acc = dogruluk_hesapla(model, test_loader, device)
        print(f"  Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}%")

    sure = time.perf_counter() - start
    final_acc = dogruluk_hesapla(model, test_loader, device)
    final_loss = loss_hesapla(model, test_loader, criterion, device)
    return final_acc, final_loss, sure


def sonuclari_kaydet(sonuclar, dosya=SONUC_DOSYASI):
    fieldnames = [
        "deney", "conv_channels", "conv_katman_sayisi", "dropout",
        "accuracy", "loss", "sure_sn", "cihaz", "notlar",
    ]
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sonuclar)


def tum_deneyler():
    ch_2 = (16, 32)
    ch_2_buyuk = (32, 64)
    ch_2_kucuk = (8, 16)
    ch_3 = (16, 32, 64)

    return [
        # 1) Dropout karsilastirmasi (2 conv katman, ayni kanal)
        {"ad": "CNN_dropout_0", "conv_channels": ch_2, "dropout": 0.0, "not": "Dropout yok - baz"},
        {"ad": "CNN_dropout_0.1", "conv_channels": ch_2, "dropout": 0.1, "not": "Dusuk dropout"},
        {"ad": "CNN_dropout_0.3", "conv_channels": ch_2, "dropout": 0.3, "not": "Orta dropout"},
        {"ad": "CNN_dropout_0.5", "conv_channels": ch_2, "dropout": 0.5, "not": "Yuksek dropout"},

        # 2) Kanal sayisi karsilastirmasi (dropout yok)
        {"ad": "CNN_kanal_8_16", "conv_channels": ch_2_kucuk, "dropout": 0.0, "not": "Az kanal"},
        {"ad": "CNN_kanal_16_32", "conv_channels": ch_2, "dropout": 0.0, "not": "Orta kanal"},
        {"ad": "CNN_kanal_32_64", "conv_channels": ch_2_buyuk, "dropout": 0.0, "not": "Cok kanal"},

        # 3) Conv katman sayisi karsilastirmasi (dropout yok)
        {"ad": "CNN_2conv", "conv_channels": ch_2, "dropout": 0.0, "not": "2 conv katman"},
        {"ad": "CNN_3conv", "conv_channels": ch_3, "dropout": 0.0, "not": "3 conv katman"},

        # 4) Kanal + dropout birlikte
        {"ad": "CNN_16_32_d0.3", "conv_channels": ch_2, "dropout": 0.3, "not": "Orta kanal + orta dropout"},
        {"ad": "CNN_32_64_d0.3", "conv_channels": ch_2_buyuk, "dropout": 0.3, "not": "Cok kanal + orta dropout"},
        {"ad": "CNN_32_64_d0.5", "conv_channels": ch_2_buyuk, "dropout": 0.5, "not": "Cok kanal + yuksek dropout"},

        # 5) Derinlik + dropout (MLP'deki gibi artan dropout)
        {"ad": "CNN_2conv_d0.1", "conv_channels": ch_2, "dropout": 0.1, "not": "2 conv + d0.1"},
        {"ad": "CNN_3conv_d0.3", "conv_channels": ch_3, "dropout": 0.3, "not": "3 conv + d0.3"},
        {"ad": "CNN_3conv_d0.5", "conv_channels": ch_3, "dropout": 0.5, "not": "3 conv + d0.5"},
    ]


def main():
    device = get_device()
    train_loader, test_loader = veri_yukle()
    deneyler = tum_deneyler()
    sonuclar = []
    genel_baslangic = time.perf_counter()

    print(f"=== MNIST CNN Deneyleri | Toplam: {len(deneyler)} | Cihaz: {CIHAZ} ({device}) ===\n")

    for i, deney in enumerate(deneyler, 1):
        print(
            f"[{i}/{len(deneyler)}] {deney['ad']} | "
            f"Kanallar: {deney['conv_channels']} | Dropout: {deney['dropout']}"
        )
        model = CNN(deney["conv_channels"], dropout=deney["dropout"]).to(device)
        acc, loss, sure = train_model(model, train_loader, test_loader, device)

        sonuc = {
            "deney": deney["ad"],
            "conv_channels": str(deney["conv_channels"]),
            "conv_katman_sayisi": len(deney["conv_channels"]),
            "dropout": deney["dropout"],
            "accuracy": round(acc * 100, 2),
            "loss": round(loss, 4),
            "sure_sn": round(sure, 2),
            "cihaz": CIHAZ,
            "notlar": deney["not"],
        }
        sonuclar.append(sonuc)
        sonuclari_kaydet(sonuclar)  # her deneyden sonra kaydet

        print(
            f"  -> Accuracy: {acc * 100:.2f}% | Loss: {loss:.4f} | "
            f"Sure: {sure:.1f} sn\n"
        )

    toplam_sure = time.perf_counter() - genel_baslangic
    print("=" * 55)
    print("TUM CNN DENEYLERI BITTI")
    print(f"Toplam sure: {toplam_sure:.1f} sn ({toplam_sure / 60:.1f} dk)")
    print(f"Sonuclar: {SONUC_DOSYASI}")
    print("=" * 55)
    print(f"{'Deney':<22} {'Acc%':>7} {'Loss':>8} {'Sure':>7}")
    print("-" * 55)
    for s in sonuclar:
        print(f"{s['deney']:<22} {s['accuracy']:>7} {s['loss']:>8} {s['sure_sn']:>7}")


if __name__ == "__main__":
    main()
