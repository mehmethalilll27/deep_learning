import csv
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# 1) AYARLAR
# EPOCHS        -> kac tur egitecegiz
# BATCH_SIZE    -> her adimda kac goruntu
# LEARNING_RATE -> guncelleme adim buyuklugu
# SONUC_DOSYASI -> sonuclari kaydetmek icin dosya adi
# CIHAZ -> cihaz adi (GPU)
EPOCHS = 5
BATCH_SIZE = 128
LEARNING_RATE = 0.001
SONUC_DOSYASI = "sonuclar_no_cnn_gpu.csv"
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

    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        # 1) egit
        loss = bir_epoch_egit(model, train_loader, criterion, optimizer, device)
        # 2) test accuracy olc
        acc = dogruluk_hesapla(model, test_loader, device)
        print(f"  Epoch {epoch}/{epochs} | Loss: {loss:.4f} | Accuracy: {acc * 100:.2f}%")

    sure = time.perf_counter() - start
    final_acc = dogruluk_hesapla(model, test_loader, device)
    final_loss = loss_hesapla(model, test_loader, criterion, device)

    return final_acc, final_loss, sure


# =========================================================
# 9) SONUCLARI CSV'YE KAYDET
# =========================================================

def sonuclari_kaydet(sonuclar, dosya=SONUC_DOSYASI):
    fieldnames = [
        "deney", "katmanlar", "katman_sayisi", "dropout",
        "accuracy", "loss", "sure_sn", "cihaz", "notlar",
    ]
    with open(dosya, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sonuclar)


# =========================================================
# 10) DENEY LISTESI
# =========================================================
# hidden = gizli katman boyutlari
# ornek: [256, 128] -> 2 gizli katman

def tum_deneyler():
    h1 = [128]
    h2 = [256, 128]
    h4 = [512, 256, 128, 64]
    h6 = [512, 256, 256, 128, 64, 32]

    return [
        # Grup 1: dropout 0.1
        {"ad": "MLP_1_katman", "hidden": h1, "dropout": 0.0, "not": "Grup1 - tek gizli katman"},
        {"ad": "MLP_2_katman", "hidden": h2, "dropout": 0.0, "not": "Grup1 - iki gizli katman"},
        {"ad": "MLP_dropout_0.1", "hidden": h2, "dropout": 0.1, "not": "Grup1 - dropout 0.1"},

        # Grup 2: dropout 0.3
        {"ad": "MLP_1_katman_g2", "hidden": h1, "dropout": 0.0, "not": "Grup2 - tek gizli katman"},
        {"ad": "MLP_2_katman_g2", "hidden": h2, "dropout": 0.0, "not": "Grup2 - iki gizli katman"},
        {"ad": "MLP_dropout_0.3", "hidden": h2, "dropout": 0.3, "not": "Grup2 - dropout 0.3"},

        # Grup 3: dropout 0.5
        {"ad": "MLP_1_katman_g3", "hidden": h1, "dropout": 0.0, "not": "Grup3 - tek gizli katman"},
        {"ad": "MLP_2_katman_g3", "hidden": h2, "dropout": 0.0, "not": "Grup3 - iki gizli katman"},
        {"ad": "MLP_dropout_0.5", "hidden": h2, "dropout": 0.5, "not": "Grup3 - dropout 0.5"},

        # Grup 4: katman sayisi
        {"ad": "MLP_2_katman_derinlik", "hidden": h2, "dropout": 0.0, "not": "Grup4 - 2 katman"},
        {"ad": "MLP_4_katman", "hidden": h4, "dropout": 0.0, "not": "Grup4 - 4 katman"},
        {"ad": "MLP_6_katman", "hidden": h6, "dropout": 0.0, "not": "Grup4 - 6 katman"},

        # Grup 5: katman + dropout
        {"ad": "MLP_2_katman_d0.1", "hidden": h2, "dropout": 0.1, "not": "Grup5 - 2 katman + dropout 0.1"},
        {"ad": "MLP_4_katman_d0.3", "hidden": h4, "dropout": 0.3, "not": "Grup5 - 4 katman + dropout 0.3"},
        {"ad": "MLP_6_katman_d0.5", "hidden": h6, "dropout": 0.5, "not": "Grup5 - 6 katman + dropout 0.5"},
    ]


# =========================================================
# 11) CALISTIR
# =========================================================
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
        acc, loss, sure = train_model(model, train_loader, test_loader, device)

        # sonuclari topla
        sonuc = {
            "deney": deney["ad"],
            "katmanlar": str(deney["hidden"]),
            "katman_sayisi": len(deney["hidden"]),
            "dropout": deney["dropout"],
            "accuracy": round(acc * 100, 2),
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
