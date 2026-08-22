"""
GOREV 4 - ADIM 4 / A3: PRETRAINED ENCODER

Gorev 3'te en buyuk kazanc transfer learning'den gelmisti (13,33 puanin 8,30'u).
Segmentasyonda ayni fikir U-Net'in ENCODER yarisina uygulanir.

MANTIK: U-Net'in encoder'i aslinda bir siniflandirma agindan farksiz — goruntuyu
kucultup ozellik cikariyor. O halde ImageNet'te egitilmis bir siniflandirma agi
(ResNet34, EfficientNet-B0...) dogrudan encoder olarak kullanilabilir. Decoder
rastgele baslar ve sifirdan ogrenir, ama encoder isini zaten biliyor.

Ilk satir kasitli: ayni mimari (Unet + resnet34) ama encoder_weights=None.
Boylece "pretrained mi yoksa mimari mi kazandirdi" sorusu ayrisir — Gorev 3'teki
ResNet18 scratch/donuk/finetune uclusunun ayni mantigi.

LEARNING RATE konfige gore degisiyor:
  sifirdan encoder -> 0,001   (ogrenecek her sey var)
  pretrained       -> 0,0001  (yuksek lr hazir filtreleri ilk epoch'ta bozar)

Ilk calistirmada ImageNet agirliklari indirilir (~100 MB toplam, internet gerekir).

Sure: ~25 dk (RTX 4060 Laptop)
"""

import segmentation_models_pytorch as smp
import torch

import ortak
from egitim_augmentation import PROFILLER

# A1 ve A2 olcumlerine gore secildi.
#   LOSS: BCE+Dice - her iki kopyada da BCE'yi recall ve goruntu-basina IoU'da
#         gecti; Dice ve Focal gurultunun cok otesinde kotu.
#   AUGMENT_PROFIL: hafif - A2'de dort profil de 2,33 puanlik gurultu tabaninin
#         icinde kaldi, yani olcum karar veremedi. "hafif" ilke geregi secildi:
#         cevirme/90 derece dondurme uydu goruntusunde fiziksel olarak gecerli,
#         maliyeti sifira yakin, ve pretrained encoder overfit etmeye baslarsa
#         hazir. "guclu"nun elastik deformasyonu su kiyilarini bozma riski tasir.
LOSS = "BCE+Dice"
AUGMENT_PROFIL = "hafif"

KONFIGLER = [
    {"encoder": "resnet34", "pretrained": False, "lr": 0.001,
     "not": "A3 - resnet34 encoder, SIFIRDAN (referans satiri)"},
    {"encoder": "resnet34", "pretrained": True, "lr": 0.0001,
     "not": "A3 - resnet34 encoder, ImageNet pretrained"},
    {"encoder": "efficientnet-b0", "pretrained": True, "lr": 0.0001,
     "not": "A3 - efficientnet-b0 encoder, ImageNet pretrained"},
    {"encoder": "mobilenet_v2", "pretrained": True, "lr": 0.0001,
     "not": "A3 - mobilenet_v2 encoder, ImageNet pretrained"},
]


def model_kurucu(encoder, pretrained):
    def kur():
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=3,
            classes=1,
        )
    return kur


def konfig_uret(k):
    etiket = "pretrained" if k["pretrained"] else "scratch"
    return {
        "ad": f"enc_{k['encoder'].replace('-', '_')}_{etiket}",
        "asama": "A3 Encoder",
        "model": "Unet",
        "encoder": k["encoder"],
        "pretrained": k["pretrained"],
        "loss": LOSS,
        "augmentation": AUGMENT_PROFIL,
        "lr": k["lr"],
        "not": k["not"],
    }


def main():
    device = ortak.get_device()
    ortak.seed_ayarla()

    adlar = ortak.dosya_listesi()
    setler, loaderlar, oranlar, (tr_idx, va_idx, te_idx) = ortak.veri_hazirla(
        PROFILLER[AUGMENT_PROFIL])
    gorsel_idx = ortak.gorsel_ornekler_sec(oranlar, te_idx)

    print("=" * 78)
    print(f"GOREV 4 / A3 - PRETRAINED ENCODER | {len(KONFIGLER)} deney")
    print(f"Loss SABIT: {LOSS} | Augmentation SABIT: {AUGMENT_PROFIL}")
    print(f"Tek degisen: encoder ve pretrained")
    print(f"Cihaz: {ortak.CIHAZ} ({torch.cuda.get_device_name(0)})")
    print("=" * 78 + "\n")

    sonuclar = []
    for i, k in enumerate(KONFIGLER, 1):
        print(f"--- [{i}/{len(KONFIGLER)}] ---")
        sonuclar.append(ortak.deney_calistir(
            konfig_uret(k), model_kurucu(k["encoder"], k["pretrained"]),
            loaderlar, adlar, gorsel_idx, device))

    ortak.ozet_yazdir(sonuclar, "A3 ENCODER DENEYLERI BITTI")

    # Ayni mimarinin iki kullanimi yan yana - transfer learning'in net katkisi
    scratch = next((s for s in sonuclar
                    if s["encoder"] == "resnet34" and not s["pretrained"]), None)
    pre = next((s for s in sonuclar
                if s["encoder"] == "resnet34" and s["pretrained"]), None)
    if scratch and pre:
        print(f"\nAYNI MIMARI (Unet + resnet34), tek fark baslangic agirliklari:")
        print(f"  Sifirdan   : IoU %{scratch['test_iou']:.2f} | "
              f"Dice %{scratch['test_dice']:.2f}")
        print(f"  Pretrained : IoU %{pre['test_iou']:.2f} | "
              f"Dice %{pre['test_dice']:.2f}")
        print(f"  Fark       : {pre['test_iou'] - scratch['test_iou']:+.2f} puan IoU")

    print("\nBir sonraki adimda (A4 mimari) kazanan encoder sabitlenecek.")


if __name__ == "__main__":
    main()
