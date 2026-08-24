"""
GOREV 5 / BOLUTLEME - ADIM 3: EGITIM DUZENI

A1'in kazanan augmentation profili sabitlenir; bu adimda modelin kendisi degil
NASIL EGITILDIGI degisir. Uc soru:

  bolut_scratch    : ImageNet agirliklari olmasaydi ne olurdu?
      Gorev 3'te ayni ResNet18 sifirdan %79.20, pretrained %86.17 vermisti;
      Gorev 4'te sifirdan resnet34 (24.4M param) el yazmasi U-Net'in (7.8M)
      altinda kalmisti. 1.464 goruntude bu farkin cok daha buyuk olmasi
      bekleniyor — ama beklemek olcmek degil.

  bolut_aux        : yardimci (auxiliary) kayip ise yariyor mu?
      DeepLabV3'un ResNet'in 3. blogundan cikan ikinci bir siniflandirici
      basligi var. Kaybi 0.4 agirlikla ana kayba eklenir. Amaci gradyani
      govdenin ortasina daha dogrudan ulastirmak. Kisa egitimlerde etkisinin
      olculmesi gerek; torchvision referansi uzun egitimler icin acik birakir.

  bolut_sabit_lr   : poly cizelgesi ne kadar katki veriyor?
      Diger iki deney poly lr kullaniyor (lr * (1-adim/toplam)^0.9). Bu deney
      lr'yi sabit birakir. Cizelgeyi "zaten herkes kullaniyor" diye eklemek ile
      katkisini olcmek ayri seyler.

Sure: ~50 dk (RTX 4060 Laptop)
"""

import bolutleme
import ortak

# A1'in kazanani. Olculen degere gore guncellenir.
TABAN_AUG = "orta"

KONFIGLER = [
    {
        "ad": "bolut_scratch",
        "asama": "A2 Egitim duzeni",
        "pretrained": "yok",
        "augmentation": TABAN_AUG,
        "not": "hazir agirlik yok - govde de sifirdan",
    },
    {
        "ad": "bolut_aux",
        "asama": "A2 Egitim duzeni",
        "pretrained": "imagenet",
        "augmentation": TABAN_AUG,
        "aux_loss": True,
        "not": "yardimci kayip acik (agirlik 0.4)",
    },
    {
        "ad": "bolut_sabit_lr",
        "asama": "A2 Egitim duzeni",
        "pretrained": "imagenet",
        "augmentation": TABAN_AUG,
        "cizelge": "sabit",
        "not": "poly cizelge yerine sabit learning rate",
    },
]


def main():
    device = ortak.get_device()
    ortak.seed_ayarla()

    gorsel_idx = bolutleme.gorsel_ornekler_sec()
    _, loaderlar = bolutleme.veri_hazirla(TABAN_AUG)

    print("=" * 82)
    print("GOREV 5 / BOLUTLEME - A2 EGITIM DUZENI")
    print("=" * 82)
    print(f"Sabit  : augmentation '{TABAN_AUG}', mimari, epoch "
          f"({bolutleme.EPOCHS}), batch, seed")
    print("Degisen: baslangic agirliklari / yardimci kayip / lr cizelgesi")
    print("=" * 82 + "\n")

    sonuclar = [bolutleme.deney_calistir(k, loaderlar, gorsel_idx, device)
                for k in KONFIGLER]

    ortak.ozet_yazdir(sonuclar, "BOLUTLEME - A2 EGITIM DUZENI",
                      bolutleme.OZET_SUTUNLARI, "val_miou")
    print(f"\n  Sonuclar : {ortak.BOLUTLEME_SONUC}")


if __name__ == "__main__":
    main()
