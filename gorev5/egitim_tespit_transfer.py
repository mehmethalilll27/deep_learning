"""
GOREV 5 / TESPIT - ADIM 2: TRANSFER LEARNING

Belge madde 1: "Sonuclar dusukse farkli seyler deneyerek skoru yukseltmeye
calisin."

Baseline'da yalnizca ResNet50 govdesi hazir agirliklarla geliyordu; FPN, RPN ve
ROI basligi sifirdan basliyordu. Burada TUM dedektor COCO agirliklariyla
baslatilir, yalnizca son siniflandirici katman 21 cikisli yenisiyle degistirilir.

Gorev 3'te transfer learning diger tum cabalarin toplamindan fazla getirmisti
(+8.30 puan / toplam +13.33). Ayni kaldiracin tespitte ne kadar ise yaradigini
olcuyoruz. Beklenti daha da guclu olmasi: siniflandirmada yalnizca govde
tasiniyordu, burada RPN ve ROI basligi da tasiniyor — yani "nesne nerede"
bilgisi de hazir geliyor.

Iki deney:
  tespit_coco_ft         : COCO agirliklari, govdenin son 3 blogu egitilebilir
  tespit_coco_ft_katman5 : ayni ama govdenin TAMAMI egitilebilir

Ikincisi daha fazla kapasite acar. 1.500 goruntuluk bir egitim kumesinde bunun
faydali mi zararli mi oldugu onceden belli degil — olculmesi gerek.

Sure: ~45 dk (RTX 4060 Laptop)
"""

import torch

import ortak
import tespit

KONFIGLER = [
    {
        "ad": "tespit_coco_ft",
        "asama": "A1 Transfer",
        "pretrained": "coco",
        "augmentation": "yok",
        "egitilebilir_katman": 3,
        "not": "COCO pretrained dedektor, govdenin son 3 blogu egitilebilir",
    },
    {
        "ad": "tespit_coco_ft_katman5",
        "asama": "A1 Transfer",
        "pretrained": "coco",
        "augmentation": "yok",
        "egitilebilir_katman": 5,
        "not": "COCO pretrained dedektor, govdenin tamami egitilebilir",
    },
]


def main():
    device = ortak.get_device()
    ortak.seed_ayarla()

    setler, loaderlar = tespit.veri_hazirla(None)
    gorsel_idx = tespit.gorsel_ornekler_sec()

    print("=" * 82)
    print("GOREV 5 / TESPIT - A1 TRANSFER LEARNING")
    print("=" * 82)
    print(f"Train / Val / Test : {len(setler['train'])} / {len(setler['val'])} / "
          f"{len(setler['test'])} goruntu")
    print(f"Degisen tek sey    : baslangic agirliklari ve acilan govde katmani")
    print(f"Sabit              : mimari, epoch ({tespit.EPOCHS}), lr, batch, seed")
    print("=" * 82 + "\n")

    sonuclar = [tespit.deney_calistir(k, loaderlar, gorsel_idx, device)
                for k in KONFIGLER]

    ortak.ozet_yazdir(sonuclar, "TESPIT - A1 TRANSFER LEARNING",
                      tespit.OZET_SUTUNLARI, "val_map50")
    print(f"\n  Sonuclar : {ortak.TESPIT_SONUC}")


if __name__ == "__main__":
    main()
