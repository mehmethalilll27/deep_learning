"""
GOREV 4 - ADIM 4 / A2: AUGMENTATION TARAMASI

A1'de loss sabitlendi; simdi veri artirma. Gorev 3'te augmentation overfit'i
17 puandan 8 puana indirmisti, burada da benzer bir etki bekleniyor.

SEGMENTASYONDA AUGMENTATION FARKLI: goruntu donerken MASKE DE AYNI SEKILDE
donmeli. albumentations bunu otomatik yapiyor — image ve mask ayni cagriya
verilir, ayni rastgele parametrelerle donusturulur. Elle yapilsaydi en sik
hata burada olurdu.

UYDU GORUNTUSUNE OZGU BIR AVANTAJ: bu veri setinde dikey cevirme ve 90 derece
dondurme de gecerli. Ustten cekilen goruntude "yukari" diye bir yon yok.
Gorev 3'teki manzara fotograflarinda dikey cevirme YANLIS olurdu (bas asagi
dag diye bir sey yok), MNIST'te yatay cevirme bile felaket olurdu (aynalanmis
2 artik 2 degil). Augmentation secimi veri setine bagli, kopyala-yapistir
edilecek bir liste degil.

4 profil:
  yok     A1 kazananinin tekrari (referans)
  hafif   yatay/dikey cevirme + 90 derece dondurme
  orta    hafif + kaydirma/olcekleme/dondurme + parlaklik-kontrast
  guclu   orta + elastik deformasyon + izgara bozulmasi

Sure: ~20 dk (RTX 4060 Laptop)
"""

import albumentations as A
import torch

import ortak
from unet import unet_kur

# A1 kazananina gore guncellenmeli. Varsayilan "BCE+Dice".
LOSS = "BCE+Dice"

PROFILLER = {
    "yok": None,
    "hafif": [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ],
    "orta": [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                           rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2,
                                   contrast_limit=0.2, p=0.5),
    ],
    "guclu": [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.3,
                           rotate_limit=45, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.3,
                                   contrast_limit=0.3, p=0.7),
        A.ElasticTransform(alpha=1, sigma=50, p=0.3),
        A.GridDistortion(p=0.3),
    ],
}


def konfig_uret(profil):
    return {
        "ad": f"aug_{profil}",
        "asama": "A2 Augmentation",
        "model": "UNet",
        "encoder": "-",
        "pretrained": False,
        "loss": LOSS,
        "augmentation": profil,
        "not": f"A2 - augmentation profili: {profil} (loss sabit: {LOSS})",
    }


def main():
    device = ortak.get_device()
    ortak.seed_ayarla()

    adlar = ortak.dosya_listesi()

    print("=" * 78)
    print(f"GOREV 4 / A2 - AUGMENTATION TARAMASI | {len(PROFILLER)} deney")
    print(f"Loss SABIT: {LOSS} | Tek degisen: augmentation profili")
    print(f"Cihaz: {ortak.CIHAZ} ({torch.cuda.get_device_name(0)})")
    print("=" * 78 + "\n")

    sonuclar = []
    for i, (profil, adimlar) in enumerate(PROFILLER.items(), 1):
        # her profil kendi veri hattini kurar (train'e augmentation, val/test'e YOK)
        setler, loaderlar, oranlar, (tr_idx, va_idx, te_idx) = ortak.veri_hazirla(adimlar)
        gorsel_idx = ortak.gorsel_ornekler_sec(oranlar, te_idx)

        print(f"--- [{i}/{len(PROFILLER)}] ---")
        sonuclar.append(ortak.deney_calistir(
            konfig_uret(profil), lambda: unet_kur(taban=32),
            loaderlar, adlar, gorsel_idx, device))

    ortak.ozet_yazdir(sonuclar, "A2 AUGMENTATION TARAMASI BITTI")

    yok = next((s for s in sonuclar if s["augmentation"] == "yok"), None)
    en_iyi = max(sonuclar, key=lambda s: s["val_iou"])
    if yok:
        print(f"\nOverfit farki: {yok['overfit_farki']:+.2f} -> "
              f"{en_iyi['overfit_farki']:+.2f} puan")
        print(f"Test IoU     : {yok['test_iou']:.2f}% -> {en_iyi['test_iou']:.2f}%  "
              f"({en_iyi['test_iou'] - yok['test_iou']:+.2f} puan)")
    print("\nBir sonraki adimda (A3 encoder) kazanan profil sabitlenecek.")


if __name__ == "__main__":
    main()
