"""
GOREV 4 - ADIM 4 / A4: SEGMENTASYON MIMARILERI

Loss, augmentation ve encoder sabitlendi; son degisken mimarinin kendisi.
Hepsi ayni encoder'i kullaniyor, yalnizca DECODER tarafi farkli.

DORT MIMARI, DORT FARKLI FIKIR:

  Unet          Klasik. Encoder'in her seviyesi decoder'in ayni seviyesine
                skip baglantiyla tasinir. Basit ve saglam.

  UnetPlusPlus  Skip baglantilari YOGUNLASTIRIR. Encoder ile decoder arasina
                ara evrisim katmanlari koyar, boylece farkli olceklerdeki
                bilgi kademeli birlestirilir. Daha agir, sinirlarda daha keskin.

  DeepLabV3Plus Skip yerine ATROUS (delikli) konvolusyon kullanir. Filtreye
                bosluk koyarak alici alani buyutur — cozunurluk kaybetmeden
                genis baglam gorur. Buyuk, butunlukluk nesnelerde guclu.

  FPN           Ozellik piramidi. Her olcekten ayri tahmin uretip birlestirir.
                Farkli buyuklukteki nesnelerin ayni anda bulundugu sahnelerde iyi.

Bu veri seti icin beklenti: su kutleleri genelde BUYUK ve BUTUNLUK bolgeler,
ince yapi az. Bu DeepLabV3+ ve FPN'in lehine, UnetPlusPlus'in ince sinir
avantajinin ise pek ise yaramayacagi anlamina gelir. Olcelim.

Sure: ~30 dk (RTX 4060 Laptop)
"""

import segmentation_models_pytorch as smp
import torch

import ortak
from egitim_augmentation import PROFILLER

# A1 ve A2 olcumlerine gore secildi (gerekce egitim_encoder.py'de).
# ENCODER: A3'te efficientnet-b0 (%81,80) ile resnet34 (%81,55) arasindaki
# 0,25 puanlik fark 2,33 puanlik gurultu tabaninin icinde — skor karar
# veremiyor. efficientnet-b0 secildi cunku ayni skoru 3,9 KAT AZ parametreyle
# (6,25M vs 24,44M) ve daha kisa surede (362 sn vs 378 sn) veriyor.
LOSS = "BCE+Dice"
AUGMENT_PROFIL = "hafif"
ENCODER = "efficientnet-b0"
PRETRAINED = True
LR = 0.0001

MIMARILER = {
    "Unet": smp.Unet,
    "UnetPlusPlus": smp.UnetPlusPlus,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
    "FPN": smp.FPN,
}


def model_kurucu(mimari_adi):
    def kur():
        return MIMARILER[mimari_adi](
            encoder_name=ENCODER,
            encoder_weights="imagenet" if PRETRAINED else None,
            in_channels=3,
            classes=1,
        )
    return kur


def konfig_uret(mimari_adi):
    return {
        "ad": f"mim_{mimari_adi}",
        "asama": "A4 Mimari",
        "model": mimari_adi,
        "encoder": ENCODER,
        "pretrained": PRETRAINED,
        "loss": LOSS,
        "augmentation": AUGMENT_PROFIL,
        "lr": LR,
        "not": f"A4 - {mimari_adi} decoder ({ENCODER} encoder sabit)",
    }


def main():
    device = ortak.get_device()
    ortak.seed_ayarla()

    adlar = ortak.dosya_listesi()
    setler, loaderlar, oranlar, (tr_idx, va_idx, te_idx) = ortak.veri_hazirla(
        PROFILLER[AUGMENT_PROFIL])
    gorsel_idx = ortak.gorsel_ornekler_sec(oranlar, te_idx)

    print("=" * 78)
    print(f"GOREV 4 / A4 - SEGMENTASYON MIMARILERI | {len(MIMARILER)} deney")
    print(f"Encoder SABIT: {ENCODER} (pretrained: {PRETRAINED}) | "
          f"Loss SABIT: {LOSS} | Aug SABIT: {AUGMENT_PROFIL}")
    print("Tek degisen: decoder mimarisi")
    print(f"Cihaz: {ortak.CIHAZ} ({torch.cuda.get_device_name(0)})")
    print("=" * 78 + "\n")

    sonuclar = []
    for i, mimari_adi in enumerate(MIMARILER, 1):
        print(f"--- [{i}/{len(MIMARILER)}] ---")
        sonuclar.append(ortak.deney_calistir(
            konfig_uret(mimari_adi), model_kurucu(mimari_adi),
            loaderlar, adlar, gorsel_idx, device))

    ortak.ozet_yazdir(sonuclar, "A4 MIMARI DENEYLERI BITTI")

    print(f"\n{'Mimari':<18}{'Parametre':>12}{'Test IoU':>11}{'Test Dice':>12}"
          f"{'Sure (sn)':>11}")
    print("-" * 64)
    for s in sorted(sonuclar, key=lambda x: -x["test_iou"]):
        print(f"{s['model']:<18}{s['parametre']:>12}{s['test_iou']:>11.2f}"
              f"{s['test_dice']:>12.2f}{s['sure_sn']:>11.1f}")
    print("-" * 64)
    print("\nParametre ve sure sutunlarina da bakin: en yuksek skor her zaman")
    print("en iyi tercih degil, maliyet/kazanc dengesi onemli.")


if __name__ == "__main__":
    main()
