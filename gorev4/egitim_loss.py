"""
GOREV 4 - ADIM 4 / A1: LOSS FONKSIYONU TARAMASI

Segmentasyonun siniflandirmadan en cok ayrildigi yer burasi. Gorev 1-3'te
CrossEntropy disinda pratikte secenek yoktu; segmentasyonda loss secimi
cogu zaman EN BUYUK tek iyilestirme kaldiraci.

Sebep: BCE her pikseli esit agirlikta sayar. Bir goruntude su orani %2 ise,
model "hepsi kara" dediginde BCE'nin %98'i zaten memnun olur. Model dogru
cevaba degil, kolay cevaba yonelir. IoU ve Dice ise yalnizca hedef bolgeyi
olctugu icin bu tuzagi yakalar.

4 loss:
  BCE       piksel bazli, referans (baseline ile ayni)
  Dice      dogrudan metrigin kendisini optimize eder
  BCE+Dice  ikisinin yarim yarim toplami - pratikte en yaygin tercih
  Focal     kolay pikselleri bastirip zor olanlara odaklanir

Model, epoch, seed, lr, augmentation BASELINE ILE AYNI. Tek degisen loss.

Sure: ~20 dk (RTX 4060 Laptop)
"""

import torch

import ortak
from unet import unet_kur

LOSSLAR = ["BCE", "Dice", "BCE+Dice", "Focal"]


def konfig_uret(loss_adi):
    return {
        "ad": f"loss_{loss_adi.replace('+', '_')}",
        "asama": "A1 Loss",
        "model": "UNet",
        "encoder": "-",
        "pretrained": False,
        "loss": loss_adi,
        "augmentation": "yok",
        "not": f"A1 - loss fonksiyonu: {loss_adi}",
    }


def main():
    device = ortak.get_device()
    ortak.seed_ayarla()

    setler, loaderlar, oranlar, (tr_idx, va_idx, te_idx) = ortak.veri_hazirla(None)
    adlar = ortak.dosya_listesi()
    gorsel_idx = ortak.gorsel_ornekler_sec(oranlar, te_idx)

    print("=" * 78)
    print(f"GOREV 4 / A1 - LOSS TARAMASI | {len(LOSSLAR)} deney")
    print("Model, epoch, seed, lr, augmentation baseline ile AYNI. "
          "Tek degisen: loss.")
    print(f"Train: {len(setler['train'])} | Val: {len(setler['val'])} | "
          f"Test: {len(setler['test'])}")
    print(f"Cihaz: {ortak.CIHAZ} ({torch.cuda.get_device_name(0)})")
    print("=" * 78 + "\n")

    sonuclar = []
    for i, loss_adi in enumerate(LOSSLAR, 1):
        print(f"--- [{i}/{len(LOSSLAR)}] ---")
        sonuclar.append(ortak.deney_calistir(
            konfig_uret(loss_adi), lambda: unet_kur(taban=32),
            loaderlar, adlar, gorsel_idx, device))

    ortak.ozet_yazdir(sonuclar, "A1 LOSS TARAMASI BITTI")

    # BCE ile en iyi loss arasindaki farki ayrica goster
    bce = next((s for s in sonuclar if s["loss"] == "BCE"), None)
    en_iyi = max(sonuclar, key=lambda s: s["val_iou"])
    if bce and en_iyi["deney"] != bce["deney"]:
        print(f"\nBCE -> {en_iyi['loss']}:")
        print(f"  Test IoU  : {bce['test_iou']:.2f}% -> {en_iyi['test_iou']:.2f}%  "
              f"({en_iyi['test_iou'] - bce['test_iou']:+.2f} puan)")
        print(f"  Test Dice : {bce['test_dice']:.2f}% -> {en_iyi['test_dice']:.2f}%  "
              f"({en_iyi['test_dice'] - bce['test_dice']:+.2f} puan)")
        print(f"  Recall    : {bce['test_recall']:.2f}% -> "
              f"{en_iyi['test_recall']:.2f}%  "
              f"({en_iyi['test_recall'] - bce['test_recall']:+.2f} puan)")
        print("\n  Recall'daki degisim onemli: BCE'nin 'kolay cevaba kacma'")
        print("  egilimi en cok recall'u dusurur (su bolgelerini kacirir).")

    print("\nBir sonraki adimda (A2 augmentation) kazanan loss sabitlenecek.")


if __name__ == "__main__":
    main()
