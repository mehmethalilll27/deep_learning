"""
GOREV 4 - ADIM 2: BASELINE

Belge madde 2: "Modeli egitin ve ilk sonuclari kaydedin - bu sizin baslangic
skorunuz."

Baseline bilerek zayif kuruldu (Gorev 3'teki ayni gerekce): amac iyi bir ilk
skor almak degil, sonraki adimlarin neyi duzelttigini olculebilir kilmak.

Baseline'da bilerek YOK:
  - augmentation       (A2'de eklenecek)
  - pretrained encoder (A3'te eklenecek)
  - Dice loss          (A1'de denenecek) -> duz BCE kullaniliyor
  - scheduler

BCE'nin neden zayif bir baslangic oldugu onemli: BCE her pikseli esit agirlikta
sayar. Bu veri setinde su orani ortalama %35,8 ama bazi goruntulerde %2. O
goruntulerde model "hepsi kara" dese BCE'yi buyuk olcude memnun eder ve piksel
accuracy %98 cikar — ama IoU sifir olur. A1'de bunu olcecegiz.

Sure: ~5 dk (RTX 4060 Laptop)
"""

import torch

import ortak
from unet import unet_kur

KONFIG = {
    "ad": "baseline_unet",
    "asama": "Baseline",
    "model": "UNet",
    "encoder": "-",
    "pretrained": False,
    "loss": "BCE",
    "augmentation": "yok",
    "not": "baseline - sifirdan U-Net, BCE loss, augmentation yok",
}


def main():
    device = ortak.get_device()
    ortak.seed_ayarla()

    setler, loaderlar, oranlar, (tr_idx, va_idx, te_idx) = ortak.veri_hazirla(None)
    adlar = ortak.dosya_listesi()
    gorsel_idx = ortak.gorsel_ornekler_sec(oranlar, te_idx)

    print("=" * 78)
    print("GOREV 4 - BASELINE | Su Kutlesi Segmentasyonu")
    print("=" * 78)
    print(f"Train      : {len(setler['train'])} goruntu")
    print(f"Validation : {len(setler['val'])} goruntu")
    print(f"Test       : {len(setler['test'])} goruntu")
    print(f"Girdi      : {ortak.IMG_SIZE}x{ortak.IMG_SIZE}x3 | "
          f"Batch: {ortak.BATCH_SIZE} | Epoch: {ortak.EPOCHS} | "
          f"LR: {ortak.LEARNING_RATE}")
    print(f"Su orani   : ortalama %{100 * oranlar.mean():.1f} | "
          f"medyan %{100 * float(sorted(oranlar)[len(oranlar) // 2]):.1f} | "
          f"min %{100 * oranlar.min():.1f} | max %{100 * oranlar.max():.1f}")
    print(f"Cihaz      : {ortak.CIHAZ} ({torch.cuda.get_device_name(0)})")
    print(f"Gorsel ornekleri (test, su oranina gore siralanmis): "
          f"{[adlar[i] for i in gorsel_idx]}")
    print("=" * 78 + "\n")

    sonuc = ortak.deney_calistir(
        KONFIG, lambda: unet_kur(taban=32), loaderlar, adlar, gorsel_idx, device)

    print("=" * 78)
    print("BASELINE SONUCU")
    print("=" * 78)
    print(f"  Test IoU        : {sonuc['test_iou']:.2f}%   <- BASELINE SKORU")
    print(f"  Test Dice       : {sonuc['test_dice']:.2f}%")
    print(f"  Piksel accuracy : {sonuc['test_piksel_acc']:.2f}%   "
          f"<- tek basina yaniltici")
    print(f"  Precision       : {sonuc['test_precision']:.2f}%")
    print(f"  Recall          : {sonuc['test_recall']:.2f}%")
    print(f"  Overfit farki   : {sonuc['overfit_farki']:+.2f} puan (train IoU - val IoU)")
    print()
    print(f"  Goruntu bazli IoU  : {sonuc['test_iou_goruntu']:.2f}%   "
          f"(veri seti geneli: {sonuc['test_iou']:.2f}%)")
    print(f"  Goruntu bazli Dice : {sonuc['test_dice_goruntu']:.2f}%   "
          f"(veri seti geneli: {sonuc['test_dice']:.2f}%)")
    print()
    print(f"  Piksel accuracy ile IoU arasindaki fark "
          f"{sonuc['test_piksel_acc'] - sonuc['test_iou']:.2f} puan.")
    print("  Belgenin 'accuracy tek basina yeterli degil' uyarisinin olculmus hali.")
    print("=" * 78)
    print(f"\n  Sonuclar : {ortak.SONUC_DOSYASI}")
    print(f"  Gorseller: {ortak.GORSEL_DIR}")


if __name__ == "__main__":
    main()
