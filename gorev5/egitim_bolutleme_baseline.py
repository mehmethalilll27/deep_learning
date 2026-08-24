"""
GOREV 5 / BOLUTLEME - ADIM 1: REFERANS VE BASELINE

Belge madde 2: "Modelin her pikseli dogru sinifa atayip atamadigini gozlemleyin."

Iki olcum:

  1) ref_hazir_voc
     torchvision'in COCO-with-VOC-labels agirliklariyla gelen DeepLabV3.
     VOC train/val goruntulerini gormemis ama VOC siniflarini ogrenmis.
     Egitim suresi 0. Ulasilmasi zor bir tavan.

  2) bolut_baseline
     Ayni mimari, govde ImageNet'ten, ASPP basligi sifirdan. Augmentation yok,
     aux loss yok, 20 epoch, 1.464 goruntu.

VOC BOLUTLEME EGITIM SETI KUCUK: 1.464 goruntu. Gorev 4'teki su veri seti
2.841 goruntuydu ve orada tek sinif vardi; burada 21 sinif var. Yani goruntu
basina dusen "ogrenilecek sey" cok daha fazla, veri ise daha az. Bu, hazir
agirliklarin neden bu kadar belirleyici oldugunu onceden aciklar.

Sure: ~35 dk (RTX 4060 Laptop)
"""

import torch

import bolutleme
import ortak

KONFIGLER = [
    {
        "ad": "ref_hazir_voc",
        "asama": "Referans",
        "pretrained": "coco+voc (egitimsiz)",
        "augmentation": "-",
        "hazir_voc": True,
        "not": "torchvision DeepLabV3 COCO_WITH_VOC_LABELS, egitim yok",
    },
    {
        "ad": "bolut_baseline",
        "asama": "Baseline",
        "pretrained": "imagenet",
        "augmentation": "yok",
        "aux_loss": False,
        "not": "baseline - ImageNet govde, ASPP sifirdan, augmentation yok",
    },
]


def main():
    device = ortak.get_device()
    ortak.seed_ayarla()

    setler, loaderlar = bolutleme.veri_hazirla(None)
    gorsel_idx = bolutleme.gorsel_ornekler_sec()

    print("=" * 82)
    print("GOREV 5 / BOLUTLEME - REFERANS VE BASELINE | PASCAL VOC 2012")
    print("=" * 82)
    print("Model      : DeepLabV3 + ResNet50 (torchvision)")
    print(f"Train      : {len(setler['train'])} goruntu (resmi Segmentation/train)")
    print(f"Validation : {len(setler['val'])} goruntu (resmi val'in yarisi)")
    print(f"Test       : {len(setler['test'])} goruntu (resmi val'in diger yarisi)")
    print(f"Girdi      : {bolutleme.KIRP}x{bolutleme.KIRP} | "
          f"Batch: {bolutleme.BATCH_SIZE} | Epoch: {bolutleme.EPOCHS} | "
          f"LR: {bolutleme.LEARNING_RATE} (SGD + poly)")
    print(f"Sinif      : {ortak.BOLUT_SINIF_SAYISI} (arka plan + "
          f"{ortak.SINIF_SAYISI} nesne) | void=255 olcume katilmaz")
    print("Olcum      : orijinal cozunurlukte (tahmin geri buyutulur)")
    print(f"Cihaz      : {ortak.CIHAZ} ({torch.cuda.get_device_name(0)})")
    print(f"Gorsel ornekleri: {gorsel_idx}")
    print("=" * 82 + "\n")

    sonuclar = [bolutleme.deney_calistir(k, loaderlar, gorsel_idx, device)
                for k in KONFIGLER]

    ortak.ozet_yazdir(sonuclar, "BOLUTLEME - REFERANS VE BASELINE",
                      bolutleme.OZET_SUTUNLARI, "val_miou")

    ref, base = sonuclar
    print()
    print(f"  Hazir VOC modeli (0 sn egitim) : mIoU %{ref['test_miou']:.2f}")
    print(f"  Baseline ({base['sure_sn'] / 60:.0f} dk egitim)          : "
          f"mIoU %{base['test_miou']:.2f}")
    print(f"  Fark: {ref['test_miou'] - base['test_miou']:+.2f} puan")
    print()
    print(f"  Baseline detay | Piksel accuracy %{base['test_piksel_acc']:.2f}  "
          f"<- tek basina yaniltici")
    print(f"                 | mIoU            %{base['test_miou']:.2f}")
    print(f"                 | Fark            "
          f"{base['test_piksel_acc'] - base['test_miou']:.2f} puan")
    print("  Piksellerin buyuk cogunlugu arka plan; accuracy kolay kismi olcuyor.")
    print(f"                 | Arka plansiz mIoU %{base['test_miou_arkaplansiz']:.2f}")
    print(f"                 | Precision %{base['test_precision']:.2f} · "
          f"Recall %{base['test_recall']:.2f} · F1 %{base['test_f1']:.2f}")
    print("=" * 82)
    print(f"\n  Sonuclar : {ortak.BOLUTLEME_SONUC}")
    print(f"  Gorseller: {ortak.GORSEL_DIR}")


if __name__ == "__main__":
    main()
