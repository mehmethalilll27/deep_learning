"""
GOREV 5 / TESPIT - ADIM 1: REFERANS VE BASELINE

Belge madde 1: "Modelin goruntudeki nesneleri bounding box ile dogru tespit
edip edemedigini gozlemleyin."

Bu script iki olcum yapar:

  1) ref_coco_egitimsiz
     COCO uzerinde egitilmis hazir Faster R-CNN, VOC'a HIC dokunmadan.
     COCO'nun 80 sinifi VOC'un 20 sinifini kapsiyor, ciktilar esleniyor.
     "Egitime baslamadan once neredeyiz" sorusunun cevabi. Egitim suresi 0.

  2) tespit_baseline
     Ayni mimari ama yalnizca ImageNet govdesiyle; FPN, RPN ve ROI basligi
     sifirdan. Augmentation yok, 6 epoch.

Baseline bilerek zayif kuruldu (Gorev 3 ve 4'teki ayni gerekce): amac iyi bir
ilk skor almak degil, sonraki adimlarin neyi duzelttigini olculebilir kilmak.

Ikisinin yan yana durmasi Gorev 5'in en ogretici karsilastirmasi: hazir model
sifir saniye egitimle, sifirdan egitilen modelin yarim saatte ulastigi yerin
neresinde?

Sure: ~25 dk (RTX 4060 Laptop)
"""

import torch

import ortak
import tespit

KONFIGLER = [
    {
        "ad": "ref_coco_egitimsiz",
        "asama": "Referans",
        "pretrained": "coco (egitimsiz)",
        "augmentation": "-",
        "egitimsiz": True,
        "not": "COCO agirliklari, VOC'ta hic egitim yok, etiketler eslendi",
    },
    {
        "ad": "tespit_baseline",
        "asama": "Baseline",
        "pretrained": "imagenet",
        "augmentation": "yok",
        "not": "baseline - yalnizca ImageNet govdesi, FPN/RPN/ROI sifirdan",
    },
]


def main():
    device = ortak.get_device()
    ortak.seed_ayarla()

    setler, loaderlar = tespit.veri_hazirla(None)
    gorsel_idx = tespit.gorsel_ornekler_sec()

    print("=" * 82)
    print("GOREV 5 / TESPIT - REFERANS VE BASELINE | PASCAL VOC 2012")
    print("=" * 82)
    print(f"Model      : Faster R-CNN + ResNet50-FPN (torchvision)")
    print(f"Train      : {len(setler['train'])} goruntu "
          f"(resmi train listesinden alt kume)")
    print(f"Validation : {len(setler['val'])} goruntu (resmi val'in yarisi)")
    print(f"Test       : {len(setler['test'])} goruntu (resmi val'in diger yarisi)")
    print(f"Girdi      : kisa kenar {tespit.MIN_BOYUT} px | "
          f"Batch: {tespit.BATCH_SIZE} | Epoch: {tespit.EPOCHS} | "
          f"LR: {tespit.LEARNING_RATE} (SGD)")
    print(f"Sinif      : {ortak.SINIF_SAYISI} nesne kategorisi + arka plan")
    print(f"Cihaz      : {ortak.CIHAZ} ({torch.cuda.get_device_name(0)})")
    print(f"Gorsel ornekleri (nesne sayisina gore siralanmis): {gorsel_idx}")
    print("=" * 82 + "\n")

    sonuclar = [tespit.deney_calistir(k, loaderlar, gorsel_idx, device)
                for k in KONFIGLER]

    ortak.ozet_yazdir(sonuclar, "TESPIT - REFERANS VE BASELINE",
                      tespit.OZET_SUTUNLARI, "val_map50")

    ref, base = sonuclar
    print()
    print(f"  Hazir COCO modeli (0 sn egitim) : mAP@0.5 %{ref['test_map50']:.2f}")
    print(f"  Sifirdan baseline ({base['sure_sn'] / 60:.0f} dk egitim) : "
          f"mAP@0.5 %{base['test_map50']:.2f}")
    print(f"  Fark: {ref['test_map50'] - base['test_map50']:+.2f} puan")
    print()
    print(f"  Baseline detay | Precision %{base['test_precision']:.2f} · "
          f"Recall %{base['test_recall']:.2f} · F1 %{base['test_f1']:.2f}")
    print(f"                 | Dogru bulunan kutularin ortalama IoU'su "
          f"%{base['test_ort_iou']:.2f}")
    print(f"                 | Sinif dogrulugu %{base['test_sinif_dogrulugu']:.2f} "
          f"(nesneyi dogru yerde bulanlarin sinifi da dogru mu)")
    print("=" * 82)
    print(f"\n  Sonuclar : {ortak.TESPIT_SONUC}")
    print(f"  Gorseller: {ortak.GORSEL_DIR}")


if __name__ == "__main__":
    main()
