"""
GOREV 5 / BOLUTLEME - ADIM 2: AUGMENTATION

Baseline'in uzerine augmentation eklenir, diger her sey sabit.

Gorev 4'te augmentation HICBIR SEY degistirmemisti — cunku oradaki baseline'da
overfit yoktu (train IoU test IoU'nun altindaydi), sorun ezber degil yetersiz
ogrenmeydi. Burada durum farkli olabilir: 1.464 goruntude 21 sinif ogrenmeye
calisan 40 milyon parametreli bir model ezberlemeye cok daha yatkin. Baseline'in
overfit farki bu adima girmeden once olculdu.

Profiller:
  hafif : yatay cevirme
  orta  : rastgele olcekleme (0.5x - 1.5x) + kirpma + yatay cevirme
  guclu : orta + renk oynamasi

Olcekleme + kirpma bolutlemenin standart augmentation'i (DeepLab makalesi de
bunu kullanir): model ayni nesneyi farkli buyukluklerde gorur, bu da ASPP'nin
ogrenmesi gereken sey ile dogrudan ortusur.

DOLGU DEGERI TUZAGI: kirpma sirasinda goruntunun disinda kalan alan maskede
255 (void) ile doldurulur, 0 (arka plan) ile DEGIL. 0 kullanilsaydi model bos
alani "arka plan" diye ogrenirdi ve bu hata hicbir yerde hata mesaji uretmezdi.

Sure: ~50 dk (RTX 4060 Laptop)
"""

import bolutleme
import ortak

KONFIGLER = [
    {
        "ad": "bolut_aug_hafif",
        "asama": "A1 Augmentation",
        "pretrained": "imagenet",
        "augmentation": "hafif",
        "not": "yatay cevirme",
    },
    {
        "ad": "bolut_aug_orta",
        "asama": "A1 Augmentation",
        "pretrained": "imagenet",
        "augmentation": "orta",
        "not": "olcekleme 0.5x-1.5x + kirpma + yatay cevirme",
    },
    {
        "ad": "bolut_aug_guclu",
        "asama": "A1 Augmentation",
        "pretrained": "imagenet",
        "augmentation": "guclu",
        "not": "olcekleme + kirpma + cevirme + renk oynamasi",
    },
]


def main():
    device = ortak.get_device()
    ortak.seed_ayarla()

    gorsel_idx = bolutleme.gorsel_ornekler_sec()
    sonuclar = []

    print("=" * 82)
    print("GOREV 5 / BOLUTLEME - A1 AUGMENTATION")
    print("=" * 82)
    print("Her profil icin loader yeniden kurulur. val ve test setlerine")
    print("augmentation UYGULANMAZ; olcum her deneyde ayni goruntulerle yapilir.")
    print("=" * 82 + "\n")

    for konfig in KONFIGLER:
        _, loaderlar = bolutleme.veri_hazirla(konfig["augmentation"])
        sonuclar.append(bolutleme.deney_calistir(konfig, loaderlar, gorsel_idx,
                                                 device))

    ortak.ozet_yazdir(sonuclar, "BOLUTLEME - A1 AUGMENTATION",
                      bolutleme.OZET_SUTUNLARI, "val_miou")
    print(f"\n  Sonuclar : {ortak.BOLUTLEME_SONUC}")


if __name__ == "__main__":
    main()
