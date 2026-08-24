"""
GOREV 5 / TESPIT - ADIM 3: AUGMENTATION

A1'in kazananinin uzerine augmentation eklenir. Diger her sey sabit.

TESPITTE AUGMENTATION SINIFLANDIRMADAN FARKLI
Siniflandirmada goruntuyu cevirmek yeterliydi, etiket degismiyordu. Tespitte
etiket goruntunun ICINDE: goruntu cevrilince kutu koordinatlari da cevrilmeli,
olceklenince kutular da olceklenmeli. Bunu yapmayi unutmak hata vermez —
model tutarsiz hedeflerle egitilir ve skor sessizce duser. (Gorev 4'te ayni
tuzak maskeler icin vardi; orada albumentations isi otomatik yapiyordu, burada
donusum elle yazildi: tespit.Augment.)

Profiller:
  hafif : yatay cevirme (%50)
  orta  : yatay cevirme + renk oynamasi + rastgele olcekleme (0.8x - 1.25x)

Dikey cevirme bilerek YOK: VOC dogal fotograflardan olusuyor, bas asagi otobus
diye bir sey yok. Egitim dagilimini test dagilimindan uzaklastirmak
augmentation degil gurultudur.

Sure: ~45 dk (RTX 4060 Laptop)
"""

import ortak
import tespit

# A1'in kazanani buraya tasinir. Ikisi de COCO pretrained; katman sayisi
# A1'de olculen degere gore sabitlenir.
TABAN_KATMAN = 3

KONFIGLER = [
    {
        "ad": "tespit_aug_hafif",
        "asama": "A2 Augmentation",
        "pretrained": "coco",
        "augmentation": "hafif",
        "egitilebilir_katman": TABAN_KATMAN,
        "not": "COCO ft + yatay cevirme",
    },
    {
        "ad": "tespit_aug_orta",
        "asama": "A2 Augmentation",
        "pretrained": "coco",
        "augmentation": "orta",
        "egitilebilir_katman": TABAN_KATMAN,
        "not": "COCO ft + yatay cevirme + renk oynamasi + olcekleme",
    },
]


def main():
    device = ortak.get_device()
    ortak.seed_ayarla()

    gorsel_idx = tespit.gorsel_ornekler_sec()
    sonuclar = []

    print("=" * 82)
    print("GOREV 5 / TESPIT - A2 AUGMENTATION")
    print("=" * 82)
    print("Her profil icin loader yeniden kurulur (augmentation veri setinde,")
    print("modelde degil). val ve test setlerine augmentation UYGULANMAZ.")
    print("=" * 82 + "\n")

    for konfig in KONFIGLER:
        _, loaderlar = tespit.veri_hazirla(konfig["augmentation"])
        sonuclar.append(tespit.deney_calistir(konfig, loaderlar, gorsel_idx, device))

    ortak.ozet_yazdir(sonuclar, "TESPIT - A2 AUGMENTATION",
                      tespit.OZET_SUTUNLARI, "val_map50")
    print(f"\n  Sonuclar : {ortak.TESPIT_SONUC}")


if __name__ == "__main__":
    main()
