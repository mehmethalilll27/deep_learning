"""
GOREV 4 - SIFIRDAN U-NET

Belge "U-Net onerilir" diyor. Hazir kutuphane yerine once sifirdan yaziyoruz ki
mimarinin ne yaptigi gorunsun; A3'te hazir encoder'li surumle karsilastirilacak.

U-NET NEDEN BOYLE:
Siniflandirmada ag giderek kuculur ve sonunda tek bir sayi uretir — uzamsal
bilgi kasitli olarak atilir. Segmentasyonda ise cikti girdiyle ayni boyutta
olmali, yani atilan uzamsal bilgi geri kazanilmali.

U-Net bunu iki parcayla cozer:
  ENCODER (inis)  : 256 -> 128 -> 64 -> 32 -> 16, kanal sayisi artar.
                    "Ne var?" sorusunu cevaplar, ama nerede oldugunu kaybeder.
  DECODER (cikis) : 16 -> 32 -> 64 -> 128 -> 256, kanal sayisi azalir.
                    "Nerede?" sorusunu cevaplamaya calisir.
  SKIP BAGLANTI   : Encoder'in her seviyesindeki ozellik haritasi, decoder'in
                    ayni seviyesine DOGRUDAN tasinir ve birlestirilir.

Skip baglantilar U-Net'in can damari. Onlarsiz decoder yalnizca 16x16'lik
sikistirilmis bilgiden 256x256 maske uretmeye calisir ve sinirlar bulanik cikar.
Skip sayesinde decoder, encoder'in yuksek cozunurluklu kenar bilgisine erisir.
"""

import torch
import torch.nn as nn


class CiftKonvolusyon(nn.Module):
    """U-Net'in temel yapi tasi: conv -> BN -> ReLU, iki kez.

    Orijinal 2015 makalesinde BatchNorm yok (o zaman yeni cikmisti), ama
    gunumuzde standart. Egitimi belirgin kararlilastiriyor.
    """

    def __init__(self, girdi_kanal, cikti_kanal):
        super().__init__()
        self.blok = nn.Sequential(
            nn.Conv2d(girdi_kanal, cikti_kanal, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cikti_kanal),
            nn.ReLU(inplace=True),
            nn.Conv2d(cikti_kanal, cikti_kanal, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(cikti_kanal),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.blok(x)


class UNet(nn.Module):
    """Klasik U-Net. taban=32 ile ~7,8 milyon parametre.

    kanallar = (32, 64, 128, 256), darbogaz = 512
    """

    def __init__(self, girdi_kanal=3, cikti_kanal=1, taban=32):
        super().__init__()
        kanallar = [taban, taban * 2, taban * 4, taban * 8]

        # ENCODER
        self.inis = nn.ModuleList()
        onceki = girdi_kanal
        for k in kanallar:
            self.inis.append(CiftKonvolusyon(onceki, k))
            onceki = k
        self.havuz = nn.MaxPool2d(2)

        # DARBOGAZ
        self.darbogaz = CiftKonvolusyon(kanallar[-1], kanallar[-1] * 2)

        # DECODER
        self.yukselt = nn.ModuleList()
        self.cikislar = nn.ModuleList()
        for k in reversed(kanallar):
            # ConvTranspose2d ogrenilebilir yukseltme yapar (interpolasyon degil)
            self.yukselt.append(nn.ConvTranspose2d(k * 2, k, kernel_size=2, stride=2))
            # girdi = yukseltilen (k) + skip (k) = 2k
            self.cikislar.append(CiftKonvolusyon(k * 2, k))

        # 1x1 konvolusyon: kanal sayisini sinif sayisina indirir
        self.son = nn.Conv2d(kanallar[0], cikti_kanal, kernel_size=1)

    def forward(self, x):
        skipler = []

        for asagi in self.inis:
            x = asagi(x)
            skipler.append(x)      # havuzlamadan ONCE sakla - yuksek cozunurluk
            x = self.havuz(x)

        x = self.darbogaz(x)

        for i, (yukari, cikis) in enumerate(zip(self.yukselt, self.cikislar)):
            x = yukari(x)
            skip = skipler[-(i + 1)]
            # Girdi 2'nin kuvveti degilse boyutlar 1 piksel kayabilir.
            # IMG_SIZE=256 kullandigimiz icin olmuyor ama guvenlik icin duruyor.
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(x, size=skip.shape[-2:],
                                              mode="bilinear", align_corners=False)
            x = torch.cat([skip, x], dim=1)   # SKIP BAGLANTI
            x = cikis(x)

        return self.son(x)   # logit dondurur, sigmoid loss icinde uygulanir


def unet_kur(taban=32):
    return UNet(girdi_kanal=3, cikti_kanal=1, taban=taban)
