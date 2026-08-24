"""
GOREV 5 / BOLUM 2 - SEMANTIK BOLUTLEME ORTAK KODU

Model: DeepLabV3 + ResNet50 (torchvision)
  Gorev 4'te U-Net ailesi (Unet / Unet++ / FPN / DeepLabV3+) denenmisti ve
  DeepLabV3+ orada SONUNCU olmustu — su kutlelerinin sinirlari icin skip
  baglantisi kritikti. VOC farkli bir problem: 21 sinif, cok nesneli sahneler,
  genis olcek degisimi. Atrous Spatial Pyramid Pooling (ASPP) tam da bunun
  icin tasarlandi: ayni katmanda farkli genislemelerde konvolusyon uygulayip
  hem kucuk hem buyuk nesneyi ayni anda gorur.
  Ayni modelin iki veri setinde ters sonuc vermesi Gorev 5'in ana gozlemi.

VOC BOLUTLEMESINDE UC TUZAK
  1) void (255) pikseller. Nesne sinirlarindaki belirsiz seride VOC 255 etiketi
     kullanir. Bunlar ne arka plan ne nesne — loss'ta ve metrikte YOK SAYILMALI
     (ignore_index=255). Sayilsaydi model imkansiz bir hedefi ogrenmeye
     calisirdi ve mIoU yapay olarak duserdi.
  2) Arka plan bir sinif. 21 sinifin biri arka plan ve piksellerin ~%74'u o.
     mIoU'yu 21 sinif uzerinden ortalamak literaturun standardi, ama arka plan
     dahil edildiginde skor oldugundan iyi gorunur; arka plansiz mIoU da
     ayrica kaydediliyor.
  3) Maske PNG'leri paletli (mode "P"). np.array(Image.open(...)) dogrudan
     sinif indekslerini verir. RGB'ye cevirmek (convert("RGB")) etiketleri
     renklere donusturur ve sessizce bozar.
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.segmentation import (
    DeepLabV3_ResNet50_Weights,
    deeplabv3_resnet50,
)

import ortak
from ortak import BOLUT_SINIF_SAYISI, VOC_SINIFLAR, VOID

# 1) AYARLAR
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 0.01           # SGD + poly; DeepLab makalesinin degeri
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
TABAN_BOYUT = 384              # olcekleme oncesi referans boyut
KIRP = 320                     # egitimde ag'a giren kare boyutu
SABIT_BOYUT = 384              # epoch sonu hizli olcumde kullanilan boyut
AUX_AGIRLIK = 0.4

TAKIP_ADET = 300               # epoch sonu takip olcumu (hizli olmali)
TRAIN_OLCUM_ADET = 400         # overfit farki icin egitim setinden ornek

ALAN_ADLARI = [
    "deney", "asama", "model", "backbone", "pretrained", "augmentation",
    "aux_loss", "seed", "epochs", "kirp", "batch", "lr", "parametre",
    "val_miou", "val_dice",
    "test_miou", "test_dice", "test_miou_arkaplansiz",
    "test_piksel_acc", "test_sinif_acc",
    "test_precision", "test_recall", "test_f1", "test_fwiou",
    "train_miou", "overfit_farki", "test_loss",
    "en_iyi_val_miou", "en_iyi_epoch", "sure_sn", "cihaz", "notlar",
]

SINIF_ALANLARI = ["deney", "sinif", "iou", "dice", "precision", "recall",
                  "piksel_orani"]


# 2) VERI
def maske_oku(ad):
    """Paletli PNG'yi sinif indeksi dizisine cevirir (0-20 ve 255)."""
    return np.array(Image.open(ortak.maske_yolu(ad)), dtype=np.uint8)


def transform_olustur(profil):
    """Augmentation profilleri.

    ONEMLI: kirpma ve doldurmada maskenin dolgu degeri 255 (void) olmali, 0
    (arka plan) DEGIL. 0 kullanilsaydi goruntunun disinda kalan bos alan
    "burada arka plan var" diye ogretilirdi — sessiz bir etiket bozulmasi.
    """
    normalize = [A.Normalize(mean=ortak.NORM_ORTALAMA, std=ortak.NORM_STD),
                 ToTensorV2()]

    if profil in (None, "yok"):
        return A.Compose([A.Resize(KIRP, KIRP)] + normalize)

    if profil == "hafif":
        return A.Compose([A.Resize(KIRP, KIRP),
                          A.HorizontalFlip(p=0.5)] + normalize)

    olcek_kirp = [
        A.Resize(TABAN_BOYUT, TABAN_BOYUT),
        A.RandomScale(scale_limit=(-0.5, 0.5), p=1.0),
        A.PadIfNeeded(KIRP, KIRP, border_mode=cv2.BORDER_CONSTANT,
                      value=0, mask_value=VOID),
        A.RandomCrop(KIRP, KIRP),
        A.HorizontalFlip(p=0.5),
    ]
    if profil == "orta":
        return A.Compose(olcek_kirp + normalize)
    if profil == "guclu":
        return A.Compose(olcek_kirp + [
            A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4,
                          hue=0.1, p=0.5)] + normalize)

    raise ValueError(f"Bilinmeyen augmentation profili: {profil}")


OLCUM_TRANSFORM = A.Compose([
    A.Resize(SABIT_BOYUT, SABIT_BOYUT),
    A.Normalize(mean=ortak.NORM_ORTALAMA, std=ortak.NORM_STD),
    ToTensorV2(),
])
# "tam" modda da ag'a giren goruntu ayni sekilde hazirlanir; farki yalnizca
# maskenin orijinal cozunurlukte tutulmasi.
TAM_TRANSFORM = OLCUM_TRANSFORM


class VOCBolutlemeVeriSeti(Dataset):
    """mod:
      "egitim" -> augmentation uygulanir, sabit KIRP x KIRP dondurur
      "olcum"  -> augmentation yok, sabit SABIT_BOYUT (hizli, batch'lenebilir)
      "tam"    -> goruntu ag icin yeniden boyutlandirilir ama maske ORIJINAL
                  cozunurlukte doner; tahmin geri buyutulup orijinal maskeyle
                  karsilastirilir. Rapor edilen skorlar bu moddan gelir.
    """

    def __init__(self, adlar, mod="olcum", profil=None):
        self.adlar = list(adlar)
        self.mod = mod
        if mod == "egitim":
            self.transform = transform_olustur(profil)
        else:
            self.transform = OLCUM_TRANSFORM

    def __len__(self):
        return len(self.adlar)

    def __getitem__(self, i):
        ad = self.adlar[i]
        goruntu = np.array(ortak.goruntu_oku(ad))
        maske = maske_oku(ad)

        if self.mod == "tam":
            cikti = self.transform(image=goruntu, mask=maske)
            return cikti["image"], torch.from_numpy(maske.astype(np.int64)), ad

        cikti = self.transform(image=goruntu, mask=maske)
        return cikti["image"], cikti["mask"].long(), ad


def tam_harmanla(parti):
    """'tam' modunda maskeler farkli boyutta; yigilamaz."""
    return tuple(zip(*parti))


def veri_hazirla(augment_profili=None):
    """Bolutleme kumeleri.

    train : ImageSets/Segmentation/train.txt (1.464 goruntu, tamami)
    val   : resmi val listesinin yarisi (karar seti)
    test  : diger yarisi (rapor seti, konfigurasyon basina bir kez olculur)
    """
    train_adlari = ortak.kume_listesi("train", "Segmentation")
    val_adlari, test_adlari = ortak.val_bol(
        ortak.kume_listesi("val", "Segmentation"))

    setler = {
        "train": VOCBolutlemeVeriSeti(train_adlari, "egitim", augment_profili),
        "takip": VOCBolutlemeVeriSeti(
            ortak.alt_kume(val_adlari, TAKIP_ADET), "olcum"),
        "val": VOCBolutlemeVeriSeti(val_adlari, "tam"),
        "test": VOCBolutlemeVeriSeti(test_adlari, "tam"),
        "train_olcum": VOCBolutlemeVeriSeti(
            ortak.alt_kume(train_adlari, TRAIN_OLCUM_ADET), "tam"),
    }

    tam_ayar = {"batch_size": 1, "shuffle": False, "num_workers": ortak.NUM_WORKERS,
                "collate_fn": tam_harmanla, "pin_memory": True}
    loaderlar = {
        "train": DataLoader(setler["train"], batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=ortak.NUM_WORKERS, pin_memory=True,
                            drop_last=True, persistent_workers=True),
        "takip": DataLoader(setler["takip"], batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=ortak.NUM_WORKERS, pin_memory=True),
        "val": DataLoader(setler["val"], **tam_ayar),
        "test": DataLoader(setler["test"], **tam_ayar),
        "train_olcum": DataLoader(setler["train_olcum"], **tam_ayar),
    }
    return setler, loaderlar


# 3) MODEL
def model_kur(pretrained="imagenet", aux_loss=False):
    """DeepLabV3-ResNet50, 21 cikis.

    pretrained="imagenet" : govde ImageNet'ten, ASPP basligi sifirdan
    pretrained="yok"      : her sey sifirdan (1.464 goruntude umutsuz — olculdu)
    """
    if pretrained == "imagenet":
        govde = "IMAGENET1K_V1"
    elif pretrained == "yok":
        govde = None
    else:
        raise ValueError(f"Bilinmeyen pretrained: {pretrained}")

    return deeplabv3_resnet50(weights=None, weights_backbone=govde,
                              num_classes=BOLUT_SINIF_SAYISI, aux_loss=aux_loss)


def hazir_voc_modeli():
    """torchvision'in COCO-with-VOC-labels agirliklariyla gelen DeepLabV3.

    Bu model COCO'nun VOC siniflarini iceren alt kumesinde egitilmis; VOC train
    ve val goruntulerini gormemis. Yani "hic egitmeden nerede duruyoruz"
    referansi olarak adil, ayni zamanda bizim 1.464 goruntuluk egitimimizin
    ulasmasi zor bir tavan.
    """
    return deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)


# 4) METRIKLER
def confusion_guncelle(cm, gercek, tahmin):
    """21x21 karisiklik matrisini gunceller; void (255) pikseller atlanir."""
    gecerli = gercek != VOID
    g = gercek[gecerli].to(torch.int64)
    t = tahmin[gecerli].to(torch.int64)
    cm += torch.bincount(BOLUT_SINIF_SAYISI * g + t,
                         minlength=BOLUT_SINIF_SAYISI ** 2
                         ).reshape(BOLUT_SINIF_SAYISI, BOLUT_SINIF_SAYISI)
    return cm


def cm_metrikleri(cm):
    """Karisiklik matrisinden tum metrikler.

    NOT: cok sinifli bolutlemede sinif basina Dice ile F1 AYNI SEYDIR
    (ikisi de 2TP / (2TP + FP + FN)). Belge ikisini de istedigi icin ikisi de
    yaziliyor ama ayni sayi olduklari bilinerek okunmali. Bagimsiz olan cift
    IoU ile Dice: IoU = TP/(TP+FP+FN), Dice kesisimi iki kez sayar ve her
    zaman IoU'dan buyuktur.
    """
    cm = cm.double()
    tp = torch.diag(cm)
    tahmin_top = cm.sum(0)
    gercek_top = cm.sum(1)
    fp = tahmin_top - tp
    fn = gercek_top - tp

    iou = tp / (tp + fp + fn).clamp(min=1)
    dice = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
    precision = tp / tahmin_top.clamp(min=1)
    recall = tp / gercek_top.clamp(min=1)

    var = gercek_top > 0            # veri setinde hic gorulmeyen sinif ortalamayi bozmasin
    toplam = cm.sum().clamp(min=1)
    frekans = gercek_top / toplam

    return {
        "miou": float(iou[var].mean()),
        "miou_arkaplansiz": float(iou[1:][var[1:]].mean()),
        "dice": float(dice[var].mean()),
        "piksel_acc": float(tp.sum() / toplam),
        "sinif_acc": float(recall[var].mean()),
        "precision": float(precision[var].mean()),
        "recall": float(recall[var].mean()),
        "f1": float(dice[var].mean()),
        "fwiou": float((frekans[var] * iou[var]).sum()),
        "sinif_iou": iou.cpu().numpy(),
        "sinif_dice": dice.cpu().numpy(),
        "sinif_precision": precision.cpu().numpy(),
        "sinif_recall": recall.cpu().numpy(),
        "sinif_frekans": frekans.cpu().numpy(),
    }


@torch.no_grad()
def olc(model, loader, device, criterion=None, tam_cozunurluk=False):
    """tam_cozunurluk=True: tahmin orijinal goruntu boyutuna geri buyutulur.

    Sabit 384x384'te olcmek daha hizli ama maskeyi kucultmek ince yapilari
    (bisiklet teli, sandalye bacagi) yok eder ve skoru YUKSELTIR — kolaylasan
    bir problemi olcmus oluruz. Rapor edilen sayilar bu yuzden orijinal
    cozunurlukte.
    """
    model.eval()
    cm = torch.zeros(BOLUT_SINIF_SAYISI, BOLUT_SINIF_SAYISI,
                     dtype=torch.int64, device=device)
    toplam_loss, adet = 0.0, 0

    for parti in loader:
        if tam_cozunurluk:
            goruntuler, maskeler, _ = parti
            for goruntu, maske in zip(goruntuler, maskeler):
                x = goruntu.unsqueeze(0).to(device, non_blocking=True)
                maske = maske.to(device)
                with torch.amp.autocast("cuda"):
                    logit = model(x)["out"]
                logit = torch.nn.functional.interpolate(
                    logit.float(), size=maske.shape[-2:], mode="bilinear",
                    align_corners=False)
                if criterion is not None:
                    toplam_loss += float(criterion(logit, maske.unsqueeze(0)))
                    adet += 1
                confusion_guncelle(cm, maske, logit.argmax(1)[0])
        else:
            goruntuler, maskeler, _ = parti
            goruntuler = goruntuler.to(device, non_blocking=True)
            maskeler = maskeler.to(device, non_blocking=True)
            with torch.amp.autocast("cuda"):
                logit = model(goruntuler)["out"]
            logit = logit.float()
            if criterion is not None:
                toplam_loss += float(criterion(logit, maskeler))
                adet += 1
            confusion_guncelle(cm, maskeler, logit.argmax(1))

    model.train()
    sonuc = cm_metrikleri(cm)
    sonuc["loss"] = toplam_loss / max(adet, 1)
    return sonuc


# 5) EGITIM
def egit(model, loaderlar, device, epochs=EPOCHS, lr=LEARNING_RATE,
         aux_loss=False, cizelge="poly"):
    criterion = nn.CrossEntropyLoss(ignore_index=VOID)

    # Siniflandirici basligi sifirdan basliyor, govde hazir agirliklarla geliyor.
    # Ayni learning rate ikisine de uygulanirsa baslik cok yavas ogrenir ya da
    # govde bozulur. 10x carpan bu isin standart cozumu.
    gruplar = [
        {"params": [p for p in model.backbone.parameters() if p.requires_grad],
         "lr_carpani": 1.0},
        {"params": list(model.classifier.parameters()), "lr_carpani": 10.0},
    ]
    if aux_loss and model.aux_classifier is not None:
        gruplar.append({"params": list(model.aux_classifier.parameters()),
                        "lr_carpani": 10.0})

    optimizer = torch.optim.SGD(gruplar, lr=lr, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda")

    toplam_adim = epochs * len(loaderlar["train"])
    adim = 0
    gecmis = []
    en_iyi_val, en_iyi_epoch = 0.0, 0
    baslangic = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        toplam = 0.0
        for goruntuler, maskeler, _ in loaderlar["train"]:
            goruntuler = goruntuler.to(device, non_blocking=True)
            maskeler = maskeler.to(device, non_blocking=True)

            if cizelge == "poly":
                ortak.poly_lr(optimizer, lr, adim, toplam_adim)
            elif cizelge == "sabit":
                for grup in optimizer.param_groups:
                    grup["lr"] = lr * grup["lr_carpani"]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                cikti = model(goruntuler)
                kayip = criterion(cikti["out"], maskeler)
                if aux_loss and "aux" in cikti:
                    kayip = kayip + AUX_AGIRLIK * criterion(cikti["aux"], maskeler)
            scaler.scale(kayip).backward()
            scaler.step(optimizer)
            scaler.update()

            toplam += float(kayip.detach())
            adim += 1

        va = olc(model, loaderlar["takip"], device)
        if va["miou"] > en_iyi_val:
            en_iyi_val, en_iyi_epoch = va["miou"], epoch

        gecmis.append({
            "epoch": epoch,
            "train_loss": round(toplam / len(loaderlar["train"]), 4),
            "val_miou": round(va["miou"] * 100, 2),
            "val_dice": round(va["dice"] * 100, 2),
            "val_piksel_acc": round(va["piksel_acc"] * 100, 2),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
        })
        print(f"  Epoch {epoch:>2}/{epochs} | Loss: {toplam / len(loaderlar['train']):.4f} "
              f"| Val mIoU: {va['miou'] * 100:.2f}% | Dice: {va['dice'] * 100:.2f}%")

    return gecmis, time.perf_counter() - baslangic, en_iyi_val, en_iyi_epoch


# 6) GORSEL
def gorsel_ornekler_sec(adet=6, seed=ortak.SEED):
    """Test setinden sinif sayisi artan sirada sabit ornekler."""
    _, test_adlari = ortak.val_bol(ortak.kume_listesi("val", "Segmentation"))
    aday = ortak.alt_kume(test_adlari, 120, seed)
    sayilar = []
    for ad in aday:
        m = maske_oku(ad)
        sinif = np.unique(m)
        sayilar.append((ad, int(((sinif != 0) & (sinif != VOID)).sum())))
    sayilar.sort(key=lambda s: s[1])
    n = len(sayilar)
    konum = [int(round(k * (n - 1) / (adet - 1))) for k in range(adet)]
    return [sayilar[k][0] for k in konum]


@torch.no_grad()
def gorsel_kaydet(model, adlar, device, dosya, baslik):
    """Her ornek icin 4 sutun: goruntu | gercek maske | tahmin | ustuste bindirme."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model.eval()
    n = len(adlar)
    fig, eksenler = plt.subplots(n, 4, figsize=(14, 3.4 * n))
    eksenler = np.atleast_2d(eksenler).reshape(n, 4)

    for satir, ad in enumerate(adlar):
        goruntu = np.array(ortak.goruntu_oku(ad))
        maske = maske_oku(ad)

        girdi = TAM_TRANSFORM(image=goruntu, mask=maske)["image"]
        with torch.amp.autocast("cuda"):
            logit = model(girdi.unsqueeze(0).to(device))["out"]
        logit = torch.nn.functional.interpolate(
            logit.float(), size=maske.shape, mode="bilinear", align_corners=False)
        tahmin = logit.argmax(1)[0].cpu().numpy().astype(np.uint8)

        gecerli = maske != VOID
        kesisim = ((tahmin == maske) & gecerli & (maske != 0)).sum()
        birlesim = (((tahmin != 0) | (maske != 0)) & gecerli).sum()
        iou = kesisim / max(birlesim, 1)

        bindirme = np.array(goruntu, dtype=np.float32) / 255.0
        renkli_tahmin = ortak.maskeyi_renklendir(tahmin)
        nesne = tahmin != 0
        bindirme[nesne] = 0.4 * bindirme[nesne] + 0.6 * renkli_tahmin[nesne]

        siniflar = [VOC_SINIFLAR[s - 1] for s in np.unique(maske)
                    if s not in (0, VOID)]
        for sutun, (veri, alt_baslik) in enumerate([
            (np.array(goruntu) / 255.0, ad),
            (ortak.maskeyi_renklendir(maske), ", ".join(siniflar) or "-"),
            (ortak.maskeyi_renklendir(tahmin), "tahmin"),
            (bindirme, f"nesne IoU %{100 * iou:.1f}"),
        ]):
            ax = eksenler[satir, sutun]
            ax.imshow(veri)
            ax.set_title(alt_baslik, fontsize=8)
            ax.axis("off")

    model.train()
    fig.suptitle(f"{baslik}\nbeyaz = void (olcume katilmaz)  ·  "
                 f"renkler VOC resmi paleti", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(dosya, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return dosya


# 7) DENEY KOSTURUCU
def deney_calistir(konfig, loaderlar, gorsel_idx, device):
    ortak.seed_ayarla()
    criterion = nn.CrossEntropyLoss(ignore_index=VOID)

    if konfig.get("hazir_voc"):
        model = hazir_voc_modeli().to(device)
        gecmis, sure, en_iyi_val, en_iyi_epoch = [], 0.0, 0.0, 0
    else:
        model = model_kur(konfig["pretrained"],
                          aux_loss=konfig.get("aux_loss", False)).to(device)

    parametre = sum(p.numel() for p in model.parameters())
    print(f"[{konfig['ad']}] {konfig['not']}")
    print(f"      Pretrained: {konfig['pretrained']} | Aug: {konfig['augmentation']} "
          f"| Aux: {konfig.get('aux_loss', False)} | Parametre: {parametre:,}"
          .replace(",", "."))

    if not konfig.get("hazir_voc"):
        gecmis, sure, en_iyi_val, en_iyi_epoch = egit(
            model, loaderlar, device,
            epochs=konfig.get("epochs", EPOCHS),
            lr=konfig.get("lr", LEARNING_RATE),
            aux_loss=konfig.get("aux_loss", False),
            cizelge=konfig.get("cizelge", "poly"))

    tr = olc(model, loaderlar["train_olcum"], device, tam_cozunurluk=True)
    va = olc(model, loaderlar["val"], device, tam_cozunurluk=True)
    te = olc(model, loaderlar["test"], device, criterion, tam_cozunurluk=True)

    satir = {
        "deney": konfig["ad"], "asama": konfig["asama"],
        "model": konfig.get("model", "DeepLabV3"),
        "backbone": konfig.get("backbone", "ResNet50"),
        "pretrained": konfig["pretrained"],
        "augmentation": konfig["augmentation"],
        "aux_loss": konfig.get("aux_loss", False),
        "seed": ortak.SEED,
        "epochs": 0 if konfig.get("hazir_voc") else konfig.get("epochs", EPOCHS),
        "kirp": KIRP, "batch": BATCH_SIZE,
        "lr": 0 if konfig.get("hazir_voc") else konfig.get("lr", LEARNING_RATE),
        "parametre": parametre,
        "val_miou": round(va["miou"] * 100, 2),
        "val_dice": round(va["dice"] * 100, 2),
        "test_miou": round(te["miou"] * 100, 2),
        "test_dice": round(te["dice"] * 100, 2),
        "test_miou_arkaplansiz": round(te["miou_arkaplansiz"] * 100, 2),
        "test_piksel_acc": round(te["piksel_acc"] * 100, 2),
        "test_sinif_acc": round(te["sinif_acc"] * 100, 2),
        "test_precision": round(te["precision"] * 100, 2),
        "test_recall": round(te["recall"] * 100, 2),
        "test_f1": round(te["f1"] * 100, 2),
        "test_fwiou": round(te["fwiou"] * 100, 2),
        "train_miou": round(tr["miou"] * 100, 2),
        "overfit_farki": round((tr["miou"] - va["miou"]) * 100, 2),
        "test_loss": round(te["loss"], 4),
        "en_iyi_val_miou": round(en_iyi_val * 100, 2),
        "en_iyi_epoch": en_iyi_epoch,
        "sure_sn": round(sure, 2), "cihaz": ortak.CIHAZ,
        "notlar": konfig["not"],
    }

    ortak.sonuc_ekle([satir], ortak.BOLUTLEME_SONUC, ALAN_ADLARI)
    ortak.sinif_bazli_kaydet(
        [{"deney": konfig["ad"],
          "sinif": "arkaplan" if i == 0 else VOC_SINIFLAR[i - 1],
          "iou": round(float(te["sinif_iou"][i]) * 100, 2),
          "dice": round(float(te["sinif_dice"][i]) * 100, 2),
          "precision": round(float(te["sinif_precision"][i]) * 100, 2),
          "recall": round(float(te["sinif_recall"][i]) * 100, 2),
          "piksel_orani": round(float(te["sinif_frekans"][i]) * 100, 3)}
         for i in range(BOLUT_SINIF_SAYISI)],
        ortak.BOLUTLEME_SINIF_SONUC, SINIF_ALANLARI)

    if gecmis:
        ortak.gecmis_kaydet(gecmis, konfig["ad"])

    gorsel = gorsel_kaydet(
        model, gorsel_idx, device,
        f"{ortak.GORSEL_DIR}/bolut_{konfig['ad']}.png",
        f"{konfig['ad']}  —  Test mIoU %{satir['test_miou']:.2f} / "
        f"Dice %{satir['test_dice']:.2f}")

    print(f"  -> SONUC | Test mIoU: {satir['test_miou']:.2f}% | "
          f"Dice: {satir['test_dice']:.2f}% | "
          f"Piksel Acc: {satir['test_piksel_acc']:.2f}% | "
          f"Overfit: {satir['overfit_farki']:+.2f} | Sure: {sure:.1f} sn")
    print(f"     Gorsel: {gorsel}\n")

    del model
    torch.cuda.empty_cache()
    return satir


OZET_SUTUNLARI = [
    ("Deney", "deney", 30), ("Param", "parametre", 11),
    ("TrainmIoU", "train_miou", 11), ("ValmIoU", "val_miou", 9),
    ("TestmIoU", "test_miou", 10), ("TestDice", "test_dice", 10),
    ("PikselAcc", "test_piksel_acc", 11), ("Overfit", "overfit_farki", 9),
    ("Sure", "sure_sn", 10),
]
