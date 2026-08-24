"""
GOREV 5 / BOLUM 1 - NESNE TESPITI ORTAK KODU

Model: Faster R-CNN + ResNet50-FPN (torchvision)
  Iki asamali dedektor. Once RPN aday bolge onerir, sonra ROI basligi her
  adayi siniflandirip kutusunu duzeltir. Tek asamali dedektorlerden (SSD,
  RetinaNet, YOLO) daha yavas ama kucuk nesnelerde daha guclu — VOC'ta
  goruntu basina ortalama 2,4 nesne var ve cogu goruntunun kucuk bir
  bolumunu kapliyor.

OLCUM
  mAP@0.5      : VOC'un klasik metrigi. Her sinif icin PR egrisi altindaki
                 alan (tum nokta interpolasyonu), 20 sinif uzerinden ortalama.
  mAP@0.5:0.95 : COCO tarzi; 0.50'den 0.95'e 0.05 adimlarla on esik, ortalama.
                 Kutunun ne kadar "tam oturdugunu" olcer, mAP@0.5 gevsektir.
  Precision / Recall / F1 : skor esigi 0.5 ve IoU esigi 0.5 ile, tum siniflar
                 uzerinden mikro toplama.
  Sinif dogrulugu (accuracy): bir nesneyi DOGRU YERDE bulan tahminlerin kacinin
                 SINIFI da dogru. Tespitte "accuracy" standart bir metrik degil;
                 anlamli olan tek tanim bu — belgenin istedigi siniflandirma
                 performansi olcusu.
"""

import time

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights,
    fasterrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF

import ortak
from ortak import VOC_SINIFLAR, SINIF_SAYISI

# 1) AYARLAR
EPOCHS = 6
BATCH_SIZE = 4
LEARNING_RATE = 0.005          # SGD, torchvision referans degerinin yarisi
                               # (referans batch 8 icin 0.01 kullanir)
MOMENTUM = 0.9
WEIGHT_DECAY = 5e-4
MIN_BOYUT = 512                # kisa kenar; torchvision varsayilani 800
MAX_BOYUT = 853
ISINMA_ADIM = 250

# Alt kume boyutlari (bkz. ortak.alt_kume aciklamasi).
# Olculdu: min_boyut=512 / batch=4 ile 0.274 sn/adim (RTX 4060 Laptop). Batch 8
# denendi ve 8.49 sn/adim'a cikti — 8 GB VRAM yetmiyor, surekli takas oluyor.
EGITIM_ADET = 2000
VAL_ADET = 500                 # epoch sonu takip olcumu (hizli olmali)
TEST_ADET = 1000               # yalnizca bir kez, rapor icin

SKOR_ESIK = 0.05               # mAP hesabina giren minimum tahmin skoru
KARAR_ESIK = 0.5               # precision/recall/F1 icin skor esigi
IOU_ESIK = 0.5

ALAN_ADLARI = [
    "deney", "asama", "model", "backbone", "pretrained", "augmentation",
    "seed", "epochs", "min_boyut", "batch", "lr", "parametre", "egitim_goruntu",
    "val_map50", "val_map",
    "test_map50", "test_map75", "test_map",
    "test_precision", "test_recall", "test_f1", "test_sinif_dogrulugu",
    "test_ort_iou", "test_tp", "test_fp", "test_fn",
    "en_iyi_val_map50", "en_iyi_epoch", "sure_sn", "cihaz", "notlar",
]

SINIF_ALANLARI = ["deney", "sinif", "ap50", "ap50_95", "precision", "recall",
                  "f1", "gercek_nesne", "tahmin"]


# 2) VERI
class VOCTespitVeriSeti(Dataset):
    """VOC XML annotation'larini Faster R-CNN'in bekledigi formata cevirir.

    Egitimde difficult isaretli nesneler ATILIR (VOC'un kendi kurali), olcumde
    ise KORUNUR cunku degerlendirme onlari yok sayarken bilmek zorunda.
    """

    def __init__(self, adlar, augment=None, egitim=True):
        self.adlar = list(adlar)
        self.augment = augment
        self.egitim = egitim

    def __len__(self):
        return len(self.adlar)

    def __getitem__(self, i):
        ad = self.adlar[i]
        goruntu = ortak.goruntu_oku(ad)
        annot = ortak.annot_oku(ad)

        kutular = annot["boxes"]
        etiketler = annot["labels"]
        zorlar = annot["difficult"]

        if self.egitim and len(kutular):
            tut = ~zorlar
            kutular, etiketler, zorlar = kutular[tut], etiketler[tut], zorlar[tut]

        if self.augment is not None:
            goruntu, kutular = self.augment(goruntu, kutular)

        hedef = {
            "boxes": torch.from_numpy(np.asarray(kutular, dtype=np.float32)
                                      .reshape(-1, 4)),
            "labels": torch.from_numpy(np.asarray(etiketler, dtype=np.int64)),
        }
        return TF.to_tensor(goruntu), hedef, ad


def harmanla(parti):
    """Tespitte goruntuler farkli boyutta; yigilamaz, liste olarak gecilir."""
    return tuple(zip(*parti))


class Augment:
    """Tespitte augmentation goruntuyle birlikte KUTULARI da donusturmek zorunda.

    Bolutlemede maske otomatik dondurulebiliyordu (albumentations isi yapiyordu);
    burada kutu koordinatlarini elle guncellemek gerekiyor. Yatay cevirmede
    x -> W - x, olceklemede tum koordinatlar ayni carpanla buyur.

    Profiller:
      hafif : yatay cevirme
      orta  : yatay cevirme + renk oynamasi + rastgele olcekleme
    """

    def __init__(self, profil, seed=ortak.SEED):
        self.profil = profil
        self.rng = np.random.default_rng(seed)
        self.renk = torchvision.transforms.ColorJitter(
            brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05)

    def __call__(self, goruntu, kutular):
        kutular = np.asarray(kutular, dtype=np.float32).reshape(-1, 4).copy()

        if self.rng.random() < 0.5:
            g = goruntu.width
            goruntu = TF.hflip(goruntu)
            if len(kutular):
                x1 = kutular[:, 0].copy()
                kutular[:, 0] = g - kutular[:, 2]
                kutular[:, 2] = g - x1

        if self.profil == "orta":
            if self.rng.random() < 0.5:
                goruntu = self.renk(goruntu)
            if self.rng.random() < 0.5:
                carpan = float(self.rng.uniform(0.8, 1.25))
                yeni = (max(32, int(round(goruntu.height * carpan))),
                        max(32, int(round(goruntu.width * carpan))))
                gercek_x = yeni[1] / goruntu.width
                gercek_y = yeni[0] / goruntu.height
                goruntu = TF.resize(goruntu, yeni)
                if len(kutular):
                    kutular[:, [0, 2]] *= gercek_x
                    kutular[:, [1, 3]] *= gercek_y

        return goruntu, kutular


def veri_hazirla(augment_profili=None, egitim_adet=EGITIM_ADET,
                 val_adet=VAL_ADET, test_adet=TEST_ADET):
    """train / val / test loader'larini kurar.

    train'e augmentation uygulanir, val ve test'e UYGULANMAZ — aksi halde
    olcum her koşuda degisir ve hangi denemenin daha iyi oldugu soylenemezdi
    (Gorev 3 ve 4'teki ayni gerekce).
    """
    train_adlari = ortak.alt_kume(ortak.kume_listesi("train"), egitim_adet)
    val_tum, test_tum = ortak.val_bol(ortak.kume_listesi("val"))
    val_adlari = ortak.alt_kume(val_tum, val_adet)
    test_adlari = ortak.alt_kume(test_tum, test_adet)

    augment = Augment(augment_profili) if augment_profili else None

    setler = {
        "train": VOCTespitVeriSeti(train_adlari, augment, egitim=True),
        "val": VOCTespitVeriSeti(val_adlari, None, egitim=False),
        "test": VOCTespitVeriSeti(test_adlari, None, egitim=False),
    }
    ortak_ayar = {"num_workers": ortak.NUM_WORKERS, "collate_fn": harmanla,
                  "pin_memory": True, "persistent_workers": ortak.NUM_WORKERS > 0}
    loaderlar = {
        "train": DataLoader(setler["train"], batch_size=BATCH_SIZE, shuffle=True,
                            **ortak_ayar),
        "val": DataLoader(setler["val"], batch_size=BATCH_SIZE, shuffle=False,
                          **ortak_ayar),
        "test": DataLoader(setler["test"], batch_size=BATCH_SIZE, shuffle=False,
                           **ortak_ayar),
    }
    return setler, loaderlar


# 3) MODEL
def model_kur(pretrained, egitilebilir_katman=3, min_boyut=MIN_BOYUT):
    """Faster R-CNN ResNet50-FPN.

    pretrained="coco"     : tum agirliklar COCO'dan gelir, son siniflandirici
                            21 cikisli yeniyle degistirilir (transfer learning)
    pretrained="imagenet" : yalnizca ResNet50 govdesi ImageNet'ten; FPN, RPN ve
                            ROI basligi sifirdan. Baseline bu.

    egitilebilir_katman: ResNet'in kac blogu guncellenecek (0-5). torchvision
    varsayilani 3; 5 tum govdeyi acar, daha fazla kapasite ama daha yavas ve
    kucuk veride ezberlemeye daha acik.
    """
    if pretrained == "coco":
        model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            trainable_backbone_layers=egitilebilir_katman,
            min_size=min_boyut, max_size=MAX_BOYUT)
        girdi = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(girdi, SINIF_SAYISI + 1)
    elif pretrained == "imagenet":
        model = fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone="IMAGENET1K_V1",
            num_classes=SINIF_SAYISI + 1,
            trainable_backbone_layers=egitilebilir_katman,
            min_size=min_boyut, max_size=MAX_BOYUT)
    else:
        raise ValueError(f"Bilinmeyen pretrained: {pretrained}")
    return model


def coco_modeli_kur(min_boyut=MIN_BOYUT):
    """Hic egitilmemis, dogrudan COCO agirliklariyla gelen model.

    COCO'nun 80 sinifi VOC'un 20 sinifinin tamamini kapsiyor. Bu model VOC'u
    hic gormeden de kutu uretebilir; ciktisi COCO etiketlerinden VOC
    etiketlerine cevrilir, VOC'ta olmayan siniflar (kus yerine 'zebra' gibi)
    atilir. "Egitmeden once nerede duruyoruz" sorusunun cevabi.
    """
    agirlik = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=agirlik, min_size=min_boyut,
                                    max_size=MAX_BOYUT)
    coco_adlari = agirlik.meta["categories"]

    eslesme = {}
    for i, voc_ad in enumerate(VOC_SINIFLAR):
        coco_ad = ortak.VOC_COCO_ADI.get(voc_ad, voc_ad)
        if coco_ad not in coco_adlari:
            raise RuntimeError(f"COCO'da bulunamadi: {coco_ad}")
        eslesme[coco_adlari.index(coco_ad)] = i + 1
    return model, eslesme


# 4) METRIKLER
def iou_matris(a, b):
    """(N,4) x (M,4) -> (N,M) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    alan_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    alan_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    kesisim = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return kesisim / (alan_a[:, None] + alan_b[None, :] - kesisim + 1e-9)


def _ort(dizi):
    """Sinifsiz (hepsi nan) durumda uyari basmadan nan dondurur.

    Bir sinif olcum kumesinde hic gecmiyorsa AP tanimsizdir; np.nanmean bu
    durumda "Mean of empty slice" uyarisi basar ve cikti kirlenir.
    """
    dizi = np.asarray(dizi, dtype=float)
    gecerli = dizi[~np.isnan(dizi)]
    return float(gecerli.mean()) if len(gecerli) else float("nan")


def _ap(recall, precision):
    """PR egrisi altindaki alan, tum nokta interpolasyonu (VOC2010+ / COCO).

    Eski 11 nokta interpolasyonu (VOC2007) daha kaba ve gunumuzde
    kullanilmiyor; ikisi ayni veri setinde 1-2 puan farkli sonuc verir, bu
    yuzden hangisinin kullanildigi belirtilmeli.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def _sinif_ap(tahminler, gercekler, sinif, iou_esik):
    """Tek sinif icin AP, recall ve precision. difficult nesneler yok sayilir."""
    gt = {}
    npos = 0
    for ad, g in gercekler.items():
        secim = g["labels"] == sinif
        kutular = g["boxes"][secim]
        zorlar = g["difficult"][secim]
        gt[ad] = {"boxes": kutular, "difficult": zorlar,
                  "eslesti": np.zeros(len(kutular), dtype=bool)}
        npos += int((~zorlar).sum())

    if npos == 0:
        return np.nan, np.nan, np.nan, 0, 0

    kayitlar = []
    for ad, t in tahminler.items():
        secim = t["labels"] == sinif
        for kutu, skor in zip(t["boxes"][secim], t["scores"][secim]):
            kayitlar.append((float(skor), ad, kutu))
    if not kayitlar:
        return 0.0, 0.0, 0.0, npos, 0

    kayitlar.sort(key=lambda k: -k[0])
    tp = np.zeros(len(kayitlar), dtype=np.float64)
    fp = np.zeros(len(kayitlar), dtype=np.float64)

    for i, (_, ad, kutu) in enumerate(kayitlar):
        g = gt[ad]
        if len(g["boxes"]) == 0:
            fp[i] = 1
            continue
        iou = iou_matris(kutu[None, :], g["boxes"])[0]
        en_iyi = int(np.argmax(iou))
        if iou[en_iyi] < iou_esik:
            fp[i] = 1
        elif g["difficult"][en_iyi]:
            pass                       # ne dogru ne yanlis — VOC kurali
        elif not g["eslesti"][en_iyi]:
            tp[i] = 1
            g["eslesti"][en_iyi] = True
        else:
            fp[i] = 1                  # ayni nesneyi ikinci kez bulmak hatadir

    tp_k, fp_k = np.cumsum(tp), np.cumsum(fp)
    recall = tp_k / npos
    precision = tp_k / np.maximum(tp_k + fp_k, 1e-9)
    return _ap(recall, precision), float(recall[-1]), float(precision[-1]), npos, len(kayitlar)


def metrikler_hesapla(tahminler, gercekler, karar_esik=KARAR_ESIK, tam=True):
    """Tum metrikleri tek yerde hesaplar.

    tahminler / gercekler: {goruntu_adi: {"boxes","labels","scores"/"difficult"}}

    tam=False yalnizca IoU 0.5'te olcer. Epoch sonu takip olcumu icin kullanilir:
    10 IoU esigi x 20 sinif eslestirmesi her epoch'ta ~20 saniye ekliyordu,
    egitimin kendisinin yanina konunca ciddi bir maliyet.
    """
    esikler = np.arange(0.5, 0.96, 0.05) if tam else np.array([0.5])
    ap_tablo = np.full((SINIF_SAYISI, len(esikler)), np.nan)
    sinif_ozet = []

    for s in range(1, SINIF_SAYISI + 1):
        for j, esik in enumerate(esikler):
            ap, rec, prec, npos, ntah = _sinif_ap(tahminler, gercekler, s, esik)
            ap_tablo[s - 1, j] = ap
            if j == 0:
                ap50_rec, ap50_prec, ap50_npos, ap50_ntah = rec, prec, npos, ntah
        f1 = (2 * ap50_prec * ap50_rec / (ap50_prec + ap50_rec)
              if ap50_prec and ap50_rec and not np.isnan(ap50_prec) else 0.0)
        sinif_ozet.append({
            "sinif": VOC_SINIFLAR[s - 1],
            "ap50": ap_tablo[s - 1, 0],
            "ap50_95": _ort(ap_tablo[s - 1]),
            "precision": ap50_prec, "recall": ap50_rec, "f1": f1,
            "gercek_nesne": ap50_npos, "tahmin": ap50_ntah,
        })

    # Karar esigi uzerindeki tahminlerle TP/FP/FN ve sinif dogrulugu
    tp = fp = fn = 0
    iou_toplam, iou_adet = 0.0, 0
    yerinde, yerinde_dogru = 0, 0

    for ad, g in gercekler.items():
        t = tahminler.get(ad, {"boxes": np.zeros((0, 4), np.float32),
                               "labels": np.zeros(0, np.int64),
                               "scores": np.zeros(0, np.float32)})
        tut = t["scores"] >= karar_esik
        t_kutu, t_etiket = t["boxes"][tut], t["labels"][tut]
        sira = np.argsort(-t["scores"][tut])
        t_kutu, t_etiket = t_kutu[sira], t_etiket[sira]

        kolay = ~g["difficult"]
        g_kutu, g_etiket = g["boxes"][kolay], g["labels"][kolay]
        eslesti = np.zeros(len(g_kutu), dtype=bool)

        if len(g_kutu):
            iou_tum = iou_matris(t_kutu, g_kutu) if len(t_kutu) else None
        else:
            iou_tum = None

        for i in range(len(t_kutu)):
            if iou_tum is None:
                fp += 1
                continue
            # sinif dogrulugu: sinifa BAKMADAN en iyi ortusen gercek nesne
            en_iyi_genel = int(np.argmax(iou_tum[i]))
            if iou_tum[i][en_iyi_genel] >= IOU_ESIK:
                yerinde += 1
                if g_etiket[en_iyi_genel] == t_etiket[i]:
                    yerinde_dogru += 1

            uygun = np.where((g_etiket == t_etiket[i]) & ~eslesti)[0]
            if len(uygun) == 0:
                fp += 1
                continue
            en_iyi = uygun[int(np.argmax(iou_tum[i][uygun]))]
            if iou_tum[i][en_iyi] >= IOU_ESIK:
                tp += 1
                eslesti[en_iyi] = True
                iou_toplam += float(iou_tum[i][en_iyi])
                iou_adet += 1
            else:
                fp += 1
        fn += int((~eslesti).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "map50": _ort(ap_tablo[:, 0]),
        "map75": _ort(ap_tablo[:, 5]) if tam else float("nan"),
        "map": _ort(ap_tablo),
        "precision": precision, "recall": recall, "f1": f1,
        "sinif_dogrulugu": yerinde_dogru / max(yerinde, 1),
        "ort_iou": iou_toplam / max(iou_adet, 1),
        "tp": tp, "fp": fp, "fn": fn,
        "sinif_ozet": sinif_ozet,
    }


# 5) TAHMIN TOPLAMA
@torch.no_grad()
def tahmin_topla(model, loader, device, eslesme=None, skor_esik=SKOR_ESIK):
    """Modeli olcum modunda calistirip tahminleri ve gercekleri toplar.

    eslesme verilirse (COCO modeli) etiketler VOC'a cevrilir, esleşmeyen
    siniflarin tahminleri atilir.
    """
    model.eval()
    tahminler, gercekler = {}, {}

    for goruntuler, _, adlar in loader:
        goruntuler = [g.to(device, non_blocking=True) for g in goruntuler]
        with torch.amp.autocast("cuda"):
            ciktilar = model(goruntuler)

        for cikti, ad in zip(ciktilar, adlar):
            kutular = cikti["boxes"].float().cpu().numpy()
            skorlar = cikti["scores"].float().cpu().numpy()
            etiketler = cikti["labels"].cpu().numpy()

            if eslesme is not None:
                tut = np.array([e in eslesme for e in etiketler], dtype=bool)
                kutular, skorlar = kutular[tut], skorlar[tut]
                etiketler = np.array([eslesme[e] for e in etiketler[tut]],
                                     dtype=np.int64)

            tut = skorlar >= skor_esik
            tahminler[ad] = {"boxes": kutular[tut], "scores": skorlar[tut],
                             "labels": etiketler[tut]}

            annot = ortak.annot_oku(ad)
            gercekler[ad] = {"boxes": annot["boxes"], "labels": annot["labels"],
                             "difficult": annot["difficult"]}

    model.train()
    return tahminler, gercekler


def olc(model, loader, device, eslesme=None, tam=True):
    return metrikler_hesapla(*tahmin_topla(model, loader, device, eslesme), tam=tam)


# 6) EGITIM
def egit(model, loaderlar, device, epochs=EPOCHS, lr=LEARNING_RATE, deney_adi=""):
    parametreler = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(parametreler, lr=lr, momentum=MOMENTUM,
                                weight_decay=WEIGHT_DECAY)
    # Son iki epoch'ta lr 10'da 1'e iner. Kisa egitimde sabit lr birakmak
    # skoru 2-3 puan dusuruyor; model son epoch'larda salinip duruyor.
    kilometre = [max(1, int(epochs * 0.67)), max(2, int(epochs * 0.89))]
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, kilometre, gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")

    gecmis = []
    en_iyi_val, en_iyi_epoch = 0.0, 0
    adim = 0
    baslangic = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        toplam = 0.0
        for goruntuler, hedefler, _ in loaderlar["train"]:
            goruntuler = [g.to(device, non_blocking=True) for g in goruntuler]
            hedefler = [{k: v.to(device) for k, v in h.items()} for h in hedefler]

            if adim < ISINMA_ADIM and epoch == 1:
                ortak.isinma_lr(optimizer, lr, adim, ISINMA_ADIM)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                kayiplar = model(goruntuler, hedefler)
                kayip = sum(kayiplar.values())
            scaler.scale(kayip).backward()
            scaler.step(optimizer)
            scaler.update()

            toplam += float(kayip.detach())
            adim += 1

        scheduler.step()
        va = olc(model, loaderlar["val"], device, tam=False)
        if va["map50"] > en_iyi_val:
            en_iyi_val, en_iyi_epoch = va["map50"], epoch

        gecmis.append({
            "epoch": epoch,
            "train_loss": round(toplam / len(loaderlar["train"]), 4),
            "val_map50": round(va["map50"] * 100, 2),
            "val_precision": round(va["precision"] * 100, 2),
            "val_recall": round(va["recall"] * 100, 2),
            "lr": round(optimizer.param_groups[0]["lr"], 6),
        })
        print(f"  Epoch {epoch:>2}/{epochs} | Loss: {toplam / len(loaderlar['train']):.4f} "
              f"| Val mAP@0.5: {va['map50'] * 100:.2f}% "
              f"| Recall: {va['recall'] * 100:.2f}%")

    return gecmis, time.perf_counter() - baslangic, en_iyi_val, en_iyi_epoch


# 7) GORSEL
def gorsel_kaydet(model, adlar, device, dosya, baslik, eslesme=None,
                  skor_esik=KARAR_ESIK):
    """Her ornek icin gercek kutular (beyaz) ve tahminler (sinif rengi) yan yana."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches

    model.eval()
    n = len(adlar)
    satir = (n + 2) // 3
    fig, eksenler = plt.subplots(satir, 3, figsize=(15, 5.0 * satir))
    eksenler = np.atleast_2d(eksenler).reshape(satir, 3)

    with torch.no_grad():
        for i, ad in enumerate(adlar):
            goruntu = ortak.goruntu_oku(ad)
            x = TF.to_tensor(goruntu).to(device)
            with torch.amp.autocast("cuda"):
                cikti = model([x])[0]

            kutular = cikti["boxes"].float().cpu().numpy()
            skorlar = cikti["scores"].float().cpu().numpy()
            etiketler = cikti["labels"].cpu().numpy()
            if eslesme is not None:
                tut = np.array([e in eslesme for e in etiketler], dtype=bool)
                kutular, skorlar = kutular[tut], skorlar[tut]
                etiketler = np.array([eslesme[e] for e in etiketler[tut]],
                                     dtype=np.int64)
            tut = skorlar >= skor_esik
            kutular, skorlar, etiketler = kutular[tut], skorlar[tut], etiketler[tut]

            annot = ortak.annot_oku(ad)
            ax = eksenler[i // 3, i % 3]
            ax.imshow(goruntu)

            for kutu in annot["boxes"]:
                ax.add_patch(patches.Rectangle(
                    (kutu[0], kutu[1]), kutu[2] - kutu[0], kutu[3] - kutu[1],
                    fill=False, edgecolor="white", linewidth=2.2, linestyle="--"))

            for kutu, skor, etiket in zip(kutular, skorlar, etiketler):
                renk = ortak.PALET[etiket] / 255.0
                ax.add_patch(patches.Rectangle(
                    (kutu[0], kutu[1]), kutu[2] - kutu[0], kutu[3] - kutu[1],
                    fill=False, edgecolor=renk, linewidth=2.0))
                ax.text(kutu[0], max(kutu[1] - 4, 8),
                        f"{VOC_SINIFLAR[etiket - 1]} {skor:.2f}",
                        fontsize=7.5, color="black",
                        bbox=dict(facecolor=renk, edgecolor="none", pad=1.0))

            ax.set_title(f"{ad}  ({len(annot['boxes'])} gercek / "
                         f"{len(kutular)} tahmin)", fontsize=9)
            ax.axis("off")

    for j in range(n, satir * 3):
        eksenler[j // 3, j % 3].axis("off")

    model.train()
    fig.suptitle(f"{baslik}\nbeyaz kesikli = gercek kutu  ·  renkli = tahmin "
                 f"(skor >= {skor_esik})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(dosya, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return dosya


def gorsel_ornekler_sec(adet=6, seed=ortak.SEED):
    """Test setinden nesne sayisi artan sirada sabit ornekler.

    Ayni goruntuler her deneyde kullanilir; farkli goruntulere bakip
    "bu model daha iyi" demek en kolay yanilma yolu.
    """
    _, test_tum = ortak.val_bol(ortak.kume_listesi("val"))
    aday = ortak.alt_kume(test_tum, 300, seed)
    sayilar = [(ad, len(ortak.annot_oku(ad)["boxes"])) for ad in aday]
    sayilar = [s for s in sayilar if s[1] > 0]
    sayilar.sort(key=lambda s: s[1])
    n = len(sayilar)
    konum = [int(round(k * (n - 1) / (adet - 1))) for k in range(adet)]
    return [sayilar[k][0] for k in konum]


# 8) DENEY KOSTURUCU
def deney_calistir(konfig, loaderlar, gorsel_idx, device):
    """Tek deneyi bastan sona: kur, egit, olc, kaydet, gorsel uret."""
    ortak.seed_ayarla()
    eslesme = None

    if konfig.get("egitimsiz"):
        model, eslesme = coco_modeli_kur()
        model = model.to(device)
        gecmis, sure, en_iyi_val, en_iyi_epoch = [], 0.0, 0.0, 0
    else:
        model = model_kur(konfig["pretrained"],
                          konfig.get("egitilebilir_katman", 3)).to(device)

    parametre = sum(p.numel() for p in model.parameters())
    print(f"[{konfig['ad']}] {konfig['not']}")
    print(f"      Pretrained: {konfig['pretrained']} | "
          f"Aug: {konfig['augmentation']} | Parametre: {parametre:,}"
          .replace(",", "."))

    if not konfig.get("egitimsiz"):
        gecmis, sure, en_iyi_val, en_iyi_epoch = egit(
            model, loaderlar, device, epochs=konfig.get("epochs", EPOCHS),
            lr=konfig.get("lr", LEARNING_RATE), deney_adi=konfig["ad"])

    va = olc(model, loaderlar["val"], device, eslesme)
    te = olc(model, loaderlar["test"], device, eslesme)

    satir = {
        "deney": konfig["ad"], "asama": konfig["asama"],
        "model": konfig.get("model", "FasterRCNN"),
        "backbone": konfig.get("backbone", "ResNet50-FPN"),
        "pretrained": konfig["pretrained"],
        "augmentation": konfig["augmentation"],
        "seed": ortak.SEED, "epochs": 0 if konfig.get("egitimsiz")
                            else konfig.get("epochs", EPOCHS),
        "min_boyut": MIN_BOYUT, "batch": BATCH_SIZE,
        "lr": 0 if konfig.get("egitimsiz") else konfig.get("lr", LEARNING_RATE),
        "parametre": parametre,
        "egitim_goruntu": 0 if konfig.get("egitimsiz")
                          else len(loaderlar["train"].dataset),
        "val_map50": round(va["map50"] * 100, 2),
        "val_map": round(va["map"] * 100, 2),
        "test_map50": round(te["map50"] * 100, 2),
        "test_map75": round(te["map75"] * 100, 2),
        "test_map": round(te["map"] * 100, 2),
        "test_precision": round(te["precision"] * 100, 2),
        "test_recall": round(te["recall"] * 100, 2),
        "test_f1": round(te["f1"] * 100, 2),
        "test_sinif_dogrulugu": round(te["sinif_dogrulugu"] * 100, 2),
        "test_ort_iou": round(te["ort_iou"] * 100, 2),
        "test_tp": te["tp"], "test_fp": te["fp"], "test_fn": te["fn"],
        "en_iyi_val_map50": round(en_iyi_val * 100, 2),
        "en_iyi_epoch": en_iyi_epoch,
        "sure_sn": round(sure, 2), "cihaz": ortak.CIHAZ,
        "notlar": konfig["not"],
    }

    ortak.sonuc_ekle([satir], ortak.TESPIT_SONUC, ALAN_ADLARI)
    ortak.sinif_bazli_kaydet(
        [{"deney": konfig["ad"],
          "sinif": s["sinif"],
          "ap50": round(float(s["ap50"]) * 100, 2),
          "ap50_95": round(float(s["ap50_95"]) * 100, 2),
          "precision": round(float(s["precision"]) * 100, 2),
          "recall": round(float(s["recall"]) * 100, 2),
          "f1": round(float(s["f1"]) * 100, 2),
          "gercek_nesne": s["gercek_nesne"], "tahmin": s["tahmin"]}
         for s in te["sinif_ozet"]],
        ortak.TESPIT_SINIF_SONUC, SINIF_ALANLARI)

    if gecmis:
        ortak.gecmis_kaydet(gecmis, konfig["ad"])

    gorsel = gorsel_kaydet(
        model, gorsel_idx, device,
        f"{ortak.GORSEL_DIR}/tespit_{konfig['ad']}.png",
        f"{konfig['ad']}  —  Test mAP@0.5 %{satir['test_map50']:.2f} / "
        f"mAP@0.5:0.95 %{satir['test_map']:.2f}", eslesme)

    print(f"  -> SONUC | Test mAP@0.5: {satir['test_map50']:.2f}% | "
          f"mAP@0.5:0.95: {satir['test_map']:.2f}% | "
          f"F1: {satir['test_f1']:.2f}% | Ort IoU: {satir['test_ort_iou']:.2f}% | "
          f"Sure: {sure:.1f} sn")
    print(f"     Gorsel: {gorsel}\n")

    del model
    torch.cuda.empty_cache()
    return satir


OZET_SUTUNLARI = [
    ("Deney", "deney", 30), ("ValmAP50", "val_map50", 10),
    ("TestmAP50", "test_map50", 11), ("mAP50-95", "test_map", 10),
    ("Prec", "test_precision", 8), ("Recall", "test_recall", 8),
    ("F1", "test_f1", 8), ("OrtIoU", "test_ort_iou", 8),
    ("Sure", "sure_sn", 10),
]
