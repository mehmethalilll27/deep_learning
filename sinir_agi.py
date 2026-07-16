import numpy as np

# 1) AKTIVASYON FONKSIYONLARI

# sigmoid: 0 ile 1 arasi
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ReLU: negatifleri 0 yapar
def relu(x):
    return np.maximum(0, x)


# 2) PARAMETRE BASLATMA
# w1, b1 = giris -> gizli katman
# w2, b2 = gizli -> cikis katmani

def init_parameters(input_size, hidden_size, output_size, seed=27):
    rng = np.random.default_rng(seed)  # ayni seed = ayni rastgele baslangic

    w1 = rng.normal(0, 1, (input_size, hidden_size)) # .normal(ortalama,standart sapma, (satır,sütun))
    b1 = np.zeros((1, hidden_size)) # içinde sadece sıfır olan matris

    w2 = rng.normal(0, 1, (hidden_size, output_size))
    b2 = np.zeros((1, output_size))

    return w1, b1, w2, b2


# 3) ILERI BESLEME (FORWARD)
# x @ w + b  ->  aktivasyon  ->  tahmin

def forward_pass(x, w1, b1, w2, b2, activation="sigmoid"):
    # 1. gizli katman
    z1 = x @ w1 + b1
    if activation == "relu":
        a1 = relu(z1)
    else:
        a1 = sigmoid(z1)

    # 2. cikis katmani
    z2 = a1 @ w2 + b2
    output = sigmoid(z2)  # 0-1 arasi tahmin

    # geri yayilim icin ara degerler
    cache = {"x": x, "z1": z1, "a1": a1, "w2": w2}
    return output, cache


# 4) KAYIP (LOSS)
# tahmin ile gercek arasindaki hata

def loss_function(y_pred, y_true, loss_type="bce"):
    eps = 1e-8 #log(0)ı önlemek için

    if loss_type == "mse":
        # ortalama karesel hata
        return np.mean(np.square(y_pred - y_true)) # np.square(y_pred - y_true) -> (y_pred - y_true)²
    else:
        # binary cross-entropy (0/1 siniflandirma icin)
        y_pred = np.clip(y_pred, eps, 1 - eps) # y_pred'i 0 ve 1 arasına kırpar
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)) 
        # y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred) -> y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred)


# 5) GERI YAYILIM (BACKWARD)
# hata cikistan geriye gider, her parametrenin gradyani bulunur

def backward_pass(y_true, y_pred, cache, w1, b1, w2, b2, loss_type="bce", activation="sigmoid"):
    m = y_true.shape[0]  # ornek sayisi
    x = cache["x"] # x -> giris verisi
    z1 = cache["z1"] # z1 -> gizli katmanin aktivasyonu
    a1 = cache["a1"] 

    # --- cikis katmani gradyanlari ---
    if loss_type == "mse":
        dz2 = (2 * (y_pred - y_true) / m) * (y_pred * (1 - y_pred))
    else:
        dz2 = (y_pred - y_true) / m

    dw2 = a1.T @ dz2 #w2 gradyanı
    db2 = np.sum(dz2, axis=0, keepdims=True) #b2 gradyanı

    # --- gizli katman gradyanlari ---
    da1 = dz2 @ w2.T #hata gizli katmana geri gelir
    if activation == "relu":
        dz1 = da1 * (z1 > 0)
    else:
        dz1 = da1 * (a1 * (1 - a1))

    dw1 = x.T @ dz1 #w1 gradyanı
    db1 = np.sum(dz1, axis=0, keepdims=True) #b1 gradyanı

    return dw1, db1, dw2, db2


# 6) PARAMETRE GUNCELLEME (SGD)
# w = w - learning_rate * dw

def update_parameters(w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate):
    w1 = w1 - learning_rate * dw1
    b1 = b1 - learning_rate * db1
    w2 = w2 - learning_rate * dw2
    b2 = b2 - learning_rate * db2
    return w1, b1, w2, b2



# 7) DOGRULUK
# 0.5 ustu -> 1, alti -> 0

def accuracy(y_pred, y_true):
    predictions = (y_pred >= 0.5).astype(int)
    return np.mean(predictions == y_true)


# 8) EGITIM DONGUSU
# sirayla: forward -> loss -> backward -> update

def train(x, y, epochs=5000, learning_rate=0.5, hidden_size=4, activation="relu", loss_type="bce"):
    # baslangic parametreleri
    w1, b1, w2, b2 = init_parameters(x.shape[1], hidden_size, y.shape[1])

    for epoch in range(1, epochs + 1):
        # 1) ileri besleme
        y_pred, cache = forward_pass(x, w1, b1, w2, b2, activation=activation)

        # 2) kayip
        loss = loss_function(y_pred, y, loss_type=loss_type)

        # 3) geri yayilim
        dw1, db1, dw2, db2 = backward_pass( # gradyanları hesapla
            y, y_pred, cache, w1, b1, w2, b2,
            loss_type=loss_type, activation=activation
        )

        # 4) agirliklari guncelle
        w1, b1, w2, b2 = update_parameters( # agirlikları guncelle
            w1, b1, w2, b2, dw1, db1, dw2, db2, learning_rate
        )

        # her 500 epoch'ta yazdir
        if epoch % 500 == 0 or epoch == 1:
            acc = accuracy(y_pred, y) # doğruluk oranını hesapla
            print(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Accuracy: {acc * 100:.1f}%")

    return w1, b1, w2, b2


# 9) VERI: XOR
# 0,0 -> 0
# 0,1 -> 1
# 1,0 -> 1
# 1,1 -> 0

X = np.array([ # giris verileri
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
])

y = np.array([ # hedef verileri
    [0],
    [1],
    [1],
    [0],
])


# 10) CALISTIR

print("XOR egitimi basliyor...\n")

w1, b1, w2, b2 = train(
    X, y,
    epochs=5000,
    learning_rate=0.5,
    hidden_size=4,
    activation="relu",
    loss_type="bce",
)

print("\nFinal tahminler:")
y_pred, _ = forward_pass(X, w1, b1, w2, b2, activation="relu")

for i in range(len(X)):
    print(f"Input: {X[i]} -> Tahmin: {y_pred[i][0]:.4f} | Hedef: {y[i][0]}")