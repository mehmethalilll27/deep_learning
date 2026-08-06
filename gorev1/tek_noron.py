import numpy as np

# 2 girdi, 3 noron -> agirlik matrisi 2x3
w = np.array([
    [0.7, 0.3, 0.5],
    [0.2, 0.9, 0.4],
])

# 1 ornek, 2 ozellik
x = np.array([[2.0, 6.0]])

# her noron icin bias
b = np.array([[-0.2, 0.1, -0.5]])

# z = x @ w + b  ->  boyut: (1, 3)
z = x @ w + b

# sigmoid: (0 ile 1 arasi)
a_sigmoid = 1 / (1 + np.exp(-z))

# ReLU: max[0,z)
a_relu = np.maximum(0, z)

# Tanh: (-1 ile 1 arasi)
a_tanh = np.tanh(z)

# Softmax: (0 ile 1 arasi, toplami 1)
a_softmax = np.exp(z) / np.sum(np.exp(z))

print("\n w: ", w)
print("\n x: ", x)
print("\n b: ", b)
print("\n z: ", z)
print("\n a_sigmoid: ", a_sigmoid)
print("\n a_relu: ", a_relu)
print("\n a_tanh: ", a_tanh)
print("\n a_softmax: ", a_softmax)