import matplotlib.pyplot as plt

# Tap weights extracted from the image
coeffs = [
    0,
    -0.001953125,
    -0.0029296875,
    -0.0009765625,
    0.0029296875,
    -0.001953125,
    0.00390625,
    -0.0009765625,
    -0.001953125,
    -0.00390625,
    0.974609375,
    0.00390625,
    0.0029296875,
    -0.00390625,
    -0.001953125,
    -0.0048828125,
    0,
    -0.0087890625,
    0.001953125,
    0.001953125,
    0.0009765625
]

plt.figure()
plt.stem(range(len(coeffs)), coeffs)
plt.xlabel("Tap index")
plt.ylabel("Coefficient value")
plt.title("FIR Tap Weights")
plt.show()
  