import numpy as np

def tx_ffe_from_channel(h, Ntaps=5):
    """
    Calcula los taps del TX-FFE resolviendo un sistema mínimo cuadrado.
    h: respuesta al impulso del canal (numpy vector)
    Ntaps: número de taps del FFE del TX (tipicamente 3–7)
    """
    # Índice del tap deseado (tap principal)
    mid = len(h) // 2

    # Queremos que el canal ecualizado tenga:
    #   y[mid] = 1  (tap principal)
    #   y[k]  = 0  (precursor ISI)
    #
    # Usamos Ntaps vecinos alrededor del tap central
    half = Ntaps // 2
    idxs = np.arange(mid - half, mid + half + 1)

    # Construcción de la matriz A
    A = np.zeros((Ntaps, Ntaps))
    b = np.zeros(Ntaps)

    for r in range(Ntaps):
        for c in range(Ntaps):
            A[r, c] = h[idxs[r] - c]
    
    # Queremos 1 en el tap central
    b[half] = 1.0

    # Resolver
    ffe = np.linalg.solve(A, b)
    return ffe / np.sum(np.abs(ffe))   # normalizado opcional

# --------------------------
# Ejemplo de uso
# --------------------------
# Si tuvieras S21 medido:
# S21 = ...  # vector frecuencia
# h = np.fft.ifft(S21)

# si no, simulo un canal cualquiera:
h = np.exp(-np.arange(200)/30.0)  # canal ficticio

ffe = tx_ffe_from_channel(h, Ntaps=5)
print("Taps TX-FFE:", ffe)
