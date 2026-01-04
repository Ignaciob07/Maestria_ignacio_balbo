
import matplotlib.pyplot as plt
from scipy.special import erfc
import numpy as np

SNR = np.array([11, 12, 13, 14, 15, 16, 17, 18])
SER = np.array([4.85e-02, 2.95e-02, 1.65e-02, 8.27e-03,
       3.80e-03, 1.54e-03, 5.44e-04, 1.62e-04])
# --- THEORETICAL SER ---
M = 4
snr_lin = 10**(SNR / 10)

snr_eff = snr_lin   # Es = 1

Q = lambda x: 0.5 * erfc(x / np.sqrt(2))

ser_theory = 2 * (1 - 1/M) * Q(
    np.sqrt(6 * snr_eff / (M**2 - 1))
)

plt.semilogy(SNR, ser_theory, '-', label="Theory")
plt.semilogy(SNR, SER, '-o', label="ISI + Noise + FFE")
plt.grid(True, which='both')
plt.xlabel('SNR (dB)')
plt.ylabel('SER')
plt.title('SER vs SNR (Semilog Plot)')
plt.show()