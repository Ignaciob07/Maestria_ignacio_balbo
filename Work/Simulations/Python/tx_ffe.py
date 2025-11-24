import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def plot_response(w, h, title):
    "Utility function to plot response functions"
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(w, 20*np.log10(np.abs(h)))
    ax.set_ylim(-40, 5)
    ax.grid(True)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Gain (dB)')
    ax.set_title(title)

fs = 22050   # Sample rate, Hz

PAM = 4
SR = 4e9          # symbol rate (baud)
BR = SR * np.log2(PAM)   # bit rate

fs = 22050

cutoff = 5000.0    # Desired cutoff frequency, Hz
trans_width = 250  # Width of transition from pass to stop, Hz
numtaps = 125      # Size of the FIR filter.
taps = signal.remez(numtaps, [0, cutoff - trans_width, cutoff, 0.5*fs],
                    [0, 1], fs=fs)
w, h = signal.freqz(taps, [1], worN=2000, fs=fs)

plot_response(w, h, "High-pass Filter")
plt.show()