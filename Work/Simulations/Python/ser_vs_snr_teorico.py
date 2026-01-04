from ffe_func import *

# SYMBOL GENERATOR
n_symbols = int(1e7)

PAM = 4
SR = 4e9          # symbol rate (baud)
BR = SR * np.log2(PAM)   # bit rate

BW = SR/2 # Nyquist BandWidth

print(f"- - - - - - - - - - - - - - - - - - - - -")
print(f"STARTING SIMULATION: PAM {PAM} EQUALIZER")
print(f"\tBit Rate: {BR/1e9:.2f}Gbps\n\tSymbol Rate: {SR/1e9:.2f}GBd\n\tBand Width: {BW/1e9:.2f}GHz\n")

# PAM-4 average symbol energy = (M^2 - 1) / 3 = 5
Es_pam = (PAM**2 - 1) / 3
norm = np.sqrt(Es_pam)   # normalizado a Es = 1
symbols = (2*np.random.randint(0,PAM,n_symbols)-PAM+1)/norm

# APPLY CHANNEL
channel_symbols = symbols.copy()

# NOISE GENERATION
snr_dbs = np.array([ii*1+11 for ii in range(8)]) # List of snr values
# snr_dbs = np.array([18,19,20,21,22]) # manual list of snr values
print(f"SNR values: {snr_dbs}\n")

ser_awgn = np.zeros(len(snr_dbs))

for kk, snr_var in enumerate(snr_dbs):

    print(f"SNR = {snr_var} dB")

    snr_lin = 10**(snr_var/10)

    noise_power = 1 / (2 * snr_lin)   # σ² = N0/2
    noise = np.random.randn(len(symbols)) * np.sqrt(noise_power)

    rx_symbols = symbols + noise

    slicer_out = np.array([
    slicer(x * norm, PAM) / norm
    for x in rx_symbols
    ])
    ser = np.mean(slicer_out != symbols)
    print(f"SER = {ser}")
    ser_awgn[kk] = ser

print(f"SNR\tSER")
for i, ser_v in enumerate(ser_awgn):
    print(f"{snr_dbs[i]}\t{ser_awgn[i]:.2e}")

# --- THEORETICAL SER ---
M = PAM
snr_lin = 10**(snr_dbs / 10)

Es = np.mean(symbols**2)      # depende de norm
snr_eff = snr_lin   # Es = 1

Q = lambda x: 0.5 * erfc(x / np.sqrt(2))

ser_theory = 2 * (1 - 1/M) * Q(
    np.sqrt(6 * snr_eff / (M**2 - 1))
)

plt.figure()
plt.semilogy(snr_dbs, ser_awgn, 'o-', label="Simulation (AWGN ideal)")
plt.semilogy(snr_dbs, ser_theory, '-', label="Theory (AWGN PAM-4)")

plt.xlabel("SNR (dB)")
plt.ylabel("SER")
plt.title("PAM-4 SER vs SNR (AWGN)")
plt.grid(True, which="both")
plt.legend()
plt.show()