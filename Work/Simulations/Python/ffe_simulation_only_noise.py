
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

mu_list = np.array([1e-6, 5e-6, 1e-5, 5e-5])
ser_array = np.zeros((len(mu_list), len(snr_dbs)))

for im, mu_ffe in enumerate(mu_list):

    print(f"\n=== mu = {mu_ffe:.1e} ===")

    for kk,snr_var in enumerate(snr_dbs): 

        print(f"CASE {kk}: SNR DB = {snr_var}")

        snr_lin = 10**(snr_var/10)

        noise_power = 1 / (2 * snr_lin)   # σ² = N0/2
        noise = np.random.randn(len(symbols)) * np.sqrt(noise_power)

        rx_symbols = channel_symbols + noise

        # new: first remove integer offset then downsample
        samples_symbols = rx_symbols # now sample at best integer point

        # FFE implementation
        FFE_LEN             = 21
        FFE                 = np.zeros(FFE_LEN)
        CENTRAL_TAP         = FFE_LEN//2
        FFE[CENTRAL_TAP]    = 1

        mu_ffe = mu_ffe

        mem_in_data     = np.zeros(FFE_LEN)

        error_scope      = []
        ffe_scope        = []
        slicer_scope     = []
        ffe_out_scope    = []
        in_ffe_scope     = []

        CMA_len = 1e5

        FFE_history = []
        
        for ii,sample_data in enumerate(samples_symbols):
            mem_in_data [1:] = mem_in_data [:-1] # shift
            mem_in_data [0] = sample_data 

            out_ffe         = FIR(mem_in_data,FFE)
                
            out_slicer         = slicer(out_ffe*norm,PAM)/norm

            error_slicer       = out_ffe-out_slicer

            in_ffe_scope.append(sample_data)
            error_scope.append(error_slicer)
            ffe_out_scope.append(out_ffe)
            slicer_scope.append(out_slicer)

            if(ii>0*FFE_LEN):
                if ii<CMA_len:
                    FFE = CMA(FFE,mem_in_data, out_ffe,mu_ffe, norm)
                else:
                    FFE = LMS(FFE,mem_in_data, error_slicer,mu_ffe)
                FFE_history.append(FFE.copy())

        ser=GET_SER(slicer_scope, CENTRAL_TAP, symbols)
        ser_array[im, kk] = ser

        DOWN_PLOT=10
        # plot_error(error_scope, DOWN_PLOT)
        # plot_symbols(in_ffe_scope, ffe_out_scope, DOWN_PLOT)
        # plot_ffe(FFE_history, DOWN_PLOT)
        # plot_ffe_frec(FFE, BW)
        # pdf_in_out(in_ffe_scope, ffe_out_scope)

# print(f"SNR\tSER")
# for i, ser_v in enumerate(ser_array):
#     print(f"{snr_dbs[i]}\t{ser_array[i]:.2e}")

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
for im, mu_ffe in enumerate(mu_list):
    plt.semilogy(
        snr_dbs,
        ser_array[im, :],
        'o-',
        label=f"mu = {mu_ffe:.1e}"
    )

plt.semilogy(snr_dbs, ser_theory, '-', label="Theory (AWGN PAM-4)")

plt.xlabel("SNR (dB)")
plt.ylabel("SER")
plt.title("PAM-4 SER vs SNR")
plt.grid(True, which="both")
plt.legend()

plt.show()