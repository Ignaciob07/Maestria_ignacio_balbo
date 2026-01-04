
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

Es_pam = (PAM**2 - 1) / 3
# norm = 1   # normalizado a Es = 1
norm = np.sqrt(Es_pam)   # normalizado a Es = 1
symbols = (2*np.random.randint(0,PAM,n_symbols)-PAM+1)/norm

CHANNEL_UP = 4 # UP Sampling to simulate continuous time

fs_ch = CHANNEL_UP*SR # Sampling Frequency of the simulation

# CHANNEL MODEL
b,delay_ch = channel_fir(fcut=BW , fs_ch=fs_ch, plt_en=False)
print("B IS: \n",b,"\n")

# RAISED COSINE FILTER (PULSE SHAPING)
out_upsampled = rrc(CHANNEL_UP, symbols, n_symbols)

# APPLY CHANNEL
channel_symbols = np.convolve(b, out_upsampled, mode="full")
channel_symbols = channel_symbols[delay_ch : delay_ch + len(out_upsampled)]

# NOISE GENERATION
# snr_dbs = np.array([ii*1+11 for ii in range(8)]) # List of snr values
snr_dbs = [15] # manual list of snr values
print(f"SNR values: {snr_dbs}\n")

ser_array=[]

for kk,snr_var in enumerate(snr_dbs):

    print(f"CASE {kk}: SNR DB = {snr_var}")

    ## NOISE GEN
    snr_lin = 10**(snr_var / 10)

    Ps = np.mean(channel_symbols**2)

    noise_var = Ps / snr_lin
    noise = np.random.randn(len(channel_symbols)) * np.sqrt(noise_var)

    rx_channel = channel_symbols + noise

    samples_symbols = rx_channel[::CHANNEL_UP]

    # FFE implementation
    FFE_LEN             = 21
    FFE                 = np.zeros(FFE_LEN)
    CENTRAL_TAP         = FFE_LEN//2
    FFE[CENTRAL_TAP]    = 1

    mu_ffe = 1e-5

    mem_in_data     = np.zeros(FFE_LEN)

    error_scope      = []
    ffe_scope        = []
    slicer_scope     = []
    ffe_out_scope    = []
    in_ffe_scope     = []

    CMA_len = 10e5

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

        if(ii>5*FFE_LEN):
            if ii<CMA_len:
                FFE = CMA(FFE,mem_in_data, out_ffe,1e-5, norm)
            else:
                FFE = LMS(FFE,mem_in_data, error_slicer,mu_ffe)
            FFE_history.append(FFE.copy())

    ser=GET_SER(slicer_scope, CENTRAL_TAP, symbols)
    ser_array.append(ser)

    DOWN_PLOT=10
    # plot_error(error_scope, DOWN_PLOT)
    # plot_symbols(in_ffe_scope, ffe_out_scope, DOWN_PLOT)
    plot_ffe(FFE_history, DOWN_PLOT)
    # plot_ffe_frec(FFE, BW)
    pdf_in_out(in_ffe_scope, ffe_out_scope)
    impulse_response(b, FFE, CHANNEL_UP, plt_en=True)

    print(FFE)

print(f"SNR\tSER")
for i, ser_v in enumerate(ser_array):
    print(f"{snr_dbs[i]}\t{ser_array[i]:.2e}")
