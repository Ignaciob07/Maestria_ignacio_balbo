
from ffe_func import *

# SYMBOL GENERATOR
n_symbols = int(1e6)

PAM = 4
SR = 4e9          # symbol rate (baud)
BR = SR * np.log2(PAM)   # bit rate

BW = SR/2 # Nyquist BandWidth

print(f"- - - - - - - - - - - - - - - - - - - - -")
print(f"STARTING SIMULATION: PAM {PAM} EQUALIZER")
print(f"\tBit Rate: {BR/1e9:.2f}Gbps\n\tSymbol Rate: {SR/1e9:.2f}GBd\n\tBand Width: {BW/1e9:.2f}GHz\n")

norm = 1   # normalize PAM-4 to unit power
symbols = (2*np.random.randint(0,PAM,n_symbols)-PAM+1)/norm

CHANNEL_UP = 4 # UP Sampling to simulate continuous time

fs_ch = CHANNEL_UP*SR # Sampling Frequency of the simulation

# CHANNEL MODEL
b,delay_ch = channel_fir(fcut=BW , fs_ch=fs_ch, plt_en=False)

# Tx FFE
FFE_TX_LEN = 5
FFE_TX = np.zeros(FFE_TX_LEN)
FFE_TX = [-0.186, 0.493, -0.210, 0.052, -0.059]
delay_ffe_tx = FFE_TX_LEN // 2

out_ffe_tx = np.convolve(b, symbols, mode="full")
out_ffe_tx = out_ffe_tx[delay_ffe_tx: delay_ffe_tx + len(symbols)]

# RAISED COSINE FILTER (PULSE SHAPING)
out_upsampled = rrc(CHANNEL_UP, out_ffe_tx, len(out_ffe_tx))

# APPLY CHANNEL
channel_symbols = np.convolve(b, out_upsampled, mode="full")
channel_symbols = channel_symbols[delay_ch : delay_ch + len(out_upsampled)]

# NOISE GENERATION
# snr_dbs = np.array([ii*1+11 for ii in range(8)]) # List of snr values
snr_dbs = [20] # manual list of snr values
print(f"SNR values: {snr_dbs}\n")

snr_array=[]

for kk,snr_var in enumerate(snr_dbs): 

    print(f"CASE {kk}: SNR DB = {snr_var}")

    ## NOISE GEN
    signal_power = np.mean(channel_symbols**2)
    noise_power = signal_power / (10**(snr_var/10))
    noise = np.random.randn(len(channel_symbols)) * np.sqrt(noise_power)

    print(f"# NOISE DB = {db10(noise_power)}")

    noise = np.random.randn(len(channel_symbols)) * np.sqrt(noise_power)
    symbols_plus_noise = channel_symbols + noise

    # new: first remove integer offset then downsample
    samples_symbols = symbols_plus_noise[::CHANNEL_UP] # now sample at best integer point

    # FFE implementation
    FFE_LEN             = 21
    FFE                 = np.zeros(FFE_LEN)
    CENTRAL_TAP         = FFE_LEN//2
    FFE[CENTRAL_TAP]    = 1

    mu_ffe = 5e-6

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

        if(ii>5*FFE_LEN):
            if ii<CMA_len:
                FFE = CMA(FFE,mem_in_data, out_ffe,mu_ffe, norm)
            else:
                FFE = LMS(FFE,mem_in_data, error_slicer,mu_ffe)
            FFE_history.append(FFE.copy())

    snr=GET_SER(slicer_scope, CENTRAL_TAP, symbols)
    snr_array.append(snr)

    DOWN_PLOT=10
    plot_error(error_scope, DOWN_PLOT)
    plot_symbols(in_ffe_scope, ffe_out_scope, DOWN_PLOT)
    plot_ffe(FFE_history, DOWN_PLOT)
    plot_ffe_frec(FFE, BW)
    pdf_in_out(in_ffe_scope, ffe_out_scope)

