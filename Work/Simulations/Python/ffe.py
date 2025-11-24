
from ffe_func import *

# SYMBOL GENERATOR
n_symbols = int(500e3)

PAM = 4
SR = 4e9          # symbol rate (baud)
BR = SR * np.log2(PAM)   # bit rate

BW = SR/2 # Nyquist BandWidth

norm = np.sqrt(5.0)   # normalize PAM-4 to unit power
symbols = (2*np.random.randint(0,PAM,n_symbols)-PAM+1)/norm

# RAISED COSINE FILTER (PULSE SHAPING)
CHANNEL_UP = 16 # UP Sampling to simulate continuous time

fs_ch = CHANNEL_UP*SR # Sampling Frequency of the simulation

print(f"Baud Rate:  {BR/1e9:.2f}GBd --- Symbol Rate: {SR/1e9:.2f}GHz --- Band Width: {BW/1e9:.2f}GHz")

rcos_filt = rcos(0.1, 40, CHANNEL_UP, 1,'sqrt')

padding = len(rcos_filt)//2 # Filter delay

symbols_up = np.zeros(n_symbols*CHANNEL_UP+padding) # Oversampled symbol sequence
symbols_up[:-padding][::CHANNEL_UP] = symbols

out_upsampled = filt(rcos_filt, 1, symbols_up)[padding:] # Apply filter

# PLOT SYMBOL GEN AND RCOS SYMBOLS
if (False): # True: plot, False: no plot
    rx_symbols = out_upsampled[::CHANNEL_UP]
    plt.figure()
    # Plot original symbols (blue)
    plt.stem(rx_symbols, linefmt='b-', markerfmt='bo', basefmt=' ')
    # Plot upsampled symbols (orange)
    plt.stem(symbols, linefmt='orange', markerfmt='o', basefmt=' ')
    plt.title("Rcos out vs original symbols")
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.grid(True)
    plt.xlim(0, 20)   # show only up to n=20
    plt.show()
    print(f"length  symbols: {len(symbols)}")
    print(f"length rx symbols: {len(rx_symbols)}")


## CHANNEL MODEL
ORDER = 7
f_cut_channel = BW-5e8 #GHz
b,a = sig.butter(ORDER,f_cut_channel/(fs_ch/2),"low")
g = np.sum(a)/np.sum(b)
b = b

# --- Frequency response ---
w, h = sig.freqz(b, a, worN=4096, fs=fs_ch)  # w in Hz

# PLOT CHANNEL FREQUENCY RESPONSE
if (False): # True: plot, False: no plot


    # --- Plot magnitude and phase ---
    plt.figure(figsize=(10,6))
    plt.plot(w/1e9, 20*np.log10(np.abs(h)))
    plt.title("Bode Plot of Butterworth Channel Filter")
    plt.ylabel("Magnitude [dB]")
    plt.ylabel("frequency")
    plt.xlim(0, 10)   # show only up to 100 GHz
    plt.ylim(-100, 0)   # show only up to 100 GHz
    plt.grid(True)
    plt.show()

# --- Attenuation at Nyquist frequency ---
f_target = BW
idx = np.argmin(np.abs(w - f_target))
atten_50GHz = 20 * np.log10(np.abs(h[idx]))

print(f"Attenuation at {f_target/1e9:.2f} GHz: {atten_50GHz:.2f} dB")

channel_symbols = filt(b,a,out_upsampled) # Apply channel
padding = len(rcos_filt)//2

# PLOT RX SYMBOLS VS CHANNEL SYMBOLS
if (False): # True: plot, False: no plot
    plt.figure()
    # Plot original symbols (blue)
    plt.stem(channel_symbols, linefmt='b-', markerfmt='bo', basefmt=' ')
    # Plot upsampled symbols (orange)
    plt.stem(out_upsampled, linefmt='orange', markerfmt='o', basefmt=' ')
    plt.title("Channel symbols vs tx symbols")
    plt.xlabel("Time")
    plt.ylabel("Magnitude")
    plt.xlim(0, 20)   # show only up to n=20
    plt.grid(True)
    plt.show()

# NOISE GENERATION
# snr_dbs = np.array([ii*1+11 for ii in range(8)]) # List of snr values
snr_dbs = [23] # manual list of snr values
print(f"SNR values: {snr_dbs}")

for kk,snr_var in enumerate(snr_dbs): 

    print(f"# SNR DB = {snr_var}")

    ## NOISE GEN
    noise_power_db = db10(5)-snr_var
    noise_power = 10**(noise_power_db / 10)  # convert to linear

    print(f"# NOISE DB = {noise_power_db}")

    noise = np.random.randn(n_symbols*CHANNEL_UP)*np.sqrt(noise_power)

    symbols_plus_noise = channel_symbols+noise

    # TIMING RECOVERY

    GD = group_delay_ba(b,a)[0]
    # print(GD)
    frational_delay = GD-np.floor(GD)
    CDR_DATA = frac_delay_sinc(symbols_plus_noise[int(GD):],-frational_delay)
    samples_symbols = CDR_DATA[::CHANNEL_UP]

    # FFE implementation
    FFE_LEN             = 27
    FFE                 = np.zeros(FFE_LEN)
    CENTRAL_TAP         = FFE_LEN // 2
    FFE[CENTRAL_TAP]    = 1

    delay_buffer        = np.zeros(CENTRAL_TAP)

    mu_ffe = 0.0001

    mem_in_data     = np.zeros(FFE_LEN)

    error_scope      = []
    ffe_scope        = []
    slicer_scope     = []
    ffe_out_scope    = []
    in_ffe_scope     = []

    CMA_len = 0

    FFE_history = []
    
    for ii,sample_data in enumerate(samples_symbols):
        mem_in_data [1:] = mem_in_data [:-1]
        mem_in_data [0] = sample_data
        in_ffe_scope.append(sample_data)

        out_ffe         = FIR(mem_in_data,FFE)
            
        out_slicer         = slicer(out_ffe*norm,PAM)/norm

        error_slicer       = out_ffe-out_slicer

        error_scope.append(error_slicer)
        ffe_out_scope.append(out_ffe)
        slicer_scope.append(out_slicer)

        if(ii>5*FFE_LEN):
            if ii<CMA_len:
                FFE = CMA(FFE,mem_in_data, out_ffe,mu_ffe)
            else:
                FFE = LMS(FFE,mem_in_data, error_slicer,mu_ffe)
            FFE_history.append(FFE.copy())
        
        delay_buffer[1:] = delay_buffer[:-1]
        delay_buffer[ 0] = symbols[ii]

    # FFE RESPONSE + CHANNEL RESPONSE + COMBINED RESPONSE
    delta_k = np.zeros(int(10e3))
    delta_k[0] = 1
    model_fir = filt(b,a,delta_k)
    ffe_extend = np.concatenate((FFE,np.zeros(100)))
    resampled_FFE        = scF.resample_poly(FFE,CHANNEL_UP,1)
    resampled_FFE       /= np.sum(resampled_FFE)

    # Channel + FFE only
    combined_response_FFE = filt(b, a, resampled_FFE)

    # combined_response = filt(combined_response,1,resampled_DFE)

    phase = int(np.argmax(np.abs(combined_response_FFE))%CHANNEL_UP)
    print(f"FFE: {np.array2string(FFE, precision=5, floatmode="fixed", separator=", ")}")

    # SYMBOL ERROR RATE
    correlation = sig.correlate(slicer_scope,symbols)
    align = CENTRAL_TAP
    dfe_align = np.array(slicer_scope)[align:]
    symbols_align = symbols[:len(dfe_align)]
    start_ber = 200000
    errors_bool = dfe_align[start_ber:] != symbols_align[start_ber:]
    n_errors = np.sum(errors_bool)
    ncomp = len(dfe_align[start_ber:])
    SER_value = n_errors / ncomp
    print(f"SER: {SER_value:.6e}  ({n_errors}/{ncomp})")

    DOWN_PLOT = 10

    fig4, (ax00,ax01,ax02,ax03) = plt.subplots(4, 1)

    ax00.stem(FFE,'b',label="Fir")
    ax01.stem(combined_response_FFE[phase::CHANNEL_UP][:len(FFE)],'r',label="Combined Response")
    ax02.stem(model_fir[:len(FFE)],'g',label="Channel")
    ax00.grid(True)
    ax01.grid(True)
    ax02.grid(True)
    ax00.legend()
    ax01.legend()
    ax02.legend()

    ax03.plot(error_scope[::DOWN_PLOT],label="Error")
    ax03.set_xlabel("Time")
    ax03.set_ylabel("Amplitud")
    ax03.set_title("Error")
    ax03.grid(True)
    ax03.legend()

    fig4.tight_layout()
    fig4.show()

    fig5, (ax50,ax51,ax52) = plt.subplots(3, 1)

    markerline51, stemlines51, baseline51 = ax51.stem(ffe_out_scope[::DOWN_PLOT],'r',label="Out FFE")

    plt.setp(stemlines51, visible=False)   # no vertical lines
    baseline51.set_visible(False)          # no baseline
    ax51.grid(True)
    ax51.legend()

    markerline52, stemlines52, baseline52 = ax52.stem(slicer_scope[::DOWN_PLOT],'g',label="Slicer")
    plt.setp(stemlines52, visible=False)   # no vertical lines
    baseline52.set_visible(False)          # no baseline
    ax52.grid(True)
    ax52.legend()

    markerline50, stemlines50, baseline50 = ax50.stem(in_ffe_scope[::DOWN_PLOT],'b',label="In FFE")
    plt.setp(stemlines50, visible=False)   # no vertical lines
    baseline50.set_visible(False)          # no baseline
    ax50.grid(True)
    ax50.legend()
    
    fig5.tight_layout()
    fig5.show()

    plt.figure()
    plt.plot(FFE_history)
    plt.title("FFE Coefficient Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Tap value")
    plt.grid(True)
    plt.show()

    # input("Press Enter to continue...")
    plt.close('all')
