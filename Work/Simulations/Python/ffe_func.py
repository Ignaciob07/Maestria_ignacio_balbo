import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.signal import welch
from scipy import signal
import scipy.signal as scF
from rcosdesign import rcosdesign as rcos
import matplotlib
from decimal import *
from tool._fixedInt import *
from scipy.stats import gaussian_kde  # For smooth PDF estimation
from numpy import i0
from scipy.special import erfc
matplotlib.use('TkAgg')
import math

def slicer(x: float, M: int = 4):
    """
    Slices a received voltage 'x' to the nearest PAM-M level.
    Assumes M is an even integer >= 2.
    The ideal levels are: -(M-1), ..., -3, -1, +1, +3, ..., (M-1).

    Args:
        x: The received analog voltage level (float).
        M: The number of modulation levels (e.g., 4 for PAM-4).
    Returns:
        The sliced integer symbol level.
    """
    # 1. Validate M
    if M < 2 or M % 2 != 0:
        raise ValueError("M must be an even integer greater than or equal to 2.")

    # 2. Find the nearest odd integer
    # We find the nearest integer, then check if it's odd or even.
    # Python 3's round() rounds .5 to the nearest *even* number,
    # e.g., round(0.5)=0, round(1.5)=2. This logic correctly
    # handles the decision boundaries (which are at 0, +/-2, +/-4, ...).

    r = int(np.round(x))

    if r % 2 != 0:
        # If r is already odd (e.g., 1, 3, -1), it's our value.
        sliced_val = r
    else:
        # If r is even (e.g., 0, 2, -2), it's a decision boundary.
        if x > r:
            sliced_val = r + 1
        else:
            sliced_val = r - 1

    # 3. Clamp the result to the valid PAM-M range
    max_level = M - 1
    # Clamp to the minimum level: -(M-1)
    output = max(sliced_val, -max_level)
    # Clamp to the maximum level: (M-1)
    output = min(output, max_level)

    return output

def GET_SER(slicer_scope, CENTRAL_TAP, symbols):
    # SYMBOL ERROR RATE
    slicer_arr = np.array(slicer_scope)
    align = CENTRAL_TAP
    slicer_align = slicer_arr[align:]
    symbols_trunc = symbols[:-align]
    start_ber = 200000
    if len(slicer_align) <= start_ber:
        # simulation not long enough
        print("Warning: not enough samples for BER start index. Lower start_ber or increase n_symbols.")
    else:
        errors_bool = slicer_align[start_ber:] != symbols_trunc[start_ber:]
        n_errors = np.sum(errors_bool)
        ncomp = len(slicer_align[start_ber:])
        SER_value = n_errors / ncomp
        print(f"SER: {SER_value:.6e}  ({n_errors}/{ncomp})")

    return SER_value

def FIR(samples,coeffs):
    return np.dot(samples,coeffs)

def LMS(fir,samples,error,mu):
    fir = fir - samples*error*mu
    return fir

def CMA(fir,samples,yk,mu, norm):
    if norm == math.sqrt(5):
      R = 1.6328125
    else:
      R=8.2*norm/(norm**2)
    # print(R)
    error=yk**2-R
    fir = fir - samples*error*yk*mu
    return fir


def db(data):
    return 20*np.log10(np.abs(data))

def db10(data):
    return 10*np.log10(np.abs(data))

def channel_fir(fcut, fs_ch, plt_en=False):

    print("CHANNEL DESIGN: FIR")

    ORDER = 128 # filter length - 1
    f_cut = fcut - fcut/20 # Hz
    b = sig.firwin(ORDER+1, f_cut/(fs_ch/2), window='blackmanharris')
    a = 1 # FIR → no denominator

    # --- Frequency response ---
    w, h = sig.freqz(b, a, worN=4096, fs=fs_ch)  # w in Hz
    f_target = fcut

    # --- Attenuation at Nyquist frequency ---
    idx = np.argmin(np.abs(w - f_target))
    attenGHz = 20 * np.log10(np.abs(h[idx]))
    print(f"Attenuation at {f_target/1e9:.2f} GHz: {attenGHz:.2f} dB\n")


    # PLOT CHANNEL FREQUENCY RESPONSE
    if (plt_en): # True: plot, False: no plot
        # --- Plot magnitude and phase ---
        plt.figure(figsize=(10,6))
        plt.plot(w/1e9, 20*np.log10(np.abs(h)))
        plt.title("Bode Plot of FIR Channel Filter")
        plt.ylabel("Magnitude [dB]")
        plt.ylabel("frequency")
        plt.xlim(0, 10)   # show only up to 100 GHz
        plt.ylim(-100, 0)   # show only up to 100 GHz
        plt.grid(True)
        plt.show()

    delay = ORDER // 2

    return b, delay

def rrc(CHANNEL_UP, symbols, n_symbols):
    rcos_filt = rcos(0.1, 40, CHANNEL_UP, 1,'sqrt')

    rcos_delay = len(rcos_filt)//2 # Filter delay

    symbols_up = np.zeros(n_symbols*CHANNEL_UP) # Oversampled symbol sequence
    symbols_up[::CHANNEL_UP] = symbols

    out_upsampled = np.convolve(symbols_up, rcos_filt, mode="full")
    return out_upsampled[rcos_delay : rcos_delay + len(symbols_up)]   # valid samples

def extract_centered(x, center, N):
    """
    Extract N samples centered at index `center` from x.
    Zero-pad if boundaries are exceeded.
    """
    half = N // 2
    y = np.zeros(N)

    start_x = center - half
    end_x   = center + half + 1

    start_y = max(0, -start_x)
    end_y   = N - max(0, end_x - len(x))

    x_start = max(0, start_x)
    x_end   = min(len(x), end_x)

    y[start_y:end_y] = x[x_start:x_end]
    return y

def impulse_response(b, FFE, CHANNEL_UP, plt_en):

    # Create delta
    delta_k = np.zeros(int(10e3))
    delta_k[0] = 1.0

    # Channel impulse response (high rate)
    model_fir = sig.lfilter(b, 1, delta_k)

    # Upsample FFE to channel rate
    resampled_FFE = scF.resample_poly(FFE, CHANNEL_UP, 1)
    resampled_FFE /= np.sum(resampled_FFE)

    # Combined channel + FFE
    combined_response_FFE = sig.lfilter(b, 1, resampled_FFE)

    # Best integer phase
    phase = int(np.argmax(np.abs(combined_response_FFE)) % CHANNEL_UP)

    # Symbol-rate responses
    comb_sr = combined_response_FFE[phase::CHANNEL_UP]
    chan_sr = model_fir[phase::CHANNEL_UP]

    N = len(FFE)

    # Centers
    center_comb = np.argmax(np.abs(comb_sr))
    center_chan = np.argmax(np.abs(chan_sr))

    # Extract centered windows (SAFE)
    comb = extract_centered(comb_sr, center_comb, N)
    chan = extract_centered(chan_sr, center_chan, N)

    # Time axis
    n = np.arange(N) - N//2

    if plt_en:
        fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(10, 8))

        ax0.stem(n, FFE, basefmt=" ")
        ax0.set_title("FFE taps")
        ax0.grid(True)

        ax1.stem(n, comb, basefmt=" ")
        ax1.set_title("Combined Channel + FFE")
        ax1.grid(True)

        ax2.stem(n, chan, basefmt=" ")
        ax2.set_title("Channel only")
        ax2.grid(True)

        plt.tight_layout()
        plt.show()

    return phase


def plot_error(error_scope, DOWN_PLOT):
    plt.figure()
    plt.plot(error_scope[::DOWN_PLOT])
    plt.title("Slicer Error")
    plt.grid(True)
    plt.show()

def plot_symbols(in_ffe_scope, ffe_out_scope, DOWN_PLOT):
    fig5, (ax50,ax51) = plt.subplots(2, 1)

    markerline51, stemlines51, baseline51 = ax51.stem(ffe_out_scope[::DOWN_PLOT],'r',label="Out FFE")

    plt.setp(stemlines51, visible=False)   # no vertical lines
    baseline51.set_visible(False)          # no baseline
    ax51.grid(True)
    ax51.legend()

    markerline50, stemlines50, baseline50 = ax50.stem(in_ffe_scope[::DOWN_PLOT],'b',label="In FFE")
    plt.setp(stemlines50, visible=False)   # no vertical lines
    baseline50.set_visible(False)          # no baseline
    ax50.grid(True)
    ax50.legend()

    fig5.tight_layout()
    plt.show()

def plot_ffe_frec(FFE, fs):
    w, h = sig.freqz(FFE, worN=4096, fs=fs)  # w in Hz
    plt.figure(figsize=(10,6))
    plt.plot(w/1e9, 20*np.log10(np.abs(h)))
    plt.title("Bode Plot of FFE Rx")
    plt.ylabel("Magnitude [dB]")
    plt.ylabel("frequency")
    # plt.xlim(0, 10)   # show only up to 100 GHz
    # plt.ylim(-100, 0)   # show only up to 100 GHz
    plt.grid(True)
    plt.show()

def plot_ffe(FFE_history, DOWN_PLOT):
    plt.figure()
    plt.plot(FFE_history[::DOWN_PLOT])
    plt.title("FFE Coefficient Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Tap value")
    plt.grid(True)
    plt.show()

def pdf_in_out(in_ffe_scope, ffe_out_scope):
    # Plotting the PDF of the input data (in_ffe_scope)
    plt.figure(figsize=(10, 6))

    # Estimate PDF using histogram and normalize
    plt.hist(in_ffe_scope, bins=100, density=True, alpha=0.6, color='blue', label='Input PDF')

    # Plotting the PDF of the output data (ffe_out_scope)
    plt.hist(ffe_out_scope, bins=100, density=True, alpha=0.6, color='red', label='Output PDF')

    # Add labels and title
    plt.title('PDF of Input and Output Data')
    plt.xlabel('Amplitude')
    plt.ylabel('Probability Density')
    plt.legend()

    # Show the plot
    plt.grid(True)
    plt.show()