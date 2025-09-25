# signal_fft_timed.py
# Generate v(t) = 10*sin(w*t) + 6*sin(3*w*t) + 4*cos(5*w*t) + 4*sin(5*w*t)
# Fs = SAMPLES_PER_CYCLE * 60 Hz
# Exports:
#   - sinal.csv: t,v
#   - fft.csv:   f,|V|,V.arg_degrees,V.real,V.imag
# Notes:
#   - |V| is single-sided amplitude scaled by NSIG (NOT Nfft)
#   - Phase is set to 0 deg when |V| < EPS (atan2 is skipped)
#   - Prints elapsed times at the end.

import time
import numpy as np
import pandas as pd

# -------- User parameters --------
SAMPLES_PER_CYCLE = 128      # set 128 (original) or 32, etc.
CYCLES = 10                  # number of fundamental cycles
F0 = 60.0                    # fundamental frequency (Hz)
EPS = 1e-6                   # threshold for magnitude ~ 0 (single-sided)

# -------- Derived parameters --------
NSIG = SAMPLES_PER_CYCLE * CYCLES
FS = float(SAMPLES_PER_CYCLE) * F0
OMEGA = 2.0 * np.pi * F0


def next_pow2(n: int) -> int:
    """Return the smallest power of two >= n (n >= 1)."""
    if n <= 1:
        return 1
    return 1 << (int(n - 1).bit_length())


def generate_signal(nsig: int, fs: float):
    """Generate time vector t and signal v(t) for nsig samples at sampling rate fs."""
    n = np.arange(nsig, dtype=np.float64)
    t = n / fs
    v = 10.0 * np.sin(OMEGA * t)
    v = v + 6.0 * np.sin(3.0 * OMEGA * t)
    v = v + 4.0 * np.cos(5.0 * OMEGA * t)
    v = v + 4.0 * np.sin(5.0 * OMEGA * t)
    return t, v


def compute_fft_single_sided(v: np.ndarray, fs: float, nsig: int) -> pd.DataFrame:
    """Compute single-sided FFT with amplitudes referenced to nsig (not Nfft)."""
    nfft = next_pow2(nsig)
    X = np.fft.fft(v, n=nfft)
    half = nfft // 2

    # Frequency axis (0 .. Fs/2)
    k = np.arange(half + 1, dtype=np.float64)
    f = k * fs / float(nfft)

    # Raw complex bins (0..half)
    Xh = X[:half + 1]

    # Single-sided scaling (relative to nsig)
    scale = np.full(half + 1, 2.0 / float(nsig), dtype=np.float64)
    scale[0] = 1.0 / float(nsig)
    if nfft % 2 == 0:
        scale[half] = 1.0 / float(nsig)

    # Magnitude and normalized complex parts
    amp = np.abs(Xh)
    mag = amp * scale
    re_n = Xh.real * scale
    im_n = Xh.imag * scale

    # Phase in degrees, set to 0 where |V| < EPS
    phase_deg = np.zeros_like(mag)
    mask = mag > float(EPS)
    phase_deg[mask] = np.degrees(np.arctan2(Xh.imag[mask], Xh.real[mask]))

    df = pd.DataFrame({
        "f": f,
        "|V|": mag,
        "V.arg_degrees": phase_deg,
        "V.real": re_n,
        "V.imag": im_n,
    })
    return df


def main() -> None:
    t0 = time.perf_counter()

    # 1) Generate and save time-domain signal
    t, v = generate_signal(NSIG, FS)
    df_time = pd.DataFrame({"t": t, "v": v})
    df_time.to_csv("sinal.csv", index=False)

    t1 = time.perf_counter()

    # 2) FFT (single-sided, scaled by NSIG) and save
    df_fft = compute_fft_single_sided(v, FS, NSIG)
    df_fft.to_csv("fft.csv", index=False)

    t2 = time.perf_counter()

    gen_time = t1 - t0
    fft_time = t2 - t1
    total_time = t2 - t0

    print(f"Generated NSIG={NSIG} samples, Fs={FS:.0f} Hz")
    print("Files written: sinal.csv, fft.csv")
    print(f"Time: generate/save = {gen_time*1000:.2f} ms, "
          f"fft/save = {fft_time*1000:.2f} ms, "
          f"total = {total_time*1000:.2f} ms")


if __name__ == "__main__":
    main()
