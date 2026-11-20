import csv
import os

import numpy as np
import librosa
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt

# settings, i need to move these into a config file soon
SEED = 0
N_TRIALS = 300
SR = 22050
DURATION = 1.0
N_HARMONICS = 8
F0_MIN, F0_MAX = 80.0, 800.0            
PYIN_FMIN = librosa.note_to_hz("C2")    
PYIN_FMAX = librosa.note_to_hz("C6")    
GRID_RES = 0.1                          
                                        
                                        
GROSS_ERROR_CENTS = 600
OUT_DIR = "results"


def make_harmonic_tone(f0, sr, duration, n_harmonics, rng):
    """Sum of harmonics with 1/k amplitudes and random phases."""
    t = np.arange(int(sr * duration)) / sr
    y = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        phase = rng.uniform(0, 2 * np.pi)
        y += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t + phase)
    return y / np.max(np.abs(y))


def cents_error(f_est, f_true):
    """Signed error in cents: 1200 * log2(estimate / truth)."""
    return 1200.0 * np.log2(f_est / f_true)


def predicted_grid_error(f0_true):
    """
    Error (cents) we would see if pYIN simply snapped to the nearest grid
    point. Grid points sit at PYIN_FMIN * 2**(k * GRID_RES / 12).
    Positive means the nearest grid point is above the true f0.
    """
    step = 100.0 * GRID_RES                         # grid spacing in cents
    pos = 1200.0 * np.log2(f0_true / PYIN_FMIN)     
    dist_to_nearest = ((pos + step / 2) % step) - step / 2
    return -dist_to_nearest


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng_f0 = np.random.default_rng(SEED)

    
    f0_values = np.exp(rng_f0.uniform(np.log(F0_MIN), np.log(F0_MAX), N_TRIALS))

    rows = []
    for i, f0 in enumerate(f0_values):
        rng_tone = np.random.default_rng(SEED + 1 + i)  # per-trial seed -> reproducible
        y = make_harmonic_tone(f0, SR, DURATION, N_HARMONICS, rng_tone)

        f0_est, voiced, _ = librosa.pyin(
            y, fmin=PYIN_FMIN, fmax=PYIN_FMAX, sr=SR, resolution=GRID_RES
        )
        mask = voiced & ~np.isnan(f0_est)
        if not mask.any():
            rows.append((i, f0, np.nan, 0.0, np.nan, predicted_grid_error(f0)))
            continue

        err = cents_error(f0_est[mask], f0)
        rows.append((
            i,
            f0,
            float(np.median(err)),
            float(mask.mean()),
            float(np.mean(np.abs(err) > GROSS_ERROR_CENTS)),
            float(predicted_grid_error(f0)),
        ))

   
    csv_path = os.path.join(OUT_DIR, "sweep_pyin_clean.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["trial", "f0_true_hz", "median_error_cents",
                    "voiced_fraction", "gross_error_rate", "predicted_grid_error_cents"])
        w.writerows(rows)

    data = np.array(rows, dtype=float)
    f0_arr, err_arr, pred_arr = data[:, 1], data[:, 2], data[:, 5]
    ok = ~np.isnan(err_arr)

   
    print(f"Trials: {N_TRIALS}  (no voiced frames in {np.sum(~ok)})")
    print(f"Mean |median error|:   {np.mean(np.abs(err_arr[ok])):.2f} cents")
    print(f"Max  |median error|:   {np.max(np.abs(err_arr[ok])):.2f} cents")
    print(f"Mean gross-error rate: {100 * np.nanmean(data[:, 4]):.2f}%")
    corr = np.corrcoef(err_arr[ok], pred_arr[ok])[0, 1]
    print(f"Correlation, measured vs predicted-from-grid error: {corr:.3f}")

    # error vs f0 
    plt.figure(figsize=(7, 4))
    plt.scatter(f0_arr[ok], err_arr[ok], s=10)
    plt.axhline(0, color="gray", lw=0.8)
    plt.xscale("log")
    plt.xlabel("True f0 (Hz)")
    plt.ylabel("Median error (cents)")
    plt.title("pYIN error on clean synthetic tones")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sweep_error_vs_f0.png"), dpi=150)
    plt.close()

    # measured vs predicted-from-grid
    lim = 100.0 * GRID_RES / 2 + 1
    plt.figure(figsize=(4.5, 4.5))
    plt.scatter(pred_arr[ok], err_arr[ok], s=10)
    plt.plot([-lim, lim], [-lim, lim], color="gray", lw=0.8, label="y = x")
    plt.xlabel("Predicted grid error (cents)")
    plt.ylabel("Measured median error (cents)")
    plt.title("Does pYIN just snap to its grid?")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "sweep_grid_check.png"), dpi=150)
    plt.close()

    print(f"Saved: {csv_path} and two plots in {OUT_DIR}/")


if __name__ == "__main__":
    main()
