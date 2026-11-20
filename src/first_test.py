import numpy as np
import librosa


SEED = 0
SR = 22050          
DURATION = 2.0        
F0_TRUE = 113.7  #off-grid on purpose right now, so pYIN can't land exactly on it       
N_HARMONICS = 8
OCTAVE_THRESHOLD_CENTS = 600  


def make_harmonic_tone(f0, sr, duration, n_harmonics):
    """Sum of harmonics with 1/k amplitudes and random phases."""
    rng = np.random.default_rng(SEED)
    t = np.arange(int(sr * duration)) / sr
    y = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        phase = rng.uniform(0, 2 * np.pi)
        y += (1.0 / k) * np.sin(2 * np.pi * k * f0 * t + phase)
    return y / np.max(np.abs(y))  


def cents_error(f_est, f_true):
    """Signed error in cents: 1200 * log2(estimate / truth)."""
    return 1200.0 * np.log2(f_est / f_true)


def main():
    y = make_harmonic_tone(F0_TRUE, SR, DURATION, N_HARMONICS)

    f0_est, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),  
        fmax=librosa.note_to_hz("C6"),   
        sr=SR,
    )

    
    mask = voiced_flag & ~np.isnan(f0_est)
    if not mask.any():
        print("pYIN found no voiced frames.")
        return

    errors = cents_error(f0_est[mask], F0_TRUE)
    gross = np.abs(errors) > OCTAVE_THRESHOLD_CENTS

    print(f"True f0:               {F0_TRUE:.2f} Hz")
    print(f"Voiced frames:         {mask.sum()} of {len(f0_est)}")
    print(f"Mean error:            {errors.mean():+.2f} cents")
    print(f"Median error:          {np.median(errors):+.2f} cents")
    print(f"Mean absolute error:   {np.abs(errors).mean():.2f} cents")
    print(f"Gross-error rate:      {100 * gross.mean():.1f}% "
          f"(> {OCTAVE_THRESHOLD_CENTS} cents)")


if __name__ == "__main__":
    main()
