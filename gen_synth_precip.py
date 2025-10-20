
#!/usr/bin/env python3
"""
gen_synth_precip.py — create synthetic Stage IV and three model prediction files.

Example:
  python3 gen_synth_precip.py --n 5000 --outdir ./ --seed 123         --p_dd 0.92 --p_rr 0.80 --mw_bias 1.25 --thresh 0.1
"""
import argparse, numpy as np
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000, help="number of samples (e.g., hourly)")
    ap.add_argument("--outdir", type=str, default=".", help="output directory")
    ap.add_argument("--seed", type=int, default=42, help="random seed")
    ap.add_argument("--p_dd", type=float, default=0.92, help="dry -> dry prob")
    ap.add_argument("--p_rr", type=float, default=0.80, help="rain -> rain prob")
    ap.add_argument("--mw_bias", type=float, default=1.25, help="MWCOMB multiplicative bias")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    N = args.n

    state = np.zeros(N, dtype=int)
    for t in range(1, N):
        if state[t-1] == 0:
            state[t] = 0 if rng.random() < args.p_dd else 1
        else:
            state[t] = 1 if rng.random() < args.p_rr else 0

    intensity_rain = 0.6 * rng.gamma(shape=2.0, scale=0.8, size=N) +                          0.4 * rng.lognormal(mean=0.1, sigma=0.5, size=N)
    stage4 = np.where(state == 1, intensity_rain, 0.0)

    mrms_noise = rng.normal(0.0, 0.25, size=N)
    mrms = np.clip(stage4 * (1.0 + mrms_noise) + 0.05 * rng.random(N), 0.0, None)

    imerg_noise = rng.normal(0.0, 0.30, size=N)
    imerg = stage4 * (1.0 + imerg_noise) + 0.06 * rng.random(N)
    kernel = np.array([0.2, 0.6, 0.2])
    imerg = np.convolve(np.pad(imerg, (1,1), mode="edge"), kernel, mode="valid")
    imerg = np.clip(imerg, 0.0, None)

    mw_noise = rng.normal(0.0, 0.35, size=N)
    mw = np.clip(args.mw_bias * stage4 * (1.0 + 0.2 * mw_noise), 0.0, None)
    false_alarm_idx = np.where((stage4 < 0.05) & (rng.random(N) < 0.10))[0]
    mw[false_alarm_idx] += rng.gamma(shape=1.2, scale=0.6, size=false_alarm_idx.size)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    np.savetxt(outdir / "stage4.txt", stage4, fmt="%.5f")
    np.savetxt(outdir / "mrms_pred.txt", mrms, fmt="%.5f")
    np.savetxt(outdir / "imerg_pred.txt", imerg, fmt="%.5f")
    np.savetxt(outdir / "mwcomb_pred.txt", mw, fmt="%.5f")

    print("Wrote files to", outdir.resolve())

if __name__ == "__main__":
    main()
