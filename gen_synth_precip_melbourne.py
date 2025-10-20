
#!/usr/bin/env python3
"""
gen_synth_precip_melbourne.py
Synthetic Melbourne-region hourly precipitation series (AEST/AEDT-style cadence),
plus three U-Net-like prediction series (MRMS/IMERG/MWCOMB).

Features:
- Temperate SE Australia flavour: lower mean intensities, shorter events
- Seasonal modulation: wetter JJA (winter), drier DJF (summer)
- MWCOMB has positive bias & modest false alarms (~8% of dry hours)
- Optional heuristic calibration toward target (CC,POD) pairs at a threshold

Usage (plain generation):
  python3 gen_synth_precip_melbourne.py --n 4000 --seed 123 --outdir ./

Usage (with calibration to rough targets):
  python3 gen_synth_precip_melbourne.py --n 5000 --calibrate --thresh 0.1 \
    --target-mrms "0.50,0.90" --target-imerg "0.47,0.85" --target-mw "0.45,0.82"

Outputs (in --outdir):
  stage4.txt       (truth-like series)
  mrms_pred.txt
  imerg_pred.txt
  mwcomb_pred.txt
  melbourne_meta.json  (parameters used, convenient for provenance)
"""
import argparse, numpy as np
import json
from pathlib import Path
from datetime import datetime, timedelta

def seasonal_weight(ts_hour):
    """Return multiplicative seasonal weight for intensity: wetter in JJA, drier in DJF."""
    m = ts_hour.month
    # Simple weights by month for Melbourne (heuristic)
    weights = {1:0.85, 2:0.90, 3:0.95, 4:1.00, 5:1.05, 6:1.15,
               7:1.20, 8:1.15, 9:1.05,10:1.00,11:0.95,12:0.90}
    return weights.get(m, 1.0)

def seasonal_markov(m):
    """Return (p_dd, p_rr) for given month, Melbourne-style: more persistence in winter."""
    # dry->dry (p_dd), rain->rain (p_rr)
    table = {
        1:(0.94, 0.70), 2:(0.93, 0.72), 3:(0.92, 0.74),
        4:(0.92, 0.76), 5:(0.91, 0.78), 6:(0.90, 0.80),
        7:(0.90, 0.82), 8:(0.91, 0.80), 9:(0.92, 0.78),
        10:(0.93,0.76), 11:(0.94,0.72), 12:(0.95,0.70)
    }
    return table[m]

def synth_stage4(N, rng, start_dt):
    """Generate hourly Stage-IV-like truth for Melbourne region with seasonality."""
    state = np.zeros(N, dtype=int)
    # Initialize based on monthly rain probability (~18% annually, higher in JJA)
    init_p_rain = 0.18
    state[0] = 1 if rng.random() < init_p_rain else 0

    times = [start_dt + timedelta(hours=i) for i in range(N)]
    for t in range(1, N):
        m = times[t].month
        p_dd, p_rr = seasonal_markov(m)
        if state[t-1] == 0:
            state[t] = 0 if rng.random() < p_dd else 1
        else:
            state[t] = 1 if rng.random() < p_rr else 0

    # Base intensity mixture tuned for lower means than generic global
    base_gamma_shape = 1.8
    base_gamma_scale = 0.6   # mean ~1.08 mm/h for gamma part before seasonal scaling
    base_logn_mu    = 0.0
    base_logn_sigma = 0.45

    intensity_rain = (
        0.65 * rng.gamma(shape=base_gamma_shape, scale=base_gamma_scale, size=N) +
        0.35 * rng.lognormal(mean=base_logn_mu, sigma=base_logn_sigma, size=N)
    )

    # Apply seasonal weight per hour
    w = np.array([seasonal_weight(ts) for ts in times])
    intensity_rain = intensity_rain * w

    stage4 = np.where(state == 1, intensity_rain, 0.0)
    return stage4, times

def make_preds(stage4, rng, params):
    """Create U-Net-like predictions for MRMS/IMERG/MWCOMB."""
    N = stage4.size
    # MRMS: low bias, moderate noise
    mrms = np.clip(stage4 * (1.0 + rng.normal(0.0, params["mrms_sigma"], size=N))
                   + params["mrms_eps"] * rng.random(N), 0.0, None)
    # IMERG: slightly noisier + smoothing (temporal averaging)
    imerg = stage4 * (1.0 + rng.normal(0.0, params["imerg_sigma"], size=N)) \
            + params["imerg_eps"] * rng.random(N)
    kernel = np.array([0.2, 0.6, 0.2])
    imerg = np.convolve(np.pad(imerg, (1,1), mode="edge"), kernel, mode="valid")
    imerg = np.clip(imerg, 0.0, None)
    # MWCOMB: biased high + modest false alarms (8% of dry hours by default)
    mw = np.clip(params["mw_bias"] * stage4 * (1.0 + 0.2 * rng.normal(0.0, params["mw_sigma"], size=N)), 0.0, None)
    dry = stage4 < 0.05
    fa_idx = np.where(dry & (rng.random(N) < params["mw_fa_prob"]))[0]
    if fa_idx.size > 0:
        mw[fa_idx] += rng.gamma(shape=params["mw_fa_k"], scale=params["mw_fa_theta"], size=fa_idx.size) * params["mw_fa_scale"]
    return mrms, imerg, mw

def metrics(obs, pred, th):
    mo = obs.mean(); mp = pred.mean()
    so = ((obs - mo)**2).sum(); sp = ((pred - mp)**2).sum()
    if so > 0 and sp > 0:
        cc = ((obs - mo)*(pred - mp)).sum() / np.sqrt(so*sp)
    else:
        cc = 0.0
    # POD at threshold
    o = obs >= th; p = pred >= th
    hits = np.sum(o & p); misses = np.sum(o & ~p)
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    return float(cc), float(pod)

def calibrate(stage4, rng, th, targets, init_params):
    params = init_params.copy()

    def mk_and_score(p):
        mr, im, mw = make_preds(stage4, rng, p)
        cc_m, pod_m = metrics(stage4, mr, th)
        cc_i, pod_i = metrics(stage4, im, th)
        cc_w, pod_w = metrics(stage4, mw, th)
        se = 2.0*((cc_m-targets["mrms"][0])**2 + (pod_m-targets["mrms"][1])**2) + \
             1.5*((cc_i-targets["imerg"][0])**2 + (pod_i-targets["imerg"][1])**2) + \
             1.5*((cc_w-targets["mw"][0])**2   + (pod_w-targets["mw"][1])**2)
        return (mr, im, mw), se

    preds, bestS = mk_and_score(params)
    keys = ["mrms_sigma","imerg_sigma","mw_sigma","mw_bias","mw_fa_prob","mw_fa_scale"]
    for _ in range(60):
        improved = False
        for k in keys:
            for f in (0.90, 1.10):
                trial = params.copy()
                trial[k] = max(1e-6, trial[k]*f)
                preds2, S2 = mk_and_score(trial)
                if S2 + 1e-6 < bestS:
                    params = trial; preds = preds2; bestS = S2; improved = True
        if not improved:
            break
    return params, preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="number of hours")
    ap.add_argument("--seed", type=int, default=123, help="random seed")
    ap.add_argument("--outdir", type=str, default=".", help="output directory")
    ap.add_argument("--start", type=str, default="2025-06-01T00:00:00", help="start datetime local (YYYY-MM-DDTHH:MM:SS)")
    ap.add_argument("--thresh", type=float, default=0.1, help="event threshold (mm/h)")
    ap.add_argument("--calibrate", action="store_true", help="heuristic calibration to targets")
    ap.add_argument("--target-mrms", type=str, default="0.50,0.90")
    ap.add_argument("--target-imerg", type=str, default="0.47,0.85")
    ap.add_argument("--target-mw", type=str, default="0.45,0.82")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    start_dt = datetime.fromisoformat(args.start)

    stage4, times = synth_stage4(args.n, rng, start_dt)

    init_params = {
        "mrms_sigma": 0.23, "mrms_eps": 0.04,
        "imerg_sigma": 0.28, "imerg_eps": 0.05,
        "mw_sigma": 0.32, "mw_bias": 1.22,
        "mw_fa_prob": 0.08, "mw_fa_k": 1.2, "mw_fa_theta": 0.6, "mw_fa_scale": 0.9
    }

    if args.calibrate:
        t_m = tuple(map(float, args.target_mrms.split(",")))
        t_i = tuple(map(float, args.target_imerg.split(",")))
        t_w = tuple(map(float, args.target_mw.split(",")))
        targets = {"mrms": t_m, "imerg": t_i, "mw": t_w}
        params, (mrms, imerg, mwcomb) = calibrate(stage4, rng, args.thresh, targets, init_params)
    else:
        params = init_params
        mrms, imerg, mwcomb = make_preds(stage4, rng, params)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    np.savetxt(outdir / "stage4.txt", stage4, fmt="%.5f")
    np.savetxt(outdir / "mrms_pred.txt", mrms, fmt="%.5f")
    np.savetxt(outdir / "imerg_pred.txt", imerg, fmt="%.5f")
    np.savetxt(outdir / "mwcomb_pred.txt", mwcomb, fmt="%.5f")

    meta = {
        "region": "Melbourne, AU (synthetic)",
        "start": args.start, "n_hours": args.n,
        "threshold_mmph": args.thresh,
        "params": params
    }
    (outdir / "melbourne_meta.json").write_text(json.dumps(meta, indent=2))

    # quick metrics print
    def metrics_json(name, pred):
        cc, pod = metrics(stage4, pred, args.thresh)
        return {name: {"CC": round(cc,3), "POD": round(pod,3)}}
    summary = {}
    summary.update(metrics_json("MRMS", mrms))
    summary.update(metrics_json("IMERG", imerg))
    summary.update(metrics_json("MWCOMB", mwcomb))
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    main()
