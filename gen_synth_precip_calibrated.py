
#!/usr/bin/env python3
"""
gen_synth_precip_calibrated.py
Create synthetic Stage IV and predictions (MRMS/IMERG/MWCOMB).
Optionally calibrate toward target CC & POD (approximate).

Examples:
  # Plain generation
  python3 gen_synth_precip_calibrated.py --n 4000

  # Calibrate to target metrics (roughly)
  python3 gen_synth_precip_calibrated.py --n 4000 --calibrate       --target-mrms "0.50,0.90" --target-imerg "0.47,0.85" --target-mw "0.45,0.82"       --thresh 0.1
"""
import argparse, numpy as np
from pathlib import Path

def metrics(obs, pred, th):
    # Pearson r
    mo = obs.mean(); mp = pred.mean()
    so = ((obs - mo)**2).sum(); sp = ((pred - mp)**2).sum()
    if so > 0 and sp > 0:
        cc = ((obs - mo)*(pred - mp)).sum() / np.sqrt(so*sp)
    else:
        cc = 0.0
    # categorical
    o = obs >= th; p = pred >= th
    hits = np.sum(o & p); misses = np.sum(o & ~p); falsea = np.sum(~o & p)
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    return float(cc), float(pod)

def synth_stage4(N, rng, p_dd=0.92, p_rr=0.80):
    state = np.zeros(N, dtype=int)
    for t in range(1, N):
        if state[t-1] == 0:
            state[t] = 0 if rng.random() < p_dd else 1
        else:
            state[t] = 1 if rng.random() < p_rr else 0
    intensity_rain = 0.6 * rng.gamma(shape=2.0, scale=0.8, size=N) +                      0.4 * rng.lognormal(mean=0.1, sigma=0.5, size=N)
    stage4 = np.where(state == 1, intensity_rain, 0.0)
    return stage4

def make_preds(stage4, rng, params):
    # params: dict with keys for each product
    # MRMS: bias ~1, noise sigma, small baseline jitter
    mrms = np.clip(stage4 * (1.0 + rng.normal(0.0, params["mrms_sigma"], size=stage4.size))
                   + params["mrms_eps"] * rng.random(stage4.size), 0.0, None)

    # IMERG: slightly noisier, smoothed
    imerg = stage4 * (1.0 + rng.normal(0.0, params["imerg_sigma"], size=stage4.size))             + params["imerg_eps"] * rng.random(stage4.size)
    kernel = np.array([0.2, 0.6, 0.2])
    imerg = np.convolve(np.pad(imerg, (1,1), mode="edge"), kernel, mode="valid")
    imerg = np.clip(imerg, 0.0, None)

    # MWCOMB: positive bias + false alarms
    mw = np.clip(params["mw_bias"] * stage4 * (1.0 + 0.2 * rng.normal(0.0, params["mw_sigma"], size=stage4.size)), 0.0, None)
    dry = stage4 < 0.05
    # false alarms controlled by mw_fa_prob and mw_fa_scale
    fa_idx = np.where(dry & (rng.random(stage4.size) < params["mw_fa_prob"]))[0]
    if fa_idx.size > 0:
        mw[fa_idx] += rng.gamma(shape=params["mw_fa_k"], scale=params["mw_fa_theta"], size=fa_idx.size) * params["mw_fa_scale"]
    return mrms, imerg, mw

def calibrate(stage4, rng, th, targets):
    # Simple heuristic search on sigma/bias/false-alarm params to hit target CC,POD roughly.
    params = {
        "mrms_sigma": 0.25, "mrms_eps": 0.05,
        "imerg_sigma": 0.30, "imerg_eps": 0.06,
        "mw_sigma": 0.35, "mw_bias": 1.25,
        "mw_fa_prob": 0.10, "mw_fa_k": 1.2, "mw_fa_theta": 0.6, "mw_fa_scale": 1.0
    }

    def step(current, key, factor_up, factor_down, score_fn):
        # Try nudging a param up or down to reduce score
        best = None; best_score = 1e9; best_params = current.copy()
        for f in [factor_down, 1.0, factor_up]:
            trial = current.copy()
            trial[key] = max(1e-6, trial[key] * f)
            mr, im, mw = make_preds(stage4, rng, trial)
            s = score_fn(mr, im, mw)
            if s < best_score:
                best = (mr, im, mw); best_score = s; best_params = trial
        return best_params, best

    def score(mr, im, mw):
        cc_m, pod_m = metrics(stage4, mr, th)
        cc_i, pod_i = metrics(stage4, im, th)
        cc_w, pod_w = metrics(stage4, mw, th)
        # weighted squared error to targets
        se = 0.0
        se += 2.0*((cc_m - targets["mrms"][0])**2 + (pod_m - targets["mrms"][1])**2)
        se += 1.5*((cc_i - targets["imerg"][0])**2 + (pod_i - targets["imerg"][1])**2)
        se += 1.5*((cc_w - targets["mw"][0])**2 + (pod_w - targets["mw"][1])**2)
        return se

    # Initial preds
    mr, im, mw = make_preds(stage4, rng, params)
    bestS = score(mr, im, mw)

    keys = ["mrms_sigma", "imerg_sigma", "mw_sigma", "mw_bias", "mw_fa_prob", "mw_fa_scale"]
    for _ in range(60):
        improved = False
        for k in keys:
            params_new, preds = step(params, k, 1.10, 0.90, score)
            S = score(*preds)
            if S + 1e-6 < bestS:
                params = params_new; bestS = S
                mr, im, mw = preds
                improved = True
        if not improved:
            break

    # Final metrics report
    return params, mr, im, mw

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--outdir", type=str, default=".")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p_dd", type=float, default=0.92)
    ap.add_argument("--p_rr", type=float, default=0.80)
    ap.add_argument("--thresh", type=float, default=0.1)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--target-mrms", type=str, default="0.50,0.90")
    ap.add_argument("--target-imerg", type=str, default="0.47,0.85")
    ap.add_argument("--target-mw", type=str, default="0.45,0.82")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    stage4 = synth_stage4(args.n, rng, args.p_dd, args.p_rr)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    if args.calibrate:
        t_m = tuple(map(float, args.target_mrms.split(",")))
        t_i = tuple(map(float, args.target_imerg.split(",")))
        t_w = tuple(map(float, args.target_mw.split(",")))
        targets = {"mrms": t_m, "imerg": t_i, "mw": t_w}
        params, mr, im, mw = calibrate(stage4, rng, args.thresh, targets)
        print("Calibration params:", params)
    else:
        params = {
            "mrms_sigma": 0.25, "mrms_eps": 0.05,
            "imerg_sigma": 0.30, "imerg_eps": 0.06,
            "mw_sigma": 0.35, "mw_bias": 1.25,
            "mw_fa_prob": 0.10, "mw_fa_k": 1.2, "mw_fa_theta": 0.6, "mw_fa_scale": 1.0
        }
        mr, im, mw = make_preds(stage4, rng, params)

    # Save
    np.savetxt(outdir / "stage4.txt", stage4, fmt="%.5f")
    np.savetxt(outdir / "mrms_pred.txt", mr, fmt="%.5f")
    np.savetxt(outdir / "imerg_pred.txt", im, fmt="%.5f")
    np.savetxt(outdir / "mwcomb_pred.txt", mw, fmt="%.5f")

    # Quick metrics print
    import json
    cc_m, pod_m = metrics(stage4, mr, args.thresh)
    cc_i, pod_i = metrics(stage4, im, args.thresh)
    cc_w, pod_w = metrics(stage4, mw, args.thresh)
    print(json.dumps({
        "MRMS": {"CC": round(cc_m,3), "POD": round(pod_m,3)},
        "IMERG": {"CC": round(cc_i,3), "POD": round(pod_i,3)},
        "MWCOMB": {"CC": round(cc_w,3), "POD": round(pod_w,3)}
    }, indent=2))

if __name__ == "__main__":
    main()
