
#!/usr/bin/env python3
"""
flatten_netcdf_to_lines.py — convert gridded precipitation datasets to
aligned one-value-per-line vectors for the Fortran evaluator.

Requires: xarray, numpy
pip install xarray netCDF4

Example:
  python3 flatten_netcdf_to_lines.py         --obs stage4.nc --obs-var precip         --pred mrms.nc --pred-var precip_hat         --time "2024-01-01T00:00:00/2024-03-31T23:00:00"         --resample 1H         --bbox 140, -40, 155, -30         --out stage4.txt mrms_pred.txt
"""
import argparse
import numpy as np
import xarray as xr
from pathlib import Path

def parse_bbox(s):
    if s is None:
        return None
    parts = [float(p.strip()) for p in s.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be: minlon, minlat, maxlon, maxlat")
    return dict(minlon=parts[0], minlat=parts[1], maxlon=parts[2], maxlat=parts[3])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--obs-var", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pred-var", required=True)
    ap.add_argument("--time", default=None, help="ISO window START/END (inclusive)")
    ap.add_argument("--resample", default=None, help="e.g., 1H, 3H, 1D")
    ap.add_argument("--bbox", default=None, help="minlon,minlat,maxlon,maxlat")
    ap.add_argument("--out", nargs=2, required=True, metavar=("OBS_TXT","PRED_TXT"))
    args = ap.parse_args()

    bbox = parse_bbox(args.bbox)

    ds_o = xr.open_dataset(args.obs)
    ds_p = xr.open_dataset(args.pred)
    da_o = ds_o[args.obs_var]
    da_p = ds_p[args.pred_var]

    # Harmonize coordinates
    if "time" in da_o.coords and "time" in da_p.coords:
        if args.time:
            t0, t1 = args.time.split("/")
            da_o = da_o.sel(time=slice(t0, t1))
            da_p = da_p.sel(time=slice(t0, t1))
        # Resample if asked (sum for accumulations, mean for rates; here assume rate -> mean)
        if args.resample:
            da_o = da_o.resample(time=args.resample).mean()
            da_p = da_p.resample(time=args.resample).mean()

    if bbox is not None and set(["lon","lat"]).issubset(da_o.coords):
        da_o = da_o.sel(lon=slice(bbox["minlon"], bbox["maxlon"]),
                        lat=slice(bbox["minlat"], bbox["maxlat"]))
    if bbox is not None and set(["lon","lat"]).issubset(da_p.coords):
        da_p = da_p.sel(lon=slice(bbox["minlon"], bbox["maxlon"]),
                        lat=slice(bbox["minlat"], bbox["maxlat"]))

    # Align (inner join on shared coords)
    da_o, da_p = xr.align(da_o, da_p, join="inner")

    # Flatten to 1D; drop NaNs synchronously
    o_vals = da_o.values.ravel()
    p_vals = da_p.values.ravel()
    mask = np.isfinite(o_vals) & np.isfinite(p_vals)
    o_vals = o_vals[mask]
    p_vals = p_vals[mask]

    np.savetxt(Path(args.out[0]), o_vals, fmt="%.5f")
    np.savetxt(Path(args.out[1]), p_vals, fmt="%.5f")
    print("Wrote", len(o_vals), "aligned pairs.")

if __name__ == "__main__":
    main()
