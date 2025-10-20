
Fortran U-Net Target Evaluation Bundle
=====================================

Contents
--------
- precip_unet_eval.f90    Fortran evaluator (CC, POD, FAR, CSI, Bias; hourly/daily/monthly)
- Makefile                `make` builds `eval`
- run_eval.sh             Compiles and runs on the four input text files
- gen_synth_precip_calibrated.py
                          Synthetic generator with optional calibration to hit target CC/POD
                          roughly matching the JHM abstract: MRMS~(0.50,0.90), IMERG~(0.47,0.85),
                          MWCOMB~(0.45,0.82) at threshold 0.1 mm/h (adjust as needed).

Quick start
-----------
1) Generate synthetic series with calibration and default targets:
   python3 gen_synth_precip_calibrated.py --n 5000 --calibrate --thresh 0.1

2) Build and run the Fortran evaluator:
   ./run_eval.sh

Customize targets
-----------------
You can choose different target CC,POD:
   python3 gen_synth_precip_calibrated.py --n 4000 --calibrate --thresh 0.1      --target-mrms "0.52,0.88" --target-imerg "0.48,0.84" --target-mw "0.44,0.80"

Bring your own data
-------------------
Use your NetCDF/HDF sources with your own Python to flatten aligned pairs into
`stage4.txt` and `*_pred.txt` files, then run `./eval ...`.

Notes
-----
- The calibration is heuristic and aims to be *close* to the targets by tuning noise,
  bias, and false-alarm rate in the synthetic generator. Exact matching is not guaranteed.
- The Fortran program performs simple non-overlapping averaging for daily/monthly scales.
