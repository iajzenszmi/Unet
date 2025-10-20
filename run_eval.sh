
#!/usr/bin/env bash
set -euo pipefail

# 1) Generate Melbourne synthetic series (calibrated to abstract targets by default)
python3 gen_synth_precip_melbourne.py --n 5000 --calibrate --thresh 0.1

# 2) Build evaluator
make

# 3) Run evaluator
./eval stage4.txt mrms_pred.txt imerg_pred.txt mwcomb_pred.txt --thresh 0.1 --perday 24 --permonth 720

# 4) Make plots (PNG always; add --show to see interactive windows if a display exists)
python3 plots.py --thresh 0.1 --perday 24 --png metrics_melbourne.png "$@"
