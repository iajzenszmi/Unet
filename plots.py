
#!/usr/bin/env python3
"""
plots.py — visualize hourly vs daily metrics (CC, POD, FAR, CSI, Bias)
for three prediction series (MRMS / IMERG / MWCOMB) against Stage IV truth.

Usage:
  python3 plots.py --thresh 0.1 --perday 24 --png metrics.png [--show]

Notes:
- Uses only matplotlib (no seaborn).
- Creates two figures: one for hourly metrics, one for daily-aggregated metrics.
- If --show is provided, interactive windows will be shown (if a display is available).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def metrics(obs, pred, th):
    # Pearson correlation
    mo = obs.mean(); mp = pred.mean()
    so = ((obs - mo)**2).sum(); sp = ((pred - mp)**2).sum()
    cc = ((obs - mo)*(pred - mp)).sum() / np.sqrt(so*sp) if (so>0 and sp>0) else 0.0

    # Categorical metrics
    o = obs >= th; p = pred >= th
    hits = np.sum(o & p)
    misses = np.sum(o & ~p)
    falsea = np.sum(~o & p)
    pod = hits / (hits + misses) if (hits + misses) > 0 else 0.0
    far = falsea / (hits + falsea) if (hits + falsea) > 0 else 0.0
    csi = hits / (hits + misses + falsea) if (hits + misses + falsea) > 0 else 0.0
    bias = pred.sum() / obs.sum() if obs.sum() > 0 else 0.0
    return cc, pod, far, csi, bias

def aggregate_mean(arr, bucket):
    nfull = arr.size // bucket
    if nfull == 0:  # no aggregation possible
        return np.array([])
    trimmed = arr[:nfull*bucket].reshape(nfull, bucket)
    return trimmed.mean(axis=1)

def load_series():
    def read(name):
        return np.loadtxt(name).astype(float)
    obs = read("stage4.txt")
    mr  = read("mrms_pred.txt")
    im  = read("imerg_pred.txt")
    mw  = read("mwcomb_pred.txt")
    return obs, {"MRMS": mr, "IMERG": im, "MWCOMB": mw}

def plot_metrics(ax, labels, data, title):
    # data is dict: metric_name -> list of values aligned with labels
    # Plot each metric as a separate bar group figure (single axes per call, per requirement).
    # To comply with the "single plot per chart" rule, we draw a grouped bar chart in one axes.
    metrics_list = ["CC","POD","FAR","CSI","Bias"]
    x = np.arange(len(labels))
    width = 0.15

    # Order bars by metric; stack groups next to each other per model label
    # We'll offset positions for each metric.
    offsets = np.linspace(-2, 2, num=len(metrics_list)) * width * 0.9
    for i, m in enumerate(metrics_list):
        ax.bar(x + offsets[i], data[m], width, label=m)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.set_ylim(0, max(1.0, max(data.get("Bias",[1])) if "Bias" in data else 1.0))
    ax.legend()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=0.1)
    ap.add_argument("--perday", type=int, default=24)
    ap.add_argument("--png", type=str, default=None)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    obs, preds = load_series()

    # Hourly metrics
    labels = list(preds.keys())
    hourly = {"CC":[], "POD":[], "FAR":[], "CSI":[], "Bias":[]}
    for name in labels:
        cc,pod,far,csi,bias = metrics(obs, preds[name], args.thresh)
        hourly["CC"].append(cc); hourly["POD"].append(pod)
        hourly["FAR"].append(far); hourly["CSI"].append(csi); hourly["Bias"].append(bias)

    # Daily aggregated metrics
    obs_d = aggregate_mean(obs, args.perday)
    daily = {"CC":[], "POD":[], "FAR":[], "CSI":[], "Bias":[]}
    for name in labels:
        pred_d = aggregate_mean(preds[name], args.perday)
        cc,pod,far,csi,bias = metrics(obs_d, pred_d, args.thresh)
        daily["CC"].append(cc); daily["POD"].append(pod)
        daily["FAR"].append(far); daily["CSI"].append(csi); daily["Bias"].append(bias)

    # Create two figures (hourly and daily)
    fig1, ax1 = plt.subplots()
    plot_metrics(ax1, labels, hourly, "Hourly Metrics")

    fig2, ax2 = plt.subplots()
    plot_metrics(ax2, labels, daily, "Daily Metrics")

    if args.png:
        fig1.savefig(args.png.replace(".png","_hourly.png"), bbox_inches="tight", dpi=150)
        fig2.savefig(args.png.replace(".png","_daily.png"), bbox_inches="tight", dpi=150)

    if args.show:
        plt.show()

if __name__ == "__main__":
    main()
