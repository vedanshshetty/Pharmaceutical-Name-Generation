"""Generate the data figures for the manuscript into paper/figures/.

Reads only committed artifacts under paper/results/ (manifest.json and the three
per-seed sweep CSVs), so the figures are reproducible offline with nothing but
matplotlib and pandas. Run from the repository root:

    python paper/make_figures.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "paper", "results")
FIGS = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIGS, exist_ok=True)
SEEDS = [1, 2, 3]

plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200,
})


def k(n):
    return f"{n:,}"


def fig_corpus():
    m = json.load(open(os.path.join(RESULTS, "manifest.json")))
    sc = m["screening_corpus"]
    tc = m["training_corpus"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.5))

    ax1.set_title("(a) corpus construction")
    labels = ["raw snapshot rows", "screening universe"]
    raw = sc["raw_rows"]
    gen, br = sc["kept_generic"], sc["kept_brand"]
    ax1.bar(labels[0], raw, color="#b0b0b0", width=0.55)
    ax1.bar(labels[1], gen, color="#4c72b0", width=0.55, label=f"generic {k(gen)}")
    ax1.bar(labels[1], br, bottom=gen, color="#c44e52", width=0.55,
            label=f"brand {k(br)}")
    ax1.text(0, raw, k(raw), ha="center", va="bottom", fontsize=8.5)
    ax1.text(1, sc["kept_total_unique"] + 500, k(sc["kept_total_unique"]),
             ha="center", va="bottom", fontsize=8.5)
    ax1.set_ylim(0, 52000)
    ax1.legend(loc="upper right", frameon=False)
    ax1.set_ylabel("names")

    ax2.set_title("(b) training assets")
    cats = ["extracted\nprefixes", "unique\nprefixes", "brand\nmarks",
            "INN/USAN\nstems"]
    vals = [tc["prefixes_extracted"], tc["unique_prefixes"],
            tc["brand_names"], tc["stems_available"]]
    cols = ["#4c72b0", "#55a868", "#c44e52", "#8172b2"]
    ax2.bar(cats, vals, color=cols, width=0.6)
    for i, v in enumerate(vals):
        ax2.text(i, v + 60, k(v), ha="center", va="bottom", fontsize=8.5)
    ax2.set_ylim(0, 5400)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_corpus.png"), bbox_inches="tight")
    print("wrote paper/figures/fig_corpus.png")


def fig_strata():
    dfs = []
    for s in SEEDS:
        df = pd.read_csv(os.path.join(RESULTS, f"sweep_all_attempts_s{s}.csv"))
        df["seed"] = s
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)

    order = ["roomy", "crowded", "saturated", "brand"]
    per = []
    for s in SEEDS:
        for d in order:
            sub = merged[(merged["seed"] == s) & (merged["difficulty"] == d)]
            acc = sub["accepted"].mean()
            eff = len(sub) / max(1, sub["accepted"].sum())
            per.append({"seed": s, "difficulty": d, "accept": acc, "effort": eff})
    p = pd.DataFrame(per)

    means = p.groupby("difficulty")[["accept", "effort"]].mean().loc[order]
    stds = p.groupby("difficulty")[["accept", "effort"]].std().loc[order]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6))
    x = range(4)
    ax1.set_title("(a) acceptance rate by difficulty")
    ax1.bar(x, means["accept"], yerr=stds["accept"], capsize=3, width=0.55,
            color=["#55a868", "#c44e52", "#8172b2", "#4c72b0"], alpha=0.9)
    for xx, s in zip(x, order):
        for r in p[p["difficulty"] == s]["accept"]:
            ax1.plot(xx + 0.18, r, "o", color="black", ms=3.5)
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(order)
    ax1.set_ylim(0, 0.62)
    ax1.set_ylabel("accepted / screened")
    for xx, s in zip(x, order):
        ax1.text(xx, means["accept"][s] + stds["accept"][s] + 0.02,
                 f"{means['accept'][s]:.2f}", ha="center", fontsize=8.5)

    ax2.set_title("(b) screening effort (screened per admitted name)")
    ax2.bar(x, means["effort"], yerr=stds["effort"], capsize=3, width=0.55,
            color=["#55a868", "#c44e52", "#8172b2", "#4c72b0"], alpha=0.9)
    for xx, s in zip(x, order):
        for r in p[p["difficulty"] == s]["effort"]:
            ax2.plot(xx + 0.18, r, "o", color="black", ms=3.5)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(order)
    ax2.set_ylim(0, 5.2)
    ax2.set_ylabel("screened per admitted name")
    for xx, s in zip(x, order):
        ax2.text(xx, means["effort"][s] + stds["effort"][s] + 0.15,
                 f"{means['effort'][s]:.1f}", ha="center", fontsize=8.5)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "fig_strata.png"), bbox_inches="tight")
    print("wrote paper/figures/fig_strata.png")


if __name__ == "__main__":
    fig_corpus()
    fig_strata()
    plt.close("all")