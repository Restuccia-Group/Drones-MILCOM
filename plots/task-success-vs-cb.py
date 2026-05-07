#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".", help="Directory containing the .txt files.")
    ap.add_argument("--out-prefix", default="Task-Success-Rate-vs-CB",
                    help="Output prefix (without extension).")
    ap.add_argument("--use-tex", action="store_true", default=False,
                    help="Enable LaTeX rendering (requires LaTeX install).")
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help="Multiply all font sizes by this factor.")
    return ap.parse_args()


# Filenames use these indices
CB_FILE_COUNTS    = [1, 2, 3, 4]
# X-axis displays these labels
CB_DISPLAY_COUNTS = [1, 2, 3, 4]

UL_TRACES = [
    ("DUSA",   "ul_dusa"),
    ("Trad",   "ul_trad"),
    ("Static", "ul_sem5qi-static"),
    ("Rand",   "ul_dynamic-random"),
]

DL_TRACES = [
    ("DUSA",   "dl_dusa"),
    ("Trad",   "dl_trad"),
    ("Static", "dl_sem5qi-static"),
    ("Rand",   "dl_dynamic-random"),
]

COLORS = {
    "DUSA":   "#F34C4C",
    "Trad":   "#719BDB",
    "Static": "#2CA02C",
    "Rand":   "#FF7F0E",
}
LINESTYLES = {
    "DUSA":   "-",
    "Trad":   "--",
    "Static": ":",
    "Rand":   (0, (3, 1, 1, 1)),
}
MARKERS = {
    "DUSA":   "o",
    "Trad":   "^",
    "Static": "D",
    "Rand":   "P",
}


def extract_stats(filepath):
    """Parse 'Overall avg' and 'Std dev' from the FINAL SUMMARY block."""
    text = filepath.read_text(errors="replace")
    m_avg = re.search(r"Overall avg\s*:\s*([0-9]+(?:\.[0-9]+)?(?:e[+\-]?\d+)?)", text, re.IGNORECASE)
    m_std = re.search(r"Std dev\s*:\s*([0-9]+(?:\.[0-9]+)?(?:e[+\-]?\d+)?)", text, re.IGNORECASE)
    if m_avg is None:
        raise ValueError("Could not find 'Overall avg' in {}".format(filepath))
    if m_std is None:
        raise ValueError("Could not find 'Std dev' in {}".format(filepath))
    return float(m_avg.group(1)), float(m_std.group(1))


def setup_style(use_tex, font_scale):
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["axes.linewidth"] = 1.5
    mpl.rcParams["font.size"] = 14 * font_scale
    mpl.rcParams["text.usetex"] = bool(use_tex)
    if use_tex:
        mpl.rcParams["text.latex.preamble"] = (
            r"\usepackage{newtxmath}"
            r"\renewcommand{\rmdefault}{ptm}\fontseries{m}\selectfont"
        )


def plot_traces(axes_avg, axes_std, traces, data_dir,
                MARKERSIZE, MARKEREDGEWIDTH, LINEWIDTH):
    """Plot avg and std traces onto the provided axes pair."""
    for label, prefix in traces:
        y_avg = []
        y_std = []
        for cb in CB_FILE_COUNTS:
            fname = data_dir / "{prefix}_{cb}cb.txt".format(prefix=prefix, cb=cb)
            if not fname.exists():
                print("[WARN] Missing file: {} -- inserting NaN".format(fname))
                y_avg.append(np.nan)
                y_std.append(np.nan)
            else:
                avg, std = extract_stats(fname)
                y_avg.append(avg)
                y_std.append(std)

        plot_kwargs = dict(
            color=COLORS[label],
            linestyle=LINESTYLES[label],
            marker=MARKERS[label],
            markersize=MARKERSIZE,
            markeredgewidth=MARKEREDGEWIDTH,
            markeredgecolor="black",
            linewidth=LINEWIDTH,
            label=label,
        )

        axes_avg.plot(CB_DISPLAY_COUNTS, y_avg, **plot_kwargs)
        axes_std.plot(CB_DISPLAY_COUNTS, y_std, **plot_kwargs)

def style_axes(ax, ylabel, AXIS_LABEL_FONTSIZE, TICK_LABEL_FONTSIZE, label_y=0.42):
    ax.grid(True, which="both", linestyle="--", linewidth=0.8, alpha=0.45)
    ax.set_xlabel("Compute Config", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=10, position=(0, label_y))
    ax.set_xticks(CB_DISPLAY_COUNTS)
    ax.tick_params(axis="both", which="both",
                   labelsize=TICK_LABEL_FONTSIZE, width=1.2, length=5, pad=6)

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    setup_style(args.use_tex, args.font_scale)

    AXIS_LABEL_FONTSIZE = 24 * args.font_scale
    TICK_LABEL_FONTSIZE = 22 * args.font_scale
    LEGEND_FONTSIZE     = 22 * args.font_scale

    LINEWIDTH       = 2.5
    MARKERSIZE      = 10
    MARKEREDGEWIDTH = 1.0

    # 2x2 grid: row 0 = UL, row 1 = DL; col 0 = Avg, col 1 = STD
    fig, axes = plt.subplots(2, 2, figsize=(8.7, 7.1))

    ax_ul_avg, ax_ul_std = axes[0]
    ax_dl_avg, ax_dl_std = axes[1]

    plot_traces(ax_ul_avg, ax_ul_std, UL_TRACES, data_dir,
                MARKERSIZE, MARKEREDGEWIDTH, LINEWIDTH)
    plot_traces(ax_dl_avg, ax_dl_std, DL_TRACES, data_dir,
                MARKERSIZE, MARKEREDGEWIDTH, LINEWIDTH)

    style_axes(ax_ul_avg, "Success Rate (Avg)", AXIS_LABEL_FONTSIZE, TICK_LABEL_FONTSIZE)
    style_axes(ax_ul_std, "Success Rate (STD)", AXIS_LABEL_FONTSIZE, TICK_LABEL_FONTSIZE)
    style_axes(ax_dl_avg, "Success Rate (Avg)", AXIS_LABEL_FONTSIZE, TICK_LABEL_FONTSIZE)
    style_axes(ax_dl_std, "Success Rate (STD)", AXIS_LABEL_FONTSIZE, TICK_LABEL_FONTSIZE)

    # Single legend above both subplots
    handles, labels = ax_ul_avg.get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=len(UL_TRACES),
        fontsize=LEGEND_FONTSIZE,
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="gray",
        handlelength=2.5,
        handletextpad=0.7,
        columnspacing=1.0,
        labelspacing=0.5,
        borderpad=0.6,
    )
    leg.get_frame().set_linewidth(1.2)

    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.91])
    plt.subplots_adjust(hspace=0.43, wspace=0.5)

    out_prefix = Path(args.out_prefix)
    plt.savefig("{}.pdf".format(out_prefix), bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.savefig("{}.png".format(out_prefix), dpi=300, bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.show()

    print("[OK] Saved: {}.pdf and {}.png".format(out_prefix, out_prefix))


if __name__ == "__main__":
    main()
