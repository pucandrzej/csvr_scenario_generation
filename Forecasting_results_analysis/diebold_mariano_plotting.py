"""Paper-style plotting helpers for Diebold-Mariano test results."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


Paper_width = 6.30045
plt.style.use("paper_style.mplstyle")

PAPER_NAMING = {
    "approach": {
        "benchmark": "naïve",
        "hist_insample": "historical",
        "weather_scenarios": "fundamental",
    },
    "model": {
        "MULTI_prediction": "multi",
        "CHAIN_prediction": "chain",
    },
    "window": {
        "None": "unrestricted",
        28: "28",
        182: "182",
    },
    "selection": {
        True: "SVS",
        False: "no SVS",
    },
}


def paper_label(config):
    """Return the paper-style label for one forecast configuration."""
    approach = PAPER_NAMING["approach"][config.approach]
    window = PAPER_NAMING["window"][config.required_scenarios]
    if config.approach == "benchmark":
        return f"{approach}, {window}"

    model = PAPER_NAMING["model"][config.column]
    selection = PAPER_NAMING["selection"][bool(config.wasserstein)]
    return f"{approach}, {model}, {selection}, {window}"


def _dm_colormap():
    """Return the green-yellow-red-black colormap used for DM p-values."""
    red = np.r_[np.linspace(0, 1, 50), np.linspace(1, 0.5, 50)[1:], 0]
    green = np.r_[np.linspace(0.5, 1, 50), np.zeros(50)]
    cmap = mpl.colors.ListedColormap(np.c_[red, green, np.zeros(100)])
    cmap.set_bad("0.65")
    return cmap


def _contiguous_spans(keys):
    """Return labels and inclusive bounds for contiguous configuration groups."""
    spans = []
    start = 0
    for stop in range(1, len(keys) + 1):
        if stop == len(keys) or keys[stop] != keys[start]:
            if keys[start] is not None:
                spans.append((keys[start][-1], start, stop - 1))
            start = stop
    return spans


def _group_spans(configs):
    """Return paper-style approach, model, and selection group spans."""
    approaches, models, selections = [], [], []
    for config in configs:
        approach = PAPER_NAMING["approach"][config.approach]
        approaches.append((approach,))
        if config.approach == "benchmark":
            models.append(None)
            selections.append(None)
            continue

        model = PAPER_NAMING["model"][config.column]
        selection = PAPER_NAMING["selection"][bool(config.wasserstein)]
        models.append((approach, model))
        selections.append((approach, model, selection))

    return {
        "approach": _contiguous_spans(approaches),
        "model": _contiguous_spans(models),
        "selection": _contiguous_spans(selections),
    }


def _add_grouped_axis_labels(ax, configs):
    """Add hierarchical labels and separators around both heatmap axes."""
    windows = [PAPER_NAMING["window"][config.required_scenarios] for config in configs]
    ticks = np.arange(len(configs))
    ax.set_xticks(ticks, windows, rotation=90)
    ax.set_yticks(ticks, windows)

    spans = _group_spans(configs)
    levels = (
        ("selection", (-0.25, -0.19), (-0.28, -0.20), 0.3),
        ("model", (-0.31, -0.25), (-0.36, -0.28), 0.5),
        ("approach", (-0.37, -0.31), (-0.44, -0.36), 0.8),
    )
    x_transform = ax.get_xaxis_transform()
    y_transform = ax.get_yaxis_transform()
    fontsize = mpl.rcParams["xtick.labelsize"]
    line_color = "0.55"

    for tick in ticks:
        ax.add_patch(
            Rectangle(
                (tick - 0.5, -0.19),
                1,
                0.19,
                transform=x_transform,
                fill=False,
                edgecolor=line_color,
                linewidth=0.3,
                clip_on=False,
            )
        )
        ax.add_patch(
            Rectangle(
                (-0.20, tick - 0.5),
                0.20,
                1,
                transform=y_transform,
                fill=False,
                edgecolor=line_color,
                linewidth=0.3,
                clip_on=False,
            )
        )

    for level, x_band, y_band, linewidth in levels:
        for label, start, stop in spans[level]:
            center = (start + stop) / 2
            current_x_band = x_band
            current_y_band = y_band
            if level == "approach" and configs[start].approach == "benchmark":
                current_x_band = (-0.37, -0.19)
                current_y_band = (-0.44, -0.20)
            x_label_position = sum(current_x_band) / 2
            y_label_position = sum(current_y_band) / 2
            ax.add_patch(
                Rectangle(
                    (start - 0.5, current_x_band[0]),
                    stop - start + 1,
                    current_x_band[1] - current_x_band[0],
                    transform=x_transform,
                    fill=False,
                    edgecolor=line_color,
                    linewidth=linewidth,
                    clip_on=False,
                )
            )
            ax.add_patch(
                Rectangle(
                    (current_y_band[0], start - 0.5),
                    current_y_band[1] - current_y_band[0],
                    stop - start + 1,
                    transform=y_transform,
                    fill=False,
                    edgecolor=line_color,
                    linewidth=linewidth,
                    clip_on=False,
                )
            )
            ax.text(
                center,
                x_label_position,
                label,
                transform=x_transform,
                ha="center",
                va="top",
                fontsize=fontsize,
                clip_on=False,
            )
            ax.text(
                y_label_position,
                center,
                label,
                transform=y_transform,
                rotation=90,
                ha="center",
                va="center",
                fontsize=fontsize,
                clip_on=False,
            )
            if stop < len(configs) - 1:
                boundary = stop + 0.5
                ax.axvline(boundary, color=line_color, linewidth=linewidth)
                ax.axhline(boundary, color=line_color, linewidth=linewidth)


def plot_dm(p_values, configs, metric, output_stem):
    """Write the complete p-value heatmap as PNG and PDF."""
    shown = p_values.copy()
    np.fill_diagonal(shown, np.nan)
    fig, ax = plt.subplots(figsize=(Paper_width, Paper_width))
    image = ax.imshow(shown, cmap=_dm_colormap(), vmin=0, vmax=0.1)
    ticks = np.arange(len(configs))
    boundaries = np.arange(-0.5, len(configs) + 0.5)
    ax.vlines(
        boundaries,
        -0.5,
        len(configs) - 0.5,
        color="0.65",
        linewidth=0.15,
    )
    ax.hlines(
        boundaries,
        -0.5,
        len(configs) - 0.5,
        color="0.65",
        linewidth=0.15,
    )
    _add_grouped_axis_labels(ax, configs)
    ax.plot(ticks, ticks, "x", color="0.2", markersize=5, markeredgewidth=0.7)
    ax.tick_params(axis="x", length=0, labelsize=5.5, pad=1)
    ax.tick_params(axis="y", length=0, labelsize=5.5)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.03)
    colorbar.set_label("p-value")
    colorbar.set_ticks([0, 0.025, 0.05, 0.075, 0.1])
    fig.tight_layout()
    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
