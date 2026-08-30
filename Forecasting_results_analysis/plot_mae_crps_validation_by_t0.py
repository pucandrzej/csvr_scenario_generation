"""Create one all-model validation plot from the per-t0 CSV."""

import argparse
import os
import shutil

import matplotlib.pyplot as plt
import pandas as pd
import plotly.graph_objects as go
from matplotlib.lines import Line2D
from plotly.subplots import make_subplots

from config.paths import GENERAL_STRATEGY_RESULTS, PAPER_FIGURES_DIR


plt.style.use("paper_style.mplstyle")
Paper_width = 6.30045
PLOT_MODEL_LABELS = {
    "_____None____": "naïve",
    "_hist_insample_None_True_dual_coeff": "historical, SVS",
    "_hist_insample_None_False_None": "historical",
    "_weather_scenarios_None_True_dual_coeff": "fundamental, SVS",
    "_weather_scenarios_None_False_None": "fundamental",
}
if shutil.which("latex") is None:
    plt.rcParams["text.usetex"] = False
    plt.rcParams["font.serif"] = ["DejaVu Serif"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_setting")
    parser.add_argument(
        "--input_file",
        default="validation_dynamic_reweighting_forecast_by_t0.csv",
    )
    parser.add_argument("--output_dir", default=PAPER_FIGURES_DIR)
    return parser.parse_args()


def _model_label(model_setting, model):
    if model_setting in PLOT_MODEL_LABELS:
        return PLOT_MODEL_LABELS[model_setting]
    setting = str(model_setting).strip("_") or "benchmark"
    return f"{setting} ({model})"


def _ordered_model_groups(results):
    order = {setting: index for index, setting in enumerate(PLOT_MODEL_LABELS)}
    return sorted(
        results.groupby(["model_setting", "model"], sort=False),
        key=lambda group: order.get(group[0][0], len(order)),
    )


def validation_figure(results, title="Validation metrics by t0"):
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("MAE", "CRPS"),
    )
    group_columns = ["model_setting", "model"]
    groups = (
        _ordered_model_groups(results)
        if set(group_columns).issubset(results.columns)
        else [((None, None), results)]
    )
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]
    styles = {
        "raw": "dash",
        "mae": "solid",
        "crps": "dot",
        "mae_weights": "dashdot",
    }

    for index, ((model_setting, model), model_results) in enumerate(groups):
        label = _model_label(model_setting, model) if model_setting is not None else None
        color = colors[index % len(colors)]
        raw = model_results.sort_values("t0").drop_duplicates("t0")

        for row, metric in ((1, "mae"), (2, "crps")):
            legendgroup = f"{model_setting}_{model}_raw"
            figure.add_trace(
                go.Scatter(
                    x=raw["t0"],
                    y=raw[f"{metric}_raw"],
                    mode="lines+markers",
                    name=f"{label} - raw" if label else "Raw ensemble",
                    line={"color": color, "dash": styles["raw"]},
                    legendgroup=legendgroup,
                    showlegend=row == 1,
                ),
                row=row,
                col=1,
            )

            for selected_by in ("mae", "crps", "mae_weights"):
                selected = model_results[model_results["selected_by"] == selected_by]
                if selected.empty:
                    continue
                method = (
                    "Inverse MAE"
                    if selected_by == "mae_weights"
                    else f"Selected by {selected_by.upper()}"
                )
                legendgroup = f"{model_setting}_{model}_{selected_by}"
                figure.add_trace(
                    go.Scatter(
                        x=selected["t0"],
                        y=selected[f"{metric}_weighted"],
                        mode="lines+markers",
                        name=(
                            f"{label} - {method}"
                            if label
                            else method
                        ),
                        line={"color": color, "dash": styles[selected_by]},
                        legendgroup=legendgroup,
                        showlegend=row == 1,
                    ),
                    row=row,
                    col=1,
                )

    figure.update_layout(
        title=title,
        template="plotly_white",
        legend={"orientation": "h", "groupclick": "togglegroup"},
    )
    figure.update_yaxes(title_text="MAE [EUR/MWh]", row=1, col=1)
    figure.update_yaxes(title_text="CRPS [EUR/MWh]", row=2, col=1)
    figure.update_xaxes(title_text="t0", row=2, col=1)
    return figure


def paper_validation_figure(results):
    """Return a compact paper figure with metric-specific kernel curves."""
    fig, axes = plt.subplots(
        2, 1, figsize=(Paper_width, 0.70 * Paper_width), sharex=True
    )
    colors = plt.get_cmap("tab10").colors
    line_styles = {"raw": "--", "mae_weights": "-.", "kernel": "-"}
    method_labels = {
        "raw": "Static",
        "mae_weights": "MAE-weighted",
        "kernel": "Kernel-weighted",
    }
    model_handles = []

    for index, ((model_setting, model), model_results) in enumerate(
        _ordered_model_groups(results)
    ):
        color = colors[index % len(colors)]
        label = _model_label(model_setting, model)
        model_handles.append(
            Line2D(
                [],
                [],
                color=color,
                linewidth=5,
                solid_capstyle="butt",
                label=label,
            )
        )
        raw = model_results.sort_values("t0").drop_duplicates("t0")
        mae_weights = model_results[model_results["selected_by"] == "mae_weights"]

        for axis, metric, selected_by in zip(
            axes, ("mae", "crps"), ("mae", "crps")
        ):
            kernel = model_results[model_results["selected_by"] == selected_by]
            for data, column, style in (
                (raw, f"{metric}_raw", "raw"),
                (mae_weights, f"{metric}_weighted", "mae_weights"),
                (kernel, f"{metric}_weighted", "kernel"),
            ):
                axis.plot(
                    data["t0"],
                    data[column],
                    color=color,
                    linestyle=line_styles[style],
                    linewidth=0.65,
                )

    axes[0].set_ylabel("MAE [EUR/MWh]")
    axes[1].set_ylabel("CRPS [EUR/MWh]")
    axes[1].set_xlabel(r"$t_0$")
    for axis in axes:
        axis.grid(alpha=0.2, linewidth=0.3)

    method_handles = [
        Line2D(
            [],
            [],
            color="0.25",
            linewidth=0.7,
            linestyle=line_styles[key],
            label=label,
        )
        for key, label in method_labels.items()
    ]
    fig.legend(
        handles=model_handles,
        loc="center left",
        ncol=len(model_handles),
        frameon=False,
        bbox_to_anchor=(0.215, 0.095),
        handlelength=1.0,
        handletextpad=0.45,
        columnspacing=1.25,
    )
    fig.legend(
        handles=method_handles,
        loc="center left",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.215, 0.055),
    )
    fig.text(0.06, 0.095, "Scenario generation:", ha="left", va="center")
    fig.text(0.06, 0.055, "Weighting method:", ha="left", va="center")
    fig.subplots_adjust(top=0.98, bottom=0.21, hspace=0.08)
    return fig


def main():
    args = parse_args()
    input_path = (
        args.input_file
        if os.path.isabs(args.input_file)
        else os.path.join(
            GENERAL_STRATEGY_RESULTS, "REWEIGHTING_MAE_CRPS", args.input_file
        )
    )
    results = pd.read_csv(input_path)
    if args.model_setting:
        results = results[results["model_setting"] == args.model_setting]
    if results.empty:
        raise ValueError("No matching validation rows found")

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(
        args.output_dir,
        "validation_dynamic_reweighting_by_t0.html",
    )
    validation_figure(results).write_html(output_path)
    print(f"Saved: {output_path}")

    pdf_path = os.path.join(
        args.output_dir,
        "validation_dynamic_reweighting_by_t0.pdf",
    )
    paper_validation_figure(results).savefig(pdf_path, bbox_inches="tight")
    plt.close("all")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
