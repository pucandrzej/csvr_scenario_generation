"""Create one all-model validation plot from the per-t0 CSV."""

import argparse
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.paths import GENERAL_STRATEGY_RESULTS, PAPER_FIGURES_DIR


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
    setting = str(model_setting).strip("_") or "benchmark"
    return f"{setting} ({model})"


def validation_figure(results, title="Validation metrics by t0"):
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("MAE", "CRPS"),
    )
    group_columns = ["model_setting", "model"]
    groups = (
        results.groupby(group_columns, sort=False)
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
    styles = {"raw": "dash", "mae": "solid", "crps": "dot"}

    for index, ((model_setting, model), model_results) in enumerate(groups):
        label = _model_label(model_setting, model) if model_setting is not None else None
        color = colors[index % len(colors)]
        raw = model_results.sort_values("t0").drop_duplicates("t0")

        for row, metric in ((1, "mae"), (2, "crps")):
            figure.add_trace(
                go.Scatter(
                    x=raw["t0"],
                    y=raw[f"{metric}_raw"],
                    mode="lines+markers",
                    name=f"{label} - raw" if label else "Raw ensemble",
                    line={"color": color, "dash": styles["raw"]},
                    showlegend=row == 1,
                ),
                row=row,
                col=1,
            )

            for selected_by in ("mae", "crps"):
                selected = model_results[model_results["selected_by"] == selected_by]
                figure.add_trace(
                    go.Scatter(
                        x=selected["t0"],
                        y=selected[f"{metric}_weighted"],
                        mode="lines+markers",
                        name=(
                            f"{label} - selected by {selected_by.upper()}"
                            if label
                            else f"Selected by {selected_by.upper()}"
                        ),
                        line={"color": color, "dash": styles[selected_by]},
                        showlegend=row == 1,
                    ),
                    row=row,
                    col=1,
                )

    figure.update_layout(
        title=title,
        template="plotly_white",
        legend={"orientation": "h", "y": -0.2},
    )
    figure.update_yaxes(title_text="MAE", row=1, col=1)
    figure.update_yaxes(title_text="CRPS", row=2, col=1)
    figure.update_xaxes(title_text="t0", row=2, col=1)
    return figure


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


if __name__ == "__main__":
    main()
