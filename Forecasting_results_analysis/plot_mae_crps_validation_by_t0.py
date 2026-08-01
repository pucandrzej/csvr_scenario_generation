"""Create validation plots from the previously generated per-t0 CSV."""

import argparse
import os

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.paths import MAE_CRPS_RESULTS_DIR, PAPER_FIGURES_DIR


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_setting")
    parser.add_argument(
        "--input_file",
        default="validation_dynamic_reweighting_forecast_by_t0.csv",
    )
    parser.add_argument("--output_dir", default=PAPER_FIGURES_DIR)
    return parser.parse_args()


def validation_figure(results, model_setting):
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        subplot_titles=("Weighted MAE", "Weighted CRPS"),
    )
    raw = results.sort_values("t0").drop_duplicates("t0")
    colors = {"mae": "#1f77b4", "crps": "#d62728"}

    for row, metric in ((1, "mae"), (2, "crps")):
        figure.add_trace(
            go.Scatter(
                x=raw["t0"],
                y=raw[f"{metric}_raw"],
                mode="lines+markers",
                name="Raw ensemble",
                line={"color": "#666666", "dash": "dash"},
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )

        for selected_by in ("mae", "crps"):
            selected = results[results["selected_by"] == selected_by]
            figure.add_trace(
                go.Scatter(
                    x=selected["t0"],
                    y=selected[f"{metric}_weighted"],
                    mode="lines+markers",
                    name=f"Selected by {selected_by.upper()}",
                    line={"color": colors[selected_by]},
                    showlegend=row == 1,
                ),
                row=row,
                col=1,
            )

    figure.update_layout(title=model_setting, template="plotly_white")
    figure.update_yaxes(title_text="MAE", row=1, col=1)
    figure.update_yaxes(title_text="CRPS", row=2, col=1)
    figure.update_xaxes(title_text="t0", row=2, col=1)
    return figure


def main():
    args = parse_args()
    input_path = (
        args.input_file
        if os.path.isabs(args.input_file)
        else os.path.join(MAE_CRPS_RESULTS_DIR, args.input_file)
    )
    results = pd.read_csv(input_path)
    if args.model_setting:
        results = results[results["model_setting"] == args.model_setting]
    if results.empty:
        raise ValueError("No matching validation rows found")

    os.makedirs(args.output_dir, exist_ok=True)
    for (model_setting, _), model_results in results.groupby(
        ["model_setting", "model"],
        sort=False,
    ):
        name = model_setting.strip("_") or "benchmark"
        output_path = os.path.join(
            args.output_dir,
            f"validation_dynamic_reweighting_by_t0_{name}.html",
        )
        validation_figure(model_results, model_setting).write_html(output_path)
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
