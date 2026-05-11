"""File parses the test (validation) results of trading strategies and prepares results in format coherent with the one used in the paper"""

import pandas as pd
import os

from config.paths import TEST_STRATEGIES_MEASURES_DIR, PAPER_TABLES_DIR

RAW_PATTERN = "test_trading_strategy_measures_"
OUTPUT_FILE = "strategies_summary_table.csv"

# HELPER


def parse_filename(fname):
    name = os.path.basename(fname).replace(".csv", "")
    base = name.replace(RAW_PATTERN, "")

    parts = base.split("_")

    if len(parts) < 7:
        raise ValueError(f"Unexpected filename format: {fname}")

    # Band type = last two tokens
    band_type = "_".join(parts[-2:])

    model_type = parts[-3]
    direction = parts[-4]
    one_sided = parts[-5] == "True"
    underlying_column = parts[-6]

    underlying_model = "_".join(parts[:-6])

    return {
        "underlying_model": underlying_model,
        "underlying_column": underlying_column,
        "one_sided": one_sided,
        "direction": direction,
        "model_type": model_type,
        "band_type": band_type,
    }


def classify_strategy(meta):
    """
    Map metadata to SELLER/PROP/MEDIAN/BANDS/etc
    """

    out = {}

    # Seller / Prop
    if meta["direction"] == "-1":
        out["agent_type"] = "SELLER"
    else:
        out["agent_type"] = "PROP"

    # Median / Bands
    if meta["model_type"] == "median":
        out["model_class"] = "MEDIAN"
    else:
        out["model_class"] = "BANDS"

    # Risk
    if meta["band_type"] == "risk_seeking":
        out["risk_type"] = "RISK SEEKING"
    elif meta["band_type"] == "risk_averse":
        out["risk_type"] = "RISK AVERSE"
    else:
        out["risk_type"] = "---"

    # SVS / ---
    if "dual_coeff" in meta["underlying_model"]:
        out["scenario_type"] = "SVS"
    else:
        out["scenario_type"] = "---"

    # HIST / WEATHER
    if "weather" in meta["underlying_model"]:
        out["data_type"] = "WEATHER"
    elif "hist" in meta["underlying_model"]:
        out["data_type"] = "HIST"
    else:
        out["data_type"] = "---"

    return out


# GATHER THE RESULTS FROM EVERY STRATEGY EVALUATION FILE

for subdir in ["kernel", "mae", "static"]:
    rows = []
    for file in os.listdir(os.path.join(TEST_STRATEGIES_MEASURES_DIR, subdir)):
        if "_naive_" in file:
            print(f"Naive results should be analysed manually. Skipping: {file}")
            continue
        meta = parse_filename(file)
        cls = classify_strategy(meta)

        df = pd.read_csv(os.path.join(TEST_STRATEGIES_MEASURES_DIR, subdir, file))

        dynamic_df = df[df["weights"] != "_"]
        baseline_df = df[df["weights"] == "_"]

        dynamic = dynamic_df.loc[dynamic_df["Sortino_ratio"].idxmax()]
        baseline = baseline_df.loc[baseline_df["Sortino_ratio"].idxmax()]

        row = {
            # Grouping
            "agent_type": cls["agent_type"],
            "model_class": cls["model_class"],
            "risk_type": cls["risk_type"],
            "scenario_type": cls["scenario_type"],
            "data_type": cls["data_type"],
            # Source
            "file": file,
            # -----------------------------
            # Dynamic
            # -----------------------------
            "SCP": dynamic["param1"],
            "p": dynamic["param2"],
            "lambda": dynamic["param3"],
            "threshold": dynamic["threshold"],
            "weights": dynamic["weights"],
            "profit": dynamic["profit"],
            "std_minus": dynamic["std_minus"],
            "rtp": dynamic["rtp"],
            "topk": dynamic["topk"],
            "sortino": dynamic["Sortino_ratio"],
            # -----------------------------
            # Baseline (non-dynamic)
            # -----------------------------
            "static_profit": baseline["profit"],
            "static_std_minus": baseline["std_minus"],
            "static_rtp": baseline["rtp"],
            "static_topk": baseline["topk"],
            "static_sortino": baseline["Sortino_ratio"],
        }

        rows.append(row)

    # BUILD FINAL TABLE

    summary = pd.DataFrame(rows)

    # Order for readability
    summary = summary.sort_values(
        [
            "agent_type",
            "model_class",
            "risk_type",
            "scenario_type",
            "data_type",
        ]
    )

    # Rename for presentation
    summary = summary.rename(
        columns={
            "agent_type": "AGENT",
            "model_class": "MODEL",
            "risk_type": "RISK",
            "scenario_type": "SCENARIO",
            "data_type": "DATA",
            # Dynamic
            "sortino": "SORTINO",
            "profit": "PROFIT",
            "std_minus": "STD-",
            "rtp": "RTP",
            "topk": "TOPK",
            # Baseline
            "_sortino": "_SORTINO",
            "_profit": "_PROFIT",
            "_std_minus": "_STD-",
            "_rtp": "_RTP",
            "_topk": "_TOPK",
        }
    )

    # Save flat CSV
    summary_table_path = os.path.join(PAPER_TABLES_DIR, subdir + "_" + OUTPUT_FILE)
    summary.to_csv(summary_table_path, index=False)

    print(f"Saved {subdir} results summary to {summary_table_path}")
