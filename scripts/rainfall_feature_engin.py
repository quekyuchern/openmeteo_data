# scripts/rainfall_feature_engin.py
# ------------------------------------------------------------
# CLI: read processed rainfall CSV -> write per-cell temporal features
# Default input: ../data/sg_rainfall_processed.csv
# Default output: ../data/rainfall_temporal_features.csv
# ------------------------------------------------------------

import os
import sys
import argparse
import pandas as pd

# Allow 'src' imports when run as a script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.feature_engin import engineer_temporal_features_all_cells


def main():
    parser = argparse.ArgumentParser(description="Engineer per-cell temporal rainfall features (no aggregation).")
    parser.add_argument(
        "--input",
        type=str,
        default="./data/processed/sg_rainfall_processed.csv",
        help="Path to processed rainfall CSV with columns 'timestamp' and 'lat,lon' headers."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/processed/rainfall_temporal_features.csv",
        help="Path to write the engineered features CSV."
    )

    # Optional knobs (you can expose more if needed)
    parser.add_argument("--dry_threshold", type=float, default=0.2, help="Threshold (mm) for dry hour.")
    parser.add_argument("--wet_threshold", type=float, default=0.2, help="Threshold (mm) for 'is raining now'.")

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input, parse_dates=["timestamp"])
    if "timestamp" not in df.columns:
        raise ValueError("Input CSV must have a 'timestamp' column.")
    df = df.set_index("timestamp").sort_index()

    # Build features (defaults match the 'starter set')
    feats = engineer_temporal_features_all_cells(
        df,
        dry_threshold=args.dry_threshold,
        wet_threshold=args.wet_threshold,
        # You can override the defaults here if you like, e.g.:
        # lag_hours=(1,2,3,4,5,6,12,24),
        # sum_windows=(3,6,12,24),
        # max_windows=(3,6),
        # api_halflives=(6,24,168),
        # future_sum_horizons=(1,3,6,12),
        # future_max_horizons=(3,6,12),
        # ttp_horizons=(6,12),
        # frontshare_h=12,
    )

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    feats.to_csv(args.output, index=True)
    print(f"Saved features to: {os.path.abspath(args.output)}")
    print(f"Shape: {feats.shape[0]} rows x {feats.shape[1]} features")


if __name__ == "__main__":
    main()
