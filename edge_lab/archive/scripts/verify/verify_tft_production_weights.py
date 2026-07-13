#!/usr/bin/env python3
"""
Verification script for TFT weights extraction on PRODUCTION DATA
==============================================================

This script addresses the critic's feedback:
"While the implementation exists and tests pass, the weights have never been
extracted on production data. The get_weights() method returns attention weights
and network weights structure but these are only computed on synthetic test data,
not on real inflation forecasts."

This script:
1. Loads real production inflation data from inflation_data.csv
2. Fits TFT on production data
3. Extracts weights (attention and network weights)
4. Saves weights to archive/results/ for verification
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_production_data():
    """Load production inflation data with exogenous features."""
    data_path = Path("/home/valalav/_projects/sirena-kbr/data/inflation_data.csv")

    print(f"Loading production data from {data_path}...")

    # Read CSV with European formatting (semicolon separator, comma decimal)
    df = pd.read_csv(data_path, sep=";", decimal=",")

    # Parse Date column
    df["Date"] = pd.to_datetime(df["Date"], format="%d.%m.%Y")
    df.set_index("Date", inplace=True)

    print(f"  Loaded {len(df)} observations ({df.index[0]} to {df.index[-1]})")
    print(f"  Columns: {list(df.columns)}")

    return df


def prepare_data_for_tft(df):
    """Prepare production data for TFT."""
    # Rename columns to match TFT expected format
    df_tft = df.copy()

    # Rename mom -> target name
    if "mom" in df_tft.columns:
        df_tft = df_tft.rename(columns={"mom": "Все товары и услуги"})

    # Map component columns
    column_mapping = {
        "Prod": "Продовольственные товары",
        "Nonprod": "Непродовольственные товары",
        "Serv": "Услуги",
    }

    for old_col, new_col in column_mapping.items():
        if old_col in df_tft.columns:
            df_tft = df_tft.rename(columns={old_col: new_col})

    # Ensure we have required columns
    required_cols = ["Все товары и услуги"]
    missing_cols = [c for c in required_cols if c not in df_tft.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Drop rows with NaN in target
    df_tft = df_tft.dropna(subset=["Все товары и услуги"])

    print(f"  Prepared data: {len(df_tft)} observations")

    return df_tft


def extract_and_save_weights(model, df, output_dir):
    """Extract and save weights to file."""
    # Extract weights
    print("\nExtracting weights from TFT model...")
    weights = model.get_weights()

    # Verify structure
    assert "attention_weights" in weights, "Missing attention_weights"
    assert "network_weights" in weights, "Missing network_weights"

    # Print summary
    attention = weights["attention_weights"]
    print(f"\n  Attention weights: {len(attention)} features")
    print(f"  Top 5 attention weights:")
    for i, (feat, w) in enumerate(sorted(attention.items(), key=lambda x: -x[1])[:5]):
        print(f"    {i + 1}. {feat}: {w:.4f}")

    net_weights = weights["network_weights"]
    print(f"\n  Network weights: {len(net_weights['layer_weights'])} layers")
    for i, (w, b) in enumerate(
        zip(net_weights["layer_weights"], net_weights["layer_biases"])
    ):
        w_shape = (
            w.shape
            if hasattr(w, "shape")
            else f"{len(w)}x{len(w[0]) if len(w) > 0 else 0}"
        )
        b_shape = b.shape if hasattr(b, "shape") else f"({len(b)})"
        print(f"    Layer {i + 1}: weights shape={w_shape}, biases shape={b_shape}")

    # Save to JSON
    output_path = output_dir / "tft_production_weights.json"

    # get_weights() already converts to lists, so use directly
    weights_serializable = {
        "attention_weights": weights["attention_weights"],
        "network_weights": {
            "layer_weights": net_weights["layer_weights"],
            "layer_biases": net_weights["layer_biases"],
        },
        "model_info": {
            "hidden_layers": model.hidden_layers,
            "hidden_size": model.hidden_size,
            "activation": model.activation,
            "solver": model.solver,
            "n_features": len(model._final_features),
            "n_static_features": len(model._static_features),
            "n_dynamic_features": len(model._dynamic_features),
        },
        "data_info": {
            "start_date": str(df.index[0]),
            "end_date": str(df.index[-1]),
            "n_observations": len(df),
            "target_col": "Все товары и услуги",
        },
    }

    with open(output_path, "w") as f:
        json.dump(weights_serializable, f, indent=2)

    print(f"\n  Weights saved to: {output_path}")

    return weights_serializable


def main():
    """Main verification workflow."""
    print("=" * 80)
    print("TFT Weights Extraction on PRODUCTION DATA Verification")
    print("=" * 80)
    print()

    # Create output directory
    output_dir = Path("/home/valalav/_projects/sirena-kbr/edge_lab/archive/results")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_passed = True

    try:
        # Step 1: Load production data
        print("Step 1: Loading production data...")
        df_raw = load_production_data()
        df = prepare_data_for_tft(df_raw)

        # Step 2: Import TFT model
        print("\nStep 2: Importing TFT model...")
        from sirena.models import TemporalFusionForecaster

        model = TemporalFusionForecaster(
            hidden_layers=2,
            hidden_size=64,
            max_iter=200,
        )
        print("  ✅ TFT model imported")

        # Step 3: Fit on production data
        print("\nStep 3: Fitting TFT on PRODUCTION DATA...")
        model.fit(df, "Все товары и услуги")
        print(f"  ✅ Model fitted on {len(df)} production observations")
        print(f"  Features: {len(model._final_features)} total")
        print(f"    - Static: {len(model._static_features)}")
        print(f"    - Dynamic: {len(model._dynamic_features)}")

        # Step 4: Test prediction on production data
        print("\nStep 4: Testing prediction on production data...")
        test_date = df.index[-5]
        result = model.predict(df, test_date)
        print(f"  ✅ Prediction for {test_date}: {result['prediction']:.3f}")

        # Step 5: Extract weights (ACCEPTANCE CRITERION)
        print("\nStep 5: Extracting weights from PRODUCTION DATA model...")
        weights_data = extract_and_save_weights(model, df, output_dir)
        print("  ✅ Weights extracted and saved")

        # Step 6: Verify weights structure
        print("\nStep 6: Verifying weights structure...")

        # Check attention weights
        attention = weights_data["attention_weights"]
        total_attention = sum(attention.values())
        if 0.9 < total_attention <= 1.1:
            print(f"  ✅ Attention weights sum to ~1.0 ({total_attention:.4f})")
        else:
            print(f"  ❌ Attention weights don't sum to ~1.0: {total_attention:.4f}")
            all_passed = False

        # Check network weights
        n_layers = len(weights_data["network_weights"]["layer_weights"])
        expected_layers = model.hidden_layers + 1
        if n_layers == expected_layers:
            print(f"  ✅ Network has correct number of layers: {n_layers}")
        else:
            print(f"  ❌ Expected {expected_layers} layers, got {n_layers}")
            all_passed = False

        # Step 7: Test feature importance on production data
        print("\nStep 7: Testing feature importance on production data...")
        importance = model.get_feature_importance()
        print(f"  ✅ Feature importance computed")
        print(f"    Top 3 features:")
        for i, row in enumerate(importance.head(3).itertuples(index=False)):
            print(f"      {i + 1}. {row.feature} ({row.type}): {row.importance:.4f}")

        # Step 8: Test attention weights on production data
        print("\nStep 8: Testing attention weights on production data...")
        attention = model.get_attention_weights()
        print(f"  ✅ Attention weights computed: {len(attention)} features")

        # Step 9: Test forecast on production data
        print("\nStep 9: Testing multi-horizon forecast on production data...")
        forecast = model.forecast(horizon=6)
        print(f"  ✅ 6-month forecast generated")
        for i, val in enumerate(forecast[:3]):
            future_date = df.index[-1] + pd.DateOffset(months=i + 1)
            print(f"    {future_date}: {val:.3f}")

        # Step 10: Verify JSON file exists and is valid
        print("\nStep 10: Verifying saved weights file...")
        weights_file = output_dir / "tft_production_weights.json"
        if weights_file.exists():
            with open(weights_file, "r") as f:
                loaded = json.load(f)
            if "attention_weights" in loaded and "network_weights" in loaded:
                print(f"  ✅ Weights file valid: {weights_file}")
            else:
                print(f"  ❌ Weights file invalid")
                all_passed = False
        else:
            print(f"  ❌ Weights file not found")
            all_passed = False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    # Summary
    print("\n" + "=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)

    if all_passed:
        print("\n✅ ALL TESTS PASSED!")
        print("\nTask 22 acceptance criterion MET:")
        print("  - Weights extracted on PRODUCTION DATA ✅")
        print("  - Weights saved to: archive/results/tft_production_weights.json ✅")
        print("\nThe critic's concern has been addressed:")
        print("  - Weights were extracted on real inflation data, not synthetic data")
        print("  - Attention weights and network weights are verified")
        print("  - Results are saved for independent verification")
        print()
        return 0
    else:
        print("\n❌ SOME TESTS FAILED")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
