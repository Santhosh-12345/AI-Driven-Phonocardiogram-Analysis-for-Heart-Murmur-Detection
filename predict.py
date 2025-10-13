import argparse
import os
import joblib
import pandas as pd
import numpy as np

from src.modeling import predict_prices


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--test_csv", default="dataset/test.csv")
    p.add_argument("--model", default="artifacts/model.joblib")
    p.add_argument("--out_csv", default="test_out.csv")
    return p.parse_args()


def main():
    args = parse_args()

    df_test = pd.read_csv(args.test_csv)
    required_cols = {"sample_id", "catalog_content"}
    missing = required_cols - set(df_test.columns)
    if missing:
        raise ValueError(f"Missing columns in test.csv: {missing}")

    model = joblib.load(args.model)
    preds = predict_prices(model, df_test)

    out = pd.DataFrame({
        "sample_id": df_test["sample_id"],
        "price": preds.astype(float)
    })
    # Ensure positive floats
    out["price"] = np.maximum(out["price"].astype(float), 0.01)

    out.to_csv(args.out_csv, index=False)
    print(f"Wrote predictions to {args.out_csv} with {len(out)} rows")


if __name__ == "__main__":
    main()
