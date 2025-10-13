import argparse
import os
import joblib
import pandas as pd

from src.modeling import train_model, TrainConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", default="dataset/train.csv")
    p.add_argument("--out_model", default="artifacts/model.joblib")
    p.add_argument("--max_features", type=int, default=200000)
    p.add_argument("--alpha", type=float, default=2.0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.out_model), exist_ok=True)

    df = pd.read_csv(args.train_csv)
    required_cols = {"sample_id", "catalog_content", "price"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in train.csv: {missing}")

    cfg = TrainConfig(max_features=args.max_features, alpha=args.alpha)
    pipe = train_model(df, cfg)

    joblib.dump(pipe, args.out_model)
    print(f"Saved model to {args.out_model}")


if __name__ == "__main__":
    main()
