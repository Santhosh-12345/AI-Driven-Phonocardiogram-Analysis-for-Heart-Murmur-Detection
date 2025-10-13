## Smart Product Pricing Challenge — Baseline Solution

This repository contains a lightweight, fully reproducible baseline for the ML Challenge 2025. It trains a text-based model on `catalog_content` and generates `test_out.csv` in the exact required format. Image download helpers are included but not required for the baseline.

### Folder structure
- `src/`: library code (features, modeling, utils)
- `train.py`: trains a model from `dataset/train.csv` and saves to `artifacts/model.joblib`
- `predict.py`: loads the saved model and writes `test_out.csv` for `dataset/test.csv`
- `artifacts/`: saved models and intermediates

### Quickstart
1) Install deps
```bash
pip install -r requirements.txt
```

2) Train the model
```bash
python train.py --train_csv dataset/train.csv --out_model artifacts/model.joblib \
  --max_features 200000 --alpha 2.0
```

3) Predict on the test set
```bash
python predict.py --test_csv dataset/test.csv --model artifacts/model.joblib --out_csv test_out.csv
```
This produces a CSV with two columns `sample_id,price` covering all rows in the test file.

### Notes
- Baseline uses TF‑IDF over normalized `catalog_content` + simple numeric features (estimated Item Pack Quantity and a percent token flag) with a Ridge regressor.
- Prices are constrained to be positive.
- A SMAPE implementation is included for validation usage.

### Optional: Image downloading
If you wish to incorporate image features, you can download images referenced by `image_link` using `src/utils.py`:
```python
from src.utils import download_images
local_paths = download_images(df.image_link, output_dir="images")
```
Handle missing downloads gracefully and avoid any external price lookup; use only the provided data.

### Reproducibility and Licensing
- Code and the exported model weights from this baseline may be submitted under the MIT License (see `LICENSE`).
- No external price lookup or scraping is used.
