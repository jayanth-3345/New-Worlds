from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "bingbangboom/exoplanet-transit-detection"
OUTPUT = Path("services/dl-service/data/processed")
OUTPUT.mkdir(parents=True, exist_ok=True)

LABELS = {
    "NO_SIGNAL": 0,
    "FALSE_POSITIVE": 1,
    "PLANET": 2,
}

def process_split(split):
    print(f"\nProcessing {split} dataset...")

    file_path = hf_hub_download(
        repo_id=REPO,
        filename=f"{split}.parquet",
        repo_type="dataset",
    )

    df = pd.read_parquet(
        file_path,
        columns=["disposition", "flux_global"],
    )

    X = np.empty((len(df), 1, 201), dtype=np.float32)
    y = df["disposition"].map(LABELS).to_numpy(dtype=np.int64)

    zero_curves = 0

    for i, raw_flux in enumerate(df["flux_global"]):
        flux = np.frombuffer(raw_flux, dtype="<f4").copy()

        if len(flux) != 201 or not np.isfinite(flux).all():
            raise ValueError(f"Invalid light curve at row {i}")

        median = np.median(flux)

        if median == 0 and np.all(flux == 0):
            X[i, 0, :] = 0.0
            zero_curves += 1
        elif median <= 0:
            raise ValueError(f"Invalid non-zero light curve at row {i}")
        else:
            X[i, 0, :] = (flux / median - 1.0) * 1000.0

    np.savez_compressed(
        OUTPUT / f"{split}.npz",
        X=X,
        y=y,
    )

    print(f"Samples: {len(y)}")
    print(f"Features: {X.shape}")
    print(f"Zero-filled curves retained: {zero_curves}")
    print(f"Labels: {dict(zip(*np.unique(y, return_counts=True)))}")
    print(f"Saved: {OUTPUT / f'{split}.npz'}")

    del df, X, y

for split in ["train", "val", "test"]:
    process_split(split)

print("\nAll datasets preprocessed successfully!")
print("Classes:", LABELS)
