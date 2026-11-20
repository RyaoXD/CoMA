"""
Build master_check.csv: align master.csv rows to entries in master_rewrite.csv.

Training with --rewrite reads master_rewrite.csv (CCD text); without rewrite reads
master.csv (impression/findings). When the two files diverge, master.csv can list
images that are missing on disk. This script uses master_rewrite.csv as the
reference index and pulls the full master.csv fields for each matching row.
"""
import argparse
import os

import pandas as pd
from tqdm import tqdm

from coma.constants import (
    MIMIC_CXR_DATA_DIR,
    MIMIC_CXR_MASTER_CCD_CSV,
    MIMIC_CXR_MASTER_CHECK_CSV,
    MIMIC_CXR_MASTER_CSV,
    MIMIC_CXR_PATH_COL,
)


def _resolve_image_path(path: str) -> str:
    return os.path.join(MIMIC_CXR_DATA_DIR, "/".join(path.split("/")[1:]))


def build_master_check(verify_images: bool = True) -> pd.DataFrame:
    ccd_df = pd.read_csv(MIMIC_CXR_MASTER_CCD_CSV)
    master_df = pd.read_csv(MIMIC_CXR_MASTER_CSV)

    ccd_df["dicom_id"] = ccd_df["dicom_id"].astype(str)
    master_df["dicom_id"] = master_df["dicom_id"].astype(str)

    master_by_id = master_df.set_index("dicom_id", drop=False)
    master_ids = set(master_by_id.index)

    missing_in_master = []
    rows = []
    for _, ccd_row in tqdm(ccd_df.iterrows(), total=len(ccd_df), desc="Align rows"):
        dicom_id = ccd_row["dicom_id"]
        if dicom_id not in master_ids:
            missing_in_master.append(dicom_id)
            continue
        rows.append(master_by_id.loc[dicom_id])

    if missing_in_master:
        print(
            f"Warning: {len(missing_in_master)} dicom_id(s) in "
            f"{MIMIC_CXR_MASTER_CCD_CSV.name} not found in "
            f"{MIMIC_CXR_MASTER_CSV.name} (skipped)."
        )

    result_df = pd.DataFrame(rows).reset_index(drop=True)

    print(f"master_rewrite.csv rows: {len(ccd_df)}")
    print(f"master.csv rows: {len(master_df)}")
    print(f"master_check.csv rows: {len(result_df)}")

    extra_in_master = set(master_df["dicom_id"]) - set(ccd_df["dicom_id"])
    if extra_in_master:
        print(
            f"Rows only in master.csv (excluded from check file): "
            f"{len(extra_in_master)}"
        )

    if verify_images:
        missing_files = 0
        for path in tqdm(result_df[MIMIC_CXR_PATH_COL], desc="Verify images"):
            if not os.path.exists(_resolve_image_path(str(path))):
                missing_files += 1
        print(f"Missing image files under {MIMIC_CXR_DATA_DIR}: {missing_files}")
        if missing_files:
            print(
                "Warning: some rows still lack images on disk; "
                "review paths or dataset completeness."
            )

    result_df.to_csv(MIMIC_CXR_MASTER_CHECK_CSV, index=False, encoding="utf-8")
    print(f"Saved: {MIMIC_CXR_MASTER_CHECK_CSV}")
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Build master_check.csv from master_rewrite + master.csv"
    )
    parser.add_argument(
        "--no-verify-images",
        action="store_true",
        help="Skip checking that image files exist on disk",
    )
    args = parser.parse_args()
    build_master_check(verify_images=not args.no_verify_images)


if __name__ == "__main__":
    main()
