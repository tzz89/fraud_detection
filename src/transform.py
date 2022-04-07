"""
Holds all the transformation functions such as
golden dataset generation and feature enrichment
"""
import os
from sklearn.model_selection import train_test_split
import pandas as pd
from pandas import DataFrame

def generate_golden_test(
    raw_data_fp: str,
    label_col_name: str,
    output_dir: str,
    test_size: float = 0.15,
    random_state: int = 42,
) -> None:
    """
    This function loads in the raw data and perform a
    stratify train test split to create the golden dataset
    """
    raw_df = pd.read_csv(raw_data_fp, index_col=0)
    train_df, golden_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df[label_col_name],
    )

    train_df.to_csv(os.path.join(output_dir, "train.csv"))
    golden_df.to_csv(os.path.join(output_dir, "golden.csv"))
