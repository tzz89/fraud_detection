"""
Holds all the transformation functions such as
golden dataset generation and feature enrichment
"""
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from pandas import DataFrame
import joblib

def generate_golden_test(
    raw_data_fp: str,
    label_col_name: str,
    output_dir: str,
    test_size: float = 0.15,
    random_state: int = 42,
    feature_cross: bool = False,
    train_filename:str = "train.csv",
    golden_filename:str = "golden.csv"
) -> None:
    """
    This function loads in the raw data and perform a
    stratify train test split to create the golden dataset
    """
    raw_df = pd.read_csv(raw_data_fp, index_col=0)
    

    if feature_cross:
        column_names = list(raw_df.columns)
        feature_names = [item for item in column_names if item !=label_col_name]
        
        poly_feature_transformer = PolynomialFeatures(degree=2)
        transformed_feature = poly_feature_transformer.fit_transform(raw_df[feature_names])
        transformed_feature_df = pd.DataFrame(transformed_feature, columns=poly_feature_transformer.get_feature_names_out())
        transformed_feature_df[label_col_name] = raw_df[label_col_name].values
        raw_df = transformed_feature_df
        #saving transformation
        joblib.dump(poly_feature_transformer, os.path.join("models", "feature_cross.pkl"))


    train_df, golden_df = train_test_split(
        raw_df,
        test_size=test_size,
        random_state=random_state,
        stratify=raw_df[label_col_name],
    )

    train_df.to_csv(os.path.join(output_dir, train_filename))
    golden_df.to_csv(os.path.join(output_dir, golden_filename))


