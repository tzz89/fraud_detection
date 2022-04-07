"""
Holds all the function to ingest data from other sources
"""
import zipfile
import os
import gdown
from .utils import load_yaml


if __name__ == "__main__":
    # load in the configuration
    configuration_fp = os.path.join("configs", "main_config.yaml")
    config = load_yaml(configuration_fp)

    # create the folder to store the data
    os.makedirs(config["data_folder"], exist_ok=True)

    # downloads the data and unzip the file
    output_file = os.path.join(config["data_folder"], "cc_fraud.zip")
    gdown.download(config["data_url"], output=output_file)
    with zipfile.ZipFile(output_file, "r") as file:
        file.extractall(config["data_folder"])
