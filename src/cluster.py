"""
This module provides helper functions to create dimensionlity
reduction plots as well as implements clustering algorithms
"""
from sklearn.decomposition import PCA, TruncatedSVD
from pandas import DataFrame
import matplotlib.pyplot as plt
import umap


def generate_pca_plot(
    features: DataFrame, label, n_components: int = 2, random_state: int = 42
):
    """
    Takes in the features and labels and decompose into color coded 2D plot
    Args:
        features (DataFrame): Features Dataframe
        label (Series): label Series
        n_components (int, optional): Number of dimension to reduce to. Defaults to 2.
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    reduced_features = pca.fit_transform(features)

    plt.figure(figsize=(5, 5))
    output_fig = plt.scatter(
        reduced_features[:, 0], reduced_features[:, 1], c=label, alpha=0.5
    )
    plt.title("PCA", fontsize=20)
    return output_fig


def generate_truncatedSVD_plot(
    features: DataFrame, label, n_components: int = 2, random_state: int = 42
):
    """
    Takes in the features and labels and decompose into color coded 2D plot
    Args:
        features (DataFrame): Features Dataframe
        label (Series): label Series
        n_components (int, optional): Number of dimension to reduce to. Defaults to 2.
    """
    truncated_svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    reduced_features = truncated_svd.fit_transform(features)

    plt.figure(figsize=(5, 5))
    output_fig = plt.scatter(
        reduced_features[:, 0], reduced_features[:, 1], c=label, alpha=0.5
    )
    plt.title("Truncated SVD", fontsize=20)
    return output_fig


def generate_UMAP_plot(
    features: DataFrame, label, n_components: int = 2, random_state: int = 42
):
    """
    Takes in the features and labels and decompose into color coded 2D plot
    Args:
        features (DataFrame): Features Dataframe
        label (Series): label Series
        n_components (int, optional): Number of dimension to reduce to. Defaults to 2.
    """
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    reduced_features = reducer.fit_transform(features)

    plt.figure(figsize=(5, 5))
    output_fig = plt.scatter(
        reduced_features[:, 0], reduced_features[:, 1], c=label, alpha=0.5
    )
    plt.title("UMAP", fontsize=20)
    return output_fig
