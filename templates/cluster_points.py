#!/usr/bin/env python3

import scanpy as sc
from anndata import AnnData, concat
import logging
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/cluster_points.txt")
    ]
)
logger = logging.getLogger(__name__)


def cluster(adata: AnnData):
    """
    Cluster the data with leiden clustering
    """

    logger.info("Running PCA")
    sc.pp.pca(adata, n_comps=min(adata.n_vars, adata.n_obs, 51)-1)
    n_neighbors = int("${params.n_neighbors}")
    logger.info(f'Using n_neighbors: {n_neighbors}')
    logger.info("Finding neighbors")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors)

    logger.info("Clustering")
    resolution = float("${params.resolution}")
    logger.info(f'Using resolution: {resolution}')
    # Run clustering
    sc.tl.leiden(
        adata,
        resolution=resolution,
        key_added="cluster",
        flavor="igraph",
        n_iterations=2,
        directed=False
    )
    logger.info("Done")


def fix_var_name(field, var_names):
    # If the name is a match, return it
    if field in var_names:
        return field

    # If there is a single lowercase match, return it
    lowercase_matches = [
        name
        for name in var_names
        if name.lower() == field.lower()
    ]
    if len(lowercase_matches) == 1:
        return lowercase_matches[0]

    # Otherwise, just return the original name
    return field


def filter_features(adata: AnnData) -> AnnData:
    """
    If the user has provided a list of features to filter to
    using the params.filter_features input, then filter the
    data to just those features.
    """

    filter_features = [
        fix_var_name(field.strip(), adata.var_names)
        for field in "${params.filter_features}".split(",")
        if field != "false"
    ]
    filter_features.sort()

    # If no features were provided, return a copy of the data as-is
    if not filter_features:
        logger.info("No features provided to filter to")
        return adata.copy()

    else:
        logger.info(
            f"User provided {len(filter_features):,} features for filtering:"
        )
        for field in filter_features:
            logger.info(f" - {field}")
            if field not in adata.var_names:
                logger.info(f" - {field} not in adata.var_names")

    # Raise an error if any of the listed features are
    # not in the adata.var_names
    missing_features = set(filter_features) - set(adata.var_names)
    if missing_features:
        raise ValueError(
            f"Missing features: {', '.join(list(missing_features))}"
        )

    logger.info(f"Filtering to {len(filter_features):,} features")
    return adata.copy()[:, filter_features]


def scale_data(norm_adata: AnnData) -> AnnData:
    """
    Scale the data independently for each image.
    """

    return concat(
        [
            sc.pp.scale(
                norm_adata[image_obs.index.values, :],
                copy=True
            )
            for _, image_obs in norm_adata.obs.groupby("image")
        ]
    )


def normalize_measurements(adata: AnnData) -> AnnData:
    """
    Normalize the measurements in the spatial data
    1. Filter to just the selected features (if any were provided)
    2. Run Z-score normalization on a per-image basis
    """

    filtered_adata = filter_features(adata)
    scaled_data = scale_data(filtered_adata)
    return scaled_data


def main():

    # Read in the spatial data
    logger.info("Reading spatial data")
    adata = sc.read_h5ad('spatialdata.h5ad')

    # Get the data type
    data_type = adata.uns["spatial_dataset"]["type"]
    logger.info(f"Data type: {data_type}")

    # Normalize values before running clustering
    norm_adata = normalize_measurements(adata)

    # Run clustering on the normalized data
    cluster(norm_adata)

    # Add the clustered data back to the original data
    for attr, kw in [
        ('obs', 'cluster'),
        ('obsm', 'X_pca'),
        ('uns', 'pca'),
        ('obsp', 'connectivities'),
        ('obsp', 'distances'),
        ('uns', 'neighbors')
    ]:
        getattr(adata, attr)[kw] = getattr(norm_adata, attr)[kw]

    # Write out the clustered data
    logger.info("Writing clustered data")
    adata.write('clustered.h5ad')
    logger.info("Done")


if __name__ == "__main__":
    main()
