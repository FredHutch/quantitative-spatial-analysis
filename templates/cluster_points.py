#!/usr/bin/env python3

import scanpy as sc
from anndata import AnnData
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


def main():

    # Read in the spatial data
    logger.info("Reading spatial data")
    adata = sc.read_h5ad('spatialdata.h5ad')

    # Get the data type
    data_type = adata.uns["spatial_dataset"]["type"]
    logger.info(f"Data type: {data_type}")

    # There is currently no difference in clustering
    # between data types
    cluster(adata)

    # Write out the clustered data
    logger.info("Writing clustered data")
    adata.write('clustered.h5ad')
    logger.info("Done")


if __name__ == "__main__":
    main()
