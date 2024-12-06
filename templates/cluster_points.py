#!/usr/bin/env python3

import scanpy as sc
from anndata import AnnData
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def cluster_xenium(adata: AnnData):
    # Preprocess the data

    # Remove any cells with zero counts
    n_zeros = sum(adata.X.sum(axis=1) == 0)
    if n_zeros > 0:
        logger.info(f"Removing {n_zeros} cells with zero counts")
        adata = adata[adata.X.sum(axis=1) > 0]

    logger.info("Preprocessing data")
    logger.info("Normalizing total counts")
    sc.pp.normalize_total(adata, target_sum=1e4)
    logger.info("Log transforming")
    sc.pp.log1p(adata)
    logger.info("Finding highly variable genes")
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    logger.info(f"Found {sum(adata.var.highly_variable)} highly variable genes")
    adata = adata[:, adata.var.highly_variable]
    logger.info("Scaling data")
    sc.pp.scale(adata, max_value=10)
    logger.info("Running PCA")
    sc.pp.pca(adata, n_comps=min(adata.n_vars, adata.n_obs, 50))
    logger.info("Finding neighbors")
    sc.pp.neighbors(adata, n_neighbors=10)

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

    return adata


def cluster_stardist(adata: AnnData):

    logger.info("Running PCA")
    sc.pp.pca(adata, n_comps=min(adata.n_vars, adata.n_obs, 50))
    logger.info("Finding neighbors")
    sc.pp.neighbors(adata, n_neighbors=10)

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

    return adata



def main():

    # Read in the spatial data
    logger.info("Reading spatial data")
    adata = sc.read_h5ad('spatial.h5ad')

    # Get the data type
    data_type = adata.uns["spatial_dataset"]["type"]
    logger.info(f"Data type: {data_type}")

    # If this is a xenium dataset
    if data_type == "xenium":
        adata = cluster_xenium(adata)

    # If it is a stardist dataset
    elif data_type == "stardist":
        adata = cluster_stardist(adata)

    # Otherwise raise an error
    else:
        raise ValueError(f"Unknown dataset type {data_type}")


    # Write out the clustered data
    logger.info("Writing clustered data")
    adata.write('clustered.h5ad')
    logger.info("Done")


if __name__ == "__main__":
    main()
