#!/usr/bin/env python3

import anndata as ad
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
import logging
import pandas as pd
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/neighborhood_analysis.txt")
    ]
)
logger = logging.getLogger(__name__)


def _find_neighbor_counts(adata: ad.AnnData, region: str, n_neighbors: int) -> pd.DataFrame:
    logger.info(f"Finding neighbors for region: {region}")
    # Build the classifier
    fit = NearestNeighbors(n_neighbors=n_neighbors).fit(adata.obsm['spatial'])

    # Return a matrix with the indices of each of those nearest neighbors
    m = fit.kneighbors(adata.obsm['spatial'], return_distance=False)

    # Make a DataFrame with the number of each cell type in the neighborhood for each cell
    return (
        pd.DataFrame(m, index=adata.obs_names)
        .map(
            lambda i: adata.obs['cluster'].iloc[i]
        )
        .apply(
            lambda row: row.value_counts(),
            axis=1
        )
    )


def main():
    # Read in the spatial dataset
    logger.info("Reading the spatial dataset")
    adata = ad.read_h5ad("input.h5ad")

    # For each region:
    #   find the N nearest neighbors
    #   count up the number of each different cell type in that neighborhood
    n_neighbors = int("${params.n_neighbors}")
    logger.info(f"Finding {n_neighbors} nearest neighbors")
    neighbor_counts = (
        pd.concat([
            _find_neighbor_counts(adata[adata.obs["region"] == region], region, n_neighbors)
            for region in adata.obs["region"].unique()
        ])
        .fillna(0)
        .map(int)
    )

    # Run KMeans clustering on the neighbor counts
    n_neighborhoods = int("${params.n_neighborhoods}")
    logger.info(f"Clustering into {n_neighborhoods} neighborhoods")
    km = MiniBatchKMeans(
        n_clusters=n_neighborhoods,
        random_state=0
    )
    labelskm = km.fit_predict(neighbor_counts.values)
    logger.info(f"Found {n_neighborhoods} neighborhoods")

    # Save the neighborhood labels
    adata.obs["neighborhood"] = labelskm

    # Save the results
    logger.info("Writing the results")
    adata.write("spatialdata.h5ad")
    logger.info("Done")


if __name__ == "__main__":
    main()
