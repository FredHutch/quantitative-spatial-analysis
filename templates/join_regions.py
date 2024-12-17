#!/usr/bin/env python3

from pathlib import Path
import anndata as ad
import pandas as pd
import logging

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/join_regions.txt")
    ]
)
logger = logging.getLogger(__name__)


def read_h5ad(fp: Path):
    logger.info(f"Reading {fp}")
    return ad.read_h5ad(fp)


# Read all of the anndata objects
adata_list = [
    read_h5ad(fp)
    for fp in Path(".").glob("*.h5ad")
]

# Find any common observations across any pair of datasets
obs_value_counts = (
    pd.Series([
        obs
        for adata in adata_list
        for obs in adata.obs.index
    ])
    .value_counts()
)

# Find any observation which is found in more than one dataset
common_obs = obs_value_counts[
    obs_value_counts > 1
].index

logging.info(f"Found {len(common_obs):,} common observations")

# Drop any common observations from the datasets
adata_list = [
    adata[~adata.obs.index.isin(common_obs)]
    for adata in adata_list
]

# Concatenate the datasets
spatial = ad.concat(adata_list, uns_merge="same")

logger.info(f"Concatenated {len(adata_list):,} datasets")
logger.info(f"Resulting dataset has {spatial.shape[0]:,} observations and {spatial.shape[1]:,} variables")

# Write the concatenated dataset to a file
logger.info("Writing the concatenated dataset to spatialdata.h5ad")
spatial.write("spatialdata.h5ad")
logger.info("Done")
