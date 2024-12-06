#!/usr/bin/env python3

from anndata import AnnData
import pandas as pd
import logging
import scanpy as sc

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Read the cell coordinates from cells.parquet
obs = pd.read_parquet('cells.parquet')
obs = obs.set_index('cell_id')

# Read the gene abundances from cell_feature_matrix/
adata: AnnData = sc.read_10x_mtx('cell_feature_matrix')
# Attach the observation metadata
adata.obs = obs

# Use the .uns to record information about the dataset
adata.uns["spatial_dataset"] = {
    "type": "xenium",
    "uri": "${uri}",
    "x_cname": "x_centroid",
    "y_cname": "y_centroid"
}

# Save the spatial coordinates in the .obsm attribute
adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values

# Save as a .h5ad file
adata.write('spatial.h5ad')
