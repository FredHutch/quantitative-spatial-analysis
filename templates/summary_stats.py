#!/usr/bin/env python3

import anndata as ad
import logging
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/summary_stats.txt")
    ]
)
logger = logging.getLogger(__name__)

# Read in the spatial dataset
adata = ad.read_h5ad("spatialdata.h5ad")

# Count up the number of cells in each region, per cluster, per neighborhood
counts = (
    adata.obs
    .groupby(["region", "cluster", "neighborhood"], observed=False)
    .size()
    .reset_index()
    .rename(columns={0: "count"})
    .query("count > 0")
)
counts.to_csv("counts.csv", index=False)
