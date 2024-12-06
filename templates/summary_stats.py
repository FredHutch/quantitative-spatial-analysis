#!/usr/bin/env python3

import anndata as ad
import pandas as pd
import logging
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Read in the spatial dataset
adata = ad.read_h5ad("spatial.h5ad")

# Count up the number of cells in each region, per cluster, per neighborhood
counts = (
    adata.obs
    .groupby(["region", "cluster", "neighborhood"], observed=False)
    .size()
    .reset_index()
    .rename(columns={0: "count"})
)
counts.to_csv("counts.csv", index=False)
