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

# For each region, make a side-by-side plot of the spatial coordinates colored by the cluster, and by the neighborhood
vc = adata.obs['region'].value_counts()
for region in adata.obs["region"].unique():
    logger.info(f"Plotting region: {region} ({vc[region]} cells)")

    region_data = adata[adata.obs["region"] == region]
    plot_df = pd.concat([
        pd.DataFrame(region_data.obsm["spatial"], columns=["x", "y"], index=region_data.obs_names),
        region_data.obs
    ], axis=1)

    # Make some linked plots
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Cluster",
        "Neighborhood"),
        shared_xaxes="all",
        shared_yaxes="all"
    )

    # Plot the spatial coordinates colored by the cluster
    for ix, grouping in enumerate(["cluster", "neighborhood"]):

        for group, group_df in plot_df.groupby(grouping):
            trace = go.Scatter(
                x=group_df["x"],
                y=group_df["y"],
                mode="markers",
                name=group,
                legendgroup=grouping,
                legendgrouptitle=dict(text=grouping.title()),
            )

            fig.add_trace(
                trace,
                row=1,
                col=ix + 1
            )

    # Format the figure,adding the title and making the background transparent
    fig.update_layout(
        title_text=f"Region: {region}",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    # Save as HTML
    fig.write_html(f"region_{region}.html")
    logger.info(f"Saved region_{region}.html")

    # Save as PNG
    fig.write_image(f"region_{region}.png")
    logger.info(f"Saved region_{region}.png")

    # Save as PDF
    fig.write_image(f"region_{region}.pdf")
    logger.info(f"Saved region_{region}.pdf")
