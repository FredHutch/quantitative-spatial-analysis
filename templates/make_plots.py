#!/usr/bin/env python3

import anndata as ad
import pandas as pd
import logging
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

Path("logs").mkdir(exist_ok=True)
Path("combined").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/make_plots.txt")
    ]
)
logger = logging.getLogger(__name__)

def plot_spatial():
    # Read in the spatial dataset
    logger.info("Reading the spatial dataset")
    adata: ad.AnnData = ad.read_h5ad("spatialdata.h5ad")

    # For each region, make a side-by-side plot of the spatial coordinates colored by the cluster, and by the neighborhood
    vc = adata.obs['region'].value_counts()
    for region in adata.obs["region"].unique():
        logger.info(f"Plotting region: {region} ({vc[region]} cells)")

        # Set up the output folder in regions/{region}, and make sure it exists
        output_dir = Path(f"regions/{region}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Subset to the region of interest
        region_data = adata[adata.obs["region"] == region]
        plot_df = pd.concat([
            pd.DataFrame(region_data.obsm["spatial"], columns=["x", "y"], index=region_data.obs_names),
            region_data.obs
        ], axis=1)
        logger.info(f"Plotting {len(plot_df):,} cells")

        # Calculate the average density of points
        x_min, x_max = plot_df["x"].min(), plot_df["x"].max()
        y_min, y_max = plot_df["y"].min(), plot_df["y"].max()
        x_range = x_max - x_min
        y_range = y_max - y_min
        area = x_range * y_range
        density = len(plot_df) / area

        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        for ax, title, color in zip(axs, ["Cell Type", "Neighborhood"], ["cluster", "neighborhood"]):
            sns.scatterplot(
                data=plot_df,
                x="x",
                y="y",
                hue=color,
                palette="tab20",
                ax=ax,
                edgecolor=None,
                s=0.01 / density
            )
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_title(title)

        plt.suptitle(f"Region: {region}")
        plt.tight_layout()
        plt.savefig(str(output_dir / "plot.png"))
        plt.savefig(str(output_dir / "plot.pdf"))
        plt.close()

    logger.info("Done")


def plot_counts():
    logger.info("Reading the counts dataset")
    df = pd.read_csv("counts.csv")

    # e.g. 
    #     region  cluster  neighborhood  count
    # 0      TMA 1        0             0    550
    # 1      TMA 1        0             1     91
    # 2      TMA 1        0             2     37
    # 3      TMA 1        0             3    156
    # 4      TMA 1        0             4    153
    # ...      ...      ...           ...    ...
    # 10290  TMA 3      292             9      3
    # 10291  TMA 3      293             7      1
    # 10292  TMA 3      294             1      1
    # 10293  TMA 3      296             5      1
    # 10294  TMA 3      306             5      1

    # Compare the number of cells in each region (summed across all clusters and neighborhoods)
    logger.info("Plotting counts by region")
    sns.barplot(
        data=df.reindex(columns=["region", "count"]).groupby("region").sum().reset_index(),
        x="region",
        y="count"
    )
    plt.xticks(rotation=45)
    plt.xlabel("")
    plt.ylabel("Number of Cells")
    plt.title("Region Size")
    plt.tight_layout()
    plt.savefig("combined/region_counts.png")
    plt.savefig("combined/region_counts.pdf")
    plt.close()

    # Plot a stacked bar graph with the proportion of cells in each neighborhood in each region
    logger.info("Plotting counts by neighborhood")

    # Normalize the counts within each region, ignoring the cluster assignment
    plot_df = (
        df
        .drop(columns=["cluster"])
        .groupby(["region", "neighborhood"])
        .sum()
        .reset_index()
        .groupby("region")
        .apply(lambda x: x.assign(count=x["count"] / x["count"].sum()), include_groups=False)
        .assign(neighborhood=lambda x: x["neighborhood"].astype(str))
        .rename(columns=dict(neighborhood="Neighborhood"))
    )

    sns.barplot(
        data=plot_df,
        x="region",
        y="count",
        hue="Neighborhood"
    )
    plt.legend(title="Neighborhood", bbox_to_anchor=(1, 1))
    plt.xticks(rotation=45)
    plt.xlabel("")
    plt.ylabel("Proportion of Cells")
    plt.title("Neighborhood Composition Across Regions")
    plt.tight_layout()
    plt.savefig("combined/neighborhood_comp_across_regions.png")
    plt.savefig("combined/neighborhood_comp_across_regions.pdf")
    plt.close()

    # Make a heatmap with the proportion of cell types in each neighborhood
    logger.info("Plotting counts by neighborhood and cluster")
    wide_df = (
        df.pivot_table(
            index="neighborhood",
            columns="cluster",
            values="count",
            fill_value=0,
            aggfunc="sum"
        )
        .apply(lambda x: x / x.sum(), axis=1)
    )
    # Do not show more than 20 cell clusters, filtering on the most common
    if wide_df.shape[1] > 20:
        wide_df = wide_df[wide_df.sum().sort_values(ascending=False).index[:20]]

    g = sns.clustermap(wide_df, cmap="viridis", cbar_kws={"label": "Proportion of Cells"})
    g.ax_heatmap.set_xlabel("Cell Cluster")
    g.ax_heatmap.set_ylabel("Neighborhood")
    plt.tight_layout()
    plt.savefig("combined/cell_clusters_across_neighborhoods.png")
    plt.savefig("combined/cell_clusters_across_neighborhoods.pdf")
    plt.close()


def main():

    # Plot the spatial coordinates
    plot_spatial()

    # Plot the summary metrics
    plot_counts()

if __name__ == "__main__":
    main()
