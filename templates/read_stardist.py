#!/usr/bin/env python3

from anndata import AnnData
from pathlib import Path
import logging
import pandas as pd

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/read_stardist.txt")
    ]
)
logger = logging.getLogger(__name__)


def format_anndata() -> AnnData:
    # Read the cell coordinates from spatial.csv
    logger.info("Reading the cell coordinates")
    spatial = pd.read_csv('stardist/spatial.csv', index_col=0)

    # Read the cell attributes
    logger.info("Reading the cell attributes")
    obs = pd.read_csv('stardist/attributes.csv', index_col=0)

    # Combine the spatial and attribute data
    logger.info("Combining the spatial and attribute data")
    obs = obs.merge(spatial, left_index=True, right_index=True)

    # Set the Object ID column as the index (without removing it from the table)
    obs.set_index("Object ID", drop=False, inplace=True)

    # Read the gene abundances from the measurements specified by the user
    logger.info("Reading the ${params.measurement_type} feature matrix")
    X = pd.read_csv('stardist/${params.measurement_type}.csv', index_col=0)

    # Apply the index from obs to X
    X.index = obs.index

    # Drop any columns which have NaN values
    logger.info("Dropping columns with NaN values")
    X = X.dropna(axis=1)
    assert X.shape[1] > 0, "No columns left after dropping NaN values"
    logger.info(f"Data now has {X.shape[1]:,} features")

    # Build the anndata object
    adata = AnnData(X=X, obs=obs)

    # Use the .uns to record information about the dataset
    adata.uns["spatial_dataset"] = {
        "type": "stardist",
        "uri": "${uri}",
        "x_cname": spatial.columns[0],
        "y_cname": spatial.columns[1]
    }

    # Save the spatial coordinates in the .obsm attribute
    logger.info("Saving the spatial coordinates")
    adata.obsm["spatial"] = spatial.values

    return adata


def main():

    # Read the cell information and gene expression data
    adata = format_anndata()

    # Save as a .h5ad file
    logger.info("Saving the AnnData object")
    adata.write('spatialdata.h5ad')

    logger.info("Done")


if __name__ == "__main__":
    main()