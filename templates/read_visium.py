#!/usr/bin/env python3

import shutil
from anndata import AnnData
from pathlib import Path
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from multiscale_spatial_image import to_multiscale
from skimage.io import imread as sk_imread
from spatialdata.models import ShapesModel, TableModel, Image2DModel
from spatialdata.transformations.transformations import Scale
from spatialdata._io.format import ShapesFormatV01
import dask.array as da
import json
import logging
import pandas as pd
import scanpy as sc
import sys
from scipy.cluster import hierarchy
from spatialdata import SpatialData

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/read_visium.txt")
    ]
)
logger = logging.getLogger(__name__)


def format_anndata() -> AnnData:
    # Read the cell coordinates from cells.parquet
    logger.info("Reading the cell coordinates")
    obs = pd.read_parquet('visium/spatial/tissue_positions.parquet')
    obs = obs.set_index('barcode', drop=False)

    # Read the gene abundances from filtered_feature_bc_matrix/
    logger.info("Reading the cell feature matrix")
    adata: AnnData = sc.read_10x_mtx('visium/filtered_feature_bc_matrix')
    # Attach the observation metadata
    adata.obs = obs.reindex(index=adata.obs_names)

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

    # Sort the genes by euclidean distance
    logger.info("Sorting genes by euclidean distance")
    adata = adata[:, sort_index(adata.to_df().T)]

    # Use the .uns to record information about the dataset
    adata.uns["spatial_dataset"] = {
        "type": "visium",
        "uri": "${uri}",
        "x_cname": "pxl_col_in_fullres",
        "y_cname": "pxl_row_in_fullres"
    }

    # Save the spatial coordinates in the .obsm attribute
    logger.info("Saving the spatial coordinates")
    adata.obsm["spatial"] = adata.obs[["pxl_col_in_fullres", "pxl_row_in_fullres"]].values

    return adata


def sort_index(df: pd.DataFrame) -> pd.Index:
    return hierarchy.leaves_list(
        hierarchy.linkage(
            df.values,
            method="ward",
            metric="euclidean"
        )
    )


def format_spatial(
    adata: AnnData,
    radius=10
):
    """Read the dataset as SpatialData and save to zarr."""

    logger.info("Formatting as SpatialData with image embedding")

    # Read the image
    image = _read_image()
    image = _format_image(image)

    # Format the table as expected by SpatialData
    adata.obs["region"] = "cell_boundaries"
    adata.obs["region"] = adata.obs["region"].astype("category")
    table = TableModel.parse(
        adata,
        region="cell_boundaries",
        region_key="region",
        instance_key="barcode"
    )

    # Format the spatial coordinates
    # Scale the point coordinates by the pixel_size
    scale = Scale([1.0, 1.0], axes=("x", "y"))
    centroids = ShapesModel.parse(
        table.obsm["spatial"],
        geometry=0,
        radius=radius,
        transformations={"global": scale},
        index=table.obs["barcode"].copy(),
    )

    return SpatialData(
        images=dict(image=image),
        shapes=dict(centroids=centroids),
        tables=dict(table=table)
    )


def _read_image() -> da.Array:
    image_fp = Path('visium/spatial/tissue_hires_image.png')

    logger.info(f"Reading the image: {image_fp}")
    try:
        image = sk_imread(image_fp, )
    except MemoryError as e:
        logger.error(f"MemoryError: {e}")
        logger.info("Exiting: 137")
        sys.exit(137)

    image = da.from_array(image)
    _log_obj(image)

    return image


def _log_obj(obj):
    for line in str(obj).split("\\n"):
        logger.info(line)


def _format_image(
    image: da.Array,
    scale_factor=2,
    min_px=1000,
    chunk_x=1000,
    chunk_y=1000,
    chunk_c=1
) -> MultiscaleSpatialImage:

    # If the array is two dimensions
    if len(image.shape) == 2:

        # Add a color dimension
        logger.info("Adding extra color dimension")
        image = da.expand_dims(image, axis=0)

    # If the array is three dimensions
    elif len(image.shape) == 3:
        # Move the color channel to the first dimension
        logger.info("Moving the color channel to the first dimension")
        image = da.moveaxis(image, -1, 0)
    else:
        raise ValueError(f"Image must be two or three dimensions, got {len(image.shape)}")

    # At this point there are only three dimensions
    assert len(image.shape) == 3, "Can only display three dimensions"

    # Set the channel names
    channel_names = ["Red", "Blue", "Green"]

    # Convert the image to multiscale and build an
    # image model which can be used in a SpatialData object

    # Build the image model
    logger.info("Building Image2DModel")
    image = Image2DModel.parse(
        image,
        dims=('c', 'y', 'x'),
        c_coords=channel_names
    )
    _log_obj(image)

    # Convert to multiscale
    # Function will pick the number of scales so that
    # the smallest is no smaller than min_px.
    # Set chunks on each level of scale.
    image: MultiscaleSpatialImage = (
        downscale_image(
            image,
            min_px=min_px,
            scale_factor=scale_factor,
            chunk_c=chunk_c,
            chunk_x=chunk_x,
            chunk_y=chunk_y
        )
    )

    _log_obj(image)
    return image


def downscale_image(
    image,
    scale_factor=2,
    min_px=1000,
    chunk_x=1000,
    chunk_y=1000,
    chunk_c=1
) -> MultiscaleSpatialImage:

    # Pick the number of scales so that the smallest
    # is no smaller than min_px
    scales = [scale_factor]
    while (
        min(image.shape[1], image.shape[2]) /
        (scale_factor**len(scales))
    ) > min_px:
        scales.append(scale_factor)
    scales_str = ', '.join(map(json.dumps, scales))

    # Convert to multiscale
    # Set chunks on each level of scale
    chunks = dict(c=chunk_c, x=chunk_x, y=chunk_y)
    chunks_str = json.dumps(chunks)
    params = f"scales={scales_str}; chunks={chunks_str}"
    logger.info(f"Converting to multiscale ({params})")
    return to_multiscale(
        image,
        scales,
        chunks=chunks
    )


def main():

    # Read the cell information and gene expression data
    adata = format_anndata()

    # Save as a .h5ad file
    logger.info("Saving the AnnData object")
    adata.write('spatialdata.h5ad')

    # Format the image + cell data as SpatialData
    sdata = format_spatial(adata)

    # Save to Zarr
    zarr_path = "spatialdata.zarr"
    logger.info(f"Saving to {zarr_path}")
    sdata.write(zarr_path, format=ShapesFormatV01())

    # Zip up the spatialdata.zarr folder using shutil
    logger.info("Zipping up the Zarr folder")
    shutil.make_archive(
        "spatialdata.zarr",
        "zip",
        root_dir=".",
        base_dir="spatialdata.zarr"
    )

    # Remove the spatialdata.zarr folder
    logger.info("Removing the Zarr folder")
    shutil.rmtree("spatialdata.zarr")

    logger.info("Done")


if __name__ == "__main__":
    main()