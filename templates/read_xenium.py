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
        logging.FileHandler("logs/read_xenium.txt")
    ]
)
logger = logging.getLogger(__name__)


def format_anndata() -> AnnData:
    # Read the cell coordinates from cells.parquet
    logger.info("Reading the cell coordinates")
    obs = pd.read_parquet('xenium/cells.parquet')
    obs = obs.set_index('cell_id', drop=False)

    # If the cell_feature_matrix.tar.gz file is present, extract it
    if Path("xenium/cell_feature_matrix.tar.gz").exists():
        logger.info("Extracting cell_feature_matrix.tar.gz")
        shutil.unpack_archive("xenium/cell_feature_matrix.tar.gz", "xenium")

    # Read the gene abundances from cell_feature_matrix/
    logger.info("Reading the cell feature matrix")
    adata: AnnData = sc.read_10x_mtx('xenium/cell_feature_matrix')
    # Attach the observation metadata
    adata.obs = obs

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
        "type": "xenium",
        "uri": "${uri}",
        "x_cname": "x_centroid",
        "y_cname": "y_centroid"
    }

    # Save the spatial coordinates in the .obsm attribute
    logger.info("Saving the spatial coordinates")
    adata.obsm["spatial"] = adata.obs[["x_centroid", "y_centroid"]].values

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

    # Read the experiment metadata as JSON from xenium/experiment.xenium
    logger.info("Reading the experiment metadata")
    experiment = json.load(open('xenium/experiment.xenium'))

    # Read the image
    image = _read_image(experiment)
    image = _format_image(image)

    # Format the table as expected by SpatialData
    adata.obs["region"] = "cell_boundaries"
    adata.obs["region"] = adata.obs["region"].astype("category")
    table = TableModel.parse(
        adata,
        region="cell_boundaries",
        region_key="region",
        instance_key="cell_id"
    )

    # Format the spatial coordinates
    # Scale the point coordinates by the pixel_size
    scale = Scale([1.0, 1.0], axes=("x", "y"))
    pixel_size = experiment["pixel_size"]
    centroids = ShapesModel.parse(
        table.obsm["spatial"] / pixel_size,
        geometry=0,
        radius=radius,
        transformations={"global": scale},
        index=table.obs["cell_id"].copy(),
    )

    return SpatialData(
        images=dict(image=image),
        shapes=dict(centroids=centroids),
        tables=dict(table=table)
    )


def _read_image(experiment) -> da.Array:
    for key in ["morphology_focus_filepath", "morphology_filepath"]:
        if key in experiment["images"]:
            image_fp = "xenium/" + experiment["images"][key]
            break

    logger.info(f"Reading the image: {image_fp}")
    try:
        image = sk_imread(image_fp, plugin="tifffile")
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
    # The array must be two dimensions
    assert len(image.shape) == 2, "Image must be two dimensions"

    # Add a color dimension
    logger.info("Adding extra color dimension")
    image = da.expand_dims(image, axis=0)

    # At this point there are only three dimensions
    assert len(image.shape) == 3, "Can only display three dimensions"

    # Set the channel names
    channel_names = ["DAPI"]

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