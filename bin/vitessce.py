#!/usr/bin/env python3

from copy import deepcopy
import json
from pathlib import Path
import shutil
import zipfile
import anndata as ad
from spatialdata import SpatialData
from spatialdata._io.format import ShapesFormatV01
from typing import Any, Dict, Iterator
import logging

Path("logs").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/vitessce.txt")
    ]
)

logger = logging.getLogger(__name__)


def _log_obj(obj):
    for line in str(obj).split("\\n"):
        logger.info(line)


def main():
    # Read in the dataset which has been annotated by region, cluster, and neighborhood
    logger.info("Reading the annotated dataset from spatialdata.h5ad")
    adata = ad.read_h5ad("spatialdata.h5ad")
    _log_obj(adata)

    spatial_dataset = (
        adata
        .uns
        .get("spatial_dataset", {})
    )
    logger.info(f"spatial_dataset:")
    for line in json.dumps(spatial_dataset, indent=4).splitlines():
        logger.info(line)

    # For each of the spatial datasets which were provided as inputs, generate outputs
    # for each of the regions that they contain
    for sdata in read_spatial_datasets():
        extract_regions(adata, sdata, spatial_dataset.get("type", "unknown"))

    # If there is a spatialdata.zarr folder, remove it
    if Path("spatialdata.zarr").exists():
        logger.info("Removing existing spatialdata.zarr folder")
        shutil.rmtree("spatialdata.zarr")


def read_spatial_datasets() -> Iterator[SpatialData]:
    n = 0
    for path in Path("spatialdata").glob("*.zarr.zip"):
        yield read_spatialdata_zarr(path)
        n += 1
    if n == 0:
        raise ValueError("No spatial datasets found in spatialdata/*.zarr.zip")
    logger.info(f"Read {n} spatial datasets from spatialdata/*.zarr.zip")


def read_spatialdata_zarr(path: Path) -> SpatialData:

    # If there is a spatialdata.zarr folder, remove it
    if Path("spatialdata.zarr").exists():
        logger.info("Removing existing spatialdata.zarr folder")
        shutil.rmtree("spatialdata.zarr")

    # Remove the .zip suffix
    extract_path = path.parent
    # Unzip the zarr archive
    logger.info(f"Extracting {path} to {extract_path}")
    with zipfile.ZipFile(path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # Make sure that a folder called spatialdata.zarr exists
    target_zarr = extract_path / "spatialdata.zarr"
    assert Path(target_zarr).exists(), f"No {target_zarr} folder found"

    # Read the spatial data
    logger.info(f"Reading spatial data from {target_zarr}")
    sdata = SpatialData.read(target_zarr)

    logger.info(f"Read {len(sdata.tables['table'])} points from {target_zarr}")

    return sdata


def extract_regions(adata: ad.AnnData, sdata: SpatialData, spatial_type: str):
    """
    Given a set of annotated spatial coordinates (adata) and a spatial dataset (sdata),
    write out the subsetted spatial dataset for each region in the annotated dataset.
    """

    # LEGACY COMPATIBILITY
    # If the spatial data has index values of "0", "1", "2", and also contains a column "object_id",
    # then set the "object_id" column to be the index
    if sdata.tables['table'].obs.index[:3].isin(["0", "1", "2"]).all() and "object_id" in sdata.tables['table'].obs.columns:
        logger.info("Setting the object_id column as the index")
        sdata.tables['table'].obs.set_index("object_id", inplace=True, drop=False)

    # Subset the table of the spatial data to the points within the annotated dataset
    overlap = sdata.tables['table'].obs_names.intersection(adata.obs_names)

    if len(overlap) == 0:
        logger.info("No overlap of points with the annotated dataset")
        return

    logger.info(f"Found {len(overlap):,} points in the annotated dataset")

    # Iterate over each region
    for region in adata.obs.reindex(index=overlap)["region"].unique():
        logger.info(f"Extracting region: {region}")
        # Get the points in this region
        region_points = adata.obs_names[adata.obs["region"] == region]
        logger.info(f"Found {len(region_points):,} points in region {region}")
        # Make a copy of the spatial dataset
        region_sdata: SpatialData = deepcopy(sdata)
        # Subset the spatial data to the points in the region
        logger.info("Subsetting the spatial data")
        region_sdata.tables['table'] = region_sdata.tables['table'][region_points]
        # Copy over the cluser and neighborhood annotations
        for key in ["cluster", "neighborhood"]:
            if key not in adata.obs.columns:
                raise ValueError(f"Missing column {key} in the annotated dataset")
            region_sdata.tables["table"].obs[key] = adata.obs.reindex(index=region_points)[key]
        # Find the most highly variable gene
        logger.info("Finding the most highly variable gene")
        init_gene = region_sdata.tables['table'].to_df().var(axis=0).idxmax()
        # Subset the shapes to the points in the region
        logger.info("Subsetting the spatial coordinates")
        region_sdata.shapes["centroids"] = region_sdata.shapes["centroids"].loc[region_points]
        # Write out the subsetted spatial data
        zarr_path = f"regions/{region}/{sanitize_filepath(region)}.zarr"
        logger.info(f"Writing region data to {zarr_path}")
        # If the zarr store already exists, remove it
        if Path(zarr_path).exists():
            shutil.rmtree(zarr_path)
        region_sdata.write(zarr_path, format=ShapesFormatV01())
        # Modify the zarr store to meet the Vitessce requirements
        format_zarr_vitessce(zarr_path)

        # Write out the vitessce configuration file
        write_vitessce_config(
            region,
            dict(
                description=f"Number of cells: {len(region_sdata.tables['table']):,}",
                init_gene=init_gene
            ),
            spatial_type
        )


def sanitize_filepath(fp: str) -> str:
    fp = fp.lower()
    for char in [" ", "-", "/", "\\", ":", "|", ";", "@"]:
        fp = fp.replace(char, "_")
    while "__" in fp:
        fp = fp.replace("__", "_")
    return fp


def format_zarr_vitessce(zarr_path):

    # Fix the omero metadata for any images
    logger.info(f"Fixing Zarr image metadata for {zarr_path}")
    fix_zarr_image_metadata(zarr_path)

    # Duplicate the {zarr_path}/tables/ folder to {zarr_path}/table/
    logger.info("Duplicating the tables folder")
    shutil.copytree(
        zarr_path + "/tables",
        zarr_path + "/table"
    )

    # Zip up the spatialdata.zarr folder using shutil
    logger.info("Zipping up the Zarr folder")
    shutil.make_archive(
        zarr_path,
        "zip",
        root_dir=str(Path(zarr_path).parent),
        base_dir=zarr_path.split("/")[-1]
    )

    # Remove the spatialdata.zarr folder
    logger.info("Removing the Zarr folder")
    shutil.rmtree(zarr_path)


def fix_zarr_image_metadata(zarr_path: str):
    """
    Given a zarr store, fill out any missing fields
    in the omero field of the image attributes.
    """

    # Iterate over every .zattr or zmetadata file
    for pattern in ["zmetadata", ".zattrs"]:
        for file in Path(zarr_path).rglob(pattern):

            # Open the object
            obj = json.load(file.open())

            # Recurse into the object, make updates, and
            # return a bool indicating if the object was changed
            if _update_omero_attr(obj):

                # Write out the updated object
                with file.open("w") as handle:
                    json.dump(obj, handle, indent=4)


def _update_omero_attr(obj):
    """
    If the omero attribute is present, fill in any missing fields.
    """

    _default_channel = {
        "color": "FFFFFF",
        "window": {
            kw: 0
            for kw in ['start', 'min', 'max', 'end']
        }
    }

    _default_rdefs = {
        "defaultT": 0,
        "defaultZ": 0,
        "name": "global"
    }

    was_modified = False

    if isinstance(obj, dict) and "omero" in obj:
        logger.info("Updating omero attribute")

        if "channels" in obj["omero"]:
            for channel in obj["omero"]["channels"]:
                for kw, val in _default_channel.items():
                    if kw not in channel:
                        channel[kw] = val
                        was_modified = True

            if "rdefs" not in obj["omero"]:
                obj["omero"]["rdefs"] = _default_rdefs
                was_modified = True
            else:
                for kw, val in _default_rdefs.items():
                    if kw not in obj["omero"]["rdefs"]:
                        obj["omero"]["rdefs"][kw] = val
                        was_modified = True

        logger.info(obj["omero"])

    if isinstance(obj, dict):
        for val in obj.values():
            if _update_omero_attr(val):
                was_modified = True
    elif isinstance(obj, list):
        for val in obj:
            if _update_omero_attr(val):
                was_modified = True

    return was_modified


def write_vitessce_config(region: str, vt_kwargs: Dict[str, Any], spatial_type: str):

    # Configure the viewer twice:
    #  - show cell measurements along with the cell type annotations (similar to the Xenium viewer)
    #  - show the neighborhoods alongside the cell type annotations
    for prefix, vt_config in [
        (
            "cell_types",
            format_vitessce_cell_types(
                region,
                obs_groups="cluster",
                title_suffix="Cell Types",
                spatial_type=spatial_type,
                **vt_kwargs
            )
        ),
        (
            "neighborhoods",
            format_vitessce_cell_types(
                region,
                obs_groups="neighborhood",
                title_suffix="Neighborhoods",
                spatial_type=spatial_type,
                **vt_kwargs
            )
        )
    ]:
        # Save the configuration to JSON
        with open(f"regions/{region}/{prefix}.vt.json", "w") as f:
            json.dump(vt_config, f, indent=4)


def _config_switch_spatial_type(spatial_type, value_group):
    if value_group == "A":
        if spatial_type == "visium":
             return {
                "A": "A",
                "B": "B",
                "C": "C"
            }
        else:
             return {"A": "A"}
    elif value_group == "B":
        if spatial_type == "visium":
             return {
                "D": "B",
                "E": "E",
                "F": "F"
            }
        else:
             return {"B": "B"}
    else:
        raise ValueError(f"Unexpected value_group = {value_group}")


def format_vitessce_cell_types(
    region: str,
    schema_version="1.0.16",
    obs_type="cell",
    obs_groups="cluster",
    init_gene="CD45",
    description="",
    title_suffix="Cell Types",
    spatial_type="unknown"
):
    """
    Format a Vitessce configuration for the cell types in a region, similar to the Xenium viewer.
    """

    logger.info(f"obs_type = {obs_type}")
    logger.info(f"obs_groups = {obs_groups}")
    logger.info(f"init_gene = {init_gene}")
    logger.info(f"description = {description}")
    logger.info(f"title_suffix = {title_suffix}")
    logger.info(f"spatial_type = {spatial_type}")

    # The spot radius varies by spatial type
    if spatial_type == "xenium" or spatial_type == "visium":
        radius = 7
    else:
        radius = 20

    zarr_path = sanitize_filepath(region) + ".zarr.zip"

    return {
        "version": schema_version,
        "name": f"{region} - {title_suffix}",
        "description": description,
        "datasets": [
            {
                "uid": "A",
                "name": region,
                "files": [
                    {
                        "url": zarr_path,
                        "fileType": "image.spatialdata.zarr",
                        "coordinationValues": {
                            "fileUid": "image",
                            "obsType": obs_type
                        },
                        "options": {
                            "path": 'images/image'
                        }
                    },
                    {
                        "url": zarr_path,
                        "fileType": "obsFeatureMatrix.spatialdata.zarr",
                        "coordinationValues": {
                            "obsType": obs_type
                        },
                        "options": {
                            "path": "tables/table/X"
                        }
                    },
                    {
                        "url": zarr_path,
                        "fileType": "obsSpots.spatialdata.zarr",
                        "coordinationValues": {
                            "obsType": obs_type
                        },
                        "options": {
                            "path": "shapes/centroids",
                            "tablePath": "tables/table"
                        }
                    },
                    {
                        "url": zarr_path,
                        "fileType": "obsSets.spatialdata.zarr",
                        "coordinationValues": {
                            "obsType": obs_type
                        },
                        "options": {
                            "obsSets": [
                                {
                                    "name": "Cell Type",
                                    "path": f"tables/table/obs/{obs_groups}"
                                }
                            ]
                        }
                    }
                ]
            }
        ],
        "coordinationSpace": {
            "dataset": {
                "A": "A"
            },
            "featureSelection": {
                "A": [
                    init_gene
                ],
                "B": [
                    init_gene
                ]
            },
            "obsType": {
                "A": obs_type
            },
            "featureType": {
                "A": "gene"
            },
            "featureValueType": {
                "A": "expression"
            },
            "obsColorEncoding": {
                "A": "cellSetSelection",
                "B": "geneSelection"
            },
            "spatialTargetZ": {
                "A": 0
            },
            "spatialTargetT": {
                "A": 0
            },
            "imageLayer": {
                "A": "__dummy__",
                "B": "__dummy__"
            },
            "fileUid": {
                "A": "image",
                "B": "image"
            },
            "spatialLayerOpacity": {
                "A": 1,
                "B": 0.5,
                "C": 1,
                "D": 0.5
            },
            "spatialLayerVisible": {
                "A": True,
                "B": True,
                "C": True,
                "D": True
            },
            "photometricInterpretation": {
                "A": "BlackIsZero",
                "B": "BlackIsZero"
            },
            "imageChannel": {
                "A": "__dummy__",
                "B": "__dummy__"
            },
            "spatialTargetC": (
                {
                    "A": 0,
                    "B": 1,
                    "C": 2,
                    "D": 0,
                    "E": 1,
                    "F": 2
                }
                if spatial_type == "visium" else
                {
                    "A": 0,
                    "B": 0
                }
            ),
            "spatialChannelColor": (
                {
                    "A": [255, 0, 0],
                    "B": [0, 255, 0],
                    "C": [0, 0, 255],
                    "D": [255, 0, 0],
                    "E": [0, 255, 0],
                    "F": [0, 0, 255]
                }
                if spatial_type == "visium" else
                {
                    "A": [255, 255, 255],
                    "B": [255, 255, 255]
                }
            ),
            "spatialChannelWindow": (
                {
                    "A": None,
                    "B": None,
                    "C": None,
                    "D": None,
                    "E": None,
                    "F": None
                }
                if spatial_type == "visium" else
                {
                    "A": None,
                    "B": None
                }
            ),
            "spatialChannelVisible": (
                {
                    "A": True,
                    "B": True,
                    "C": True,
                    "D": True,
                    "E": True,
                    "F": True
                }
                if spatial_type == "visium" else
                {
                    "A": True,
                    "B": True
                }
            ),
            "spatialChannelOpacity": (
                {
                    "A": 1,
                    "B": 1,
                    "C": 1,
                    "D": 1,
                    "E": 1,
                    "F": 1
                }
                if spatial_type == "visium" else
                {
                    "A": 1,
                    "B": 1
                }
            ),
            "spotLayer": {
                "A": "__dummy__",
                "B": "__dummy__"
            },
            "spatialSpotRadius": {
                "A": radius,
                "B": radius
            },
            "spatialLayerColormap": {
                "A": None,
                "B": None
            },
            "featureValueColormap": {
                "A": "plasma",
                "B": "plasma"
            },
            "featureValueColormapRange": {
                "A": [
                    0,
                    1.0
                ],
                "B": [
                    0,
                    1.0
                ]
            },
            "metaCoordinationScopes": {
                "A": {
                    "spatialTargetZ": "A",
                    "spatialTargetT": "A",
                    "obsType": "A",
                    "imageLayer": "A",
                    "spotLayer": "A",
                    "obsColorEncoding": "A",
                    "featureSelection": "A"
                },
                "B": {
                    "spatialTargetZ": "A",
                    "spatialTargetT": "A",
                    "obsType": "A",
                    "imageLayer": "B",
                    "spotLayer": "B",
                    "obsColorEncoding": "B",
                    "featureSelection": "B"
                }
            },
            "metaCoordinationScopesBy": {
                "A": {
                    "imageLayer": {
                        "fileUid": {
                            "A": "A"
                        },
                        "spatialLayerOpacity": {
                            "A": "A"
                        },
                        "spatialLayerVisible": {
                            "A": "A"
                        },
                        "photometricInterpretation": {
                            "A": "A"
                        },
                        "spatialLayerColormap": {
                            "A": "A"
                        },
                        "imageChannel": {
                            "A": (
                                ["A", "B", "C"]
                                if spatial_type == "visium"
                                else ["A"]
                            )
                        }
                    },
                    "spotLayer": {
                        "spatialLayerOpacity": {
                            "A": "B"
                        },
                        "spatialLayerVisible": {
                            "A": "B"
                        },
                        "spatialLayerColor": {
                            "A": "A"
                        },
                        "obsColorEncoding": {
                            "A": "A"
                        },
                        "spatialSpotRadius": {
                            "A": "A"
                        }
                    },
                    "imageChannel": {
                        "spatialTargetC": _config_switch_spatial_type(spatial_type, "A"),
                        "spatialChannelColor": _config_switch_spatial_type(spatial_type, "A"),
                        "spatialChannelWindow": _config_switch_spatial_type(spatial_type, "A"),
                        "spatialChannelVisible": _config_switch_spatial_type(spatial_type, "A"),
                        "spatialChannelOpacity": _config_switch_spatial_type(spatial_type, "A")
                    }
                },
                "B": {
                    "imageLayer": {
                        "fileUid": {
                            "B": "B"
                        },
                        "spatialLayerOpacity": {
                            "B": "C"
                        },
                        "spatialLayerVisible": {
                            "B": "C"
                        },
                        "photometricInterpretation": {
                            "B": "B"
                        },
                        "spatialLayerColormap": {
                            "B": "B"
                        },
                        "imageChannel": {
                            "B": (
                                ["D", "E", "F"]
                                if spatial_type == "visium"
                                else ["B"]
                            )
                        }
                    },
                    "spotLayer": {
                        "spatialLayerOpacity": {
                            "B": "D"
                        },
                        "spatialLayerVisible": {
                            "B": "D"
                        },
                        "spatialLayerColor": {
                            "B": "B"
                        },
                        "obsColorEncoding": {
                            "B": "B"
                        },
                        "spatialSpotRadius": {
                            "B": "B"
                        }
                    },
                    "imageChannel": {
                        "spatialTargetC": _config_switch_spatial_type(spatial_type, "B"),
                        "spatialChannelColor": _config_switch_spatial_type(spatial_type, "B"),
                        "spatialChannelWindow": _config_switch_spatial_type(spatial_type, "B"),
                        "spatialChannelVisible": _config_switch_spatial_type(spatial_type, "B"),
                        "spatialChannelOpacity": _config_switch_spatial_type(spatial_type, "B")
                    }
                }
            }
        },
        "layout": [
            {
                "component": "spatialBeta",
                "coordinationScopes": {
                    "dataset": "A",
                    "metaCoordinationScopes": [
                        "A"
                    ],
                    "metaCoordinationScopesBy": [
                        "A"
                    ]
                },
                "x": 0,
                "y": 0,
                "w": 4,
                "h": 6
            },
            {
                "component": "layerControllerBeta",
                "coordinationScopes": {
                    "dataset": "A",
                    "metaCoordinationScopes": [
                        "A"
                    ],
                    "metaCoordinationScopesBy": [
                        "A"
                    ]
                },
                "x": 0,
                "y": 6,
                "w": 4,
                "h": 3
            },
            {
                "component": "spatialBeta",
                "coordinationScopes": {
                    "dataset": "A",
                    "metaCoordinationScopes": [
                        "B"
                    ],
                    "metaCoordinationScopesBy": [
                        "B"
                    ]
                },
                "x": 4,
                "y": 0,
                "w": 4,
                "h": 6
            },
            {
                "component": "layerControllerBeta",
                "coordinationScopes": {
                    "dataset": "A",
                    "metaCoordinationScopes": [
                        "B"
                    ],
                    "metaCoordinationScopesBy": [
                        "B"
                    ]
                },
                "x": 4,
                "y": 6,
                "w": 4,
                "h": 3
            },
            {
                "component": "heatmap",
                "coordinationScopes": {
                    "dataset": "A",
                    "featureSelection": "B"
                },
                "x": 0,
                "y": 9,
                "w": 8,
                "h": 3
            },
            {
                "component": "obsSetSizes",
                "coordinationScopes": {
                    "obsType": "A",
                    "dataset": "A"
                },
                "x": 8,
                "y": 0,
                "w": 2,
                "h": 6
            },
            {
                "component": "featureList",
                "coordinationScopes": {
                    "dataset": "A",
                    "featureSelection": "B"
                },
                "x": 10,
                "y": 0,
                "w": 2,
                "h": 6
            },
            {
                "component": "obsSetFeatureValueDistribution",
                "coordinationScopes": {
                    "dataset": "A",
                    "featureSelection": "B"
                },
                "x": 8,
                "y": 6,
                "w": 4,
                "h": 6
            }
        ],
        "initStrategy": "auto"
    }


if __name__ == "__main__":
    main()
