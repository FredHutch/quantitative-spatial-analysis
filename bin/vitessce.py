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


def main():
    # Read in the dataset which has been annotated by region, cluster, and neighborhood
    adata = ad.read_h5ad("spatialdata.h5ad")

    # For each of the spatial datasets which were provided as inputs, generate outputs
    # for each of the regions that they contain
    for sdata in read_spatial_datasets():
        extract_regions(adata, sdata)

    # If there is a spatialdata.zarr folder, remove it
    if Path("spatialdata.zarr").exists():
        logger.info("Removing existing spatialdata.zarr folder")
        shutil.rmtree("spatialdata.zarr")

    
def read_spatial_datasets() -> Iterator[SpatialData]:
    for path in Path("spatialdata").glob("*.zarr.zip"):
        yield read_spatialdata_zarr(path)


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


def extract_regions(adata: ad.AnnData, sdata: SpatialData):
    """
    Given a set of annotated spatial coordinates (adata) and a spatial dataset (sdata),
    write out the subsetted spatial dataset for each region in the annotated dataset.
    """

    # Subset the table of the spatial data to the points within the annotated dataset
    overlap = sdata.tables['table'].obs_names.intersection(adata.obs_names)

    if len(overlap) == 0:
        logger.info(f"No overlap of points with the annotated dataset")
        return
    
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
        zarr_path = f"regions/{region}/spatialdata.zarr"
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
            )
        )
        

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
        base_dir="spatialdata.zarr"
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


def write_vitessce_config(region: str, vt_kwargs: Dict[str, Any]):

    # Configure the viewer twice:
    #  - show cell measurements along with the cell type annotations (similar to the Xenium viewer)
    #  - show the neighborhoods alongside the cell type annotations
    for prefix, vt_config in [
        ("cell_types", format_vitessce_cell_types(region, **vt_kwargs)),
        ("neighborhoods", format_vitessce_cell_types(region, **vt_kwargs))
    ]:
        # Save the configuration to JSON
        with open(f"regions/{region}/{prefix}.vt.json", "w") as f:
            json.dump(vt_config, f, indent=4)


def format_vitessce_cell_types(
    region: str,
    schema_version = "1.0.16",
    obs_type = "cell",
    init_gene = "CD45",
    radius = 10,
    description = ""
):
    """
    Format a Vitessce configuration for the cell types in a region, similar to the Xenium viewer.
    """

    return {
        "version": schema_version,
        "name": f"{region} - Cell Types",
        "description": description,
        "datasets": [
            {
                "uid": "A",
                "name": region,
                "files": [
                    {
                        "url": "spatialdata.zarr.zip",
                        "fileType": "image.spatialdata.zarr",
                        "coordinationValues": {
                            "fileUid": "image",
                            "obsType": obs_type
                        },
                        "options": {
                            "path": f'images/image'
                        }
                    },
                    {
                        "url": "spatialdata.zarr.zip",
                        "fileType": "obsFeatureMatrix.spatialdata.zarr",
                        "coordinationValues": {
                            "obsType": obs_type
                        },
                        "options": {
                            "path": "tables/table/X"
                        }
                    },
                    {
                        "url": "spatialdata.zarr.zip",
                        "fileType": "obsSpots.spatialdata.zarr",
                        "coordinationValues": {
                            "obsType": obs_type
                        },
                        "options": {
                            "path": f"shapes/centroids",
                            "tablePath": "tables/table"
                        }
                    },
                    {
                        "url": "spatialdata.zarr.zip",
                        "fileType": "obsSets.spatialdata.zarr",
                        "coordinationValues": {
                            "obsType": obs_type
                        },
                        "options": {
                            "obsSets": [
                                {
                                    "name": "Cell Type",
                                    "path": f"tables/table/obs/cluster"
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
            "spatialTargetC": {
                "A": 0,
                "B": 0,
            },
            "spatialChannelColor": {
                "A": [255, 255, 255],
                "B": [255, 255, 255]
            },
            "spatialChannelWindow": {
                "A": None,
                "B": None
            },
            "spatialChannelVisible": {
                "A": True,
                "B": True
            },
            "spatialChannelOpacity": {
                "A": 1,
                "B": 1
            },
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
                            "A": [
                                "A"
                            ]
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
                        "spatialTargetC": {
                            "A": "A"
                        },
                        "spatialChannelColor": {
                            "A": "A"
                        },
                        "spatialChannelWindow": {
                            "A": "A"
                        },
                        "spatialChannelVisible": {
                            "A": "A"
                        },
                        "spatialChannelOpacity": {
                            "A": "A"
                        }
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
                            "B": [
                                "B"
                            ]
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
                        "spatialTargetC": {
                            "B": "B"
                        },
                        "spatialChannelColor": {
                            "B": "B"
                        },
                        "spatialChannelWindow": {
                            "B": "B"
                        },
                        "spatialChannelVisible": {
                            "B": "B"
                        },
                        "spatialChannelOpacity": {
                            "B": "B"
                        }
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
