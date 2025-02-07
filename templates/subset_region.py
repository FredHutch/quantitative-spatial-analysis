#!/usr/bin/env python3

from pathlib import Path
from typing import Any, Dict
import anndata as ad
import json
from shapely import Polygon, Point


def extract_region(
    spatial: ad.AnnData,
    region_def: Dict[str, Any]
) -> ad.AnnData:

    # Set up a vector of False values for each point in the spatial dataset
    mask = [False] * spatial.shape[0]

    # Get the coordinates for the outline of each sub-region
    for outline in region_def["outline"]:
        x_coords = outline["x"]
        y_coords = outline["y"]

        # Create a polygon from the coordinates
        polygon = Polygon(zip(x_coords, y_coords))

        # Get the coordinates of every point in the spatial dataset
        coords = spatial.obsm["spatial"]

        # Check if each point is within the polygon
        mask = [
            polygon.contains(Point(coord)) or in_mask
            for coord, in_mask in zip(coords, mask)
        ]

    # Subset the spatial dataset to the points within the region
    subset = spatial[mask, :]

    # Add the image ID to the metacata
    subset.obs["image"] = "${image_id}"

    # Add the region ID to the metadata
    subset.obs["region"] = "${region_id}"

    return subset


def main():

    # Read in the spatial dataset
    spatial = ad.read_h5ad("spatialdata.h5ad")

    # Read in the region definition
    with open("region.json") as handle:
        region_def = json.load(handle)

    # Subset to the points within the region
    region = extract_region(spatial, region_def)

    # Save to a new file
    region.write("region.h5ad")

    # Save the region definition
    region_json = Path("${region_id}") / "region.json"
    region_json.parent.mkdir(exist_ok=True, parents=True)
    with open(region_json, "w") as handle:
        json.dump(region_def, handle)


if __name__ == "__main__":
    main()
