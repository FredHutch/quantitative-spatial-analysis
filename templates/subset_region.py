#!/usr/bin/env python3

import anndata as ad
import json
from shapely import Polygon, Point


def extract_region(spatial: ad.AnnData):
    with open("region.json") as handle:
        region = json.load(handle)

    # Set up a vector of False values for each point in the spatial dataset
    mask = [False] * spatial.shape[0]

    # Get the coordinates for the outline of each sub-region
    for outline in region["outline"]:
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

    # Add the region ID to the metadata
    subset.obs["region"] = "${region_id}"

    return subset


def main():

    # Read in the spatial dataset
    spatial = ad.read_h5ad("spatial.h5ad")

    # Subset to the points within the region
    region = extract_region(spatial)

    # Save to a new file
    region.write("region.h5ad")


if __name__ == "__main__":
    main()
