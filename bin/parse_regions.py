#!/usr/bin/env python3

import json
import os
import sys


def main():
    # Read in the region definition file
    region_file = "input.region.json"
    with open(region_file) as f:
        regions = json.load(f)

    # Get the command line argument for the region ID using sys.argv
    if len(sys.argv) != 2:
        print("Usage: python parse_regions.py <region_id>")
        sys.exit(1)
    region_id = sys.argv[1]

    # If the region file contains multiple regions,
    # prepend the region ID to each region's definition
    # and write them to separate files
    if isinstance(regions, list):
        for region in regions:
            region["region_id"] = f'{region_id}_{region["region_id"]}'
            write_region(region)
    else:
        # If there's only one region, just use the provided region ID
        regions["region_id"] = region_id
        write_region(regions)


def write_region(region):
    ix = 0
    while os.path.exists(f"region.{ix}.json"):
        ix += 1
    with open(f"region.{ix}.json", "w") as f:
        json.dump(region, f, indent=4)
        print(f"Written region {region['region_id']} to region.{ix}.json")


if __name__ == "__main__":
    main()