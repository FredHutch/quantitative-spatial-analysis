include { extract_regions_xenium } from './extract_regions'
include { summarize } from './summarize'

process cluster_points {
    input:
    path "spatialdata.h5ad"

    output:
    path "clustered.h5ad", emit: anndata
    path "logs/*", emit: logs

    script:
    template "cluster_points.py"
}

process neighborhood_analysis {
    input:
    path "input.h5ad"

    output:
    path "spatialdata.h5ad", emit: anndata
    path "logs/*", emit: logs

    script:
    template "neighborhood_analysis.py"
}

process vitessce {
    input:
    path "spatialdata.h5ad"
    path "spatialdata/spatialdata.*.zarr.zip"

    output:
    path "logs/*", emit: logs
    path "regions/*/*", emit: zarr

    script:
    """#!/bin/bash
set -e
vitessce.py
"""

}

workflow analyze_regions {
    take:
    regions

    main:

    // Parse each region file as JSON and extract
    // the URI of the data file it references, as well as the type
    regions
        .map { region ->
            def obj = file(region.uri, checkIfExists: true)
            def json = new groovy.json.JsonSlurper().parseText(obj.text)
            return [
                json["dataset"]["uri"].replaceAll("s3:/", "s3://"),
                json["dataset"]["type"],
                [region.id, obj]
            ]
        }
        // Group by the input dataset
        .groupTuple(by: [0, 1])
        // Branch based on the type of the dataset
        .branch {
            xenium: it[1] == "xenium"
            stardist: it[1] == "stardist"
            other: true
        }
        .set { source_datasets }

    // Raise an error if there is anything in the .other branch
    source_datasets.other.map { error "Unsupported dataset type: ${it[1]}" }

    // Extract the points encoded by each region
    extract_regions_xenium(source_datasets.xenium)

    // Run clustering on the extracted points
    cluster_points(extract_regions_xenium.out.anndata)

    // Run neighborhood analysis on the clustered points
    neighborhood_analysis(cluster_points.out.anndata)

    // Make plots and summarize the results
    summarize(neighborhood_analysis.out.anndata)

    // Format a vitessce display for each region using the cluster and neighborhood annotations
    vitessce(
        neighborhood_analysis.out.anndata.toSortedList(),
        extract_regions_xenium.out.spatialdata
            .map { it[1] }
            .toSortedList()
    )

    emit:
    region_defs = extract_regions_xenium.out.region_defs
    spatial = neighborhood_analysis.out.anndata
    plots = summarize.out.plots
    summary = summarize.out.summary
    zarr = vitessce.out.zarr
    logs = summarize.out.logs
        .mix(cluster_points.out.logs)
        .mix(neighborhood_analysis.out.logs)
        .mix(vitessce.out.logs)

}