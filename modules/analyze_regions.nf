include { extract_regions } from './extract_regions'
include { summarize } from './summarize'

process parse_regions {
    input:
    tuple val(region_id), path("input.region.json")

    output:
    path "region.*.json"

    script:
    """#!/bin/bash
set -e
parse_regions.py "${region_id}"
    """
}

process integrate_measurements_scvi {
    input:
    path "spatialdata.h5ad"

    output:
    path "integrated_spatialdata.h5ad", emit: anndata
    path "logs/*", emit: logs

    script:
    template "integrate_measurements_scvi.py"
}

process cluster_points {
    publishDir "${params.outdir}", mode: 'copy', overwrite: true, pattern: "**.txt"
    publishDir "${params.outdir}/combined", mode: 'copy', overwrite: true, pattern: "cluster_feature_metrics.csv"

    input:
    path "spatialdata.h5ad"

    output:
    path "clustered.h5ad", emit: anndata
    path "logs/*", emit: logs
    path "cluster_feature_metrics.csv", emit: cluster_feature_metrics

    script:
    template "cluster_points.py"
}

process neighborhood_analysis {
    publishDir "${params.outdir}/combined", mode: 'copy', overwrite: true, pattern: "*.h5ad"
    publishDir "${params.outdir}/combined", mode: 'copy', overwrite: true, pattern: "*.csv.gz"
    publishDir "${params.outdir}/combined", mode: 'copy', overwrite: true, pattern: "*.feather"
    publishDir "${params.outdir}", mode: 'copy', overwrite: true, pattern: "**.txt"

    input:
    path "input.h5ad"

    output:
    path "spatialdata.h5ad", emit: anndata
    path "*.csv.gz", emit: csv
    path "*.feather", emit: feather
    path "logs/*", emit: logs

    script:
    template "neighborhood_analysis.py"
}

process vitessce {
    publishDir "${params.outdir}", mode: 'copy', overwrite: true

    input:
    path "spatialdata.h5ad"
    path "spatialdata/spatialdata.*.zarr.zip"

    output:
    path "logs/*", emit: logs
    path "regions/**", emit: zarr, optional: true

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

    // Parse each region file, to account for the case when a single JSON file
    // contains multiple regions (e.g. TMA cores)
    parse_regions(regions)

    // Parse each region file as JSON and extract
    // the URI of the data file it references, as well as the type
    parse_regions
        .out
        .flatten()
        .map { obj ->
            def json = new groovy.json.JsonSlurper().parseText(obj.text)
            return [
                "uri": json["dataset"]["uri"].replaceAll("s3:/", "s3://"),
                "type": json["dataset"]["type"],
                "contents": [json["region_id"], obj]
            ]
        }
        .map { it -> [it['uri'], it['type'], it['contents']]}
        // Group by the input dataset
        .groupTuple(by: [0, 1])
        // Branch based on the type of the dataset
        .branch {
            xenium: it[1] == "xenium"
            visium: it[1] == "visium"
            stardist: it[1] == "stardist"
            other: true
        }
        .set { source_datasets }

    // Raise an error if there is anything in the .other branch
    source_datasets.other.map { error "Unsupported dataset type: ${it[1]}" }

    // Extract the points encoded by each region
    extract_regions(
        source_datasets.xenium,
        source_datasets.visium,
        source_datasets.stardist
    )

    // If the user has not selected a method for integration
    if ( "${params.integrate_measurements}" == "none" ) {

        anndata = extract_regions.out.anndata
        integrate_measurements_logs = Channel.empty()

    } else if ( "${params.integrate_measurements}" == "scvi" ) {

        // Integrate measurements across all regions
        integrate_measurements_scvi(extract_regions.out.anndata)
        anndata = integrate_measurements_scvi.out.anndata
        integrate_measurements_logs = integrate_measurements_scvi.out.logs

    } else {

        error "Option not supported: integrate_measurements=${params.integrate_measurements}"

    }


    // Run clustering on the extracted points
    cluster_points(anndata)

    // Run neighborhood analysis on the clustered points
    neighborhood_analysis(cluster_points.out.anndata)

    // Make plots and summarize the results
    summarize(neighborhood_analysis.out.anndata)

    // Format a vitessce display for each region using the cluster and neighborhood annotations
    vitessce(
        neighborhood_analysis.out.anndata.toSortedList(),
        extract_regions.out.spatialdata
            .map { it -> it[1] }
            .filter { it -> it.exists() }
            .toSortedList()
    )

    emit:
    region_defs = extract_regions.out.region_defs
    spatial = neighborhood_analysis.out.anndata
    plots = summarize.out.plots
    summary = summarize.out.summary
    zarr = vitessce.out.zarr
    logs = summarize.out.logs
        .mix(cluster_points.out.logs)
        .mix(neighborhood_analysis.out.logs)
        .mix(vitessce.out.logs)
        .mix(integrate_measurements_logs)

}