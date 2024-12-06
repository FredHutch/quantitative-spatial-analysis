include { extract_regions_xenium } from './extract_regions'
include { publish } from './publish'

process cluster_points {
    input:
    path "spatial.h5ad"

    output:
    path "clustered.h5ad"

    script:
    template "cluster_points.py"
}

process neighborhood_analysis {
    input:
    path "input.h5ad"

    output:
    path "spatial.h5ad"

    script:
    template "neighborhood_analysis.py"
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
    cluster_points(extract_regions_xenium.out)

    // Run neighborhood analysis on the clustered points
    neighborhood_analysis(cluster_points.out)

    // Make plots and publish the results
    publish(neighborhood_analysis.out)

}