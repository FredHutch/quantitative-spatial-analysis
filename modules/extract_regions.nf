process read_xenium {
    input:
    tuple val(uri), path("cells.parquet"), path("cell_feature_matrix")

    output:
    tuple val(uri), path("spatial.h5ad")

    script:
    template "read_xenium.py"
}

process subset_region {
    input:
    tuple path("spatial.h5ad"), val(region_id), path("region.json")

    output:
    path "region.h5ad"

    script:
    template "subset_region.py"
}

process join_regions {
    input:
    path "region.*.h5ad"

    output:
    path "spatial.h5ad"

    script:
    template "join_regions.py"
}


workflow extract_regions_xenium {
    take:
    source_datasets

    main:

    // For each input dataset (given as a URI), get the files needed to parse the features and spatial data
    source_datasets
        .map {
            return [
                it[0],
                file(it[0] + "/cells.parquet", type: "file", checkIfExists: true),
                file(it[0] + "/cell_feature_matrix", type: "dir", checkIfExists: true)
            ]
        }
        // Convert the datasets to AnnData (h5ad) format
        | read_xenium

    // Extract the data from each region
    read_xenium
        .out
        .join(
            source_datasets
                .map { [it[0], it[2]] }
        )
        .transpose()
        .map { [it[1], it[2][0], it[2][1]] }
        | subset_region

    // Merge the regions into a single object
    join_regions(subset_region.out.toSortedList())

    emit:
    join_regions.out
}