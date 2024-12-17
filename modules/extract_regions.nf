process read_xenium {
    publishDir "${params.outdir}", mode: 'copy', overwrite: true, pattern: "**.txt"
    input:
    tuple val(uri), path("xenium")

    output:
    tuple val(uri), path("spatialdata.h5ad"), emit: anndata
    tuple val(uri), path("spatialdata.zarr.zip"), emit: spatialdata
    path "logs/*", emit: logs

    script:
    template "read_xenium.py"
}

process read_stardist {
    publishDir "${params.outdir}", mode: 'copy', overwrite: true, pattern: "**.txt"
    input:
    tuple val(uri), path("stardist")

    output:
    tuple val(uri), path("spatialdata.h5ad"), emit: anndata
    path "logs/*", emit: logs

    script:
    template "read_stardist.py"
}

process subset_region {
    publishDir "${params.outdir}/regions", mode: 'copy', overwrite: true, pattern: "*.json"

    input:
    tuple path("spatialdata.h5ad"), val(region_id), path("region.json")

    output:
    path "region.h5ad", emit: anndata
    path "*/region.json", emit: region_json

    script:
    template "subset_region.py"
}

process join_regions {
    input:
    path "region.*.h5ad"

    output:
    path "spatialdata.h5ad"

    script:
    template "join_regions.py"
}


workflow extract_regions {
    take:
    source_datasets_xenium
    source_datasets_stardist

    main:
    // Extract the points encoded by each region
    extract_regions_xenium(source_datasets_xenium)

    // Extract the points encoded by each region
    extract_regions_stardist(source_datasets_stardist)

    // Make sure that at least one region was found
    extract_regions_xenium.out.anndata
        .mix (extract_regions_stardist.out.anndata)
        .ifEmpty { error "No regions were found" }
        .set { all_regions }

    // Merge the regions into a single object
    join_regions(all_regions.toSortedList())

    emit:
    anndata = join_regions.out
    region_defs = extract_regions_xenium.out.region_json.mix(extract_regions_stardist.out.region_json)
    spatialdata = extract_regions_xenium.out.spatialdata.mix(extract_regions_stardist.out.spatialdata)
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
                file(it[0], type: "dir", checkIfExists: true)
            ]
        }
        // Convert the datasets to AnnData (h5ad) format
        | read_xenium

    // Extract the data from each region
    read_xenium
        .out
        .anndata
        .join(
            source_datasets
                .map { [it[0], it[2]] }
        )
        .transpose()
        .map { [it[1], it[2][0], it[2][1]] }
        | subset_region

    emit:
    anndata = subset_region.out.anndata
    region_json = subset_region.out.region_json
    spatialdata = read_xenium.out.spatialdata
}

workflow extract_regions_stardist {
    take:
    source_datasets

    main:

    // For each input dataset (given as a URI), get the files needed to parse the features and spatial data
    source_datasets
        .map {
            return [
                it[0],
                file(it[0] + "/cell_measurements", type: "dir", checkIfExists: true)
            ]
        }
        // Convert the datasets to AnnData (h5ad) format
        | read_stardist

    // Extract the data from each region
    read_stardist
        .out
        .anndata
        .join(
            source_datasets
                .map { [it[0], it[2]] }
        )
        .transpose()
        .map { [it[1], it[2][0], it[2][1]] }
        | subset_region

    // Get the spatialdata which is already formatted nicely in the stardist outputs
    source_datasets
        .map {
            return [
                it[0],
                file(it[0] + "/dashboard/spatialdata.zarr.zip", checkIfExists: true)
            ]
        }
        // Convert the datasets to AnnData (h5ad) format
        .set { spatialdata }
    

    emit:
    anndata = subset_region.out.anndata
    region_json = subset_region.out.region_json
    spatialdata = spatialdata
}