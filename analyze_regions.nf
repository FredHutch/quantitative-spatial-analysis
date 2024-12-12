#!/usr/bin/env nextflow

// Workflow used to analyze a collection of regions

include { analyze_regions } from './modules/analyze_regions'
nextflow.preview.output = true

workflow {

    main:
    if (!params.regions) {
        error "Parameter 'regions' is required but not provided."
    }

    if (!params.outdir) {
        error "Parameter 'outdir' is required but not provided."
    }

    log.info """
#################################
# QUANTITATIVE SPATIAL ANALYSIS #
#################################

Analyzing regions: ${params.regions}
    """

    Channel
        .fromPath(params.regions, checkIfExists: true)
        .splitCsv(header: true, sep: ',')
        .set { regions }

    analyze_regions(regions)
    
    publish:
    analyze_regions.out.region_defs >> "region_defs"
    analyze_regions.out.logs >> "logs"
    analyze_regions.out.plots >> "plots"
    analyze_regions.out.summary >> "summary"
    analyze_regions.out.spatial >> "spatial"
    analyze_regions.out.zarr >> "zarr"
}

output {
    "region_defs" {
        mode "copy"
        overwrite true
        path "${params.outdir}/regions"
    }
    "logs" {
        mode "copy"
        overwrite true
        path "${params.outdir}"
    }
    "plots" {
        mode "copy"
        overwrite true
        path "${params.outdir}"
    }
    "zarr" {
        mode "copy"
        overwrite true
        path "${params.outdir}"
    }
    "summary" {
        mode "copy"
        overwrite true
        path "${params.outdir}/combined"
    }
    "spatial" {
        mode "copy"
        overwrite true
        path "${params.outdir}/combined"
    }
}