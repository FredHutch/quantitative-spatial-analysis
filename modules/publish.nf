process make_plots {
    input:
    path "spatial.h5ad"

    output:
    path "*"

    script:
    template "make_plots.py"
}

process summary_stats {
    input:
    path "spatial.h5ad"

    output:
    path "*"

    script:
    template "summary_stats.py"
}

workflow publish {
    take:
    spatial_data

    main:
    make_plots(spatial_data)
    summary_stats(spatial_data)

    publish:
    make_plots.out >> "${params.outdir}/plots"
    summary_stats.out >> "${params.outdir}/summary"
    spatial_data >> "${params.outdir}/spatial.h5ad"

}