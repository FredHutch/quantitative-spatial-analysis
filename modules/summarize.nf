process make_plots {
    publishDir "${params.outdir}", mode: 'copy', overwrite: true

    input:
    path "spatialdata.h5ad"
    path "counts.csv"

    output:
    path "regions/*", emit: region_plots
    path "combined/*", emit: combined_plots
    path "logs/*", emit: logs

    script:
    template "make_plots.py"
}

process summary_stats {
    publishDir "${params.outdir}", mode: 'copy', overwrite: true

    input:
    path "spatialdata.h5ad"

    output:
    path "logs/*", emit: logs
    path "*.csv", emit: summary

    script:
    template "summary_stats.py"
}

workflow summarize {
    take:
    spatial_data

    main:
    summary_stats(spatial_data)
    make_plots(spatial_data, summary_stats.out.summary)

    emit:
    plots = make_plots.out.region_plots.mix(make_plots.out.combined_plots)
    summary = summary_stats.out.summary
    logs = make_plots.out.logs.mix(summary_stats.out.logs)

}
