# Quantitative Spatial Analysis
Analyze collections of spatial data objects

## Background

Spatial data in this case is any dataset in which measurements on a set of observations
have been collected, where each observation also has a location in physical space.

A good example of this is spatial transcriptomics, where each cell (observation) has
been measured for the expression of a collection of genes (features), and where the location
of each cell is given in x/y coordinates.

## Purpose

The goal of this project is to allow users to:

- Select and annotate regions from one or more spatial datasets (e.g. cores from a TMA)
- Build an aggregate collection which consists of multiple regions from one or more source datasets
- Save that collection as a cohesive data package
- Perform clustering on cells (e.g. k-means, leiden, louvain) and annotate each cluster (i.e. cell type)
- Perform neighborhood analysis across the aggregate collection
- Compute summary metrics on the cell type vs. neighborhood vs. region levels
- Visualize the entire spatial data collection in an interactive way

## Components

### Interactive App

The interactive process of region selection can be performed with the [Streamlit](https://streamlit.io/)
app defined by `app.py` and `app/`.

First install prerequisites with `pip install -r requirements.txt`, and then run the app with

```shell
streamlit run app.py
```

### Region Definition

Each region is defined on the basis of:

1. A source dataset with a particular type (e.g. `"xenium"`) which
is accessible from a recognizable data repository (e.g. `"cirro"`).
2. One or more outlines given as x/y coordinate arrays for which
any points within those shapes are included in the region.

The JSON-serializable format for the object follows the pattern:

```json
{
    "dataset": {
        "cirro_source": {
          "domain": "organization.cirro.bio",
            "project": "000000000-0000-0000-0000-000000000000",
            "dataset": "00000000-0000-0000-0000-000000000000",
            "path": "data/analysis-subfolder"
        },
        "type": "xenium"
    },
    "outline": [
        {
            "x": [
                1671.0085411393027,
                1497.2881668808081,
                1005.0804398150743,
                3431.375000292045,
                2811.7723321034155
            ],
            "xref": "x",
            "y": [
                4049.5208189474565,
                3861.008928715988,
                3528.3408871310435,
                4265.75504597767
            ],
            "yref": "y"
        }
    ]
}
```

## Analysis Workflow

The `analyze_regions.nf` workflow can be used to analyze one or more
spatial regions using the analysis steps outlined above.
It is a Nextflow workflow which can be used following the
[the official documentation](https://nextflow.io).

### Inputs

- `regions`: The path to a CSV file containing a list of regions, with the columns `id` (the name/identifier of the region) and `uri` (the path to the `region.json` file defined above)
- `resolution`: The leiden clustering resolution used for determining cell types
- `n_neighbors`: The number of neighbors for each individual cell to consider when performing neighborhood anlaysis
- `n_neighborhoods`: The number of neighborhoods to return (provided as the `k` for k-means clustering)
- `outdir`: The base path for all output files. Note that this cannot have a leading or trailing slash (`/`) based on Nextflow publishing implementation.

### Outputs

```
├── combined
│   ├── counts.csv # The number of cells per region, per cell type, per neighborhood
│   └── spatialdata.h5ad # The spatial coordinates, measurements (e.g. gene expression) and annotations for each individual cell
├── logs
│   ├── cluster_points.txt
│   ├── make_plots.txt
│   ├── neighborhood_analysis.txt
│   ├── summary_stats.txt
│   └── vitessce.txt
└── regions
    ├── <regionId> # Information for each individual region
    │   ├── cell_types.vt.json
    │   ├── neighborhoods.vt.json
    │   └── spatialdata.zarr.zip # SpatialData object in zarr format
    └── <regionId>.json # Definition of the region as shown above
```