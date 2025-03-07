import io
from cirro import DataPortalDataset, DataPortalProject
from matplotlib import pyplot as plt
import seaborn as sns
from anndata import AnnData
from app import html
from app.datasets.ui import back_button
import streamlit as st
from app.analyses.data import get_catalog, SpatialAnalysisCatalog
from app.analyses.plots import sort_table
from app.analyses import plots
from app.streamlit import get_query_param, clear_query_param
from app.cirro import pick_dataset
from app.html import linked_header
import pandas as pd
import plotly.express as px
from time import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def main(project: DataPortalProject):

    # If there is no dataset selected
    if get_query_param("dataset") is None:
        logger.info("Clearing params from analyses.main")
        clear_query_param("pick_region")
        clear_query_param("show_region")

        # Show the dataset selection menu
        dataset = select_dataset(project)

        if dataset is None:

            # Show a button that lets the user kick off their own analysis
            html.cirro_analysis_button(
                "Run a new analysis",
                None,
                "process-hutch-quantitative-spatial-analysis-1_0"
            )

        # Otherwise, redraw the page
        else:
            st.rerun()

    # Otherwise, show the dataset
    else:
        show_dataset(project)


def select_dataset(project: DataPortalProject) -> DataPortalDataset:
    """
    Show a menu that allows the user to select a dataset from the catalog.
    """

    catalog = get_catalog_cached(project.id)

    # If there are no datasets available
    if catalog.analyses is None:
        st.write("Select a data collection from the menu")

    elif catalog.analyses.shape[0] == 0:
        st.write("Data collection does not contain any recognized spatial analyses")

    # If there is a dataset selected and it is part of the catalog
    elif get_query_param("dataset") is not None and get_query_param("dataset") in catalog.df["id"].values:
        return project.get_dataset_by_id(get_query_param("dataset"))

    else:

        # Show the table of datasets which can be selected
        return pick_dataset(
            project,
            catalog.analyses,
            ["Name", "Created", "Description"],
            {
                "Name": st.column_config.TextColumn(width="medium", disabled=True),
                "Description": st.column_config.TextColumn(width="medium", disabled=True),
                "Created": st.column_config.TextColumn(max_chars=14, disabled=True),
            },
            "Select an analysis to view its contents"
        )


def update_refresh_time():
    st.session_state["refresh_time"] = time()


def get_catalog_cached(project_id: str) -> SpatialAnalysisCatalog:
    # Get the data catalog, respecting the refresh time
    with st.spinner("Loading catalog..."):
        return get_catalog(
            st.session_state.get("refresh_time"),
            project_id
        )


def show_dataset(project: DataPortalProject):

    # Get the catalog
    catalog = get_catalog_cached(project.id)

    # Get the dataset ID
    dataset_id = get_query_param("dataset")

    # If the dataset is not in the catalog for the selected project
    if catalog is None or dataset_id not in catalog.datasets:
        # Deselect the project and the dataset
        logger.info("Clearing params from analyses.show_dataset")
        clear_query_param("dataset")
        st.rerun()
        return

    # Let the user navigate back to the dataset selection
    back_button("dataset", label="Switch Dataset")

    # Get the dataset
    dataset = catalog.datasets[dataset_id]

    # Display the dataset information
    # Show the ingest dataset
    with st.container(key=dataset_id):
        html.card_content(
            title=dataset.name.strip(),
            content=[
                (
                    dataset.description.strip()
                    if dataset.description
                    else ""
                ),
                f"<b>Created</b>: {dataset.created_at.strftime('%Y-%m-%d %H:%M')}"
            ]
        )
        # Show the dataset in Cirro
        html.cirro_dataset_button(dataset_id)
    html.card_style(dataset_id)

    # Show a display of the analysis results
    explore_analysis()

    back_button("dataset", label="Switch Dataset")


def read_file(file_path: str, filetype="csv", **kwargs):
    """Read a file from the currently selected dataset."""

    # Get the dataset and project ID
    dataset_id = get_query_param("dataset")
    project_id = get_query_param("project")
    assert dataset_id is not None, "Cannot read file - no dataset selected"
    assert project_id is not None, "Cannot read file - no project selected"

    # Read the file, caching on dataset ID and file path
    return read_file_cached(project_id, dataset_id, file_path, filetype, **kwargs)


@st.cache_data
def read_file_cached(project_id: str, dataset_id: str, file_path: str, filetype: str, **kwargs):

    # Get the catalog
    catalog = get_catalog_cached(project_id)

    # Get the dataset
    dataset = catalog.datasets[dataset_id]

    # Get the file
    file = dataset.list_files().get_by_name("data/" + file_path)

    # If the filetype is feather, read the object as bytes to a stream
    # and then use the pandas read_feather method
    if filetype == "feather":
        return pd.read_feather(io.BytesIO(file._get()))

    else:

        # Otherwise use the file reader method
        return getattr(file, f"read_{filetype}")(**kwargs)


def has_file(file_path: str):
    """Check to see if the selected dataset conatins a particular file."""
    # Get the dataset and project ID
    dataset_id = get_query_param("dataset")
    project_id = get_query_param("project")
    assert dataset_id is not None, "Cannot read file - no dataset selected"
    assert project_id is not None, "Cannot read file - no project selected"

    return has_file_cached(project_id, dataset_id, file_path)


@st.cache_data
def has_file_cached(project_id: str, dataset_id: str, file_path: str):

    # Get the catalog
    catalog = get_catalog_cached(project_id)

    # Get the dataset
    dataset = catalog.datasets[dataset_id]

    # Get the file
    return any([
        file.name == "data/" + file_path
        for file in dataset.list_files()
    ])


def show_umap():
    linked_header("UMAP Embedding")

    # Check to see if the UMAP embedding has been computed
    umap_fp = "combined/spatialdata.umap.feather"
    if not has_file(umap_fp):
        st.write("No UMAP coordinates found - please run the latest analysis version")
        return

    # Otherwise, read in the UMAP coordinates
    with st.spinner("Reading Data..."):
        umap = read_file(umap_fp, filetype="feather")

    # Read in the annotations
    annot_fp = "combined/spatialdata.annotations.feather"
    if not has_file(annot_fp):
        st.write("No cluster annotations found - please run the latest analysis version")
        return

    # Otherwise, read in the annotations
    with st.spinner("Reading Data..."):
        annot = read_file(annot_fp, filetype="feather")

    # Merge the tables
    umap = umap.merge(
        annot.reindex(columns=['cluster', 'neighborhood', 'region']),
        left_index=True,
        right_index=True
    )

    # Let the user filter out cells based on their cluster, neighborhood, or region
    display_clusters = st.multiselect(
        "Show Clusters:",
        options=umap["cluster"].unique(),
        default=umap["cluster"].unique()
    )
    display_neighborhoods = st.multiselect(
        "Show Neighborhoods:",
        options=umap["neighborhood"].unique(),
        default=umap["neighborhood"].unique()
    )
    display_regions = st.multiselect(
        "Show Regions:",
        options=umap["region"].unique(),
        default=umap["region"].unique()
    )

    # Filter the data
    umap = umap[
        umap["cluster"].isin(display_clusters) &
        umap["neighborhood"].isin(display_neighborhoods) &
        umap["region"].isin(display_regions)
    ]

    # If there is no data left, stop here
    if umap.shape[0] == 0:
        st.write("Please select more data to plot")
        return

    cols_a = st.columns(3)

    # Let the user pick the point size
    size = cols_a[0].number_input(
        "Point Size",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    # Let the user pick the width and height of the plot
    width = cols_a[1].number_input(
        "Figure Width",
        min_value=1,
        max_value=20,
        value=10,
        step=1
    )
    height = cols_a[2].number_input(
        "Figure Height",
        min_value=1,
        max_value=20,
        value=6,
        step=1
    )

    cols_b = st.columns(3)

    # Let the user decide which annotations to show
    color = cols_b[0].selectbox(
        "Color By:",
        options=["cluster", "neighborhood", "region"],
        index=0
    )

    # Let the user select how many points to show
    subsample = cols_b[1].checkbox("Subsample Points", value=True)
    if subsample:
        subsample_size = cols_b[2].number_input(
            "Subsample Size",
            min_value=1,
            max_value=umap.shape[0],
            value=min(100000, umap.shape[0]),
            step=1
        )
        umap = umap.sample(subsample_size)

    with st.spinner("Plotting..."):
        # Make a plot
        fig, ax = plt.subplots(figsize=(width, height))
        sns.scatterplot(
            (
                umap
                .sort_values(by=color)
                .assign(
                    **{
                        color: umap[color].astype(str)
                    }
                )
            ),
            x=umap.columns.values[0],
            y=umap.columns.values[1],
            hue=color,
            s=size,
            linewidth=0,
            ax=ax
        )
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        # Format the legend
        ax.legend(title=color.title(), bbox_to_anchor=[1, 1], loc='upper left', markerscale=10/size)
        # Remove the box around the plot
        sns.despine(ax=ax, left=True, bottom=True)
        # Remove the ticks
        ax.set_xticks([])
        ax.set_yticks([])
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)

    with st.spinner("Rasterizing UMAP..."):
        # Make a button to download as PNG
        with io.BytesIO() as buf:
            plt.savefig(buf, format="png")
            plot_bytes = buf.getvalue()
    st.download_button("Download PNG", data=plot_bytes, file_name="umap.png")


def show_features_by_cluster():
    linked_header("Compare Features Across Clusters")

    # Check to see if the feature summary metrics have been computed across clusters
    metrics_fp = "combined/cluster_feature_metrics.csv"
    if not has_file(metrics_fp):
        st.write("No feature summary table found -- please run the latest analysis version")
        return

    # Otherwise, read in the table
    with st.spinner("Reading Data..."):
        metrics = read_file(metrics_fp, filetype="csv", index_col=[0, 1])

    # For every feature, calculate the ratio of the std across clusters vs. the mean std within clusters
    feature_spread = pd.Series({
        feature: (
            dat.loc[(slice(None), 'mean')].std() /
            dat.loc[(slice(None), 'std')].mean()
        )
        for feature, dat in metrics.items()
    }).sort_values(ascending=False)

    # Let the user decide what clusters to show, defaulting to all of them
    all_clusters = metrics.reset_index()['cluster'].unique()
    all_clusters.sort()
    selected_clusters = st.multiselect(
        label="Select Cluster:",
        options=all_clusters,
        default=all_clusters
    )

    # Let the user decide what features to show, defaulting to the top 30
    selected_features = st.multiselect(
        label="Show Features:",
        options=feature_spread.index.values,
        default=feature_spread.head(30).index.values
    )

    # If none were selected, stop here
    if len(selected_features) == 0:
        return

    # Make a long DataFrame for plotting
    plot_df: pd.DataFrame = (
        metrics
        .loc[(selected_clusters, slice(None))]
        .reindex(columns=selected_features)
        .reset_index()
        .melt(id_vars=["cluster", "level_1"])
        .rename(
            columns=dict(
                level_1="metric",
                variable="feature"
            )
        )
        .pivot_table(
            index=["cluster", "feature"],
            columns="metric",
            values="value"
        )
        .reset_index()
    )

    # Use the values to sort the display
    wide_df = plot_df.pivot(
        index="cluster",
        columns="feature",
        values="mean"
    )
    wide_df = sort_table(wide_df)
    wide_df = sort_table(wide_df.T).T

    if st.selectbox(
        "Plot Type:",
        options=["Bubble Plot", "Heatmap"],
        index=1
    ) == "Bubble Plot":

        fig = px.scatter(
            plot_df.assign(adj_mean=plot_df['mean']-plot_df['mean'].min()),
            x="feature",
            y="cluster",
            color="mean",
            size="adj_mean",
            size_max=10,
            category_orders=dict(
                feature=wide_df.columns.values,
                cluster=wide_df.index.values
            )
        )

    else:

        fig = px.imshow(
            wide_df.values,
            color_continuous_scale="Blues",
            labels=dict(
                x="Feature",
                y="Cluster",
                color="Mean Abundance (Normalized)"
            ),
            aspect=(
                None if st.checkbox(label="Square Aspect", value=False, key="show-features-square-aspect") else "auto"
            )
        )
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(wide_df.shape[1])),
            ticktext=wide_df.columns.values
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(wide_df.shape[0])),
            ticktext=wide_df.index.values
        )
    )

    st.plotly_chart(fig, use_container_width=False)


def get_annotations(counts: pd.DataFrame, annotation_type: str):
    """Let the user supply a table of annotations."""

    linked_header(f"Annotate {annotation_type.title()}s")

    # If there are no annotations in the state, set them up
    session_state_kw = f"{annotation_type}_annotations_working"
    if st.session_state.get(session_state_kw) is None:

        # Use the counts to show the user what proportion of cells belong to each cluster
        st.session_state[session_state_kw] = (
            counts
            .groupby(annotation_type)["count"]
            .sum()
            .reset_index()
            .assign(Percent=lambda d: 100 * d['count'] / d['count'].sum())
            .drop(columns='count')
            .sort_values(by="Percent", ascending=False)
            .astype(str)
        )

    # Present the annotations in two columns
    col1, col2 = st.columns(2)

    # Let the user provide a table
    user_provided = col2.file_uploader(f"Upload {annotation_type.title()} Annotations:")

    if user_provided:
        # Read the table
        user_provided = pd.read_csv(user_provided).astype(str)

        # Add the user provided annotations
        st.session_state[session_state_kw] = (
            st.session_state[session_state_kw]
            .drop(
                columns=[
                    cname for cname in st.session_state[session_state_kw].columns
                    if (
                        (cname in user_provided.columns)
                        and
                        (cname not in ["Percent", annotation_type])
                    )
                ]
            )
            .merge(
                user_provided.drop(columns=["Percent"] if "Percent" in user_provided.columns else []),
                how="left",
                left_on=annotation_type,
                right_on=annotation_type
            )
        )

    # Show the table
    col1.dataframe(
        st.session_state[session_state_kw],
        column_config=dict(
            Percent=st.column_config.NumberColumn(format="%.2f")
        ),
        hide_index=True,
        use_container_width=True
    )

    # Let the user download the table
    col2.download_button(
        "Download Annotations (CSV)",
        st.session_state[session_state_kw].to_csv(index=None),
        file_name=f"{annotation_type}_annotations.csv",
        help=f"Download the {annotation_type} annotation table to reuse in future analysis"
    )

    # Return the table
    return st.session_state[session_state_kw]


def new_category_name(annot_df: pd.DataFrame) -> str:
    """Pick a name for a new category that doesn't already exist."""
    i = 1
    while f"Category {i}" in annot_df.columns.values:
        i += 1
    return f"Category {i}"


def read_spatial(cluster_annot_df: pd.DataFrame, neighborhood_annot_df: pd.DataFrame) -> AnnData:
    """Read the spatial attributes as AnnData."""

    # Read the spatialdata object
    with st.spinner("Reading Spatial Coordinates"):
        adata: AnnData = read_file("combined/spatialdata.h5ad", filetype="h5ad")

    # Merge in the cluster and neighborhod annotations
    with st.spinner("Adding Annotations"):
        adata.obsm["cell_annotations"] = adata.obs.reindex(columns=["cluster", "neighborhood"]).astype(str)
        adata.obsm["cell_annotations"] = adata.obsm["cell_annotations"].assign(**{
            annot_name: adata.obs["cluster"].apply(annot_values.get)
            for annot_name, annot_values in cluster_annot_df.astype(str).set_index("cluster").items()
            if annot_name not in ['Percent']
        })
        adata.obsm["cell_annotations"] = adata.obsm["cell_annotations"].assign(**{
            annot_name: adata.obs["neighborhood"].apply(annot_values.get)
            for annot_name, annot_values in neighborhood_annot_df.astype(str).set_index("neighborhood").items()
            if annot_name not in ['Percent']
        })

    return adata


def show_spatial(adata: AnnData):
    """Show the viewer the spatial distribution of cells."""

    # Add a header that has a link in the sidebar
    linked_header("View Spatial Coordinates")

    # Pick the region to display
    region = st.selectbox(
        label="Select Region:",
        options=adata.obs["region"].unique()
    )

    # Select the annotation to display
    annotation = st.selectbox(
        label="Select Annotation:",
        options=list(adata.obsm["cell_annotations"].keys())
    )

    # Subset the points to this particular region
    region_adata = adata[adata.obs["region"] == region]

    # Get the vector of labels
    labels: pd.Series = region_adata.obsm["cell_annotations"][annotation]

    # If there are no labels
    if labels.dropna().shape[0] == 0:
        st.write(f"No annotations found for group '{annotation}'")

    # Select the groups to display
    groups = st.multiselect(
        "Select Groups:",
        options=labels.dropna().drop_duplicates().sort_values().values,
        default=labels.dropna().drop_duplicates().sort_values().values
    )
    if len(groups) == 0:
        st.write("Provide Groups to Display")
        return

    # Subset to the selected points
    to_plot = region_adata[labels.isin(groups), :]

    # Make a plot
    fig, ax = plt.subplots(
        figsize=(
            st.number_input("Figure Width", min_value=1., max_value=20., value=5., step=0.1),
            st.number_input("Figure Height", min_value=1., max_value=20., value=3., step=0.1)
        )
    )
    sns.scatterplot(
        data=pd.concat([
            pd.DataFrame(to_plot.obsm["spatial"], index=to_plot.obs_names, columns=["x_centroid", "y_centroid"]),
            to_plot.obsm["cell_annotations"].astype(str)
        ], axis=1).sort_values(by=annotation),
        x="x_centroid",
        y="y_centroid",
        hue=annotation,
        linewidth=0,
        s=st.number_input("Point Size", min_value=0.1, max_value=20., value=1.0),
        ax=ax
    )
    if st.checkbox("Invert X axis"):
        ax.invert_xaxis()
    if st.checkbox("Invert Y axis"):
        ax.invert_yaxis()
    lgnd = plt.legend(bbox_to_anchor=[1, 1])
    for handle in lgnd.legend_handles:
        handle.set_markersize(6.0)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
    # Let the user download the image
    with io.BytesIO() as buf:
        plt.savefig(buf, format="png")
        plot_bytes = buf.getvalue()
    st.download_button("Download PNG", data=plot_bytes, file_name=f"{region}.{annotation}.png")


def explore_analysis():
    """
    Provide a set of displays summarizing the analysis results for the dataset.
    """

    # Display the UMAP embedding
    show_umap()

    # Display the marker abundances by cluster
    show_features_by_cluster()

    # Read the counts
    counts = read_file("combined/counts.csv", filetype="csv")

    # Let the user annotate / merge clusters for analysis
    cluster_annot_df = get_annotations(counts, "cluster")

    # Let the user annotate / merge neighborhoods for analysis
    neighborhood_annot_df = get_annotations(counts, "neighborhood")

    # Show the user the spatial distribution of each cluster/neighborhood/annotation
    adata = read_spatial(cluster_annot_df, neighborhood_annot_df)
    show_spatial(adata)

    # Let the user perform numeric comparisons
    compare_counts(counts, cluster_annot_df, neighborhood_annot_df)


def compare_counts(counts: pd.DataFrame, cluster_annot_df: pd.DataFrame, neighborhood_annot_df: pd.DataFrame):
    linked_header("Compare Cell Counts")

    # Merge the counts with the annotations
    counts = (
        counts
        .assign(
            **{
                cname: counts[cname].astype(str)
                for cname in ["region", "cluster", "neighborhood"]
            }
        )
        .merge(
            cluster_annot_df.drop(columns=["Percent"]).assign(
                cluster=lambda d: d["cluster"].astype(str)
            ),
            left_on="cluster",
            right_on="cluster"
        )
        .merge(
            neighborhood_annot_df.drop(columns=["Percent"]).assign(
                neighborhood=lambda d: d["neighborhood"].astype(str)
            ),
            left_on="neighborhood",
            right_on="neighborhood"
        )
    )

    # Let the user configure the plot
    plots.compare_counts(counts)
