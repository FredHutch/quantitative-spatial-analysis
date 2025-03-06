import io
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
from app.cirro import show_menu
from app.html import linked_header
import pandas as pd
import plotly.express as px
from time import time


def main():

    # If there is no dataset selected
    if get_query_param("dataset") is None:
        clear_query_param("pick_region")
        clear_query_param("show_region")
        # Show the dataset selection menu
        select_dataset()

        # Show a button that lets the user kick off their own analysis
        html.cirro_analysis_button(
            "Run a new analysis",
            None,
            "process-hutch-quantitative-spatial-analysis-1_0"
        )

    # Otherwise, show the dataset
    else:
        show_dataset()


def select_dataset():
    """
    Show a menu that allows the user to select a dataset from the catalog.
    """

    catalog = get_catalog_cached()

    # If there are no datasets available
    if catalog.analyses is None:
        st.write("Select a data collection from the menu")

    elif catalog.analyses.shape[0] == 0:
        st.write("Data collection does not contain any recognized spatial analyses")

    else:
        print(catalog.analyses)

        # Show the table of datasets which can be selected
        show_menu(
            "dataset",
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


def get_catalog_cached() -> SpatialAnalysisCatalog:
    # Get the data catalog, respecting the refresh time
    with st.spinner("Loading catalog..."):
        return get_catalog(
            st.session_state.get("refresh_time"),
            get_query_param("project")
        )


def show_dataset():

    # Get the catalog
    catalog = get_catalog_cached()

    # Get the dataset ID
    dataset_id = get_query_param("dataset")

    # If the dataset is not in the catalog for the selected project
    if catalog is None or dataset_id not in catalog.datasets:
        # Deselect the project and the dataset
        clear_query_param("project")
        clear_query_param("dataset")
        st.rerun()
        return

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

    # Get the dataset ID
    dataset_id = get_query_param("dataset")
    assert dataset_id is not None, "Cannot read file - no dataset selected"

    # Read the file, caching on dataset ID and file path
    return read_file_cached(dataset_id, file_path, filetype, **kwargs)


@st.cache_data
def read_file_cached(dataset_id: str, file_path: str, filetype: str, **kwargs):

    # Get the catalog
    catalog = get_catalog_cached()

    # Get the dataset
    dataset = catalog.datasets[dataset_id]

    # Get the file
    file = dataset.list_files().get_by_name("data/" + file_path)

    # Read the file
    return getattr(file, f"read_{filetype}")(**kwargs)


def has_file(file_path: str):
    """Check to see if the selected dataset conatins a particular file."""
    # Get the dataset ID
    dataset_id = get_query_param("dataset")
    assert dataset_id is not None, "Cannot read file - no dataset selected"

    return has_file_cached(dataset_id, file_path)


@st.cache_data
def has_file_cached(dataset_id: str, file_path: str):

    # Get the catalog
    catalog = get_catalog_cached()

    # Get the dataset
    dataset = catalog.datasets[dataset_id]

    # Get the file
    return any([
        file.name == "data/" + file_path
        for file in dataset.list_files()
    ])


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
