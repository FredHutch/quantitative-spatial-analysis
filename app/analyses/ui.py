from app import html
from app.datasets.ui import back_button
import streamlit as st
from app.analyses.data import get_catalog, SpatialAnalysisCatalog
from app.analyses import plots
from app.streamlit import get_query_param, clear_query_param
from app.cirro import select_project, show_menu


def main():
    st.write("#### Spatial Data Analysis")

    # If there is no project selected
    if get_query_param("project") is None:
        clear_query_param("dataset")
        clear_query_param("pick_region")
        clear_query_param("show_region")
        # Show the project selection menu
        select_project()
    
    # If there is no dataset selected
    elif get_query_param("dataset") is None:
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
        return
    elif catalog.analyses.shape[0] == 0:
        st.write("Data collection does not contain any recognized spatial analyses")
        return

    # Show the table of datasets which can be selected
    show_menu(
        "dataset",
        catalog.analyses,
        ["Name", "Created", "Type"],
        {
            "Name": st.column_config.TextColumn(width="medium", disabled=True),
            "Description": st.column_config.TextColumn(width="medium", disabled=True),
            "Created": st.column_config.TextColumn(max_chars=14, disabled=True),
        },
        "Select a dataset to view its contents"
    )

    # Show the back button
    back_button("project")


def get_catalog_cached() -> SpatialAnalysisCatalog:
    # Get the data catalog, respecting the refresh time
    with st.spinner(f"Loading catalog..."):

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
            title=catalog.datasets[dataset_id].name.strip(),
            content=[
                (
                    catalog.datasets[dataset_id].description.strip()
                    if catalog.datasets[dataset_id].description
                    else ""
                ),
                f"<b>Created</b>: {catalog.datasets[dataset_id].created_at.strftime('%Y-%m-%d %H:%M')}"
            ]
        )
        # Show the dataset in Cirro
        html.cirro_dataset_button(dataset_id)
    html.card_style(dataset_id)

    # Show a display of the analysis results
    explore_analysis(dataset_id)
    
    back_button("dataset")


def explore_analysis(dataset_id: str):
    """
    Provide a set of displays summarizing the analysis results for the dataset.
    """

    # Get the catalog
    catalog = get_catalog_cached()

    # Get the dataset
    ds = catalog.datasets[dataset_id]

    # Read the counts
    counts = ds.list_files().get_by_name("data/counts.csv").read_csv()

    counts.assign(
        **{
            cname: counts[cname].astype(str)
            for cname in ["region", "cluster", "neighborhood"]
        }
    )

    # Get the number of cells, clusters, neighborhoods, and regions
    n_cells = counts["count"].sum()
    n_clusters = counts["cluster"].nunique()
    n_neighborhoods = counts["neighborhood"].nunique()
    n_regions = counts["region"].nunique()
    # Show the summary information
    st.write(f"""
        - **Regions**: {n_regions:,}
        - **Neighborhoods**: {n_neighborhoods:,} 
        - **Clusters**: {n_clusters:,} 
        - **Cells**: {n_cells:,}
    """)

    # Let the user select the display format
    plot_type = st.selectbox(
        "Display",
        [
            "Cell Clusters Across Regions",
            "Neighborhoods Across Regions",
            "Cell Clusters Across Neighborhoods",
            "Cell Clusters Across Regions and Neighborhoods"
        ]
    )
    getattr(plots, plot_type.replace(" ", "_").lower())(counts)


main()
