from time import sleep
from typing import List, Tuple, Union
import pandas as pd
import streamlit as st
import plotly.express as px
from app import html
from app.cirro import save_region, show_menu
from app.datasets.data import get_catalog, SpatialDataCatalog
from app.models.points import SpatialPoints, SpatialRegion
from app.streamlit import set_query_param, clear_query_param, get_query_param


def get_catalog_cached() -> SpatialDataCatalog:
    # Get the data catalog, respecting the refresh time
    with st.spinner("Loading catalog..."):

        return get_catalog(
            st.session_state.get("refresh_time"),
            get_query_param("project")
        )


def main():

    # If there is no dataset selected
    if get_query_param("dataset") is None:
        clear_query_param("pick_region")
        clear_query_param("show_region")
        # Show the dataset selection menu
        select_dataset()

    # If the user has selected a dataset to pick regions for, show that interface
    elif get_query_param("pick_region") is not None:
        pick_region()

    # If the user has selected a region to display, show that region
    elif get_query_param("show_region") is not None:
        show_region()

    # Otherwise, show the dataset
    else:
        show_dataset()


def select_dataset():
    """
    Show a menu that allows the user to select a dataset from the catalog.
    """

    catalog = get_catalog_cached()

    # If there are no datasets available
    if catalog.df is None:
        st.write("Select a data collection from the menu")

    elif catalog.df.shape[0] == 0:
        st.write("Data collection does not contain any recognized spatial datasets")

    else:
        # Show the table of datasets which can be selected
        show_menu(
            "dataset",
            catalog.df,
            ["Name", "Created", "Type", "Analysis Outputs"],
            {
                "Name": st.column_config.TextColumn(width="medium", disabled=True),
                "Created": st.column_config.TextColumn(max_chars=14, disabled=True),
                "Type": st.column_config.TextColumn(width="small", disabled=True),
                "Analysis Outputs": st.column_config.ListColumn()
            },
            "Select a dataset to view its contents",
            clear_params=["pick_region", "show_region"]
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

    # Show the ingest dataset
    with st.container(key=dataset_id):
        html.card_content(
            title=catalog.datasets[dataset_id].name.strip(),
            content=[
                catalog.datasets[dataset_id].description,
                f"<b>Created</b>: {catalog.datasets[dataset_id].created_at.strftime('%Y-%m-%d %H:%M')}",
                f"<b>Type</b>: {catalog.dataset_types[dataset_id]}"
            ]
        )
        dataset_buttons(catalog, dataset_id)
    html.card_style(dataset_id)

    # Make a list of the analysis outputs derived from this dataset
    analysis_outputs = catalog.groups[dataset_id]

    # Ignore the ingest dataset itself
    analysis_outputs = [child_id for child_id in analysis_outputs if child_id != dataset_id]

    # If there are none, stop
    if len(analysis_outputs) == 0:
        back_button("dataset", label="Switch Dataset")
        return

    # Group the analysis outputs by type
    analysis_output_types = [catalog.dataset_types[dataset_id] for dataset_id in analysis_outputs]
    ix = 0
    for dataset_type, dataset_list in pd.Series(analysis_outputs).groupby(analysis_output_types):

        # Make a card for each analysis type
        card_key = f"{dataset_id}-{ix}"
        with st.container(key=card_key):
            html.card_content(
                title=dataset_type[0] if isinstance(dataset_type, (list, tuple)) else dataset_type,
                content=[]
            )
            for dataset_id in dataset_list:
                html.paragraph([
                    f"{catalog.datasets[dataset_id].name.strip()}",
                    f"<b>Created</b>: {catalog.datasets[dataset_id].created_at.strftime('%Y-%m-%d %H:%M')}"
                ])
                dataset_buttons(catalog, dataset_id)
        html.card_style(card_key)
        ix += 1
    back_button("dataset", label="Switch Dataset")


def dataset_buttons(catalog: SpatialDataCatalog, dataset_id: str):
    """
    Show all of the action buttons available for a particular dataset.
    """

    # Show the dataset in Cirro
    html.cirro_dataset_button(dataset_id)

    # Get the dataset type
    process_id = catalog.process_id(dataset_id)

    # Images: Run StarDist / Cellpose
    if process_id == "images":
        html.cirro_analysis_button("Run Segmentation (StarDist)", dataset_id, "process-hutch-qupath-stardist-1_0")
        html.cirro_analysis_button("Run Segmentation (Cellpose)", dataset_id, "process-hutch-cellpose-1_0")

    # StarDist / Cellpose: Pick Region
    elif process_id in ["process-hutch-qupath-stardist-1_0", "process-hutch-cellpose-1_0"]:
        if st.button("Pick Region", key=f"pick-regions-{dataset_id}"):
            set_query_param("pick_region", dataset_id)
            st.rerun()

    # Xenium: Pick Region, Run Proseg
    elif process_id == "xenium":
        html.cirro_analysis_button("Run Segmentation (Proseg)", dataset_id, "proseg-resegment-1-0")
        if st.button("Pick Region", key=f"pick-regions-{dataset_id}"):
            set_query_param("pick_region", dataset_id)
            st.rerun()

    elif process_id == "region":
        if st.button("Show Region", key=f"show-region-{dataset_id}"):
            set_query_param("show_region", dataset_id)
            st.rerun()


def _calc_plot_size(points: SpatialPoints) -> Tuple[int, int]:
    # Use the dimensions to make the coordinates square
    x_range = points.coords[points.xcol].max() - points.coords[points.xcol].min()
    y_range = points.coords[points.ycol].max() - points.coords[points.ycol].min()

    # Set the default size such that the longer dimension is 800 pixels
    if x_range > y_range:
        default_width = 800
    else:
        default_height = 800
        default_width = int(default_height * x_range / y_range)

    # Let the user modify the width
    width = st.number_input("Width", value=default_width, min_value=100, max_value=2000, step=100)
    height = int(width * y_range / x_range)

    # If the user wants to keep the fixed aspect ratio, the height picker is disabled
    if st.checkbox(f"Fixed Aspect Ratio ({height:,}px)", value=True):
        return width, height
    else:
        return width, st.number_input("Height", value=height)


def back_button(session_key: Union[str, List[str]], label="Back"):
    if st.button(label, key=f"back-{session_key}"):
        if isinstance(session_key, list):
            for key in session_key:
                clear_query_param(key)
        else:
            clear_query_param(session_key)
        st.rerun()


def pick_region():
    # Get the catalog
    catalog = get_catalog_cached()
    # Get the dataset ID which the region will be picked from
    dataset_id = get_query_param("pick_region")

    # Print the dataset name
    st.write(catalog.datasets[dataset_id].name)

    # Get the coordinates of points for this dataset
    with st.spinner("Loading points..."):
        try:
            points: SpatialPoints = catalog.get_points(dataset_id)
        except Exception as e:
            st.exception(e)
            back_button("pick_region", label="Back to Dataset")
            return

    # Get the size of the plot to show
    width, height = _calc_plot_size(points)

    with st.spinner("Loading plot..."):
        # Set up the scatter plot
        fig = px.scatter(
            points.coords,
            x=points.xcol,
            y=points.ycol,
            color=(
                None if points.clusters is None else
                points.clusters.apply(str)
            ),
            width=width,
            height=height,
            opacity=st.number_input("Opacity", value=0.05, min_value=0.001, max_value=1., step=0.01),
            labels={
                points.xcol: "X Coordinate",
                points.ycol: "Y Coordinate"
            }
        )
        # Optionally invert the axes
        if st.checkbox("Invert X-axis", value=False):
            fig.update_xaxes(autorange='reversed')
        if st.checkbox("Invert Y-axis", value=False):
            fig.update_yaxes(autorange='reversed')
        # Display in streamlit and let the user select a region
        region = st.plotly_chart(
            fig,
            selection_mode="lasso",
            on_select="rerun",
            use_container_width=False
        )
    st.write("Use the lasso tool to select a region of interest")
    n_points = len(region["selection"]["points"])
    st.write(f"Selected {n_points:,} points")

    if n_points == 0:
        st.warning("Select some points to define a region")

    name = st.text_input("Name of region")

    if name:
        if st.button(f"Save Region - {name}"):
            with st.spinner("Saving region..."):
                try:
                    ds = save_region(
                        points,
                        name,
                        region["selection"]["lasso"]
                    )
                except Exception as e:
                    st.exception(e)
                    back_button("pick_region", label="Back to Dataset")
                    return
                if ds is not None:
                    try:
                        catalog.add_dataset(ds)
                    except Exception as e:
                        st.exception(e)
                        back_button("pick_region", label="Back to Dataset")
                        return
            sleep(1)
    else:
        st.warning("Enter a name for the region")

    back_button("pick_region", label="Back to Dataset")


def show_region():
    # Get the catalog
    catalog = get_catalog_cached()

    # Get the region ID
    region_id = get_query_param("show_region")

    # Get the region
    region: SpatialRegion = catalog.regions[region_id]

    # Get the coordinates of points for this dataset
    with st.spinner("Loading points..."):
        try:
            points: SpatialPoints = catalog.get_points(region.dataset.cirro_source.dataset)
        except Exception as e:
            st.exception(e)
            back_button("show_region", label="Back to Dataset")
            return

    # Get the size of the plot to show
    width, height = _calc_plot_size(points)

    # Let the user modify the opacity

    # Set up the points
    fig = px.scatter(
        points.coords,
        x=points.xcol,
        y=points.ycol,
        width=width,
        height=height,
        opacity=st.number_input("Opacity", value=0.01, min_value=0.001, max_value=1., step=0.01),
        color_discrete_sequence=["blue"],
        labels={
            points.xcol: "X Coordinate",
            points.ycol: "Y Coordinate",
        }
    )
    # Optionally invert the axes
    if st.checkbox("Invert X-axis", value=False):
        fig.update_xaxes(autorange='reversed')
    if st.checkbox("Invert Y-axis", value=False):
        fig.update_yaxes(autorange='reversed')

    # Add the region outline
    for shape in region.outline:
        fig.add_trace(
            px.line(
                x=shape["x"] + [shape["x"][0]],
                y=shape["y"] + [shape["y"][0]],
                color_discrete_sequence=["orange"]
            ).data[0]
        )

    # Show the figure
    st.plotly_chart(fig, use_container_width=False)

    back_button("show_region", label="Back to Dataset")
