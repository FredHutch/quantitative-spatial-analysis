from time import sleep, time
from typing import List, Tuple, Union
from cirro import DataPortalDataset, DataPortalProject
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from app import html
from app.cirro import save_region, pick_dataset, save_tma_cores
from app.datasets.data import get_catalog, SpatialDataCatalog
from app.datasets.helpers.autodetection import find_tma_cores, name_tma_cores
from app.models.points import SpatialPoints, SpatialRegion
from app.streamlit import set_query_param, clear_query_param, get_query_param
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_catalog_cached(project_id: str) -> SpatialDataCatalog:
    # Get the data catalog, respecting the refresh time
    with st.spinner("Loading catalog..."):

        return get_catalog(
            st.session_state.get("refresh_time"),
            project_id
        )


def main(project: DataPortalProject):

    # Get the dataset
    dataset = select_dataset(project)

    # If no dataset is selected, just let the user refresh the page
    if dataset is None:
        refresh_button(key='select-dataset-refresh-button')

    # If the user has selected a dataset to pick regions for, show that interface
    elif get_query_param("pick_region") is not None:
        pick_region(project)

    # If the user has selected a region to display, show that region
    elif get_query_param("show_region") is not None:
        show_region(project)

    # Otherwise, show the dataset
    else:
        show_dataset(dataset)


def select_dataset(project: DataPortalProject) -> DataPortalDataset:
    """
    Show a menu that allows the user to select a dataset from the catalog.
    """

    catalog = get_catalog_cached(project.id)

    # If there are no datasets available
    if catalog.df is None:
        st.write("Select a data collection from the menu")

    elif catalog.df.shape[0] == 0:
        st.write("Data collection does not contain any recognized spatial datasets")

    # If there is a dataset selected and it is part of the catalog
    elif get_query_param("dataset") is not None and get_query_param("dataset") in catalog.df["id"].values:
        return project.get_dataset_by_id(get_query_param("dataset"))

    else:
        # Show the table of datasets which can be selected
        return pick_dataset(
            project,
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


def show_dataset(project: DataPortalProject):

    # Get the catalog
    catalog = get_catalog_cached(project.id)

    # Get the dataset ID
    dataset_id = get_query_param("dataset")

    # If the dataset is not in the catalog for the selected project
    if catalog is None or dataset_id not in catalog.datasets:
        # Deselect the project and the dataset
        logger.info("Clearing params from datasets.show_dataset")
        clear_query_param("project")
        clear_query_param("dataset")
        st.rerun()
        return

    # Let the user navigate back to the dataset selection
    back_button("dataset", label="Switch Dataset", key="back-button-dataset-top")
    refresh_button(label="Refresh", key="refresh-button-dataset-top")

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
    logger.info("HERE0")
    logger.info(dataset_id)
    logger.info(analysis_outputs)

    # Ignore the ingest dataset itself
    analysis_outputs = [child_id for child_id in analysis_outputs if child_id != dataset_id]
    logger.info("HERE1")
    logger.info(analysis_outputs)

    # If there are none, stop
    if len(analysis_outputs) == 0:
        back_button("dataset", label="Switch Dataset", key="back-button-dataset-bottom")
        refresh_button(label="Refresh", key="refresh-button-dataset-bottom")
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
    back_button("dataset", label="Switch Dataset", key="back-button-dataset-top-bottom")
    refresh_button(label="Refresh", key="refresh-button-dataset-top-bottom")


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

    # StarDist / Cellpose / Proseg: Pick Region
    elif process_id in ["process-hutch-qupath-stardist-1_0", "process-hutch-cellpose-1_0", "proseg-resegment-1-0"]:
        if st.button("Pick Region", key=f"pick-regions-{dataset_id}"):
            set_query_param("pick_region", dataset_id)
            st.rerun()

    # Xenium: Pick Region, Run Proseg
    elif process_id == "xenium":
        html.cirro_analysis_button("Run Segmentation (Proseg)", dataset_id, "proseg-resegment-1-0")
        if st.button("Pick Region", key=f"pick-regions-{dataset_id}"):
            set_query_param("pick_region", dataset_id)
            st.rerun()

    # Visium: Pick Region
    elif process_id == "ingest_spaceranger":
        if st.button("Pick Region", key=f"pick-regions-{dataset_id}"):
            set_query_param("pick_region", dataset_id)
            st.rerun()

    # Region: Show Region
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


def back_button(session_key: Union[str, List[str]], label="Back", key=None):
    if st.button(label, key=f"back-{session_key}" if key is None else key):
        if isinstance(session_key, list):
            for key in session_key:
                logger.info("Clearing params from back_button")
                clear_query_param(key)
        else:
            clear_query_param(session_key)
        st.rerun()


def refresh_button(label="Refresh", key="refresh-button"):
    if st.button(label, key=key):
        # Update the refresh time
        st.session_state["refresh_time"] = time()
        st.rerun()


def pick_region(project: DataPortalProject):
    # Get the catalog
    catalog = get_catalog_cached(project.id)
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
            back_button("pick_region", label="Back to Dataset", key="back-to-dataset-2")
            return
    if points is None:
        back_button("pick_region", label="Back to Dataset", key="back-to-dataset-1")
        return

    # Get the size of the plot to show
    width, height = _calc_plot_size(points)

    # Either let the user manually select a region, or automatically pick TMA cores
    selection_mode = st.selectbox(
        label="Region Selection",
        options=["Manual", "Automatic TMA Core Selection"],
        index=0
    )
    if selection_mode == "Manual":
        select_region_manually(points, width, height)
    else:
        select_tma_cores(points, width, height)

    back_button("pick_region", label="Back to Dataset", key="back-to-dataset-0")


def select_tma_cores(points: SpatialPoints, width: int, height: int):

    # Let the user manually rotate the plot
    angle = st.number_input(
        label="Rotate Dataset",
        step=1,
        value=0
    )
    min_prop_cells = st.number_input(
        "Minimum Fraction of Cells per TMA (%)",
        value=0.1,
        max_value=100.,
        min_value=0.,
        step=0.1
    ) / 100.
    with st.spinner("Finding TMA Cores"):
        cores = find_tma_cores(points, angle, min_prop_cells=min_prop_cells)

    # Let the user pick the naming scheme
    core_naming_scheme = st.selectbox(
        "TMA Core Naming Scheme:",
        options=["Row=Letter; Column=Number", "Column=Letter; Row=Number"],
        index=0
    )
    # Let the user flip the order of the rows and columns
    row_start = st.selectbox("Rows Start From:", options=["Top", "Bottom"], index=0)
    col_start = st.selectbox("Columns Start From:", options=["Left", "Right"], index=0)

    # Name the row and column
    cores = name_tma_cores(cores, core_naming_scheme, row_start, col_start)

    # Plot the points and the identified cores
    with st.spinner("Loading plot..."):
        # Set up the scatter plot
        fig = points.plotly_scatter(
            width=width,
            height=height,
            opacity=st.number_input("Opacity", value=0.05, min_value=0.001, max_value=1., step=0.01)
        )
        # Add a circle for every core
        for _, core in cores.iterrows():
            fig.add_shape(
                type="circle",
                xref="x",
                yref="y",
                x0=core['x'] - core['radius'],
                x1=core['x'] + core['radius'],
                y0=core['y'] - core['radius'],
                y1=core['y'] + core['radius'],
                label=dict(
                    text=core["name"],
                    xanchor="center",
                    yanchor="middle",
                    font=dict(color="black")
                )
            )

        # Optionally invert the axes
        if st.checkbox("Invert X-axis", value=False):
            fig.update_xaxes(autorange='reversed')
        if st.checkbox("Invert Y-axis", value=False):
            fig.update_yaxes(autorange='reversed')
        # Display the plot
        st.plotly_chart(fig, use_container_width=False)

    name = st.text_input("Name of TMA")

    if name:
        if st.button(f"Save Region - {name}"):
            with st.spinner("Saving region..."):
                try:
                    ds = save_tma_cores(points, name, cores)
                except Exception as e:
                    st.exception(e)
                    back_button("pick_region", label="Back to Dataset", key="back-to-dataset-3")
                    return
                if ds is not None:
                    st.session_state["refresh_time"] = time()
                    back_button("pick_region", label="Back to Dataset", key="back-to-dataset-4")
                    return
            sleep(1)
    else:
        st.warning("Enter a name for the TMA")


def select_region_manually(points: SpatialPoints, width: int, height: int):

    with st.spinner("Loading plot..."):
        # Set up the scatter plot
        fig = points.plotly_scatter(
            width=width,
            height=height,
            opacity=st.number_input("Opacity", value=0.05, min_value=0.001, max_value=1., step=0.01)
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
                    back_button("pick_region", label="Back to Dataset", key="back-to-dataset-5")
                    return
                if ds is not None:
                    st.session_state["refresh_time"] = time()
                    back_button("pick_region", label="Back to Dataset", key="back-to-dataset-6")
                    return
            sleep(1)
    else:
        st.warning("Enter a name for the region")


def show_region(project: DataPortalProject):
    # Get the catalog
    catalog = get_catalog_cached(project.id)

    # Get the region ID
    region_id = get_query_param("show_region")

    # Get the region
    region: Union[SpatialRegion, List[SpatialRegion]] = catalog.regions[region_id]
    logger.info(region[0].dataset.cirro_source.dataset)

    cirro_source = (
        region.dataset.cirro_source
        if isinstance(region, SpatialRegion)
        else
        region[0].dataset.cirro_source
    )

    # Get the coordinates of points for this dataset
    with st.spinner("Loading points..."):
        try:
            points: SpatialPoints = catalog.get_points(
                cirro_source.dataset,
                path=cirro_source.path
            )
        except Exception as e:
            st.exception(e)
            back_button("show_region", label="Back to Dataset")
            return
    if points is None:
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

    # Add the region outline(s)
    if isinstance(region, SpatialRegion):
        for shape in region.outline:
            fig.add_trace(
                px.line(
                    x=shape["x"] + [shape["x"][0]],
                    y=shape["y"] + [shape["y"][0]],
                    color_discrete_sequence=["black"]
                ).data[0]
            )
    else:
        for _region in region:
            for shape in _region.outline:
                fig.add_trace(
                    px.line(
                        x=shape["x"] + [shape["x"][0]],
                        y=shape["y"] + [shape["y"][0]],
                        color_discrete_sequence=["black"]
                    ).data[0]
                )
                fig.add_annotation(
                    x=np.mean(shape["x"]),
                    y=np.mean(shape["y"]),
                    text=_region.region_id,
                    showarrow=False,
                    font=dict(color="black")
                )

    # Show the figure
    st.plotly_chart(fig, use_container_width=False)

    back_button("show_region", label="Back to Dataset")
