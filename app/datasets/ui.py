from time import sleep
from typing import Tuple
import pandas as pd
import streamlit as st
from app.datasets.data import get_catalog, SpatialDataCatalog
from app.cirro import save_region
from app import html
import plotly.express as px

from app.models.points import SpatialPoints, SpatialRegion


def main():

    st.write("#### Spatial Data Catalog")

    # Get the data catalog
    with st.spinner(f"Loading catalog..."):
        catalog = get_catalog(
            st.session_state.get("refresh_time"),
            (
                None
                if st.session_state.get("project") is None
                else st.session_state["project"].name
            )
        )

    # If the user has selected a dataset to pick regions for, show that interface
    if st.session_state.get("pick-region") is not None:
        pick_region(catalog, st.session_state["pick-region"])
        return
    
    # If the user has selected a region to display, show that region
    if st.session_state.get("show-region") is not None:
        show_region(catalog, st.session_state["show-region"])
        return

    # If there are no datasets available
    if catalog.df is None:
        st.write("Select a data collection from the menu")
        return
    elif catalog.df.shape[0] == 0:
        st.write("Data collection does not contain any recognized spatial datasets")
        return

    # Show the table of contents
    st.write("Select a dataset to view its contents")
    selection = st.data_editor(
        catalog.df.assign(selected=False),
        use_container_width=True,
        hide_index=True,
        column_order=["selected", "Name", "Created", "Type", "Analysis Outputs"],
        column_config={
            "selected": st.column_config.CheckboxColumn(label="☑️"),
            "Name": st.column_config.TextColumn(width="medium", disabled=True),
            "Created": st.column_config.TextColumn(max_chars=14, disabled=True),
            "Type": st.column_config.TextColumn(width="small", disabled=True),
            "Analysis Outputs": st.column_config.ListColumn()
        }
    )

    # Show any selected datasets
    for ingest_id in selection.query("selected")["id"].values:
        st.write("---")
        show_dataset(catalog, ingest_id)


def show_dataset(catalog: SpatialDataCatalog, ingest_id: str):

    # Show the ingest dataset
    with st.container(key=ingest_id):
        html.card_content(
            title=catalog.datasets[ingest_id].name.strip(),
            content=[
                catalog.datasets[ingest_id].description,
                f"<b>Created</b>: {catalog.datasets[ingest_id].created_at.strftime('%Y-%m-%d %H:%M')}",
                f"<b>Type</b>: {catalog.dataset_types[ingest_id]}"
            ]
        )
        dataset_buttons(catalog, ingest_id)
    html.card_style(ingest_id)

    # Make a list of the analysis outputs derived from this dataset
    analysis_outputs = catalog.groups[ingest_id]

    # Ignore the ingest dataset itself
    analysis_outputs = [dataset_id for dataset_id in analysis_outputs if dataset_id != ingest_id]

    # If there are none, stop
    if len(analysis_outputs) == 0:
        return
    
    # Group the analysis outputs by type
    analysis_output_types = [catalog.dataset_types[dataset_id] for dataset_id in analysis_outputs]
    ix = 0
    for dataset_type, dataset_list in pd.Series(analysis_outputs).groupby(analysis_output_types):
        
        # Make a card for each analysis type
        card_key = f"{ingest_id}-{ix}"
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
            st.session_state["pick-region"] = dataset_id
            st.rerun()

    # Xenium: Pick Region, Run Proseg
    elif process_id == "xenium":
        html.cirro_analysis_button("Run Segmentation (Proseg)", dataset_id, "proseg-resegment-1-0")
        if st.button("Pick Region", key=f"pick-regions-{dataset_id}"):
            st.session_state["pick-region"] = dataset_id
            st.rerun()

    elif process_id == "region":
        if st.button("Show Region", key=f"show-region-{dataset_id}"):
            st.session_state["show-region"] = dataset_id
            st.rerun()


def _calc_plot_size(points: SpatialPoints) -> Tuple[int, int]:
    # Use the dimensions to make the coordinates square
    x_range = points.coords[points.xcol].max() - points.coords[points.xcol].min()
    y_range = points.coords[points.ycol].max() - points.coords[points.ycol].min()

    # Let the user modify the width
    width = st.number_input("Width", value=800, min_value=100, max_value=2000, step=100)
    height = int(width * y_range / x_range)
    return width, height

def pick_region(catalog: SpatialDataCatalog, dataset_id: str):
    # Print the dataset name
    st.write(catalog.datasets[dataset_id].name)

    # Get the coordinates of points for this dataset
    with st.spinner("Loading points..."):
        points: SpatialPoints = catalog.get_points(dataset_id)

    # Get the size of the plot to show
    width, height = _calc_plot_size(points)

    st.write("Use the lasso tool to select a region of interest")

    region = st.plotly_chart(
        px.scatter(
            points.coords,
            x=points.xcol,
            y=points.ycol,
            width=width,
            height=height,
            opacity=0.5,
            color_discrete_sequence=["blue"]
        ),
        selection_mode="lasso",
        on_select="rerun"
    )
    n_points = len(region["selection"]["points"])
    st.write(f"Selected {n_points:,} points")

    if n_points == 0:
        st.warning("Select some points to define a region")

    name = st.text_input("Name of region")

    if name:
        if st.button(f"Save Region - {name}"):
            with st.spinner("Saving region..."):
                save_region(
                    points,
                    name,
                    region["selection"]["lasso"]
                )
            sleep(1)
    else:
        st.warning("Enter a name for the region")

    if st.button("Back"):
        st.session_state["pick-region"] = None
        st.rerun()


def show_region(catalog: SpatialDataCatalog, region_id: str):
    # Get the region
    region: SpatialRegion = catalog.regions[region_id]

    # Get the coordinates of points for this dataset
    with st.spinner("Loading points..."):
        points = catalog.get_points(region.dataset.cirro_source.dataset)

    # Get the size of the plot to show
    width, height = _calc_plot_size(points)

    # Set up the points
    fig = px.scatter(
        points.coords,
        x=points.xcol,
        y=points.ycol,
        width=width,
        height=height,
        opacity=0.5,
        color_discrete_sequence=["blue"]
    )

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
    st.plotly_chart(fig)

    if st.button("Back"):
        st.session_state["show-region"] = None
        st.rerun()


main()
