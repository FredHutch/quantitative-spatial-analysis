from dataclasses import asdict
import os
from pathlib import Path
from app.models.points import CirroDataset, SpatialDataset, SpatialPoints, SpatialRegion
from app.streamlit import get_query_param, set_query_param, clear_query_param
import json
from tempfile import TemporaryDirectory
from time import sleep, time
from typing import Iterable, Optional, List, Union
from cirro import DataPortal, DataPortalProject
from cirro import DataPortalDataset
import streamlit as st
import logging
import pandas as pd
import math
from functools import lru_cache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def pick_dataset(
    project: DataPortalProject,
    df: pd.DataFrame,
    column_order: List[str],
    column_config: dict,
    header_text: str,
    clear_params=[]
):
    st.write(header_text)
    selection = st.data_editor(
        df.assign(selected=False),
        use_container_width=True,
        hide_index=True,
        column_order=["selected"] + column_order,
        column_config={
            "selected": st.column_config.CheckboxColumn(label="☑️"),
            **column_config
        }
    )
    # If a dataset was selected
    if len(selection.query("selected")["id"].values) > 0:
        # Get the selected id
        selected_id = selection.query("selected")["id"].values[0]

        # Set the query parameter
        with st.spinner("Loading..."):
            set_query_param("dataset", selected_id)

        for param in clear_params:
            logger.info("Clearing params from pick_dataset")
            clear_query_param(param)

        # Return the project
        return project.get_dataset_by_id(selected_id)


def get_project() -> Optional[DataPortalProject]:
    """
    If a project ID is present in the session state, return the project.
    """
    project_id = get_query_param("project")
    if project_id is None:
        return
    portal = st.session_state.get("data_portal")
    if portal is None:
        return
    return portal.get_project_by_id(project_id)


def list_datasets() -> Optional[List[DataPortalDataset]]:
    """
    Return the list of datasets which can be loaded from Cirro.
    """

    # Get the project selected by the user
    project: Optional[DataPortalProject] = get_project()

    # If no project was selected, or we are not logged in, stop here
    if not project:
        return

    # Return the list of datasets
    logger.info(f"Getting datasets for project: {project.name}")
    return project.list_datasets()


def select_project() -> Optional[DataPortalProject]:
    """
    Return the project selected by the user
    """

    # Get the Cirro client
    portal: DataPortal = st.session_state.get("data_portal")
    # If we are not logged in, stop here
    if not portal:
        return

    # Try to get the list of projects
    try:
        with st.spinner("Loading projects..."):
            projects = portal.list_projects()
    # If there is an error
    except Exception as e:
        # Report it to the user and stop here
        st.exception(e)
        return

    # Make a list of the available projects
    project_list = [
        f"{p.name} - {p.id}"
        for p in projects
    ]
    # Sort the projects by name
    project_list.sort()

    # Check and see if there is one preselected
    if get_query_param("project") is None:
        index = None
    else:
        project_id = get_query_param("project")
        index = None
        for i, v in enumerate(project_list):
            if v.endswith(project_id):
                index = i
                break

    # Let the user pick one
    project = st.selectbox(
        label="Select a project",
        options=project_list,
        index=index
    )

    # If they pick one
    if project is not None:
        # Get the ID
        project_id = project.rsplit(" - ", 1)[1]
        # Set the ID
        set_query_param("project", project_id)

        # Return the project
        return portal.get_project_by_id(project_id)


def cirro_dataset_link(dataset_id: str) -> str:
    """Return the URL of the dataset in Cirro."""

    return f"{cirro_project_link()}/dataset/{dataset_id}"


def cirro_project_link() -> str:
    """Return the URL of the project in Cirro."""

    # Get the Cirro domain
    domain = get_query_param("domain")
    # If we are not logged in, stop here
    if not domain:
        raise ValueError("Not logged in")

    # Get the project selected by the user
    project = get_project()

    if project is None:
        raise ValueError("No project selected")
    return f"https://{domain}/project/{project.id}"


def cirro_analysis_link(dataset_id: str, analysis_id: str) -> str:
    """Return a URL for the page to run an analysis in Cirro."""

    if dataset_id is not None:

        return f"{cirro_dataset_link(dataset_id)}/pipeline/{analysis_id}"

    else:

        return f"{cirro_project_link()}/pipeline/{analysis_id}"
    

def _get_project_safe() -> DataPortalProject:
    """Raise an error if a project is not selected, otherwise return the selected Cirro Project."""
    
    # Get the Cirro client
    portal: DataPortal = st.session_state.get("data_portal")
    # If we are not logged in, stop here
    if not portal:
        raise ValueError("Not logged in")

    # Get the project selected by the user
    project = get_project()
    if project is None:
        raise ValueError("No project selected")

    return project


def save_tma_cores(points: SpatialPoints, name: str, cores: pd.DataFrame):

    # Make a single object that has all of the regions
    regions = [
        region
        for region in _tma_cores_to_spatial_region(cores, dataset=points.dataset)
    ]

    save_region_json(
        regions,
        name,
        source_dataset=points.dataset.cirro_source.dataset
    )


def _tma_cores_to_spatial_region(cores: pd.DataFrame, dataset: CirroDataset) -> Iterable[SpatialRegion]:

    for _, core in cores.iterrows():

        yield SpatialRegion(
            outline=[
                dict(
                    xref="x",
                    yref="y",
                    x=core["shape"][:, 0].tolist(),
                    y=core["shape"][:, 1].tolist(),
                )
            ],
            dataset=dataset,
            region_id=core["name"]
        )


def save_region(
    points: SpatialPoints,
    region_id: str,
    outline: dict
) -> DataPortalDataset:

    # Format the region information
    # Note that the region_id will be saved as the Cirro dataset name,
    # which is editable while the files within the dataset are not.
    region = SpatialRegion(
        outline=outline,
        dataset=points.dataset
    )

    save_region_json(
        region,
        region_id,
        source_dataset=points.dataset.cirro_source.dataset
    )


def save_region_json(
    region: Union[SpatialRegion, List[SpatialRegion]],
    region_id: str,
    source_dataset: str
) -> DataPortalDataset:
    """
    Save a region to Cirro.
    """

    # Get Cirro information
    project = _get_project_safe()

    # Make a description of the dataset in Cirro which contains the region
    dataset_name = project.get_dataset_by_id(source_dataset).name
    description = f"{dataset_name} - {region_id}"

    # Format the JSON string to save
    json_str = (
        json.dumps(asdict(region), indent=4)
        if isinstance(region, SpatialRegion) else
        json.dumps([asdict(r) for r in region], indent=4)
    )

    # Write out the dataset to a temporary file
    # and upload it to Cirro
    with TemporaryDirectory() as tmp:

        # Write the MuData object to the file
        with open(f"{tmp}/region.json", "w") as handle:
            handle.write(json_str)

        # Upload the file to Cirro
        try:
            ds = project.upload_dataset(
                name=region_id,
                description=description,
                process="spatial_region_json",
                upload_folder=tmp
            )
        except Exception as e:
            st.exception(e)
            sleep(10)
            return

    st.write(f"Saved region: {region_id}")
    logger.info(f"Saved region: {region_id}")

    return ds


def parse_region(
    dataset: DataPortalDataset,
    parse_retry_interval=0.1,
    parse_retry_timeout=10
) -> Union[SpatialRegion, List[SpatialDataset]]:
    """
    Read region information from a dataset
    """
    parse_retry_timer = time() + parse_retry_timeout
    while time() < parse_retry_timer:
        region_json = dataset.list_files().filter_by_pattern("data/region.json")
        if region_json:
            break
        sleep(parse_retry_interval)

    if len(region_json) == 0:
        raise ValueError(f"No region.json file found in {dataset.name}")

    region = json.loads(region_json[0].read())

    if isinstance(region, list):
        return [
            parse_region_from_dict(r, r['region_id'])
            for r in region
        ]
    else:
        return parse_region_from_dict(region, dataset.name)


def parse_region_from_dict(region: dict, region_id: str):
    return SpatialRegion(
        outline=region["outline"],
        region_id=region_id,
        dataset=SpatialDataset(
            type=region["dataset"]["type"],
            uri=region["dataset"]["uri"],
            cirro_source=CirroDataset(
                domain=st.session_state["domain"],
                project=region["dataset"]["cirro_source"]["project"],
                dataset=region["dataset"]["cirro_source"]["dataset"],
                path=region["dataset"]["cirro_source"]["path"]
            )
        )
    )
