from dataclasses import asdict
from app.models.points import CirroDataset, SpatialDataset, SpatialPoints, SpatialRegion
import json
from tempfile import TemporaryDirectory
from time import sleep
from typing import Optional, List
from cirro import DataPortal, DataPortalProject
from cirro import DataPortalDataset
import streamlit as st
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def list_datasets() -> Optional[List[DataPortalDataset]]:
    """
    Return the list of datasets which can be loaded from Cirro.
    """

    # Get the project selected by the user
    project: Optional[DataPortalProject] = st.session_state.get("project")

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

    # Sort the projects by name
    projects.sort(key=lambda p: p.name)
    
    # Give the user the option to select a project
    project = st.sidebar.selectbox(
        "Select data collection (or project)",
        [p.name for p in projects],
        placeholder="< select a project >",
        index=None
    )

    # If no project was selected, stop here
    if project is None:
        return

    # Return the project object
    for p in projects:
        if p.name == project:

            # Return the project object
            return p

    raise ValueError(f"Project '{project}' not found")


def cirro_dataset_link(dataset_id: str) -> str:
    """Return the URL of the dataset in Cirro."""

    # Get the Cirro domain
    domain = st.session_state.get("domain")
    # If we are not logged in, stop here
    if not domain:
        raise ValueError("Not logged in")

    # Get the project selected by the user
    project: Optional[DataPortalProject] = st.session_state.get("project")

    if project is None:
        raise ValueError("No project selected")
    return f"https://{domain}/project/{project.id}/dataset/{dataset_id}"


def cirro_analysis_link(dataset_id: str, analysis_id: str) -> str:
    """Return a URL for the page to run an analysis in Cirro."""

    return f"{cirro_dataset_link(dataset_id)}/pipeline/{analysis_id}"


def save_region(points: SpatialPoints, region_id: str, outline: dict):
    """
    Save a region to Cirro.
    """

    # Get the Cirro client
    portal: DataPortal = st.session_state.get("data_portal")
    # If we are not logged in, stop here
    if not portal:
        raise ValueError("Not logged in")
    
    # Get the project selected by the user
    project: Optional[DataPortalProject] = st.session_state.get("project")
    if project is None:
        raise ValueError("No project selected")

    # Format the region information
    # Note that the region_id will be saved as the Cirro dataset name,
    # which is editable while the files within the dataset are not.
    region = SpatialRegion(
        outline=outline,
        dataset=points.dataset
    )

    # Make a description of the dataset in Cirro which contains the region
    dataset_name = (
        project
        .get_dataset_by_id(points.dataset.cirro_source.dataset)
        .name
    )
    description = f"{dataset_name} - {region_id}"

    # Write out the dataset to a temporary file
    # and upload it to Cirro
    with TemporaryDirectory() as tmp:

        # Write the MuData object to the file
        with open(f"{tmp}/region.json", "w") as handle:
            json.dump(
                asdict(region),
                handle,
                indent=4
            )

        # Upload the file to Cirro
        try:
            project.upload_dataset(
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


def parse_region(dataset: DataPortalDataset) -> Optional[SpatialRegion]:
    """
    Read region information from a dataset
    """
    region_json = dataset.list_files().filter_by_pattern("data/region.json")
    if len(region_json) == 0:
        raise ValueError(f"No region.json file found in {dataset.name}")

    region = json.loads(region_json[0].read())

    try:
        return SpatialRegion(
            outline=region["outline"],
            region_id=dataset.name,
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
    except Exception as _:
        return
