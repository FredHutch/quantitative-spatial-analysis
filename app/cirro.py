from dataclasses import asdict
from app.models.points import CirroDataset, SpatialDataset, SpatialPoints, SpatialRegion
from app.streamlit import get_query_param, set_query_param, clear_query_param
import json
from tempfile import TemporaryDirectory
from time import sleep, time
from typing import Optional, List
from cirro import DataPortal, DataPortalProject
from cirro import DataPortalDataset
import streamlit as st
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def show_menu(
    query_key: str,
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
            set_query_param(query_key, selected_id)

        for param in clear_params:
            clear_query_param(param)

        st.rerun()


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

    # Make a DataFrame with the project info
    project_df = pd.DataFrame(
        [
            {
                "name": p.name,
                "id": p.id,
                "description": p.description
            }
            for p in projects
        ]
    )
    # Sort the projects by name
    project_df.sort_values("name", inplace=True)

    # Show a table with the projects that the user can select
    show_menu(
        query_key="project",
        df=project_df,
        column_order=["name", "description", "id"],
        column_config={
            "description": {"maxWidth": 300}
        },
        header_text="Select a project"
    )


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


def save_region(
    points: SpatialPoints,
    region_id: str,
    outline: dict
) -> DataPortalDataset:
    """
    Save a region to Cirro.
    """

    # Get the Cirro client
    portal: DataPortal = st.session_state.get("data_portal")
    # If we are not logged in, stop here
    if not portal:
        raise ValueError("Not logged in")

    # Get the project selected by the user
    project = get_project()
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
) -> Optional[SpatialRegion]:
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
    except: # noqa
        return
