from pathlib import Path
from time import sleep

from cirro import DataPortalProject
from app.cirro import get_project, list_datasets, parse_region
from app.models.points import SpatialPoints, SpatialDataset, CirroDataset

from functools import lru_cache
from io import BytesIO
from typing import Dict, Generator, Union
from cirro.sdk.dataset import DataPortalDataset
import pandas as pd
from collections import defaultdict
from typing import TypedDict, List
import streamlit as st
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def _get_ingest_ids(
    dataset: Union[DataPortalDataset, str]
) -> Generator[str, str, str]:
    if isinstance(dataset, str):
        project = get_project()
        dataset = project.get_dataset_by_id(dataset)
    if dataset.process.executor == "INGEST":
        yield dataset.id
    else:
        for parent_dataset in dataset.source_datasets:
            yield from _get_ingest_ids(parent_dataset)


class DatasetGroup(TypedDict):
    ingest_id: str
    ingest_type: str
    datasets: List[DataPortalDataset]
    created_at: str
    name: str
    description: str
    tags: List[str]


class SpatialDataCatalog:

    filter_types = [
        "images",
        "xenium",
        "process-hutch-qupath-stardist-1_0",
        "process-hutch-cellpose-1_0"
    ]
    # Mapping of ingest dataset IDs to the datasets that were derived from them
    groups: Dict[str, List[str]]

    # Catalog of all datasets which match the filters
    datasets: Dict[str, DataPortalDataset]

    # Mapping of dataset IDs to readable dataset types
    dataset_types: Dict[str, str]

    # Region information, keyed by dataset ID for each region
    regions: Dict[str, Dict[str, str]]

    # Dataframe of the catalog
    df: pd.DataFrame

    def __init__(self):
        # Get all datasets of the expected types, and group them by the original ingest dataset
        self.groups = defaultdict(list)
        self.datasets = {}
        self.dataset_types = {}
        self.regions = {}

        # Get the list of all datasets
        datasets = list_datasets()

        # If no project was selected, stop here
        if not datasets:
            self.df = None
            return

        # Loop over every dataset
        for dataset in datasets:

            self.add_dataset(dataset)

        # Make a table of the datasets
        self. df = pd.DataFrame([
            {
                "Name": self.datasets[ingest_id].name,
                "Created": self.datasets[ingest_id].created_at.strftime("%Y-%m-%d %H:%M"),
                "Type": self.dataset_types[ingest_id],
                "Analysis Outputs": [
                    f"{dataset_type}: {count:,}"
                    for dataset_type, count in pd.Series([
                        self.dataset_types[dataset_id]
                        for dataset_id in dataset_group
                    ]).value_counts().items()
                ],
                "id": ingest_id
            }
            for ingest_id, dataset_group in self.groups.items()
        ])
        if self.df.shape[0]:
            self.df.sort_values("Created", ascending=False, inplace=True)

    def add_dataset(self, dataset: DataPortalDataset):

        # Parse region datasets in a particular way
        if dataset.process_id == "spatial_region_json":
            region = parse_region(dataset)

            self.datasets[dataset.id] = dataset
            self.dataset_types[dataset.id] = "Region"
            for ingest_id in _get_ingest_ids(region.dataset.cirro_source.dataset):
                self.groups[ingest_id].append(dataset.id)
            self.regions[dataset.id] = region

        # Skip datasets that are not of the expected types
        elif dataset.process_id not in self.filter_types:
            return

        # Skip datasets that have failed analysis
        elif dataset.status == "FAILED":
            return

        else:

            self.datasets[dataset.id] = dataset
            self.dataset_types[dataset.id] = dataset.process.name
            for ingest_id in _get_ingest_ids(dataset):
                self.groups[ingest_id].append(dataset.id)

    @lru_cache
    def process_id(_self, dataset_id):
        if dataset_id in _self.regions:
            return "region"
        else:
            return _self.datasets[dataset_id].process_id

    def get_points(self, dataset_id: str) -> SpatialPoints:
        """
        Get the spatial coordinates of the points in the dataset
        """
        process_id = self.process_id(dataset_id)

        if process_id == "images":
            raise ValueError("Images do not have points")
        elif process_id == "xenium":
            return self.get_points_xenium(dataset_id)
        elif process_id in [
            "process-hutch-qupath-stardist-1_0",
            "process-hutch-cellpose-1_0"
        ]:
            return self.get_points_stardist(dataset_id)
        
    def get_points_xenium(self, dataset_id: str) -> SpatialPoints:

        # Get the Dataset object
        ds = self.datasets[dataset_id]

        # List the files in the dataset which may contain points
        files = ds.list_files()
        files = files.filter_by_pattern("*/cells.parquet")

        # Only keep the file names
        files = [file.name for file in files]

        # If there is more than one, ask the user which one to select
        if len(files) > 1:
            file = st.selectbox("Select spatial coordinates file", files)
        else:
            file = files[0]

        # Read the file
        coords = self.read_parquet(dataset_id, file)

        for cname in ["cell_id", "x_centroid", "y_centroid", "transcript_counts"]:
            if cname not in coords.columns:
                raise ValueError(f"Missing column {cname} in {file}")
        coords = coords.set_index("cell_id").reindex(columns=["x_centroid", "y_centroid", "transcript_counts"])

        # Get the folder which contains the complete Xenium dataset (within the Cirro dataset)
        folder = str(Path(file).parent)

        # Construct the URI for the folder
        folder_uri = str(Path(ds._get_detail().s3) / folder)

        return SpatialPoints(
            coords=coords,
            xcol="x_centroid",
            ycol="y_centroid",
            meta_cols=["transcript_counts"],
            dataset=SpatialDataset(
                type="xenium",
                uri=folder_uri,
                cirro_source=CirroDataset(
                    domain=st.session_state["domain"],
                    project=ds.project_id,
                    dataset=dataset_id,
                    path=folder
                )
            )
        )
    
    def get_points_stardist(self, dataset_id: str) -> SpatialPoints:

        # Get the Dataset object
        ds = self.datasets[dataset_id]

        # Get the spatial coordinates
        coords = (
            ds
            .list_files()
            .get_by_name("data/cell_measurements/spatial.csv")
            .read_csv(index_col=0)
        )
        # Construct the URI for the folder
        folder = "data"
        folder_uri = str(Path(ds._get_detail().s3) / folder)

        return SpatialPoints(
            coords=coords,
            xcol=coords.columns.values[0],
            ycol=coords.columns.values[1],
            meta_cols=[],
            dataset=SpatialDataset(
                type="stardist",
                uri=folder_uri,
                cirro_source=CirroDataset(
                    domain=st.session_state["domain"],
                    project=ds.project_id,
                    dataset=dataset_id,
                    path=folder
                )
            )
        )

    @lru_cache
    def read_parquet(_self, dataset_id: str, file_name: str):

        ds = _self.datasets[dataset_id]
        file = ds.list_files().get_by_name(file_name)

        # Read the contents of that file
        with BytesIO(file._get()) as handle:
            coords = pd.read_parquet(handle)

        return coords


@st.cache_resource
def get_catalog(
    refresh_time: float,
    selected_project: str
) -> SpatialDataCatalog:
    logger.info("----------------------------------------------")
    logger.info(f"Refreshing the data catalog ({refresh_time})")
    logger.info(f"Selected project: {selected_project}")
    catalog = SpatialDataCatalog()
    if catalog.df is None:
        n = 0
    else:
        n = catalog.df.shape[0]
    logger.info(f"Project has {n} datasets")
    logger.info("----------------------------------------------")
    return catalog
