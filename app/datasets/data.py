import io
import json
from pathlib import Path

from app.cirro import get_project, list_datasets, parse_region
from app.models.points import SpatialPoints, SpatialDataset, CirroDataset

from functools import lru_cache
from io import BytesIO
from typing import Dict, Generator, Set, Union
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
        "ingest_spaceranger",  # Visium
        "process-hutch-qupath-stardist-1_0",
        "process-hutch-cellpose-1_0",
        "proseg-resegment-1-0"
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

    # All datasets that we have access to
    all_dataset_ids: Set[str]

    # Internal constants
    _visium_clusters_suffix = "/analysis/clustering/gene_expression_graphclust/clusters.csv"
    _visium_coordinates_suffix = "/spatial/tissue_positions.parquet"
    _visium_scalefactors_suffix = "/spatial/scalefactors_json.json"

    def __init__(self):
        # Get all datasets of the expected types, and group them by the original ingest dataset
        self.groups = defaultdict(list)
        self.datasets = {}
        self.dataset_types = {}
        self.regions = {}

        # Get the list of all datasets
        datasets = list_datasets()

        # Keep track of all datasets that we have access to
        self.all_dataset_ids = set([ds.id for ds in datasets])

        # If no project was selected, stop here
        if not datasets:
            self.df = None
            return

        # Loop over every dataset
        for dataset in datasets:

            self.add_dataset(dataset)

        # Make a table of the datasets
        self.df = pd.DataFrame([
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
            if ingest_id in self.datasets
        ])
        if self.df.shape[0]:
            self.df.sort_values("Created", ascending=False, inplace=True)

        logger.info(self.groups)

    def _get_ingest_ids(self, dataset: Union[DataPortalDataset, str]) -> Generator[str, str, str]:
        """Find the dataset furthest back in the chain of provenence that we can access."""

        if isinstance(dataset, str):
            project = get_project()
            dataset = project.get_dataset_by_id(dataset)
        if dataset.process.executor == "INGEST":
            yield dataset.id
        else:
            source_datasets = [
                parent_dataset
                for parent_dataset in dataset.source_datasets
                if parent_dataset.id in self.all_dataset_ids
            ]
            if len(source_datasets) > 0:
                for ds in source_datasets:
                    yield from self._get_ingest_ids(ds)
            else:
                yield dataset.id

    def add_dataset(self, dataset: DataPortalDataset):

        # Parse region datasets in a particular way
        if dataset.process_id == "spatial_region_json":
            region = parse_region(dataset)

            self.datasets[dataset.id] = dataset
            self.dataset_types[dataset.id] = "Region"
            logger.info("--------------------HERE0")
            for ingest_id in self._get_ingest_ids(
                (
                    region[0]
                    if isinstance(region, list)
                    else region
                ).dataset.cirro_source.dataset
            ):
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
            for ingest_id in self._get_ingest_ids(dataset):
                self.groups[ingest_id].append(dataset.id)

    @lru_cache
    def process_id(_self, dataset_id):
        if dataset_id in _self.regions:
            return "region"
        else:
            return _self.datasets[dataset_id].process_id

    def get_points(self, dataset_id: str, path=None) -> SpatialPoints:
        """
        Get the spatial coordinates of the points in the dataset
        """
        process_id = self.process_id(dataset_id)

        if process_id == "images":
            raise ValueError("Images do not have points")
        elif process_id == "xenium":
            return self.get_points_xenium(dataset_id)
        elif process_id == "ingest_spaceranger":

            # For Visium datasets, the user must select which resolution to use
            if path is not None:
                # If a path is provided, use it directly
                points_folder = path
            else:
                # Otherwise, ask the user to select the resolution
                points_folder = self.select_visium_resolution(dataset_id)
            if points_folder is None:
                return

            # Get the points for the selected resolution
            return self.get_points_visium(dataset_id, points_folder)

        elif process_id in [
            "process-hutch-qupath-stardist-1_0",
            "process-hutch-cellpose-1_0"
        ]:
            return self.get_points_stardist(dataset_id)

        elif process_id == "proseg-resegment-1-0":
            return self.get_points_proseg(dataset_id)

    def get_points_proseg(self, dataset_id: str) -> SpatialPoints:

        # Get the Dataset object
        ds = self.datasets[dataset_id]

        # List the files in the dataset which may contain points
        files = ds.list_files()

        # Read in the file with the cell coordinates
        coords = files.get_by_id("data/proseg/cell-metadata.csv.gz").read_csv()

        _expected_cnames = ["centroid_x", "centroid_y", "cluster"]
        for cname in ["cell"] + _expected_cnames:
            if cname not in coords.columns:
                raise ValueError(f"Missing column {cname} in proseg/cell-metadata.csv.gz")
        coords = coords.set_index("cell").reindex(columns=_expected_cnames)

        # Construct the URI for the folder
        folder = "data/proseg"
        folder_uri = str(Path(ds._get_detail().s3) / folder)

        return SpatialPoints(
            coords=coords,
            clusters=coords["cluster"].astype(str),
            xcol="centroid_x",
            ycol="centroid_y",
            meta_cols=[],
            dataset=SpatialDataset(
                type="proseg",
                uri=folder_uri,
                cirro_source=CirroDataset(
                    domain=st.session_state["domain"],
                    project=ds.project_id,
                    dataset=dataset_id,
                    path=folder
                )
            )
        )

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
            clusters=None,
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

    def select_visium_resolution(self, dataset_id: str) -> str:

        # Get the Dataset object
        ds = self.datasets[dataset_id]

        # The user must pick which level of binning to use
        points_options = ds.list_files().filter_by_pattern(f"*{self._visium_coordinates_suffix}")

        if len(points_options) == 0:
            raise Exception("No spatial coordinates found in the dataset")

        elif len(points_options) == 1:
            # If there is only one option, use it
            points_file = points_options[0]
            points_folder = points_file.name[:-len(self._visium_coordinates_suffix)]
        else:
            # If there are multiple options, ask the user to select one
            points_folder = st.selectbox(
                "Select spatial coordinates to use",
                [file.name[:-len(self._visium_coordinates_suffix)] for file in points_options],
                index=None
            )
            if points_folder is None:
                st.write("Please select spatial coordinates to continue.")
                return
        return points_folder

    def get_points_visium(self, dataset_id: str, points_folder: str) -> SpatialPoints:

        # Get the Dataset object
        ds = self.datasets[dataset_id]

        points_file = points_folder + self._visium_coordinates_suffix

        # Get the spatial coordinates
        logger.info(f"Reading the spatial coordinates from {points_file}")
        coords = (
            pd.read_parquet(
                io.BytesIO(
                    ds
                    .list_files()
                    .get_by_name(points_file)
                    ._get()
                )
            )
            .set_index("barcode")
            .query("in_tissue == 1")
        )

        # Get the scale factors for the Visium dataset
        scalefactors_file = points_folder + self._visium_scalefactors_suffix
        logger.info(f"Reading the scaling factors for the Visium dataset ({scalefactors_file})")
        scaling_factors = json.loads(
            ds
            .list_files()
            .get_by_name(scalefactors_file)
            .read()
        )

        # Apply the scaling factors to the coordinates
        coords = coords.assign(
            pxl_col_in_fullres=coords['pxl_col_in_fullres'] * scaling_factors['tissue_hires_scalef'],
            pxl_row_in_fullres=coords['pxl_row_in_fullres'] * scaling_factors['tissue_hires_scalef']
        )

        # Get the automated clusters
        cluster_file = points_folder + self._visium_clusters_suffix

        if cluster_file not in [_f.name for _f in ds.list_files()]:
            clusters = pd.Series(
                ["none" for _ in range(coords.shape[0])],
                index=coords.index,
                name="Cluster"
            )
        else:
            # If the clusters file exists, read it
            clusters = (
                ds
                .list_files()
                .get_by_name(cluster_file)
                .read_csv(index_col=0)
                ["Cluster"]
                .astype(str)
                .reindex(coords.index)
            )

        # Construct the URI for the folder
        folder_uri = str(Path(ds._get_detail().s3) / points_folder)

        return SpatialPoints(
            coords=coords,
            clusters=clusters,
            xcol="pxl_col_in_fullres",
            ycol="pxl_row_in_fullres",
            meta_cols=[],
            dataset=SpatialDataset(
                type="visium",
                uri=folder_uri,
                cirro_source=CirroDataset(
                    domain=st.session_state["domain"],
                    project=ds.project_id,
                    dataset=dataset_id,
                    path=points_folder
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

        # Get the automated clusters
        clusters = (
            ds
            .list_files()
            .get_by_name("data/cell_clustering/leiden_clusters.csv")
            .read_csv(index_col=0)
            ["leiden"]
            .astype(str)
        )
        # Construct the URI for the folder
        folder = "data"
        folder_uri = str(Path(ds._get_detail().s3) / folder)

        return SpatialPoints(
            coords=coords,
            clusters=clusters,
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
