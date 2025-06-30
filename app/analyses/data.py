
from app.cirro import list_datasets

from typing import Dict
from cirro.sdk.dataset import DataPortalDataset
import pandas as pd
import streamlit as st
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


class SpatialAnalysisCatalog:

    analysis_type = ["process-hutch-quantitative-spatial-analysis-1_0"]
    region_type = ["spatial_region_json"]

    # Catalog of all datasets which match the filters
    datasets: Dict[str, DataPortalDataset]

    # Dataframe of the regions
    regions: pd.DataFrame

    # DataFrame of the analyses
    analyses: pd.DataFrame

    def __init__(self):
        # Get all datasets of the expected types, and group them by the original ingest dataset
        self.datasets = {}

        # Get the list of all datasets
        datasets = list_datasets()

        # Initialize the dataframes
        self.regions = None
        self.analyses = None

        # If no project was selected, stop here
        if not datasets:
            return

        # Build a table with information about the datasets
        df = pd.DataFrame([
            row
            for dataset in datasets
            for row in [self.parse_dataset(dataset)]
            if row is not None
        ])

        if df.shape[0]:
            df.sort_values("Created", ascending=False, inplace=True)
            self.analyses = df.query("Type == 'Analysis'")
            self.regions = df.query("Type == 'Region'")

    def parse_dataset(self, dataset: DataPortalDataset) -> Dict[str, str]:

        # Check to see what type of dataset this is
        if dataset.process_id in self.analysis_type:
            dataset_type = "Analysis"
        elif dataset.process_id in self.region_type:
            dataset_type = "Region"
        else:
            return
        
        self.datasets[dataset.id] = dataset
        
        # Format the viewable information about the dataset
        return {
            "Name": dataset.name,
            "Description": dataset.description,
            "Created": dataset.created_at.strftime("%Y-%m-%d %H:%M"),
            "Type": dataset_type,
            "id": dataset.id
        }


@st.cache_resource
def get_catalog(
    refresh_time: float,
    selected_project: str
) -> SpatialAnalysisCatalog:
    logger.info("----------------------------------------------")
    logger.info(f"Refreshing the data catalog ({refresh_time})")
    logger.info(f"Selected project: {selected_project}")
    catalog = SpatialAnalysisCatalog()
    if catalog.analyses is None:
        n = 0
    else:
        n = catalog.analyses.shape[0]
    logger.info(f"Project has {n} analyses")
    logger.info("----------------------------------------------")
    return catalog
