#!/usr/bin/env streamlit run
from time import time
import streamlit as st
from app.login.ui import main as login
from app.cirro import select_project
from app.datasets.ui import main as show_dataset
from app.analyses.ui import main as show_analysis
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def run():

    # Set a default refresh time for reading the data catalog
    if st.session_state.get("refresh_time") is None:
        st.session_state["refresh_time"] = time()

    st.set_page_config("Quantitative Spatial Analysis", page_icon="🌌", layout="wide")

    st.write("#### Spatial Data Catalog")

    # If we are logged in, we can access the datasets and analyses pages
    if st.session_state.get("data_portal") is None:
        login()
    else:
        # Get the project containing the data of interest
        project = select_project()

        # If there is no project, stop
        if project is None:
            return

        # Ask the user if they want to inspect a primary dataset or an analysis
        dataset_type = st.selectbox(
            "Dataset Type:",
            options=["Primary Dataset", "Combined Analysis"]
        )

        if dataset_type == "Primary Dataset":
            logger.info("Showing primary datasets")
            show_dataset(project)
        else:
            logger.info("Showing combined analyses")
            show_analysis(project)


def update_refresh_time():
    st.session_state["refresh_time"] = time()


if __name__ == "__main__":
    run()
