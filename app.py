#!/usr/bin/env streamlit run
from time import time
import streamlit as st
from app.streamlit import get_query_param


def run():

    st.set_page_config("Quantitative Spatial Analysis", page_icon="🌌", layout="wide")

    # Set up the pages
    login = st.Page("app/login/ui.py", title="Login", url_path="login")
    projects = st.Page("app/projects/ui.py", title="Projects", url_path="projects")
    datasets = st.Page("app/datasets/ui.py", title="Datasets", url_path="datasets")
    analyses = st.Page("app/analyses/ui.py", title="Analyses", url_path="analyses")

    # If we are logged in, we can access the datasets and analyses pages
    if st.session_state.get("data_portal") is None:
        pages = [login]
    elif get_query_param("project") is None:
        pages = [projects]
    else:
        pages = [datasets, analyses]

    # Build the navigation menu
    pg = st.navigation({"Quantitative Spatial Analysis": pages})

    # Set a default refresh time for reading the data catalog
    if st.session_state.get("refresh_time") is None:
        st.session_state["refresh_time"] = time()

    # Run the app
    pg.run()


def update_refresh_time():
    st.session_state["refresh_time"] = time()


if __name__ == "__main__":
    run()
