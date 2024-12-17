import streamlit as st
from app.analyses.data import SpatialDataAnalyses
from app.streamlit import get_query_param, clear_query_param
from app.cirro import select_project


def main():
    st.write("#### Spatial Data Analysis")

    # If there is no project selected
    if get_query_param("project") is None:
        clear_query_param("dataset")
        clear_query_param("pick_region")
        clear_query_param("show_region")
        # Show the project selection menu
        select_project()
    
    # If there is no dataset selected
    elif get_query_param("dataset") is None:
        clear_query_param("pick_region")
        clear_query_param("show_region")
        # Show the dataset selection menu
        select_dataset()

    # Get the data catalog
    catalog = SpatialDataAnalyses()
    st.write('We are on the analysis page')
    if catalog.datasets is None:
        st.write("No datasets available")
        return

    st.write(f"There are {len(catalog.datasets):,} datasets available")


main()
