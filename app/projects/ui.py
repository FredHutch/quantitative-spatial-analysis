import streamlit as st
from time import time
from app.cirro import select_project
from app.streamlit import clear_query_param


def main():

    st.write("#### Spatial Data Catalog")

    # Make sure to clear the project-dependent query params
    clear_query_param("dataset")
    clear_query_param("pick_region")
    clear_query_param("show_region")

    # Show the project selection menu
    select_project()

    # Refresh only when the user clicks the button
    st.button(
        "Refresh",
        on_click=update_refresh_time,
    )


def update_refresh_time():
    st.session_state["refresh_time"] = time()


main()
