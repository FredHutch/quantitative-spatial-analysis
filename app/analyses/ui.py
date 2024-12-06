import streamlit as st
from app.analyses.data import SpatialDataAnalyses


def main():
    # Get the data catalog
    catalog = SpatialDataAnalyses()
    st.write('We are on the analysis page')
    if catalog.datasets is None:
        st.write("No datasets available")
        return

    st.write(f"There are {len(catalog.datasets):,} datasets available")


main()
