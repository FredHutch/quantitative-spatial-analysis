from typing import List, Tuple
import pandas as pd
import streamlit as st
import plotly.express as px
from plotly.graph_objects import Figure
from scipy.stats import chi2_contingency


def _barmode_selector():
    return (
        "group"
        if st.segmented_control("Bar Mode", ["Stacked", "Unstacked"], default="Stacked") == "Unstacked"
        else "overlay"
    )


def _show_image_and_download_button(fig: Figure):

    st.plotly_chart(fig)

    # Make a button to download the HTML image
    st.download_button(
        "Download Plot (HTML)",
        fig.to_html(),
        file_name="plot.html",
        mime="text/html"
    )


def _format_inputs(counts: pd.DataFrame, cnames: List[str]) -> Tuple[pd.DataFrame, str]:
    # Collapse by cell clusters and regions, making sure to treat the categories as strings
    df = (
        counts
        .assign(
            **{
                cname: counts[cname].astype(str)
                for cname in cnames
            }
        )
        .pivot_table(
            index=cnames,
            values="count",
            aggfunc="sum"
        )
        .reset_index()
    )

    # Compute the different metrics
    df = df.assign(
        **{
            "Percent of Total": 100 * df["count"] / df["count"].sum()
        },
        **{
            f"Percent of {cname.title()}": 100 * df["count"] / df.groupby(cname)["count"].transform("sum")
            for cname in cnames
        }
    ).rename(
        columns=dict(count="Cell Count")
    )

    # Let the user select the metric
    metric = st.selectbox(
        "Metric",
        ["Cell Count", "Percent of Total"] + [
            f"Percent of {cname.title()}" for cname in cnames
        ]
    )

    # Let the user filter the clusters / neighborhoods / regions
    for cname in cnames:
        df = df.loc[
            df[cname].isin(
                st.multiselect(f"{cname}s".title(), df[cname].unique(), df[cname].unique())
            )
        ]

    # Title case all of the columns
    df = df.rename(columns=lambda cname: cname.title())

    return df, metric


def _chi2_test(df: pd.DataFrame, group1: str, group2: str):
    # Compute the chi-squared test
    chi2 = chi2_contingency(
        pd.crosstab(
            df[group1],
            df[group2],
            df["Cell Count"],
            aggfunc="sum"
        ).fillna(0)
    )
    formatted_pvalue = (
        f"{chi2.pvalue:.2E}"
        if chi2.pvalue < 0.01
        else
        f"{chi2.pvalue:.2f}"
    )
    st.write(
        f"""
        The p-value for the hypothesis that cells are distributed independently
        across the {group1} and {group2} categories is {formatted_pvalue}
        (chi-squared contingency test).
        """
    )


def cell_clusters_across_regions(counts: pd.DataFrame):
    df, metric = _format_inputs(counts, ["region", "cluster"])

    st.dataframe(df)

    # Plot the data
    fig = px.bar(
        df,
        x="Region",
        y=metric,
        color="Cluster",
        title="Cell Clusters Across Regions",
        color_discrete_sequence=px.colors.qualitative.D3,
        barmode=_barmode_selector()
    )

    _show_image_and_download_button(fig)

    _chi2_test(df, "Region", "Cluster")


def neighborhoods_across_regions(counts: pd.DataFrame):
    df, metric = _format_inputs(counts, ["region", "neighborhood"])

    st.dataframe(df)

    # Plot the data
    fig = px.bar(
        df,
        x="Region",
        y=metric,
        color="Neighborhood",
        title="Cell Neighborhoods Across Regions",
        color_discrete_sequence=px.colors.qualitative.D3,
        barmode=_barmode_selector()
    )

    _show_image_and_download_button(fig)

    _chi2_test(df, "Region", "Neighborhood")


def cell_clusters_across_neighborhoods(counts: pd.DataFrame):
    df, metric = _format_inputs(counts, ["cluster", "neighborhood"])

    st.dataframe(df)

    # Plot the data
    fig = px.bar(
        df,
        x="Neighborhood",
        y=metric,
        color="Cluster",
        title="Cell Clusters Across Neighborhoods",
        color_discrete_sequence=px.colors.qualitative.D3,
        barmode=_barmode_selector()
    )

    _show_image_and_download_button(fig)
