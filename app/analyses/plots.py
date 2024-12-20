from typing import List, Tuple
import pandas as pd
import streamlit as st
import plotly.express as px


def format_inputs(counts: pd.DataFrame, cnames: List[str]) -> Tuple[pd.DataFrame, str]:
    # Collapse by cell clusters and regions
    df = counts.pivot_table(
        index=cnames,
        values="count",
        aggfunc="sum"
    ).reset_index()

    # Let the user select the metric
    metric = st.selectbox(
        "Metric",
        ["Cell Count", "Percent of Total"] + [
            f"Percent of {cname.title()}" for cname in cnames
        ]
    )
    if metric == "Cell Count":
        pass
    elif metric == "Percent of Total":
        df = df.assign(
            count = 100 * df["count"] / df["count"].sum()
        )
    else:
        norm_cname = metric.split("Percent of ")[1].lower()
        df = df.assign(
            count = 100 * df["count"] / df.groupby(norm_cname)["count"].transform("sum")
        )

    # Let the user filter the clusters / neighborhoods / regions
    for cname in cnames:
        df = df.loc[
            df[cname].isin(
                st.multiselect(f"{cname}s".title(), df[cname].unique(), df[cname].unique())
            )
        ]

    return df, metric


def cell_clusters_across_regions(counts: pd.DataFrame):
    df, metric = format_inputs(counts, ["region", "cluster"])

    st.dataframe(df)

    # Plot the data
    fig = px.bar(
        df.assign(cluster=df["cluster"].astype(str)),
        x="region",
        y="count",
        color="cluster",
        title="Cell Clusters Across Regions",
        labels={"count": metric, "cluster": "Cluster", "region": "Region"},
        color_discrete_sequence=px.colors.qualitative.D3
    )

    st.plotly_chart(fig)


def neighborhoods_across_regions(counts: pd.DataFrame):
    pass


def cell_clusters_across_neighborhoods(counts: pd.DataFrame):
    pass


def cell_clusters_across_regions_and_neighborhoods(counts: pd.DataFrame):
    pass
