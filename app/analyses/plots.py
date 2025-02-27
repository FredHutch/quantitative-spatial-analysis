from typing import List, Tuple
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import chi2_contingency
from scipy.cluster import hierarchy


def _barmode_selector():
    return (
        "group"
        if st.segmented_control("Bar Mode", ["Stacked", "Unstacked"], default="Stacked") == "Unstacked"
        else "stack"
    )


def _show_image_and_download_button(fig: go.Figure):

    st.plotly_chart(fig, use_container_width=False)

    # Make a button to download the HTML image
    st.download_button(
        "Download Plot (HTML)",
        fig.to_html(),
        file_name="plot.html",
        mime="text/html"
    )


def _format_inputs(counts: pd.DataFrame, cnames: List[str]) -> Tuple[pd.DataFrame, str]:
    # Collapse by any number of categories
    cnames = list(set(cnames))

    # Make sure to treat the categories as strings
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
            f"Percent of {cname.title()}": 100 * (df["count"] / df.groupby(cname)["count"].transform("sum"))
            for cname in cnames
        }
    ).rename(
        columns=dict(count="Cell Count")
    )

    # assert False, df.query("neighborhood == '0'")

    # Let the user select the metric
    metric = st.selectbox(
        "Metric",
        ["Cell Count", "Percent of Total"] + [
            f"Percent of {cname.title()}" for cname in cnames
        ]
    )

    return df, metric


def _chi2_test(df: pd.DataFrame, group1: str, group2: str):

    # Sum up the counts across groups
    vals = pd.crosstab(
        df[group1],
        df[group2],
        df["Cell Count"],
        aggfunc="sum"
    ).T.fillna(0)

    # Compute the chi-squared test
    chi2 = chi2_contingency(vals)
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

    st.download_button(
        label="Download Counts (CSV)",
        data=vals.to_csv(),
        file_name=f"{group1}.{group2}.counts.csv"
    )


@st.cache_data
def col_options(counts: pd.DataFrame):
    options = list(counts.columns.values)
    options.sort()
    return [
        val for val in options if val != "count"
    ]


def plot_bars(counts: pd.DataFrame):
    # Let the user select a grouping for the X axis, and the color
    selected_x = st.selectbox("X-Axis", options=col_options(counts), index=col_options(counts).index("region"))
    selected_color = st.selectbox("Color", options=col_options(counts), index=col_options(counts).index("neighborhood"))

    # Format the data to display, and ask the user to select which one to use as the value
    df, metric = _format_inputs(counts, [selected_color, selected_x])

    # Set up the plot
    fig = px.bar(
        data_frame=df,
        x=selected_x,
        y=metric,
        color=selected_color,
        color_discrete_sequence=px.colors.qualitative.D3,
        barmode=_barmode_selector(),
        labels={
            cname: cname.title()
            for cname in df.columns.values
        }

    )

    _show_image_and_download_button(fig)

    if selected_x != selected_color:
        _chi2_test(df, selected_x, selected_color)


def sort_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.shape[0] < 3:
        return df

    return df.loc[
        df.index.values[
            hierarchy.leaves_list(
                hierarchy.linkage(
                    df.values,
                    method="ward",
                    metric="euclidean"
                )
            )
        ]
    ]


def plot_heatmap(counts: pd.DataFrame):
    # Let the user select groupings for the X axis and Y axis, and the value used for the color
    selected_x = st.selectbox("X-Axis", options=col_options(counts), index=col_options(counts).index("region"))
    selected_y = st.selectbox("Y-Axis", options=col_options(counts), index=col_options(counts).index("neighborhood"))

    if selected_y == selected_x:
        st.write("Select different variables for X and Y")
        return

    # Format the data to display, and ask the user to select which one to use as the value
    df, metric = _format_inputs(counts, [selected_x, selected_y])

    # Make a wide DataFrame
    wide_df = df.pivot_table(
        index=selected_x,
        columns=selected_y,
        values=metric
    ).fillna(0)

    # Sort the columns and rows
    wide_df = sort_table(wide_df)
    wide_df = sort_table(wide_df.T).T

    # Make the plot
    fig = px.imshow(
        wide_df.values,
        color_continuous_scale="blues",
        labels=dict(
            x=selected_x,
            y=selected_y
        ),
        aspect=(
            None if st.checkbox(label="Square Aspect", value=False) else "auto"
        )
    )
    fig.update_layout(
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(wide_df.shape[1])),
            ticktext=wide_df.columns.values
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(wide_df.shape[0])),
            ticktext=wide_df.index.values
        )
    )

    _show_image_and_download_button(fig)

    _chi2_test(df, selected_x, selected_y)


def compare_counts(counts: pd.DataFrame):

    # Let the user filter based on groups
    filters = {
        cname: st.multiselect(
            f"Filter on {cname}",
            options=cvals.sort_values().unique(),
            default=cvals.sort_values().unique()
        )
        for cname, cvals in counts.items()
        if cname != "count"
    }
    for cname, selected in filters.items():
        counts = counts.loc[counts[cname].isin(selected)]

    st.write(f"Total number of cells: {counts.shape[0]:,}")

    plot_type = st.selectbox(
        label="Plot Type",
        options=["Bars", "Heatmap"]
    )

    if plot_type == "Bars":
        plot_bars(counts)
    elif plot_type == "Heatmap":
        plot_heatmap(counts)

