from typing import List
import streamlit as st
from app.cirro import cirro_dataset_link, cirro_analysis_link


def card_style(key: str):
    html = """
<style>
.st-key-{} {{
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 10px;
    background-color: rgb(240, 242, 246);
}}
</style>
    """.format(key)

    st.markdown(html, unsafe_allow_html=True)


def linked_header(title: str):
    st.write(f"#### {title}")
    st.sidebar.markdown(f"[{title}](#{title.lower().replace(' ', '-')})")


def card_content(title: str, content: List[str]):
    st.markdown(
        f"""
        <h5 level="5" style="margin-bottom: 5px; padding-right: 10px;">
            {title}
        </h5>
        <hr style="margin: 5px 15px;">
        <p>
        {'<br>'.join(content)}
        </p>
        """,
        unsafe_allow_html=True
    )


def paragraph(content: List[str]):
    return st.markdown(
        f"""
        <p>
        {'<br>'.join(content)}
        </p>
        """,
        unsafe_allow_html=True
    )


def cirro_dataset_button(dataset_id: str):
    _make_button("View / Edit Dataset", cirro_dataset_link(dataset_id))


def cirro_analysis_button(label: str, dataset_id: str, analysis_id: str):
    _make_button(label, cirro_analysis_link(dataset_id, analysis_id))


def _make_button(title: str, url: str):
    st.markdown(
        f"""
        <a href="{url}" target="_blank" style="
            display: inline-block;
            padding: 10px 20px;
            font-size: 16px;
            color: #fff;
            background-color: #077C9E;
            border: none;
            border-radius: 5px;
            text-decoration: none;
            text-align: center;
            cursor: pointer;
            transition: background-color 0.3s ease;
        ">
            {title}
        </a>
        """,
        unsafe_allow_html=True
    )
