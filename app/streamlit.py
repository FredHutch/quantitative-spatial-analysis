import streamlit as st
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


def get_query_param(key: str):
    logger.info(f"get_query_param {key}")
    # If the query param is not set, try the session state
    val = st.query_params.get(key)
    if val is not None:
        logger.info(f"get_query_param {key} - from query param")
        return val

    # If the session_state is set
    elif st.session_state.get(key) is not None:
        logger.info(f"get_query_param {key} - from session state")
        val = st.session_state.get(key)
        # Set the query param
        set_query_param(key, val)
        return val

    else:
        return None


def set_query_param(
    key: str,
    value: str,
    timeout=5,
    poll_interval=0.1
):
    """
    Set the query param and also the session state.
    Wait and make sure that the query param was set.
    """
    logger.info(f"set_query_param {key}: {value}")
    st.session_state[key] = value
    st.query_params[key] = value
    while st.query_params.get(key) != value:
        timeout -= poll_interval
        if timeout < 0:
            raise TimeoutError(f"Timeout waiting to set query parameter: {key}")


def clear_query_param(
    key: str,
    timeout=5,
    poll_interval=0.1
):
    """
    Clear the query param and also the session state.
    """
    logger.info(f"clear_query_param {key}")
    if key in st.query_params:
        del st.query_params[key]
    while st.query_params.get(key) is not None:
        timeout -= poll_interval
        if timeout < 0:
            raise TimeoutError(f"Timeout waiting to clearn query parameter: {key}")
    if key in st.session_state:
        del st.session_state[key]

