import streamlit as st


def get_query_param(key: str):
    # If the query param is not set, try the session state
    val = st.query_params.get(key)
    if val is not None:
        return val

    # If the session_state is set
    if st.session_state.get(key) is not None:
        val = st.session_state.get(key)
        # Set the query param
        set_query_param(key, val)
        return val


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
    st.query_params[key] = value
    while st.query_params.get(key) != value:
        timeout -= poll_interval
        if timeout < 0:
            raise TimeoutError(f"Timeout waiting to set query parameter: {key}")
    st.session_state[key] = value


def clear_query_param(
    key: str,
    timeout=5,
    poll_interval=0.1
):
    """
    Clear the query param and also the session state.
    """
    if key in st.query_params:
        del st.query_params[key]
    while st.query_params.get(key) is not None:
        timeout -= poll_interval
        if timeout < 0:
            raise TimeoutError(f"Timeout waiting to clearn query parameter: {key}")
    if key in st.session_state:
        del st.session_state[key]

