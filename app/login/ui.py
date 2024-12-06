import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from cirro import CirroApi, DataPortal
from cirro.auth.device_code import DeviceCodeAuth
from cirro.config import AppConfig, list_tenants
from io import StringIO
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
try:
    from streamlit.runtime.scriptrunner_utils import script_run_context
except ImportError:
    from streamlit.runtime.scriptrunner import script_run_context
from threading import Thread
from time import sleep


def setup_data_portal():

    # If we are already logged in, stop here
    if st.session_state.get("data_portal"):
        return

    st.write("#### Spatial Data Catalog")

    # The domain may be passed in as a query parameter, or it may be stored in the session state
    domain = "dev.cirro.bio" # FIXME
    # domain = st.query_params.get("domain", st.session_state.get("domain")) # FIXME
    if domain is None:
        tenant_dict = {
            tenant['displayName']: tenant['domain']
            for tenant in list_tenants()
        }

        # Let the user select a tenant
        st.write("### Quantitative Spatial Analysis")
        st.write("Log in to the [Cirro](https://cirro.bio) data backend to get started.")
        tenant = st.selectbox(
            "Select Organization",
            ["< select for login >"] + list(tenant_dict.keys())
        )

        domain = tenant_dict.get(tenant)

    if domain:
        st.session_state["domain"] = domain
        _cirro_login(domain, st.empty())


def _cirro_login(domain: str, container: DeltaGenerator):

    # Connect to Cirro - capturing the login URL
    auth_io = StringIO()
    cirro_login_thread = Thread(
        target=_cirro_login_sub,
        args=(auth_io, domain)
    )
    script_run_context.add_script_run_ctx(cirro_login_thread)

    cirro_login_thread.start()

    login_string = auth_io.getvalue()

    while len(login_string) == 0 and cirro_login_thread.is_alive():
        sleep(1)
        login_string = auth_io.getvalue()

    container.write(login_string)
    cirro_login_thread.join()
    container.empty()
    st.rerun()


def _cirro_login_sub(auth_io: StringIO, base_url: str):

    app_config = AppConfig(base_url=base_url)

    st.session_state['Cirro-auth_info'] = DeviceCodeAuth(
        region=app_config.region,
        client_id=app_config.client_id,
        auth_endpoint=app_config.auth_endpoint,
        enable_cache=False,
        auth_io=auth_io
    )

    st.session_state['Cirro-client'] = CirroApi(
        auth_info=st.session_state['Cirro-auth_info'],
        base_url=base_url
    )
    st.session_state['data_portal'] = DataPortal(
        client=st.session_state['Cirro-client']
    )


cols = st.columns([1, 2 ,1])
with cols[1]:
    setup_data_portal()
