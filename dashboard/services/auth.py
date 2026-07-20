"""Admin lock for the dashboard's write operations when deployed.

Locally (no S3 secrets), everything is unlocked: nothing changes for
development. In the cloud, buttons that *write* or trigger heavy jobs (data
refresh, live real forecast) are hidden until the admin password (secret
`admin_password`) has been entered.
"""
from __future__ import annotations

import streamlit as st

from src.storage import admin_password, admin_unlocked, is_cloud


def admin_gate() -> bool:
    """Show (in the cloud) a password field in the sidebar and return True
    when the user is unlocked. Locally, return True without rendering."""
    if not is_cloud():
        return True

    if admin_unlocked():
        st.sidebar.success("🔓 Admin mode")
        return True

    with st.sidebar.expander("🔒 Admin access"):
        pwd = st.text_input("Password", type="password", key="_admin_pwd_input")
        if st.button("Unlock", key="_admin_unlock_btn"):
            expected = admin_password()
            if expected and pwd == expected:
                st.session_state["_admin_ok"] = True
                st.rerun()
            else:
                st.error("Incorrect password.")
    return False
