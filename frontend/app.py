from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="FraudShield — Executive Analytics",
    page_icon="public/fraudshield-logo.svg",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={"About": "FraudShield executive analytics demo. All records and actions are simulated."},
)

pages = [
    st.Page("pages/overview.py", title="Executive Overview", icon=":material/space_dashboard:", url_path="", default=True),
    st.Page("pages/transactions.py", title="Transactions & Analytics", icon=":material/receipt_long:", url_path="transactions"),
    st.Page("pages/users.py", title="User Management", icon=":material/group:", url_path="users"),
    st.Page("pages/settings.py", title="System Settings", icon=":material/settings:", url_path="settings"),
]

current_page = st.navigation(pages, position="hidden")
current_page.run()
