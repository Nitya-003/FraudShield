from __future__ import annotations

import html
import json
import secrets

import streamlit as st
import streamlit.components.v1 as components

from components.local_storage import load_json, save_json
from components.shell import render_shell
from components.ui import page_header, section_heading
from utils import valid_email


DEFAULT_SETTINGS = {
    "organization": "FraudShield Analytics",
    "primary_email": "ops@northstarrisk.demo",
    "timezone": "America/New_York",
    "two_factor": True,
    "session_timeout": "30 minutes",
    "notifications": {
        "Risk alerts": {"Email": True, "In-App": True, "SMS": False},
        "Transaction reviews": {"Email": True, "In-App": True, "SMS": True},
        "Account changes": {"Email": True, "In-App": False, "SMS": False},
        "Weekly reports": {"Email": True, "In-App": False, "SMS": False},
    },
    "api_key": "fs_demo_2a71d80b1f2c4c9e81d3f0ce",
}


def _get_settings() -> dict:
    if "fraudshield_settings" not in st.session_state:
        saved = load_json("fraudshield_settings", {})
        settings = {**DEFAULT_SETTINGS, **(saved if isinstance(saved, dict) else {})}
        settings["notifications"] = {**DEFAULT_SETTINGS["notifications"], **settings.get("notifications", {})}
        st.session_state["fraudshield_settings"] = settings
    return st.session_state["fraudshield_settings"]


def _save_settings(settings: dict, message: str) -> None:
    st.session_state["fraudshield_settings"] = settings
    save_json("fraudshield_settings", settings, component_key="fraudshield_save_settings")
    st.toast(message, icon=":material/check_circle:")


render_shell("settings")
page_header("System settings", "Tune workspace preferences, access controls, and notification behavior.")
settings = _get_settings()

general_tab, security_tab, notifications_tab, api_tab = st.tabs(["General", "Security", "Notifications", "API & Integrations"])

with general_tab:
    section_heading("Organization profile", "These details are stored locally in this demo browser.")
    with st.form("general_settings_form"):
        organization = st.text_input("Organization name", value=settings.get("organization", ""), max_chars=80)
        primary_email = st.text_input("Primary email", value=settings.get("primary_email", ""))
        timezones = ["America/New_York", "America/Los_Angeles", "America/Chicago", "Europe/London", "Europe/Paris", "Asia/Singapore", "Asia/Tokyo", "UTC"]
        current_timezone = settings.get("timezone", "America/New_York")
        timezone = st.selectbox("Timezone", timezones, index=timezones.index(current_timezone) if current_timezone in timezones else 0)
        avatar_file = st.file_uploader("Profile avatar", type=["png", "jpg", "jpeg", "webp"], help="Optional image preview; the uploaded file is not sent to a server.")
        submitted = st.form_submit_button("Save general settings", type="primary", icon=":material/save:")
    if avatar_file is not None:
        st.image(avatar_file, caption=f"Local preview · {avatar_file.name}", width=100)
    if submitted:
        errors = []
        if len(organization.strip()) < 2:
            errors.append("Organization name must contain at least two characters.")
        if not valid_email(primary_email):
            errors.append("Enter a valid primary email address.")
        if errors:
            for error in errors:
                st.error(error)
        else:
            settings.update({"organization": organization.strip(), "primary_email": primary_email.strip(), "timezone": timezone, "avatar_name": avatar_file.name if avatar_file else settings.get("avatar_name", "")})
            _save_settings(settings, "General settings saved in this browser.")

with security_tab:
    section_heading("Security controls", "Account settings are a local demonstration; no real password is changed.")
    with st.form("security_preferences_form"):
        two_factor = st.toggle("Two-factor authentication", value=bool(settings.get("two_factor", True)), help="Demo preference only; no authentication provider is connected.")
        session_timeout = st.selectbox("Session timeout", ["15 minutes", "30 minutes", "1 hour", "4 hours", "8 hours"], index=["15 minutes", "30 minutes", "1 hour", "4 hours", "8 hours"].index(settings.get("session_timeout", "30 minutes")))
        security_saved = st.form_submit_button("Save security preferences", type="primary")
    if security_saved:
        settings.update({"two_factor": two_factor, "session_timeout": session_timeout})
        _save_settings(settings, "Security preferences saved in this browser.")
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown('<div class="chart-card-title">Update password</div><div class="chart-card-subtitle">Validate a new password locally; no credential is stored or sent.</div>', unsafe_allow_html=True)
        with st.form("password_update_form"):
            current_password = st.text_input("Current password", type="password", autocomplete="current-password")
            new_password = st.text_input("New password", type="password", autocomplete="new-password")
            confirm_password = st.text_input("Confirm new password", type="password", autocomplete="new-password")
            password_saved = st.form_submit_button("Validate password update", type="primary")
        if new_password:
            checks = [len(new_password) >= 12, any(char.isupper() for char in new_password), any(char.islower() for char in new_password), any(char.isdigit() for char in new_password), any(not char.isalnum() for char in new_password)]
            st.caption(f"Password strength checks: {sum(checks)}/5 met · 12+ characters, upper/lowercase, number, and symbol.")
        if password_saved:
            if not current_password:
                st.error("Enter the current password to continue.")
            elif not all([len(new_password) >= 12, any(char.isupper() for char in new_password), any(char.islower() for char in new_password), any(char.isdigit() for char in new_password), any(not char.isalnum() for char in new_password)]):
                st.error("Use at least 12 characters with uppercase, lowercase, a number, and a symbol.")
            elif new_password != confirm_password:
                st.error("The new password and confirmation do not match.")
            else:
                st.toast("Password validation passed in this demo; no credentials were changed.", icon=":material/check_circle:")

with notifications_tab:
    section_heading("Notification preferences", "Choose which channels should receive each event category.")
    notification_rows = []
    for category, channels in settings.get("notifications", DEFAULT_SETTINGS["notifications"]).items():
        notification_rows.append({"Event category": category, "Email": bool(channels.get("Email", False)), "In-App": bool(channels.get("In-App", False)), "SMS": bool(channels.get("SMS", False))})
    edited = st.data_editor(
        notification_rows,
        hide_index=True,
        use_container_width=True,
        num_rows="fixed",
        disabled=["Event category"],
        column_config={
            "Event category": st.column_config.TextColumn("Event category", width="large"),
            "Email": st.column_config.CheckboxColumn("Email"),
            "In-App": st.column_config.CheckboxColumn("In-App"),
            "SMS": st.column_config.CheckboxColumn("SMS"),
        },
        key="notification_matrix",
    )
    if st.button("Save notification preferences", type="primary", icon=":material/save:", key="save_notification_matrix"):
        settings["notifications"] = {row["Event category"]: {channel: bool(row[channel]) for channel in ["Email", "In-App", "SMS"]} for row in edited}
        _save_settings(settings, "Notification preferences saved in this browser.")

with api_tab:
    section_heading("API key management", "Placeholder credentials for this demo workspace only.")
    api_key = settings.get("api_key", DEFAULT_SETTINGS["api_key"])
    masked_key = f"fs_demo_{'•' * max(8, len(api_key) - 8)}{api_key[-4:]}"
    left, right = st.columns([1.7, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown('<div class="chart-card-title">Workspace API key</div><div class="chart-card-subtitle">Use this key in a connected integration. This demo key is not valid for any service.</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="api-key">{html.escape(masked_key)}</div>', unsafe_allow_html=True)
            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            key_literal = json.dumps(api_key)
            components.html(
                f'''<button onclick="navigator.clipboard.writeText({key_literal}).then(()=>this.innerText='Copied').catch(()=>this.innerText='Copy unavailable')" style="border:1px solid #dce5ec;border-radius:9px;padding:9px 14px;background:#fff;color:#20354a;font-weight:650;cursor:pointer;font-family:system-ui">Copy to clipboard</button>
                <span style="margin-left:9px;color:#8292a3;font:11px system-ui">Copies the simulated key only</span>''',
                height=42,
                scrolling=False,
            )
            if st.button("Regenerate key", icon=":material/autorenew:", key="regenerate_api_key"):
                settings["api_key"] = "fs_demo_" + secrets.token_hex(14)
                _save_settings(settings, "A new simulated API key was generated.")
                st.rerun()
    with right:
        st.markdown('<div class="callout"><strong>Demo mode</strong><br>Copy and regenerate operate on a placeholder value stored in this browser. No external API, integration, or secret is connected.</div>', unsafe_allow_html=True)
        st.markdown('<div class="detail-label">Access scope</div><div class="detail-value">Read-only analytics · Demo</div><div class="detail-label">Key status</div><div class="detail-value">Active placeholder</div>', unsafe_allow_html=True)
