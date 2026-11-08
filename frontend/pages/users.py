from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from components.shell import render_shell
from components.ui import avatar, metric_card, page_header, section_heading, status_badge
from data.mock_data import get_users
from utils import valid_email


def _ensure_users() -> list[dict]:
    if "fraudshield_users" not in st.session_state:
        st.session_state["fraudshield_users"] = get_users()
    return st.session_state["fraudshield_users"]


def _save_users(users: list[dict]) -> None:
    st.session_state["fraudshield_users"] = users


def _invite_dialog() -> None:
    @st.dialog("Invite a team member")
    def show() -> None:
        st.caption("Add a person to this simulated FraudShield workspace.")
        with st.form("invite_user_form", clear_on_submit=False):
            name = st.text_input("Full name", placeholder="Alex Morgan")
            email = st.text_input("Work email", placeholder="alex@company.com")
            role = st.selectbox("Role", ["Admin", "Editor", "Viewer"])
            submitted = st.form_submit_button("Send invite", type="primary", use_container_width=True)
        if submitted:
            users = _ensure_users()
            errors = []
            if len(name.strip()) < 2:
                errors.append("Enter a name with at least two characters.")
            if not valid_email(email):
                errors.append("Enter a valid email address.")
            if any(person["email"].lower() == email.strip().lower() for person in users):
                errors.append("That email is already in this demo workspace.")
            if errors:
                for error in errors:
                    st.error(error)
            else:
                initials = "".join(part[0] for part in name.strip().split()[:2]).upper()
                users.append({"id": f"USR-{1001 + len(users)}", "name": name.strip(), "email": email.strip(), "role": role, "status": "Active", "last_active": "Just invited", "department": "New member", "initials": initials, "joined": "Today"})
                _save_users(users)
                st.session_state["show_invite_dialog"] = False
                st.toast(f"Invite queued for {email.strip()} in this demo.", icon=":material/mark_email_read:")
                st.rerun()
    show()


def _edit_dialog(user: dict) -> None:
    @st.dialog("Edit team member")
    def show() -> None:
        with st.form("edit_user_form"):
            name = st.text_input("Full name", value=user["name"])
            email = st.text_input("Work email", value=user["email"])
            role = st.selectbox("Role", ["Admin", "Editor", "Viewer"], index=["Admin", "Editor", "Viewer"].index(user["role"]))
            submitted = st.form_submit_button("Save changes", type="primary", use_container_width=True)
        if submitted:
            if len(name.strip()) < 2 or not valid_email(email):
                st.error("Enter a valid name and email address.")
            else:
                users = _ensure_users()
                for person in users:
                    if person["id"] == user["id"]:
                        person.update({"name": name.strip(), "email": email.strip(), "role": role, "initials": "".join(part[0] for part in name.strip().split()[:2]).upper()})
                _save_users(users)
                st.session_state.pop("edit_user_id", None)
                st.toast("User details updated in this demo.", icon=":material/check_circle:")
                st.rerun()
    show()


def _delete_dialog(user: dict) -> None:
    @st.dialog("Remove team member")
    def show() -> None:
        st.write(f"Remove **{user['name']}** from the simulated workspace?")
        st.caption("This only changes the current demo session.")
        confirm, cancel = st.columns(2)
        with confirm:
            if st.button("Remove user", type="primary", use_container_width=True, key="confirm_delete_user"):
                _save_users([person for person in _ensure_users() if person["id"] != user["id"]])
                st.session_state.pop("delete_user_id", None)
                st.toast("User removed from this demo workspace.", icon=":material/person_remove:")
                st.rerun()
        with cancel:
            if st.button("Cancel", use_container_width=True, key="cancel_delete_user"):
                st.session_state.pop("delete_user_id", None)
                st.rerun()
    show()


def _action_menu(user: dict, suffix: str) -> None:
    with st.popover("⋯", help=f"Actions for {user['name']}"):
        if st.button("Edit", key=f"edit-{suffix}", use_container_width=True):
            st.session_state["edit_user_id"] = user["id"]
            st.rerun()
        status_action = "Suspend" if user["status"] == "Active" else "Reactivate"
        if st.button(status_action, key=f"suspend-{suffix}", use_container_width=True):
            users = _ensure_users()
            for person in users:
                if person["id"] == user["id"]:
                    person["status"] = "Inactive" if person["status"] == "Active" else "Active"
            _save_users(users)
            st.toast(f"{user['name']} status updated in this demo.", icon=":material/check_circle:")
            st.rerun()
        if st.button("Delete", key=f"delete-{suffix}", use_container_width=True):
            st.session_state["delete_user_id"] = user["id"]
            st.rerun()


render_shell("users")
page_header("User management", "Keep the right people connected to the right parts of your workspace.")
users = _ensure_users()
active_count = sum(person["status"] == "Active" for person in users)
active_today = sum(person["status"] == "Active" and ("min ago" in person["last_active"] or "Today" in person["last_active"] or "Just invited" in person["last_active"]) for person in users)
stat_cols = st.columns(3, gap="medium")
for column, metric in zip(
    stat_cols,
    [
        {"title": "Total Accounts", "value": f"{len(users):,}", "delta": "+3", "caption": "this quarter", "trend": "up", "spark": [12, 13, 14, 15, 15, 17, len(users)], "icon": "◉", "color": "cyan"},
        {"title": "Active Today", "value": f"{active_today:,}", "delta": "+2", "caption": "since yesterday", "trend": "up", "spark": [4, 6, 5, 7, 6, 8, active_today], "icon": "↗", "color": "green"},
        {"title": "Churn Rate", "value": "2.8%", "delta": "−0.4%", "caption": "vs. last month", "trend": "down_good", "spark": [34, 31, 32, 28, 27, 24, 21], "icon": "◌", "color": "violet"},
    ],
):
    with column:
        metric_card(metric)

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
filter_cols = st.columns([2.1, 1.25, 1.25, 1.2, 1], vertical_alignment="bottom")
with filter_cols[0]:
    search_value = st.text_input("Search users", value=st.query_params.get("user_search", ""), placeholder="Name or email", key="user_search_widget")
with filter_cols[1]:
    role_options = ["Admin", "Editor", "Viewer"]
    role_value = st.selectbox("Role", ["All roles"] + role_options, index=(["All roles"] + role_options).index(st.query_params.get("user_role", "All roles")) if st.query_params.get("user_role", "All roles") in ["All roles"] + role_options else 0, key="user_role_widget")
with filter_cols[2]:
    status_options = ["All statuses", "Active", "Inactive"]
    status_value = st.selectbox("Status", status_options, index=status_options.index(st.query_params.get("user_status", "All statuses")) if st.query_params.get("user_status", "All statuses") in status_options else 0, key="user_status_widget")
with filter_cols[3]:
    view_mode = st.radio("View", ["Table", "Cards"], horizontal=True, key="user_view_mode", label_visibility="collapsed")
with filter_cols[4]:
    if st.button("Invite New User", type="primary", icon=":material/person_add:", use_container_width=True, key="invite_new_user"):
        st.session_state["show_invite_dialog"] = True

st.query_params["user_search"] = search_value
st.query_params["user_role"] = role_value
st.query_params["user_status"] = status_value

filtered_users = list(users)
if search_value.strip():
    token = search_value.strip().lower()
    filtered_users = [person for person in filtered_users if token in person["name"].lower() or token in person["email"].lower()]
if role_value != "All roles":
    filtered_users = [person for person in filtered_users if person["role"] == role_value]
if status_value != "All statuses":
    filtered_users = [person for person in filtered_users if person["status"] == status_value]

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
with st.container(border=True):
    header_left, header_right = st.columns([3, 1], vertical_alignment="center")
    with header_left:
        section_heading("Workspace members", f"{len(filtered_users)} of {len(users)} accounts · {active_count} active")
    with header_right:
        st.markdown('<div style="text-align:right;color:var(--fs-muted);font-size:10px">User actions are simulated</div>', unsafe_allow_html=True)
    if not filtered_users:
        st.markdown('<div class="empty-state"><div class="empty-mark">⌕</div><strong>No users found</strong><span>Try changing the search, role, or status filters.</span></div>', unsafe_allow_html=True)
    elif view_mode == "Cards":
        for start in range(0, len(filtered_users), 3):
            cards = st.columns(3, gap="medium")
            for column, user in zip(cards, filtered_users[start : start + 3]):
                with column:
                    tone = ["cyan", "violet", "green", "amber", "coral"][int(user["id"].split("-")[-1]) % 5]
                    st.markdown(
                        f'''<div class="user-card"><div class="user-card-head">{avatar(user['name'], user['initials'], 42, tone)}
                        <div><div class="user-card-name">{html.escape(user['name'])}</div><div class="user-card-email">{html.escape(user['email'])}</div></div></div>
                        <div style="display:flex;gap:7px;align-items:center">{status_badge(user['status'])}<span class="activity-tag">{html.escape(user['role'])}</span></div>
                        <div class="user-card-meta"><span>{html.escape(user['department'])}</span><span>{html.escape(user['last_active'])}</span></div></div>''',
                        unsafe_allow_html=True,
                    )
                    _action_menu(user, f"card-{user['id']}")
    else:
        headings = st.columns([2.4, 3.2, 1.25, 1.2, 1.5, 0.45], vertical_alignment="center")
        for column, label in zip(headings, ["User", "Email", "Role", "Status", "Last active", ""]):
            with column:
                st.markdown(f'<div class="small-muted" style="font-weight:700;text-transform:uppercase;letter-spacing:.6px;padding:6px 0">{label}</div>', unsafe_allow_html=True)
        st.markdown("<hr style='margin:2px 0 8px'>", unsafe_allow_html=True)
        for index, user in enumerate(filtered_users):
            row = st.columns([2.4, 3.2, 1.25, 1.2, 1.5, 0.45], vertical_alignment="center")
            with row[0]:
                st.markdown(f'{avatar(user["name"], user["initials"], 30, "violet")} <span style="margin-left:7px;font-size:11px;font-weight:650">{html.escape(user["name"])}</span>', unsafe_allow_html=True)
            with row[1]:
                st.markdown(f'<span style="font-size:11px;color:var(--fs-muted)">{html.escape(user["email"])}</span>', unsafe_allow_html=True)
            with row[2]:
                st.markdown(f'<span class="activity-tag">{html.escape(user["role"])}</span>', unsafe_allow_html=True)
            with row[3]:
                st.markdown(status_badge(user["status"]), unsafe_allow_html=True)
            with row[4]:
                st.markdown(f'<span style="font-size:10px;color:var(--fs-muted)">{html.escape(user["last_active"])}</span>', unsafe_allow_html=True)
            with row[5]:
                _action_menu(user, f"table-{user['id']}")
            if index != len(filtered_users) - 1:
                st.markdown("<hr style='margin:1px 0 4px'>", unsafe_allow_html=True)

edit_id = st.session_state.get("edit_user_id")
if edit_id:
    edit_user = next((person for person in _ensure_users() if person["id"] == edit_id), None)
    if edit_user:
        _edit_dialog(edit_user)
delete_id = st.session_state.get("delete_user_id")
if delete_id:
    delete_user = next((person for person in _ensure_users() if person["id"] == delete_id), None)
    if delete_user:
        _delete_dialog(delete_user)
if st.session_state.get("show_invite_dialog"):
    _invite_dialog()
