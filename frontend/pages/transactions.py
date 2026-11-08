from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from components.shell import render_shell
from components.ui import page_header, section_heading, status_badge
from data.mock_data import get_transactions
from utils import csv_bytes, format_currency, safe_int


def _refund_dialog(transaction: dict) -> None:
    @st.dialog("Confirm simulated refund")
    def show() -> None:
        st.markdown(f"You're about to mark **{transaction['ID']}** for a refund review.")
        st.caption("Demo action only — no payment provider will be contacted and no funds will move.")
        confirm, cancel = st.columns(2)
        with confirm:
            if st.button("Confirm review", type="primary", use_container_width=True, key="confirm_refund_action"):
                st.toast(f"Refund marked for review in this demo · {transaction['ID']}", icon=":material/check_circle:")
                st.session_state.pop("confirm_refund_id", None)
                st.rerun()
        with cancel:
            if st.button("Cancel", use_container_width=True, key="cancel_refund_action"):
                st.session_state.pop("confirm_refund_id", None)
                st.rerun()
    show()


date_range = render_shell("transactions")
page_header("Transactions & analytics", "Explore payment activity, review exceptions, and export the current view.")

all_records = get_transactions(42)
df = pd.DataFrame(all_records)
df["Date"] = pd.to_datetime(df["Date"])

query_search = st.query_params.get("tx_search", "")
query_status = st.query_params.get("tx_status", "")
query_page = safe_int(st.query_params.get("tx_page", 1))
search_col, status_col, sort_col, direction_col, export_col = st.columns([2.3, 1.45, 1.6, 1.2, 1.0], vertical_alignment="bottom")
with search_col:
    search_term = st.text_input("Search transactions", value=query_search, placeholder="Customer, email, or ID", key="tx_search_widget")
with status_col:
    status_options = ["Completed", "Pending", "Failed"]
    has_status_query = "tx_status" in st.query_params
    default_statuses = [item for item in query_status.split(",") if item in status_options] if has_status_query else status_options
    selected_statuses = st.multiselect("Status", status_options, default=default_statuses, key="tx_status_widget")
with sort_col:
    sort_fields = st.multiselect("Sort priority", ["Date", "Amount", "Customer", "Status"], default=["Date"], key="tx_sort_fields")
with direction_col:
    direction = st.selectbox("Order", ["Newest / high first", "Oldest / low first"], label_visibility="collapsed", key="tx_sort_direction")

filtered = df.copy()
range_start, range_end = st.session_state.get("global_date_bounds", (None, None))
if range_start and range_end:
    filtered = filtered[filtered["Date"].dt.date.between(range_start, range_end)]
if search_term.strip():
    term = search_term.strip().lower()
    searchable = filtered[["ID", "Customer", "Email", "Company"]].astype(str).agg(" ".join, axis=1).str.lower()
    filtered = filtered[searchable.str.contains(term, regex=False)]
if selected_statuses:
    filtered = filtered[filtered["Status"].isin(selected_statuses)]
else:
    filtered = filtered.iloc[0:0]

if sort_fields:
    ascending = direction == "Oldest / low first"
    filtered = filtered.sort_values(by=sort_fields, ascending=ascending, kind="stable")

st.query_params["tx_search"] = search_term
if selected_statuses:
    st.query_params["tx_status"] = ",".join(selected_statuses)
else:
    st.query_params["tx_status"] = ""

fields = ["ID", "Customer", "Email", "Date", "Status", "Payment Method", "Amount"]
export_rows = filtered[fields].copy()
export_bytes = csv_bytes(export_rows.to_dict("records"), fields)
with export_col:
    st.download_button("Export CSV", data=export_bytes, file_name="fraudshield-transactions.csv", mime="text/csv", icon=":material/download:", use_container_width=True, disabled=filtered.empty)

page_size = 10
max_page = max(1, (len(filtered) + page_size - 1) // page_size)
page_number = min(query_page, max_page)
if page_number != query_page:
    st.query_params["tx_page"] = str(page_number)
start = (page_number - 1) * page_size
page_df = filtered.iloc[start : start + page_size]

st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
show_sheet = bool(st.session_state.get("show_transaction_sheet")) and bool(st.session_state.get("selected_tx_id"))
if show_sheet:
    left, right = st.columns([1.8, 1], gap="large")
else:
    left, right = st.container(), None
with left:
    with st.container(border=True):
        title_col, count_col = st.columns([2.4, 1], vertical_alignment="center")
        with title_col:
            section_heading("Payment activity", f"{len(filtered)} matching records · {date_range.lower()}")
        with count_col:
            st.markdown(f'<div style="text-align:right;color:var(--fs-muted);font-size:11px">Page {page_number} of {max_page}</div>', unsafe_allow_html=True)
        if filtered.empty:
            st.markdown('<div class="empty-state"><div class="empty-mark">⌕</div><strong>No transactions found</strong><span>Adjust your search or status filters to see matching records.</span></div>', unsafe_allow_html=True)
        else:
            st.caption("Select a row number to open its right-side detail sheet.")
            display = page_df[["ID", "Customer", "Date", "Status", "Payment Method", "Amount"]].copy().reset_index(drop=True)
            display["Date"] = display["Date"].dt.strftime("%b %d, %Y")
            display["Amount"] = display["Amount"].map(lambda value: f"${value:,.2f}")
            event = st.dataframe(
                display,
                use_container_width=True,
                hide_index=False,
                height=425,
                on_select="rerun",
                selection_mode="single-row",
                key="transaction_table",
                column_config={
                    "ID": st.column_config.TextColumn("Transaction ID", width="medium"),
                    "Customer": st.column_config.TextColumn("Customer", width="medium"),
                    "Date": st.column_config.TextColumn("Date", width="small"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Payment Method": st.column_config.TextColumn("Payment method", width="medium"),
                    "Amount": st.column_config.TextColumn("Amount", width="small"),
                },
            )
            selected_rows = getattr(getattr(event, "selection", None), "rows", [])
            if selected_rows:
                selected_position = selected_rows[0]
                if selected_position < len(page_df):
                    selected_tx_id = page_df.iloc[selected_position]["ID"]
                    if selected_tx_id != st.session_state.get("selected_tx_id"):
                        st.session_state["selected_tx_id"] = selected_tx_id
                        st.session_state["show_transaction_sheet"] = True
                        st.rerun()
            elif st.session_state.get("selected_tx_id") and st.session_state.get("show_transaction_sheet", False):
                st.session_state["selected_tx_id"] = None
                st.session_state["show_transaction_sheet"] = False
        page_controls_left, page_controls_right = st.columns([2, 1], vertical_alignment="center")
        with page_controls_left:
            st.caption(f"Showing {min(start + 1, len(filtered))}–{min(start + page_size, len(filtered))} of {len(filtered)} transactions")
        with page_controls_right:
            if st.button("Previous", disabled=page_number <= 1, use_container_width=True, key="tx_previous"):
                st.query_params["tx_page"] = str(page_number - 1)
                st.rerun()
            if st.button("Next", disabled=page_number >= max_page, use_container_width=True, key="tx_next"):
                st.query_params["tx_page"] = str(page_number + 1)
                st.rerun()

if show_sheet and right is not None:
    with right:
        selected_id = st.session_state.get("selected_tx_id")
        selected = next((record for record in all_records if record["ID"] == selected_id), None)
        if selected and st.session_state.get("show_transaction_sheet", False):
            with st.container(border=True, key="transaction-detail-sheet"):
                title_col, close_col = st.columns([3, 1], vertical_alignment="center")
                with title_col:
                    section_heading("Transaction details", "Full record breakdown")
                with close_col:
                    if st.button("×", help="Close transaction details", use_container_width=True, key="close_transaction_sheet"):
                        st.session_state["show_transaction_sheet"] = False
                        st.rerun()
                st.markdown(f'<div style="font-size:20px;font-weight:760;letter-spacing:-.5px">{html.escape(selected["ID"])}</div>', unsafe_allow_html=True)
                st.markdown(f'{status_badge(str(selected["Status"]))}', unsafe_allow_html=True)
                st.markdown(f'<div class="detail-label">Customer</div><div class="detail-value">{html.escape(selected["Customer"])}</div><div class="small-muted">{html.escape(selected["Email"])}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="detail-label">Company</div><div class="detail-value">{html.escape(selected["Company"])}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="detail-label">Amount</div><div class="detail-value" style="font-size:22px">{format_currency(float(selected["Amount"]))}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="detail-label">Payment method</div><div class="detail-value">{html.escape(selected["Payment Method"])}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="detail-label">Product</div><div class="detail-value">{html.escape(selected["Product"])}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="detail-label">Date · Invoice · Reference</div><div class="detail-value">{selected["Date"]:%b %d, %Y} · {html.escape(selected["Invoice"])} · {html.escape(selected["Reference"])}</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="detail-label">Risk score</div><div class="detail-value">{selected["Risk score"]} / 100 <span class="small-muted">· illustrative demo metric</span></div>', unsafe_allow_html=True)
                st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
                if st.button("Issue refund", type="primary", icon=":material/currency_exchange:", use_container_width=True, key="issue_refund"):
                    st.session_state["confirm_refund_id"] = selected["ID"]
                    st.rerun()
                st.caption("Simulated action only. No funds will be moved.")

confirm_id = st.session_state.get("confirm_refund_id")
if confirm_id:
    transaction_for_dialog = next((record for record in all_records if record["ID"] == confirm_id), None)
    if transaction_for_dialog:
        _refund_dialog(transaction_for_dialog)
