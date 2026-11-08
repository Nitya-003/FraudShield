from __future__ import annotations

import html
import time

import streamlit as st

from components.charts import category_figure, revenue_figure
from components.shell import render_shell
from components.ui import avatar, metric_card, page_header, section_heading, skeleton_cards
from data.mock_data import get_activities, get_metric_data


render_shell("overview")

head_left, head_right = st.columns([4, 1], vertical_alignment="bottom")
with head_left:
    page_header("Executive overview", "A clear view of revenue, adoption, and the work that needs attention.")
with head_right:
    if st.button("Refresh data", icon=":material/refresh:", use_container_width=True, key="overview_refresh"):
        st.session_state["overview_refreshing"] = True
        st.rerun()

if st.session_state.pop("overview_refreshing", False):
    skeleton_cards(4)
    time.sleep(0.35)

metrics = get_metric_data()
metric_columns = st.columns(4, gap="medium")
for column, metric in zip(metric_columns, metrics):
    with column:
        metric_card(metric)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
chart_col, breakdown_col = st.columns([1.7, 1], gap="medium")
with chart_col:
    with st.container(border=True):
        controls_left, controls_right = st.columns([3, 1.1], vertical_alignment="center")
        with controls_left:
            st.markdown('<div class="chart-card-title">Revenue &amp; expenses</div><div class="chart-card-subtitle">Income movement over your selected period</div>', unsafe_allow_html=True)
        with controls_right:
            period = st.selectbox("Chart period", ["7D", "30D", "1Y"], index=1, label_visibility="collapsed", key="overview_chart_period")
        st.plotly_chart(revenue_figure(period, dark=st.session_state.get("theme") == "dark"), use_container_width=True, config={"displayModeBar": False})
        st.markdown(f'<div class="small-muted">Chart window: {html.escape(period)} · Revenue remains ahead of the expense curve.</div>', unsafe_allow_html=True)
with breakdown_col:
    with st.container(border=True):
        st.markdown('<div class="chart-card-title">Sales by product</div><div class="chart-card-subtitle">Revenue mix across active plans</div>', unsafe_allow_html=True)
        st.plotly_chart(category_figure(dark=st.session_state.get("theme") == "dark"), use_container_width=True, config={"displayModeBar": False})

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
activity_col, insight_col = st.columns([1.5, 1], gap="medium")
with activity_col:
    with st.container(border=True):
        section_heading("Recent activity", "The latest changes across your workspace")
        for item in get_activities():
            tag_class = f"tag-{item['tone']}"
            st.markdown(
                f'''<div class="activity-row">{avatar(item['name'], item['initials'], 36, item['tone'])}
                <div class="activity-copy"><strong>{html.escape(item['name'])}</strong> {html.escape(item['action'])}
                <span class="activity-target">{html.escape(item['target'])}</span></div>
                <div class="activity-meta"><span class="activity-time">{html.escape(item['time'])}</span>
                <span class="activity-tag {tag_class}">{html.escape(item['tag'])}</span></div></div>''',
                unsafe_allow_html=True,
            )
with insight_col:
    with st.container(border=True):
        section_heading("Signal check", "A quick read on operational momentum")
        st.markdown(
            '''<div class="callout"><strong>Review queue is steady.</strong><br>Open issues are down 6.4% from last month, while total revenue is tracking 12.8% higher.</div>
            <div style="height:12px"></div>
            <div class="detail-label">Team response time</div><div class="detail-value">2h 18m <span style="color:#16845f;font-size:11px">↓ 14 min</span></div>
            <div class="detail-label">Review completion</div><div class="detail-value">94.2% <span style="color:#16845f;font-size:11px">↑ 2.1 pts</span></div>
            <div class="detail-label">Dataset status</div><div class="detail-value">Demo data · refreshed just now</div>''',
            unsafe_allow_html=True,
        )
