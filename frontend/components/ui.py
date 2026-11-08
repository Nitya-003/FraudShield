from __future__ import annotations

import html
from typing import Iterable

import streamlit as st


def _sparkline(values: Iterable[float], color: str) -> str:
    points = list(values)
    if not points:
        return ""
    width, height, pad = 120, 34, 3
    low, high = min(points), max(points)
    spread = max(high - low, 1)
    coords = []
    for index, value in enumerate(points):
        x = pad + index * (width - 2 * pad) / max(len(points) - 1, 1)
        y = height - pad - (value - low) / spread * (height - 2 * pad)
        coords.append(f"{x:.1f},{y:.1f}")
    return (
        f'<svg class="sparkline" viewBox="0 0 {width} {height}" role="img" aria-label="Trend sparkline">'
        f'<polyline points="{" ".join(coords)}" fill="none" stroke="{color}" stroke-width="2.2" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )


def metric_card(metric: dict) -> None:
    colors = {"cyan": "#20bfd0", "violet": "#907cf5", "green": "#32bd91", "amber": "#e8a74d"}
    color = colors.get(metric.get("color", "cyan"), colors["cyan"])
    title = html.escape(str(metric["title"]))
    value = html.escape(str(metric["value"]))
    delta = html.escape(str(metric["delta"]))
    caption = html.escape(str(metric.get("caption", "vs. last month")))
    icon = html.escape(str(metric.get("icon", "•")))
    trend = "positive" if metric.get("trend") in {"up", "down_good"} else "negative"
    svg = _sparkline(metric.get("spark", []), color)
    st.markdown(
        f'''<section class="metric-card" style="--metric-accent:{color}">
          <div class="metric-top"><span class="metric-label">{title}</span><span class="metric-icon">{icon}</span></div>
          <div class="metric-value">{value}</div>
          <div class="metric-bottom"><span class="metric-delta {trend}">{delta}</span><span class="metric-caption">{caption}</span>{svg}</div>
        </section>''',
        unsafe_allow_html=True,
    )


def page_header(title: str, description: str, eyebrow: str = "FRAUDSHIELD · OPERATIONS") -> None:
    st.markdown(
        f'''<div class="page-heading">
          <div class="eyebrow">{html.escape(eyebrow)}</div>
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(description)}</p>
        </div>''',
        unsafe_allow_html=True,
    )


def section_heading(title: str, subtitle: str | None = None) -> None:
    sub = f'<p class="section-subtitle">{html.escape(subtitle)}</p>' if subtitle else ""
    st.markdown(
        f'<div class="section-heading"><h2>{html.escape(title)}</h2>{sub}</div>',
        unsafe_allow_html=True,
    )


def avatar(name: str, initials: str | None = None, size: int = 38, tone: str = "cyan") -> str:
    colors = {"cyan": "#dff7f8;color:#117f8a", "violet": "#eee9ff;color:#6650c4", "green": "#e0f6ed;color:#20825e", "amber": "#fff1da;color:#99621a", "coral": "#ffebe7;color:#a84e3e"}
    style = colors.get(tone, colors["cyan"])
    letters = initials or "".join(part[:1] for part in name.split()[:2]).upper()
    return f'<span class="avatar" style="width:{size}px;height:{size}px;{style}">{html.escape(letters)}</span>'


def status_badge(status: str) -> str:
    tones = {"Completed": "success", "Active": "success", "Resolved": "success", "Pending": "warning", "Inactive": "neutral", "Failed": "danger", "Flagged": "danger"}
    tone = tones.get(status, "neutral")
    return f'<span class="status-badge {tone}"><i></i>{html.escape(status)}</span>'


def empty_state(title: str, message: str) -> None:
    st.markdown(
        f'''<div class="empty-state"><div class="empty-mark">⌕</div><strong>{html.escape(title)}</strong>
        <span>{html.escape(message)}</span></div>''',
        unsafe_allow_html=True,
    )


def skeleton_cards(count: int = 4) -> None:
    columns = st.columns(count)
    for column in columns:
        with column:
            st.markdown(
                '<div class="skeleton-card"><div class="skeleton-line short"></div><div class="skeleton-line wide"></div><div class="skeleton-line medium"></div></div>',
                unsafe_allow_html=True,
            )
