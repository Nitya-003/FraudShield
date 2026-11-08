from __future__ import annotations

import plotly.graph_objects as go

from data.mock_data import get_category_sales, get_revenue_series


CYAN = "#20bfd0"
VIOLET = "#8f7cf4"
MUTED = "#8190a3"
GRID = "rgba(128, 149, 169, 0.16)"


def revenue_figure(period: str = "30D", dark: bool = False) -> go.Figure:
    rows = get_revenue_series(period)
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[row["date"] for row in rows],
            y=[row["revenue"] for row in rows],
            mode="lines",
            name="Revenue",
            line={"color": CYAN, "width": 3, "shape": "spline"},
            fill="tozeroy",
            fillcolor="rgba(32, 191, 208, 0.08)",
            hovertemplate="%{x}<br>Revenue · $%{y:,.0f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=[row["date"] for row in rows],
            y=[row["expenses"] for row in rows],
            mode="lines",
            name="Expenses",
            line={"color": VIOLET, "width": 2.4, "shape": "spline", "dash": "solid"},
            hovertemplate="%{x}<br>Expenses · $%{y:,.0f}<extra></extra>",
        )
    )
    text = "#e8f0f6" if dark else "#25384d"
    figure.update_layout(
        height=330,
        margin={"l": 4, "r": 12, "t": 10, "b": 4},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, ui-sans-serif, system-ui", "size": 11, "color": text},
        legend={"orientation": "h", "y": 1.12, "x": 0, "font": {"size": 11}, "tracegroupgap": 12},
        hovermode="x unified",
        xaxis={"showgrid": False, "zeroline": False, "showline": False, "tickfont": {"color": MUTED}, "nticks": 7},
        yaxis={"showgrid": True, "gridcolor": GRID, "zeroline": False, "tickprefix": "$", "tickformat": "~s", "tickfont": {"color": MUTED}},
    )
    return figure


def category_figure(dark: bool = False) -> go.Figure:
    rows = get_category_sales()
    text = "#e8f0f6" if dark else "#25384d"
    figure = go.Figure(
        go.Bar(
            x=[row["sales"] for row in rows],
            y=[row["category"] for row in rows],
            orientation="h",
            marker={"color": [CYAN, "#6ed4dc", VIOLET, "#b9afe9", "#9faec0"], "line": {"width": 0}, "cornerradius": 7},
            text=[f'${row["sales"] / 1000:.1f}k' for row in rows],
            textposition="outside",
            textfont={"size": 10, "color": text},
            hovertemplate="%{y}<br>Sales · $%{x:,.0f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=330,
        margin={"l": 2, "r": 42, "t": 10, "b": 4},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, ui-sans-serif, system-ui", "size": 11, "color": text},
        showlegend=False,
        xaxis={"visible": False, "showgrid": False, "range": [0, max(row["sales"] for row in rows) * 1.27]},
        yaxis={"showgrid": False, "zeroline": False, "tickfont": {"color": text, "size": 11}, "autorange": "reversed"},
    )
    return figure
