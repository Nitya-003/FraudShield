from __future__ import annotations

import csv
import io
import re
from datetime import date, datetime
from typing import Any, Iterable


def format_currency(value: float | int) -> str:
    return f"${value:,.2f}"


def format_short_currency(value: float | int) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if value >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def valid_email(value: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9.!#$%&'*+/=?^_`{|}~-]+@[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?)+", value.strip()))


def csv_bytes(rows: Iterable[dict[str, Any]], fields: list[str] | None = None) -> bytes:
    records = list(rows)
    if not records:
        return b""
    selected = fields or list(records[0].keys())
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=selected, extrasaction="ignore")
    writer.writeheader()
    for record in records:
        row = dict(record)
        for key, value in row.items():
            if isinstance(value, (date, datetime)):
                row[key] = value.isoformat()
        writer.writerow(row)
    return buffer.getvalue().encode("utf-8-sig")


def safe_int(value: Any, default: int = 1) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return default


def persist_query(key: str, value: str) -> None:
    import streamlit as st

    if value:
        st.query_params[key] = value
    elif key in st.query_params:
        del st.query_params[key]
