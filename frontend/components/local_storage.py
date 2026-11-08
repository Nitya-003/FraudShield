from __future__ import annotations

import json
from typing import Any

import streamlit as st
from streamlit_local_storage import LocalStorage


_STORAGE_STATE_KEY = "fraudshield_storage_adapter"
_STORAGE_COMPONENT_KEY = "fraudshield_storage_component"


def get_storage() -> LocalStorage:
    storage = st.session_state.get(_STORAGE_STATE_KEY)
    if not isinstance(storage, LocalStorage):
        storage = LocalStorage(key=_STORAGE_COMPONENT_KEY)
        st.session_state[_STORAGE_STATE_KEY] = storage
    return storage


def load_json(key: str, default: Any) -> Any:
    value = get_storage().getItem(key)
    if not isinstance(value, str):
        return default
    try:
        return json.loads(value)
    except (TypeError, ValueError):
        return default


def save_json(key: str, value: Any, component_key: str) -> None:
    get_storage().setItem(itemKey=key, itemValue=json.dumps(value, separators=(",", ":")), key=component_key)
