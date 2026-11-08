from __future__ import annotations

import base64
import html
from pathlib import Path
from datetime import date, timedelta

import streamlit as st

from components.local_storage import get_storage


LOGO_PATH = Path(__file__).resolve().parents[1] / "public" / "fraudshield-logo.svg"


def _theme_css(theme: str) -> str:
    if theme == "dark":
        colors = {
            "bg": "#0d1724", "surface": "#142235", "surface2": "#1b2b3d", "text": "#e8f0f6",
            "muted": "#93a5b8", "border": "#26384c", "cyan": "#35d2dc", "soft": "#163740",
            "shadow": "0 18px 40px rgba(0,0,0,.18)", "chart": "#132235",
        }
    else:
        colors = {
            "bg": "#f4f7fa", "surface": "#ffffff", "surface2": "#f7f9fc", "text": "#1a2a3d",
            "muted": "#77889b", "border": "#e6ecf1", "cyan": "#16b6c7", "soft": "#e8f8fa",
            "shadow": "0 16px 38px rgba(25,47,70,.06)", "chart": "#ffffff",
        }
    variables = ";".join(f"--fs-{key}:{value}" for key, value in colors.items())
    return f'''<style>
    :root {{{variables};color-scheme:{"dark" if theme == "dark" else "light"};}}
    html, body, [data-testid="stAppViewContainer"], .stApp {{background:var(--fs-bg)!important;color:var(--fs-text)!important;}}
    [data-testid="stHeader"] {{background:transparent!important;}}
    [data-testid="stToolbar"] {{right:1rem;}}
    .block-container {{max-width:1560px;padding:1.15rem 2.1rem 3.5rem;}}
    section[data-testid="stSidebar"] {{background:var(--fs-surface)!important;border-right:1px solid var(--fs-border);}}
    section[data-testid="stSidebar"] > div {{padding-top:1.3rem;}}
    [data-testid="stSidebar"] * {{color:var(--fs-text);}}
    [data-testid="stSidebarNav"] {{display:none;}}
    .brand-lockup {{display:flex;align-items:center;gap:11px;padding:4px 8px 18px;margin-bottom:14px;border-bottom:1px solid var(--fs-border);}}
    .brand-lockup img {{width:38px;height:38px;border-radius:11px;}}
    .brand-name {{font-size:18px;font-weight:760;letter-spacing:-.6px;color:var(--fs-text);}}
    .brand-name span {{color:var(--fs-cyan);}}
    .sidebar-label {{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;color:var(--fs-muted);font-weight:750;margin:14px 8px 8px;}}
    [data-testid="stSidebar"] [data-testid="stPageLink"] a {{border-radius:10px;padding:9px 11px;color:var(--fs-muted);font-size:13px;font-weight:600;}}
    [data-testid="stSidebar"] [data-testid="stPageLink"] a[aria-current="page"] {{background:var(--fs-soft);color:var(--fs-cyan);}}
    [data-testid="stSidebar"] [data-testid="stPageLink"] a:hover {{background:var(--fs-surface2);}}
    [data-testid="stSidebar"] button {{border-radius:10px;}}
    .sidebar-note {{margin:17px 8px 4px;padding:12px;border:1px solid var(--fs-border);border-radius:13px;background:var(--fs-surface2);}}
    .sidebar-note strong {{font-size:12px;display:block;margin-bottom:4px;}}
    .sidebar-note span {{font-size:11px;color:var(--fs-muted);line-height:1.45;display:block;}}
    .sidebar-profile {{margin-top:20px;padding-top:14px;border-top:1px solid var(--fs-border);}}
    .profile-row {{display:flex;align-items:center;gap:10px;padding:6px 7px;}}
    .profile-avatar {{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;background:#e2e9ff;color:#515cb0;font-weight:750;font-size:12px;}}
    .profile-meta strong {{font-size:12px;display:block;}}
    .profile-meta span {{font-size:10px;color:var(--fs-muted);}}
    .notification-count {{display:inline-flex;align-items:center;justify-content:center;min-width:17px;height:17px;padding:0 5px;border-radius:99px;background:#e26352;color:#fff!important;font-size:9px;letter-spacing:0;font-weight:750;margin-left:6px;}}
    .st-key-global-header {{position:sticky;top:0;z-index:100;backdrop-filter:blur(14px);background:color-mix(in srgb,var(--fs-bg) 90%,transparent);border-bottom:1px solid var(--fs-border);padding:.25rem 0 .65rem;margin-bottom:1.35rem;}}
    .st-key-global-header [data-testid="stPopover"] button, .st-key-global-header [data-testid="stBaseButton-secondary"] {{border-color:var(--fs-border);background:var(--fs-surface);color:var(--fs-text);}}
    .topbar-brand {{font-size:12px;color:var(--fs-muted);letter-spacing:.2px;padding-top:8px;}}
    .topbar-brand strong {{color:var(--fs-text);font-size:13px;}}
    .page-heading {{margin:4px 0 24px;}}
    .eyebrow {{font-size:10px;letter-spacing:1.7px;text-transform:uppercase;color:var(--fs-cyan);font-weight:750;margin-bottom:8px;}}
    .page-heading h1 {{font-size:30px;line-height:1.16;letter-spacing:-1.05px;font-weight:750;color:var(--fs-text);margin:0 0 8px;}}
    .page-heading p {{font-size:13px;color:var(--fs-muted);margin:0;line-height:1.55;}}
    .section-heading {{margin:5px 0 13px;}}
    .section-heading h2 {{font-size:15px;font-weight:700;color:var(--fs-text);margin:0;letter-spacing:-.2px;}}
    .section-subtitle {{font-size:11px!important;color:var(--fs-muted)!important;margin:4px 0 0!important;}}
    .metric-card {{background:var(--fs-surface);border:1px solid var(--fs-border);border-radius:16px;padding:18px 18px 14px;box-shadow:var(--shadow);min-height:146px;position:relative;overflow:hidden;}}
    .metric-top {{display:flex;align-items:center;justify-content:space-between;}}
    .metric-label {{color:var(--fs-muted);font-size:12px;font-weight:600;}}
    .metric-icon {{width:29px;height:29px;border-radius:9px;background:color-mix(in srgb,var(--metric-accent) 13%,transparent);color:var(--metric-accent);display:flex;align-items:center;justify-content:center;font-size:15px;font-weight:700;}}
    .metric-value {{font-size:27px;letter-spacing:-.75px;color:var(--fs-text);font-weight:740;margin:13px 0 10px;font-variant-numeric:tabular-nums;}}
    .metric-bottom {{display:flex;align-items:center;gap:6px;position:relative;}}
    .metric-delta {{font-size:10px;font-weight:720;padding:3px 6px;border-radius:6px;}}
    .metric-delta.positive {{color:#1b936b;background:rgba(42,187,133,.12);}}
    .metric-delta.negative {{color:#cc6856;background:rgba(224,112,92,.12);}}
    .metric-caption {{font-size:10px;color:var(--fs-muted);}}
    .sparkline {{width:76px;height:27px;margin-left:auto;overflow:visible;}}
    .chart-card {{background:var(--fs-surface);border:1px solid var(--fs-border);border-radius:16px;padding:18px 18px 10px;box-shadow:var(--shadow);}}
    .chart-card-title {{font-size:14px;color:var(--fs-text);font-weight:700;margin:0 0 2px;}}
    .chart-card-subtitle {{font-size:11px;color:var(--fs-muted);margin:0;}}
    .avatar {{display:inline-flex;align-items:center;justify-content:center;border-radius:50%;font-size:11px;font-weight:750;flex:0 0 auto;}}
    .status-badge {{display:inline-flex;align-items:center;gap:6px;border-radius:999px;padding:4px 9px;font-size:10px;font-weight:700;white-space:nowrap;}}
    .status-badge i {{width:5px;height:5px;border-radius:50%;background:currentColor;}}
    .status-badge.success {{color:#16845f;background:rgba(44,184,131,.12);}}
    .status-badge.warning {{color:#a87014;background:rgba(231,171,65,.14);}}
    .status-badge.danger {{color:#c45b4b;background:rgba(224,106,88,.12);}}
    .status-badge.neutral {{color:var(--fs-muted);background:var(--fs-surface2);}}
    .activity-row {{display:flex;align-items:flex-start;gap:11px;padding:13px 0;border-bottom:1px solid var(--fs-border);}}
    .activity-row:last-child {{border-bottom:0;}}
    .activity-copy {{flex:1;min-width:0;font-size:12px;line-height:1.5;color:var(--fs-text);}}
    .activity-copy strong {{font-weight:680;}}
    .activity-target {{display:block;color:var(--fs-muted);font-size:10px;margin-top:2px;}}
    .activity-meta {{display:flex;flex-direction:column;align-items:flex-end;gap:6px;white-space:nowrap;}}
    .activity-time {{font-size:10px;color:var(--fs-muted);}}
    .activity-tag {{font-size:9px;font-weight:700;padding:3px 7px;border-radius:6px;background:var(--fs-soft);color:var(--fs-cyan);}}
    .empty-state {{min-height:170px;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:8px;color:var(--fs-muted);text-align:center;border:1px dashed var(--fs-border);border-radius:14px;padding:22px;}}
    .empty-mark {{width:38px;height:38px;border-radius:12px;background:var(--fs-surface2);display:flex;align-items:center;justify-content:center;font-size:20px;color:var(--fs-cyan);}}
    .empty-state strong {{font-size:13px;color:var(--fs-text);}}
    .empty-state span {{font-size:11px;max-width:340px;line-height:1.5;}}
    .skeleton-card {{height:145px;border:1px solid var(--fs-border);border-radius:16px;background:var(--fs-surface);padding:20px;}}
    .skeleton-line {{height:11px;border-radius:10px;margin-bottom:17px;background:linear-gradient(90deg,var(--fs-surface2),color-mix(in srgb,var(--fs-cyan) 11%,var(--fs-surface2)),var(--fs-surface2));background-size:220% 100%;animation:shimmer 1.15s ease-in-out infinite;}}
    .skeleton-line.short {{width:46%;}}
    .skeleton-line.wide {{width:72%;height:24px;margin-top:22px;}}
    .skeleton-line.medium {{width:60%;height:9px;}}
    @keyframes shimmer {{to {{background-position:-220% 0;}}}}
    .table-container {{background:var(--fs-surface);border:1px solid var(--fs-border);border-radius:14px;overflow:hidden;}}
    [data-testid="stDataFrame"] {{border:1px solid var(--fs-border);border-radius:12px;overflow:hidden;}}
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-testid="stSelectbox"] div[data-baseweb="select"] > div, [data-testid="stMultiSelect"] div[data-baseweb="select"] > div, [data-testid="stDateInput"] input {{border-radius:9px!important;background:var(--fs-surface)!important;}}
    [data-testid="stTextInput"] input {{color:var(--fs-text)!important;}}
    [data-testid="stBaseButton-primary"] {{background:var(--fs-cyan)!important;border-color:var(--fs-cyan)!important;color:#082b34!important;font-weight:700!important;border-radius:9px!important;}}
    [data-testid="stBaseButton-secondary"] {{border-color:var(--fs-border)!important;color:var(--fs-text)!important;border-radius:9px!important;}}
    [data-testid="stForm"] {{border:1px solid var(--fs-border);border-radius:14px;padding:16px;background:var(--fs-surface);}}
    [data-testid="stTabs"] button {{font-weight:650;}}
    [data-testid="stVerticalBlock"] {{gap:0.85rem;}}
    hr {{border-color:var(--fs-border)!important;}}
    .detail-panel {{background:var(--fs-surface);border:1px solid var(--fs-border);border-radius:15px;padding:18px;box-shadow:var(--shadow);}}
    .st-key-transaction-detail-sheet {{position:sticky;top:92px;animation:transactionSheetIn .24s ease-out both;}}
    @keyframes transactionSheetIn {{from {{opacity:0;transform:translateX(14px);}}to {{opacity:1;transform:translateX(0);}}}}
    .detail-label {{font-size:10px;color:var(--fs-muted);text-transform:uppercase;letter-spacing:1.1px;font-weight:720;margin:13px 0 4px;}}
    .detail-value {{font-size:13px;color:var(--fs-text);font-weight:620;}}
    .user-card {{border:1px solid var(--fs-border);background:var(--fs-surface);border-radius:15px;padding:16px;box-shadow:var(--shadow);min-height:188px;}}
    .user-card-head {{display:flex;align-items:center;gap:10px;margin-bottom:12px;}}
    .user-card-name {{font-size:13px;font-weight:720;color:var(--fs-text);}}
    .user-card-email {{font-size:10px;color:var(--fs-muted);overflow-wrap:anywhere;}}
    .user-card-meta {{display:flex;justify-content:space-between;align-items:center;padding-top:12px;border-top:1px solid var(--fs-border);margin-top:12px;font-size:10px;color:var(--fs-muted);}}
    .setting-card {{border:1px solid var(--fs-border);border-radius:15px;padding:20px;background:var(--fs-surface);box-shadow:var(--shadow);}}
    .setting-card h3 {{font-size:14px;margin:0 0 5px;color:var(--fs-text);}}
    .setting-card p {{font-size:11px;margin:0 0 14px;color:var(--fs-muted);}}
    .api-key {{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;border:1px solid var(--fs-border);background:var(--fs-surface2);border-radius:10px;padding:13px 15px;color:var(--fs-text);letter-spacing:1px;font-size:13px;}}
    .callout {{border:1px solid color-mix(in srgb,var(--fs-cyan) 28%,var(--fs-border));background:var(--fs-soft);border-radius:11px;padding:11px 13px;color:var(--fs-text);font-size:11px;line-height:1.5;}}
    .small-muted {{font-size:10px;color:var(--fs-muted);}}
    @media (max-width: 900px) {{
      .block-container {{padding:1rem 1rem 2.5rem;}}
      .page-heading h1 {{font-size:25px;}}
      .st-key-global-header {{position:relative;}}
      .metric-card {{padding:14px;}}
    }}
    @media (prefers-reduced-motion: reduce) {{*,*::before,*::after {{animation-duration:.01ms!important;transition-duration:.01ms!important;}}}}
    </style>'''


def _load_theme(storage) -> str:
    query_theme = st.query_params.get("theme")
    saved_theme = storage.getItem("fraudshield_theme")
    candidate = query_theme or saved_theme or "light"
    return candidate if candidate in {"light", "dark"} else "light"


def _change_theme(storage) -> None:
    theme = "dark" if st.session_state.get("theme", "light") == "light" else "light"
    st.session_state["theme"] = theme
    st.query_params["theme"] = theme
    storage.setItem(itemKey="fraudshield_theme", itemValue=theme, key="fraudshield_set_theme")
    st.rerun()


def render_shell(active_page: str) -> str:
    storage = get_storage()
    if "theme" not in st.session_state:
        st.session_state["theme"] = _load_theme(storage)
    theme = st.session_state["theme"]
    st.markdown(_theme_css(theme), unsafe_allow_html=True)

    try:
        logo = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
        logo_src = f"data:image/svg+xml;base64,{logo}"
    except OSError:
        logo_src = ""

    with st.sidebar:
        st.markdown(
            f'<div class="brand-lockup"><img src="{logo_src}" alt="FraudShield shield mark"><div class="brand-name">Fraud<span>Shield</span></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-label">Workspace</div>', unsafe_allow_html=True)
        st.page_link("pages/overview.py", label="Executive Overview", icon=":material/space_dashboard:")
        st.page_link("pages/transactions.py", label="Transactions & Analytics", icon=":material/receipt_long:")
        st.page_link("pages/users.py", label="User Management", icon=":material/group:")
        st.page_link("pages/settings.py", label="System Settings", icon=":material/settings:")
        st.markdown('<div class="sidebar-label">Monitoring <span class="notification-count">3</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-note"><strong>3 items need attention</strong><span>Two transactions and one access review are waiting in the demo queue.</span></div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="sidebar-profile"><div class="profile-row"><div class="profile-avatar">SC</div><div class="profile-meta"><strong>Sarah Chen</strong><span>Workspace admin</span></div></div></div>',
            unsafe_allow_html=True,
        )
        with st.popover("Profile & workspace", use_container_width=True):
            st.markdown("**Sarah Chen**  \nWorkspace administrator")
            st.caption("FraudShield demo workspace")
            st.page_link("pages/settings.py", label="Open workspace settings", icon=":material/tune:")
            st.caption("All records shown here are simulated.")

    with st.container(key="global-header"):
        c_brand, c_search, c_date, c_alerts, c_actions, c_theme = st.columns([1.05, 2.25, 1.35, 1.0, 1.0, 1.0], vertical_alignment="center")
        with c_brand:
            st.markdown('<div class="topbar-brand"><strong>FraudShield</strong><br>Executive workspace</div>', unsafe_allow_html=True)
        with c_search:
            with st.popover("Search  ·  ⌘ K", use_container_width=True):
                search = st.text_input("Search FraudShield", placeholder="Pages, transactions, users…", label_visibility="collapsed", key="global_search")
                quick_links = [
                    ("Overview", "pages/overview.py", "overview"),
                    ("Transactions", "pages/transactions.py", "transaction"),
                    ("Users", "pages/users.py", "user"),
                    ("Settings", "pages/settings.py", "setting"),
                ]
                matches = [item for item in quick_links if not search or search.lower() in item[0].lower() or search.lower() in item[2]]
                for label, page, _ in matches:
                    st.page_link(page, label=label)
                if search and not matches:
                    st.caption("No matching page. Try a name, transaction ID, or setting.")
        with c_date:
            range_options = ["Last 30 days", "Last 7 days", "This year", "Custom range"]
            saved_range = st.query_params.get("range", "Last 30 days")
            range_index = range_options.index(saved_range) if saved_range in range_options else 0
            date_range = st.selectbox("Date range", range_options, index=range_index, key="global_date_range", help="Filters transaction dates. The Overview chart has separate 7D, 30D, and 1Y controls.", label_visibility="collapsed")
            st.query_params["range"] = date_range
            today = date.today()
            if date_range == "Last 7 days":
                range_start, range_end = today - timedelta(days=6), today
            elif date_range == "This year":
                range_start, range_end = date(today.year, 1, 1), today
            elif date_range == "Custom range":
                try:
                    default_start = date.fromisoformat(st.query_params.get("range_start", ""))
                    default_end = date.fromisoformat(st.query_params.get("range_end", ""))
                except ValueError:
                    default_start, default_end = today - timedelta(days=29), today
                chosen_dates = st.date_input("Choose dates", value=(default_start, default_end), key="global_custom_date_range", label_visibility="collapsed")
                if isinstance(chosen_dates, tuple) and len(chosen_dates) == 2:
                    range_start, range_end = chosen_dates
                else:
                    range_start, range_end = default_start, default_end
                if range_end < range_start:
                    range_start, range_end = range_end, range_start
                st.query_params["range_start"] = range_start.isoformat()
                st.query_params["range_end"] = range_end.isoformat()
            else:
                range_start, range_end = today - timedelta(days=29), today
                for range_key in ("range_start", "range_end"):
                    if range_key in st.query_params:
                        del st.query_params[range_key]
            st.session_state["global_date_bounds"] = (range_start, range_end)
        with c_alerts:
            with st.popover("Alerts · 3", use_container_width=True):
                st.markdown("**Needs your attention**")
                st.caption("2 transactions are waiting for review.")
                st.caption("1 workspace access review is due today.")
                st.page_link("pages/transactions.py", label="Review transactions")
        with c_actions:
            with st.popover("Quick actions", use_container_width=True):
                st.markdown("**Jump into a workflow**")
                st.page_link("pages/transactions.py", label="View transactions", icon=":material/receipt_long:")
                st.page_link("pages/users.py", label="Manage users", icon=":material/person_add:")
                st.page_link("pages/settings.py", label="Open settings", icon=":material/tune:")
        with c_theme:
            theme_label = "Dark" if theme == "light" else "Light"
            if st.button(theme_label, icon=":material/dark_mode:" if theme == "light" else ":material/light_mode:", help=f"Switch to {theme_label.lower()} mode and save your preference", use_container_width=True, key="theme_toggle"):
                _change_theme(storage)

    return date_range
