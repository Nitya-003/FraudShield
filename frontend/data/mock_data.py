from __future__ import annotations

from datetime import date, datetime, timedelta
import random
from typing import Any


_SEED = 731
_CUSTOMERS = [
    ("Maya Chen", "maya.chen@acme.io", "Acme Labs"),
    ("Jordan Ellis", "jordan.ellis@northstar.co", "Northstar Co."),
    ("Priya Shah", "priya.shah@brightlane.com", "Brightlane"),
    ("Ethan Brooks", "ethan.brooks@fieldwork.ai", "Fieldwork AI"),
    ("Sofia Martinez", "sofia.martinez@commonthread.com", "Common Thread"),
    ("Noah Williams", "noah.williams@evergreen.dev", "Evergreen"),
    ("Ava Robinson", "ava.robinson@orbitpay.com", "OrbitPay"),
    ("Leo Kim", "leo.kim@atlascloud.io", "Atlas Cloud"),
    ("Isabella Turner", "isabella.turner@monarch.app", "Monarch"),
    ("Owen Patel", "owen.patel@lumenworks.com", "Lumen Works"),
    ("Amara Johnson", "amara.johnson@silvershore.net", "Silvershore"),
    ("Lucas Nguyen", "lucas.nguyen@papertrail.io", "Papertrail"),
]

_USER_NAMES = [
    ("Sarah Chen", "sarah.chen@fraudshield.demo"),
    ("Marcus Reed", "marcus.reed@fraudshield.demo"),
    ("Priya Kapoor", "priya.kapoor@fraudshield.demo"),
    ("Daniel Brooks", "daniel.brooks@fraudshield.demo"),
    ("Amara Okafor", "amara.okafor@fraudshield.demo"),
    ("Theo Martin", "theo.martin@fraudshield.demo"),
    ("Elena Petrova", "elena.petrova@fraudshield.demo"),
    ("Ravi Menon", "ravi.menon@fraudshield.demo"),
    ("Grace Kim", "grace.kim@fraudshield.demo"),
    ("Oliver Scott", "oliver.scott@fraudshield.demo"),
    ("Nina Alvarez", "nina.alvarez@fraudshield.demo"),
    ("Isaac Morgan", "isaac.morgan@fraudshield.demo"),
    ("Fatima Hassan", "fatima.hassan@fraudshield.demo"),
    ("Eli Thompson", "eli.thompson@fraudshield.demo"),
    ("Zara Ali", "zara.ali@fraudshield.demo"),
    ("Milo Bennett", "milo.bennett@fraudshield.demo"),
    ("Lena Fischer", "lena.fischer@fraudshield.demo"),
    ("Kai Wilson", "kai.wilson@fraudshield.demo"),
]


def initials(name: str) -> str:
    return "".join(part[0] for part in name.split()[:2]).upper()


def get_transactions(count: int = 42) -> list[dict[str, Any]]:
    rng = random.Random(_SEED)
    today = date.today()
    statuses = ["Completed"] * 27 + ["Pending"] * 9 + ["Failed"] * 6
    payment_methods = ["Visa ···· 4242", "Mastercard ···· 8211", "Amex ···· 1008", "ACH transfer"]
    products = ["Risk Monitor", "Identity API", "Team plan", "Review credits", "Enterprise suite"]
    records: list[dict[str, Any]] = []
    for index in range(count):
        customer = _CUSTOMERS[index % len(_CUSTOMERS)]
        tx_date = today - timedelta(days=(index * 3 + rng.randint(0, 4)) % 30)
        status = statuses[(index * 7 + 3) % len(statuses)]
        amount = round(rng.uniform(95, 12400), 2)
        records.append(
            {
                "ID": f"TXN-{today.year}-{10482 + index:05d}",
                "Customer": customer[0],
                "Email": customer[1],
                "Company": customer[2],
                "Date": tx_date,
                "Status": status,
                "Payment Method": payment_methods[index % len(payment_methods)],
                "Amount": amount,
                "Product": products[index % len(products)],
                "Reference": f"FS-{rng.randint(120000, 990000)}",
                "Risk score": rng.randint(3, 96),
                "Invoice": f"INV-{today.year}-{3400 + index:04d}",
            }
        )
    return records


def get_users() -> list[dict[str, Any]]:
    roles = ["Admin", "Editor", "Viewer"]
    activity = ["2 min ago", "18 min ago", "Today, 9:42 AM", "Today, 8:16 AM", "Yesterday", "2 days ago", "4 days ago"]
    return [
        {
            "id": f"USR-{1001 + index}",
            "name": name,
            "email": email,
            "role": roles[(index * 5 + 1) % len(roles)],
            "status": "Inactive" if index in {5, 12, 16} else "Active",
            "last_active": activity[index % len(activity)],
            "department": ["Operations", "Risk", "Finance", "Product"][index % 4],
            "initials": initials(name),
            "joined": (date.today() - timedelta(days=32 + index * 9)).isoformat(),
        }
        for index, (name, email) in enumerate(_USER_NAMES)
    ]


def get_activities() -> list[dict[str, str]]:
    return [
        {"name": "Sarah Chen", "initials": "SC", "action": "approved a high-value transaction", "target": "TXN-10496", "time": "4 min ago", "tag": "Review", "tone": "cyan"},
        {"name": "Marcus Reed", "initials": "MR", "action": "invited a new team member", "target": "lena.fischer@fraudshield.demo", "time": "18 min ago", "tag": "Team", "tone": "violet"},
        {"name": "Priya Kapoor", "initials": "PK", "action": "resolved an account alert", "target": "AC-882104", "time": "42 min ago", "tag": "Resolved", "tone": "green"},
        {"name": "Daniel Brooks", "initials": "DB", "action": "exported the monthly report", "target": "October 2026", "time": "1 hr ago", "tag": "Report", "tone": "amber"},
        {"name": "Amara Okafor", "initials": "AO", "action": "flagged an unusual payment pattern", "target": "TXN-10487", "time": "2 hrs ago", "tag": "Flagged", "tone": "coral"},
        {"name": "Theo Martin", "initials": "TM", "action": "updated the review threshold", "target": "Policy · North America", "time": "3 hrs ago", "tag": "Settings", "tone": "slate"},
    ]


def get_category_sales() -> list[dict[str, Any]]:
    return [
        {"category": "Identity API", "sales": 48600},
        {"category": "Risk Monitor", "sales": 39200},
        {"category": "Team plan", "sales": 28750},
        {"category": "Enterprise", "sales": 21400},
        {"category": "Review credits", "sales": 15320},
    ]


def get_revenue_series(period: str = "30D") -> list[dict[str, Any]]:
    rng = random.Random(_SEED + len(period))
    today = date.today()
    rows: list[dict[str, Any]] = []
    if period == "7D":
        for offset in range(6, -1, -1):
            point = today - timedelta(days=offset)
            base = 8800 + (6 - offset) * 360
            revenue = base + rng.randint(-900, 1100)
            expenses = revenue * 0.58 + rng.randint(-400, 400)
            rows.append({"date": point.strftime("%b %d"), "revenue": revenue, "expenses": expenses})
    elif period == "1Y":
        for offset in range(11, -1, -1):
            month = today.replace(day=1) - timedelta(days=offset * 30)
            base = 119000 + (11 - offset) * 8700
            revenue = base + rng.randint(-8500, 9000)
            expenses = revenue * 0.61 + rng.randint(-5000, 5000)
            rows.append({"date": month.strftime("%b %y"), "revenue": revenue, "expenses": expenses})
    else:
        for offset in range(29, -1, -1):
            point = today - timedelta(days=offset)
            base = 7600 + (29 - offset) * 175
            revenue = base + rng.randint(-1250, 1450)
            expenses = revenue * 0.59 + rng.randint(-650, 650)
            rows.append({"date": point.strftime("%b %d"), "revenue": revenue, "expenses": expenses})
    return rows


def get_metric_data() -> list[dict[str, Any]]:
    return [
        {"title": "Total Revenue", "value": "$284,219", "delta": "+12.8%", "caption": "vs. last month", "trend": "up", "spark": [17, 21, 19, 24, 20, 28, 27, 34, 31, 39, 42, 47], "icon": "↗", "color": "cyan"},
        {"title": "Active Users", "value": "24,892", "delta": "+8.2%", "caption": "vs. last month", "trend": "up", "spark": [16, 18, 17, 23, 22, 25, 24, 32, 31, 36, 39, 43], "icon": "◉", "color": "violet"},
        {"title": "Conversion Rate", "value": "3.84%", "delta": "+0.6%", "caption": "vs. last month", "trend": "up", "spark": [23, 22, 26, 25, 29, 26, 31, 33, 32, 38, 37, 42], "icon": "◌", "color": "green"},
        {"title": "Open Issues", "value": "18", "delta": "−6.4%", "caption": "vs. last month", "trend": "down_good", "spark": [44, 41, 45, 38, 42, 34, 37, 30, 33, 26, 24, 20], "icon": "⚑", "color": "amber"},
    ]
