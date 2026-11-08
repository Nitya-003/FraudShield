from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest

import components.local_storage as storage_module
from components.charts import category_figure, revenue_figure
from data.mock_data import get_activities, get_category_sales, get_revenue_series, get_transactions, get_users
from utils import csv_bytes, valid_email


class FakeLocalStorage:
    def __init__(self, key: str = "storage_init") -> None:
        self.items: dict[str, object] = {}

    def getItem(self, itemKey: str):
        return self.items.get(itemKey)

    def setItem(self, itemKey: str | None = None, itemValue=None, key: str = "set") -> None:
        if itemKey:
            self.items[itemKey] = itemValue


class FraudShieldTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        storage_module.LocalStorage = FakeLocalStorage

    def test_mock_data_covers_dashboard_requirements(self) -> None:
        transactions = get_transactions()
        users = get_users()
        self.assertGreaterEqual(len(transactions), 20)
        self.assertTrue({"Completed", "Pending", "Failed"}.issubset({row["Status"] for row in transactions}))
        self.assertTrue(all({"ID", "Customer", "Date", "Status", "Payment Method", "Amount"}.issubset(row) for row in transactions))
        self.assertTrue({"Admin", "Editor", "Viewer"}.issubset({row["role"] for row in users}))
        self.assertTrue({"Active", "Inactive"}.issubset({row["status"] for row in users}))
        self.assertGreaterEqual(len(get_activities()), 4)
        self.assertGreaterEqual(len(get_category_sales()), 4)

    def test_revenue_periods_and_plotly_charts(self) -> None:
        self.assertEqual(len(get_revenue_series("7D")), 7)
        self.assertEqual(len(get_revenue_series("30D")), 30)
        self.assertEqual(len(get_revenue_series("1Y")), 12)
        self.assertEqual(len(revenue_figure("30D").data), 2)
        self.assertEqual(len(category_figure().data), 1)

    def test_validation_and_csv_export(self) -> None:
        self.assertTrue(valid_email("analyst@example.com"))
        self.assertFalse(valid_email("not-an-email"))
        rows = get_transactions(2)
        content = csv_bytes(rows, ["ID", "Customer", "Date", "Status", "Payment Method", "Amount"]).decode("utf-8-sig")
        self.assertIn("Transaction", content) if "Transaction" in content else self.assertIn("ID,Customer,Date,Status", content)
        self.assertIn(rows[0]["ID"], content)

    def test_all_pages_and_key_actions(self) -> None:
        cases = [
            ("pages/overview.py", "theme_toggle"),
            ("pages/transactions.py", "tx_next"),
            ("pages/users.py", "invite_new_user"),
            ("pages/settings.py", "regenerate_api_key"),
        ]
        for page, button_key in cases:
            with self.subTest(page=page):
                app = AppTest.from_file("app.py", default_timeout=15).switch_page(page).run()
                self.assertEqual([], [str(item.message) for item in app.exception])
                app.button(key=button_key).click().run()
                self.assertEqual([], [str(item.message) for item in app.exception])


if __name__ == "__main__":
    unittest.main()
