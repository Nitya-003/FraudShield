from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ml.config import BANKING_COVERAGE_NOTE, CHANNELS
import ml.channels.card as card_channel
import ml.channels.transfer_wallet as transfer_channel
import ml.channels.account as account_channel
import ml.channels.upi as upi_channel

CHANNEL_ALIASES = {
    
    "card": "card",
    "credit_card": "card",
    "creditcard": "card",
    "debit_card": "card",
    "cards": "card",
    "card_payment": "card",
    "card_payments": "card",

    "transfer_wallet": "transfer_wallet",
    "transfer": "transfer_wallet",
    "transfers": "transfer_wallet",
    "wallet": "transfer_wallet",
    "wallets": "transfer_wallet",
    "paysim": "transfer_wallet",
    "bank_transfer": "transfer_wallet",
    "bank_transfers": "transfer_wallet",
    "money_transfer": "transfer_wallet",
    "online_transfer": "transfer_wallet",
    "digital_wallet": "transfer_wallet",

    "account": "account",
    "bank_account": "account",
    "bank_accounts": "account",
    "baf": "account",
    "account_opening": "account",
    "bank_account_fraud": "account",
    "accounts": "account",

    "upi": "upi",
    "bhim": "upi",
    "upi_payment": "upi",
    "upi_transactions": "upi",
    "upi_synthetic": "upi",
    "upi_simulated": "upi",
    "upi_p2p": "upi",
    "upi_transfer": "upi",
    "p2p": "upi",
}


def resolve_channel(channel: str) -> str:
    key = str(channel).strip().lower().replace("-", "_").replace(" ", "_")
    if key in CHANNEL_ALIASES:
        return CHANNEL_ALIASES[key]
    raise ValueError(
        f"Unknown channel: '{channel}'. Supported channels: {list(CHANNELS.keys())}.\n"
        f"Note: {BANKING_COVERAGE_NOTE}"
    )


def get_channel_registry() -> dict[str, dict[str, Any]]:
    return CHANNELS


def get_channel_info(channel: str) -> dict[str, Any]:
    canonical = resolve_channel(channel)
    return CHANNELS[canonical]


def predict(
    channel: str,
    payload: dict[str, Any],
    top_k: int = 3,
    history_store: Any | None = None,
    commit_to_store: bool = False,
    use_group_fairness: bool = False,
) -> dict[str, Any]:

    t0 = time.perf_counter()
    canonical = resolve_channel(channel)

    if canonical == "card":
        res = card_channel.predict(payload, top_k=top_k)
    elif canonical == "transfer_wallet":
        res = transfer_channel.predict(
            payload,
            top_k=top_k,
            history_store=history_store,
            commit_to_store=commit_to_store,
        )
    elif canonical == "account":
        res = account_channel.predict(payload, top_k=top_k, use_group_fairness=use_group_fairness)
    elif canonical == "upi":
        res = upi_channel.predict(payload, top_k=top_k)
    else:
        raise ValueError(f"Unhandled channel: {canonical}")

    latency = (time.perf_counter() - t0) * 1000
    res["latency_ms"] = round(res.get("latency_ms", latency), 2)
    res["channel"] = canonical
    res["data_source"] = CHANNELS[canonical]["data_source"]
    return res


class FraudRouter:

    @staticmethod
    def predict(
        channel: str,
        payload: dict[str, Any],
        top_k: int = 3,
        history_store: Any | None = None,
        commit_to_store: bool = False,
    ) -> dict[str, Any]:
        return predict(
            channel,
            payload,
            top_k=top_k,
            history_store=history_store,
            commit_to_store=commit_to_store,
        )

    @staticmethod
    def resolve_channel(channel: str) -> str:
        return resolve_channel(channel)

    @staticmethod
    def get_channel_registry() -> dict[str, dict[str, Any]]:
        return get_channel_registry()

    @staticmethod
    def get_channel_info(channel: str) -> dict[str, Any]:
        return get_channel_info(channel)
