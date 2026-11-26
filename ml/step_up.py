from __future__ import annotations

import random
from datetime import datetime
from typing import Any


def format_currency_inr(amount: float) -> str:
    try:
        amt = float(amount)
        return f"₹{amt:,.2f}"
    except (ValueError, TypeError):
        return f"₹{amount}"


def extract_plain_reasons(
    channel: str,
    payload: dict[str, Any],
    score_result: dict[str, Any] | None = None,
) -> tuple[str, str]:

    reasons_en: list[str] = []
    reasons_hi: list[str] = []

    amount = float(payload.get("amount", payload.get("Amount", 0.0)))
    old_bal = float(payload.get("oldbalanceOrg", payload.get("current_balance", 0.0)))

    if old_bal > 0 and amount >= 0.7 * old_bal:
        reasons_en.append("This transfer amount is a large fraction of your available account balance.")
        reasons_hi.append("यह लेन-देन राशि आपके उपलब्ध बैंक खाते की शेष राशि का एक बड़ा हिस्सा है।")
    elif amount > 50000.0:
        reasons_en.append("This payment amount is higher than your usual daily transfer range.")
        reasons_hi.append("यह भुगतान राशि आपके सामान्य दैनिक लेन-देन से अधिक है।")

    tx_type = str(payload.get("type", payload.get("transaction_type", ""))).upper()
    if channel == "upi":
        if "COLLECT" in tx_type or payload.get("is_collect_request", False):
            reasons_en.append("This is an inbound collect request initiated by a third-party payee.")
            reasons_hi.append("यह किसी तीसरे पक्ष के प्राप्तकर्ता द्वारा शुरू किया गया कलेक्ट-अनुरोध है।")
        if payload.get("is_payer_off_hours") or payload.get("is_night"):
            reasons_en.append("This payment was initiated during unusual late-night hours.")
            reasons_hi.append("यह भुगतान असामान्य देर रात के समय शुरू किया गया है।")
    elif channel == "card":
        if payload.get("is_foreign_country") or payload.get("foreign_transaction"):
            reasons_en.append("This card charge originated from an unfamiliar merchant location.")
            reasons_hi.append("यह कार्ड शुल्क एक अपरिचित व्यापारी स्थान से आया है।")
    elif channel == "account":
        reasons_en.append("Additional identity verification is required to complete this application.")
        reasons_hi.append("इस आवेदन को पूरा करने के लिए अतिरिक्त पहचान सत्यापन की आवश्यकता है।")

    if not reasons_en and score_result and score_result.get("top_reasons"):
        top_r = score_result["top_reasons"][0]
        reasons_en.append(f"Security check triggered by: {top_r}.")
        reasons_hi.append(f"सुरक्षा जांच कारण: {top_r}।")

    if not reasons_en:
        reasons_en.append("Routine security verification required for this transaction.")
        reasons_hi.append("इस लेन-देन के लिए सामान्य सुरक्षा सत्यापन आवश्यक है।")

    return (" ".join(reasons_en), " ".join(reasons_hi))


def generate_step_up_message(
    channel: str,
    payload: dict[str, Any],
    score_result: dict[str, Any] | None = None,
    otp_code: str | None = None,
) -> dict[str, Any]:

    if otp_code is None:
        otp_val = random.randint(100000, 999999)
        otp_code = f"{str(otp_val)[:3]}-{str(otp_val)[3:]}"

    amount = float(payload.get("amount", payload.get("Amount", 0.0)))
    amt_fmt = format_currency_inr(amount)

    counterparty = (
        payload.get("nameDest")
        or payload.get("payee_vpa")
        or payload.get("merchant_name")
        or payload.get("beneficiary_account")
        or "Beneficiary"
    )

    tx_type = str(payload.get("type", payload.get("transaction_type", ""))).upper()
    is_collect = (
        channel == "upi"
        and (
            "COLLECT" in tx_type
            or payload.get("is_collect_request", False)
            or payload.get("payment_mode") == "COLLECT"
        )
    )

    reason_en, reason_hi = extract_plain_reasons(channel, payload, score_result)

    warning_en = ""
    if channel == "upi":
        if is_collect:
            warning_en = (
                "\n\nCRITICAL WARNING: You are SENDING money, NOT receiving money. "
                f"Entering your UPI PIN will DEDUCT {amt_fmt} from your bank account. "
                "Do NOT enter your PIN if someone told you that you are receiving a prize, refund, or payment."
            )
        else:
            warning_en = (
                f"\n\nSECURITY REMINDER: Entering your UPI PIN authorizes a payment of {amt_fmt} "
                f"to {counterparty}. Never share your PIN or OTP with anyone."
            )

    msg_en = (
        f"[FraudShield Secure Verification]\n"
        f"Your one-time passcode for authorizing payment of {amt_fmt} to {counterparty} is {otp_code}. "
        f"Valid for 5 minutes.\n"
        f"Why we ask: {reason_en}"
        f"{warning_en}"
    )

    warning_hi = ""
    if channel == "upi":
        if is_collect:
            warning_hi = (
                "\n\nमहत्वपूर्ण चेतावनी: आप पैसे भेज रहे हैं, प्राप्त नहीं कर रहे हैं। "
                f"अपना UPI पिन दर्ज करने पर आपके बैंक खाते से {amt_fmt} कट जाएंगे। "
                "यदि किसी ने आपसे कहा है कि आप इनाम, रिफंड या भुगतान प्राप्त कर रहे हैं, तो अपना पिन दर्ज न करें।"
            )
        else:
            warning_hi = (
                f"\n\nसुरक्षा सूचना: अपना UPI पिन दर्ज करने पर {counterparty} को {amt_fmt} का भुगतान स्वीकृत होगा। "
                "अपना पिन या ओटीपी किसी के साथ कभी साझा न करें।"
            )

    msg_hi = (
        f"[फ्रॉडशील्ड सुरक्षित सत्यापन]\n"
        f"{counterparty} को {amt_fmt} के भुगतान को अधिकृत करने के लिए आपका वन-टाइम पासवर्ड (ओटीपी) {otp_code} है। "
        f"यह 5 मिनट के लिए मान्य है।\n"
        f"सत्यापन का कारण: {reason_hi}"
        f"{warning_hi}"
    )

    return {
        "channel": channel,
        "otp_code": otp_code,
        "amount": amount,
        "amount_formatted": amt_fmt,
        "counterparty": str(counterparty),
        "is_collect_request": is_collect,
        "has_scam_warning": bool(warning_en),
        "reason_plain_english": reason_en,
        "reason_plain_hindi": reason_hi,
        "english_message": msg_en.strip(),
        "hindi_message": msg_hi.strip(),
        "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }

