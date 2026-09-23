"""
SPnPOP - Shared ERP auth + data fetch
===========================================================

The ONE place login + data-pull logic lives. Both auth_client.py (manual
terminal runs) and api_server.py (the always-on dashboard server) import
from here instead of each defining their own copy - so a change to the
auth flow only needs to happen once.

Credentials come from environment variables:
    SPNPOP_USER, SPNPOP_PASSWORD, SPNPOP_CCD
See README/chat notes for how to set these on Windows.
"""

import os

import requests

AUTH_URL = "https://n8n.bluelotusx.co/webhook/b3602-claudeAuth"
DATA_URL = "https://reports.bl360x.com/AIConnectorMCP/api/erp/GetSalesDataForPrediction?FrmDt=2020/07/01&ToDt=2026/09/15"

ERP_USER = os.environ.get("SPNPOP_USER", "username")
ERP_PASSWORD = os.environ.get("password")
ERP_CCD = os.environ.get("SPNPOP_CCD", "dc")


def get_token(verbose: bool = False) -> str:
    if not ERP_PASSWORD:
        raise RuntimeError(
            "SPNPOP_PASSWORD environment variable is not set. "
            "Set SPNPOP_USER / SPNPOP_PASSWORD / SPNPOP_CCD before running."
        )

    payload = {"userID": ERP_USER, "password": ERP_PASSWORD, "CCd": ERP_CCD}
    response = requests.post(AUTH_URL, json=payload, timeout=30)

    if verbose:
        print("Status code:", response.status_code)
        print("Response body:", response.text)

    response.raise_for_status()
    return response.json()["token"]


def get_training_data(token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(DATA_URL, headers=headers, timeout=60)
    response.raise_for_status()
    return response.json()


def fetch_live_payload(verbose: bool = False) -> dict:
    """Convenience wrapper: auth + fetch in one call."""
    token = get_token(verbose=verbose)
    return get_training_data(token)
