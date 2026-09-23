# SPnPOP - Sales Prediction and Purchase Order Prediction

Item-level (product-wise) demand forecasting for a Sri Lankan retail
hardware business, trained on live ERP data via `n8n.bluelotusx.co`
and `reports.bl360x.com`. Trains 7 direct-horizon (Day+1..Day+7)
XGBoost Tweedie models per item, tuned via random search scored on
WAPE, with Sri Lankan public holiday calendar features included.

## Setup

1. Create and activate a virtual environment:
   ```powershell
   python -m venv .venv
   & .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Allow local scripts to run (one-time, if not already done):
   ```powershell
   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
   ```

## Files

| File | Purpose |
|---|---|
| `model.py` | The ML pipeline - feature engineering, training, tuning, prediction. Imported, not run directly. |
| `erp_client.py` | Shared ERP login + data-fetch logic. Imported by both `auth_client.py` and `api_server.py`. |
| `auth_client.py` | Manual one-off terminal script: fetch live data, train, print forecast. Run with `python auth_client.py`. |
| `api_server.py` | The always-on server the dashboard talks to. Run with `uvicorn` (see below). |
| `start_server.ps1` | Convenience script: sets credentials as env vars and starts the server in one step. |
| `requirements.txt` | Python package dependencies. |

## Running the server (for the dashboard)

Every time, from the project folder with the venv active:
```powershell
.\start_server.ps1
```
This sets the ERP credentials for that session and starts `uvicorn` on port 8000. Leave the terminal open - closing it stops the server.

If you'd rather set credentials permanently instead of every session, see the "Credentials" section below - though `start_server.ps1` is the more reliable option on this machine.

## API Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/train/{client_id}?grain=item&n_iters=20` | Start training in the background |
| GET | `/train/{client_id}/status` | Poll training status and results |
| GET | `/forecast/{client_id}?grain=item&recency_days=90` | Get the live Day+1..7 forecast per item |

`grain` is `item` (product-wise) or `category`. `recency_days` excludes items with no recent activity from the forecast (they stay trained, just not served).

## Credentials

The pipeline needs three environment variables to authenticate against the ERP:
- `SPNPOP_USER`
- `SPNPOP_PASSWORD`
- `SPNPOP_CCD`

`start_server.ps1` sets these automatically for that session. To set them permanently instead (once, requires closing and reopening PowerShell afterward):
```powershell
[System.Environment]::SetEnvironmentVariable("SPNPOP_USER", "<value>", "User")
[System.Environment]::SetEnvironmentVariable("SPNPOP_PASSWORD", "<value>", "User")
[System.Environment]::SetEnvironmentVariable("SPNPOP_CCD", "<value>", "User")
```

## Notes on the data

Out of the full item catalog, only items with sufficient transaction history (≥30 non-zero days, configurable via `min_active_days`) are trained on. Of those, only items with a recent transaction (within `recency_days`, default 90) are included in the live forecast - older-trained items stay on disk in case they become active again.
