# Obtain the official Home Credit data

Analysis is paused until the real files are present. Do not paste API secrets into chat.

## Browser method (no Kaggle CLI needed)

1. Sign in to https://www.kaggle.com/competitions/home-credit-default-risk/data
2. Join the competition / accept its rules if prompted. Read the applicable data-use terms.
3. Choose **Download All**. Keep the official competition archive, commonly named
   `home-credit-default-risk.zip`.
4. From this project directory, import it with your actual download path:

```powershell
python scripts/download_data.py --archive "$env:USERPROFILE\Downloads\home-credit-default-risk.zip"
```

Alternatively extract the CSV files directly into `data/raw`, then run:

```powershell
python scripts/download_data.py --check-only
```

## Kaggle CLI method

Create a local environment and install only the download dependency first:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install kaggle
```

Authenticate with Kaggle's supported browser flow:

```powershell
.\.venv\Scripts\kaggle.exe auth login
```

If OAuth is unavailable, use https://www.kaggle.com/settings/api to create a token,
store it in `%USERPROFILE%\.kaggle\access_token`, or use the documented legacy
`%USERPROFILE%\.kaggle\kaggle.json` method. Keep credentials outside this repository.
Accept competition rules in your browser, then run:

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py
```

The script invokes `kaggle competitions download home-credit-default-risk -p data/raw`
using an absolute output path, imports only the required CSVs, and checks their headers.
An authentication or download error stops execution. Existing CSVs are not overwritten.
Header verification is not a full data audit or proof of provenance.

## Required files

```text
application_train.csv
bureau.csv
bureau_balance.csv
previous_application.csv
installments_payments.csv
POS_CASH_balance.csv
credit_card_balance.csv
```

Retain `HomeCredit_columns_description.csv` from the official download if available;
it will help establish field meanings. `application_test.csv` has no training labels
and will not be presented as a labeled validation set.

If an HTTP 401/403 occurs, verify login and competition-rule acceptance, or use the
browser method. If data access remains unavailable, leave the project paused.
After the helper succeeds, tell the assistant the real data is ready.

Official references:
- https://www.kaggle.com/competitions/home-credit-default-risk/data
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md
- https://github.com/Kaggle/kaggle-cli/blob/main/docs/competitions.md
