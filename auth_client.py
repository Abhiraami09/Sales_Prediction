import pandas as pd
from erp_client import fetch_live_payload
from model import run_training_job, predict_next_horizons


def prepare_data(data):
    df = pd.DataFrame(data["data"])
    return df


def main():

    # 1+2. Authenticate + fetch live data (shared logic, see erp_client.py)
    data = fetch_live_payload(verbose=True)

    # 3. Sanity check/preview of the raw payload
    df = prepare_data(data)
    print(df.head())
    print("Number of records:", len(df))

    # 4. Train item-level (product-wise) models - 7 horizons, tuned,
    #    sparse items auto-dropped and reported rather than blocking the run
    client_id = "demo_client"
    run_training_job(data, client_id, grain="item")

    # 5. Forward forecast - Day+1..Day+7 per item, from each item's own
    #    most recent complete data point
    forecast = predict_next_horizons(data, client_id, grain="item")
    print(forecast)


if __name__ == "__main__":
    main()
