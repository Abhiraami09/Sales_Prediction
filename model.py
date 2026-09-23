"""
SPnPOP - Standard Training Pipeline from ERP JSON Payload (item- or category-level)
====================================================================================

Takes the raw ERP API JSON response (the {"success": true, "data": [...]}
shape) straight from the live API call (no file save needed - pass the
dict you already fetched directly into run_training_job / predict_next_horizons),
normalizes it, builds leakage-safe features, and trains 7 independent
direct-horizon XGBoost models (Day+1 ... Day+7) using a Tweedie objective.

Grain is selectable:
  - grain="category"  -> one model set covering all categories pooled
  - grain="item"      -> one model set covering all items pooled, with
                          items that don't have enough history automatically
                          dropped from training (reported, not silently
                          discarded) so sparse items don't wreck the fit
                          for everything else.

Also included:
  - preview_prepared_data(): inspect the prepared DataFrame before training.
  - Hyperparameter tuning: random search scored on validation, early
    stopping, winner retrained on train+val, reported on untouched test.
  - Sri Lankan public holiday calendar (incl. Poya days) as features.
  - Per-entity (category/item) test-set error breakdown, not just blended
    metrics.
  - predict_next_horizons(): actual forward forecast, per entity, Day+1..7.

Requires: pip install holidays

Usage (category grain, unchanged from before):
    python train_from_erp_json.py --input erp_response.json --client-id demo_client --grain category

Usage (item grain - product-wise prediction):
    python train_from_erp_json.py --input erp_response.json --client-id demo_client --grain item
    python train_from_erp_json.py --input erp_response.json --client-id demo_client --grain item --predict

"Live" usage - call directly from auth_client.py with the fetched dict,
no file involved at all:
    from train_from_erp_json import run_training_job, predict_next_horizons
    data = get_training_data(token)                     # already a dict
    run_training_job(data, client_id, grain="item")      # train
    predict_next_horizons(data, client_id, grain="item") # forecast
"""

import argparse
import itertools
import json
import os
from datetime import datetime, timedelta

import holidays
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

# xgboost 2.0+ renamed the xgb.train() custom-eval-metric argument from
# "feval" to "custom_metric". Detect which this installation expects so
# tuning works regardless of which xgboost version is installed.
_XGB_MAJOR = int(xgb.__version__.split(".")[0])
_CUSTOM_METRIC_KWARG = "custom_metric" if _XGB_MAJOR >= 2 else "feval"

HORIZONS = [1, 2, 3, 4, 5, 6, 7]
LAG_DAYS = [1, 2, 3, 7, 14]
ROLLING_WINDOWS = [3, 7, 14]
CAT_COLS = ["ItmCat1Cd", "ItmCat2Cd", "ItmCat3Cd", "ItmCat7Cd", "ItmCat8Cd"]

# grain name -> the actual DataFrame column it groups on
GRAIN_COLUMNS = {"category": "Category", "item": "ItemCode"}

# Item-level demand is intermittent, so items with almost no history get
# auto-dropped before training rather than allowed to poison the pooled
# model or crash on an empty split. Category grain isn't filtered (0).
DEFAULT_MIN_ACTIVE_DAYS = {"category": 0, "item": 30}

# Hyperparameter search space for random search. Widen this if you have
# more time/compute; each combo trains one XGBoost model per horizon.
#
# tweedie_variance_power kept in the 1.05-1.3 range on purpose: closer to 1
# behaves more like a Poisson-ish "chase the spikes" model, while values
# near 1.9 behave more like a gamma "predict-the-typical-value" model. On
# intermittent demand (mostly-zero days with occasional spikes), values
# like 1.5 let the model win on MAE by just predicting near-zero every day
# - which is exactly the collapse we saw. Combined with scoring tuning on
# WAPE instead of MAE (see wape_eval below), this keeps the search honest.
PARAM_GRID = {
    "max_depth": [4, 6, 8],
    "eta": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "min_child_weight": [1, 5, 10],
    "tweedie_variance_power": [1.05, 1.1, 1.2, 1.3],
}
N_RANDOM_SEARCH_ITERS_DEFAULT = 20


# ---------------------------------------------------------------------------
# 1. Load + normalize the ERP payload
# ---------------------------------------------------------------------------

def load_erp_payload(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def normalize_erp_payload(payload: dict) -> pd.DataFrame:
    """Map the ERP schema onto the fields the training pipeline expects.

    Known gaps vs the original SalesData.xlsx schema:
      - No price field -> Price_Ratio cannot be built from this source.
      - No transaction ID.
      - Category is split across five columns and usually blank in all but
        one -> take the first non-empty value.
      - Qty is a signed stock movement (negative on sales) -> demand is the
        absolute value./
    """
    df = pd.DataFrame(payload["data"])

    for c in CAT_COLS:
        df[c] = df[c].replace("", pd.NA).replace(r"^\s*$", pd.NA, regex=True)
    df["Category"] = df[CAT_COLS].bfill(axis=1).iloc[:, 0].fillna("Uncategorized")

    df["Demand"] = df["Qty"].abs()
    df = df.rename(columns={
        "ItmCd": "ItemCode",
        "ItmNm": "ItemName",
        "LocNm": "Location",
        "EftvDt": "TransactionDate",
        "AdrNm": "CustomerName",
    })
    df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])

    # AvgDisPer is a newly-added field and may not be present on every
    # payload/row - default to 0 rather than erroring if it's missing.
    if "AvgDisPer" not in df.columns:
        df["AvgDisPer"] = 0.0

    # Drop rows with no real item/category identity (e.g. raw stock-count
    # adjustment rows seen in the ERP feed) - these aren't real sales and
    # would otherwise pollute an "Uncategorized" / NaN-item bucket.
    df = df.dropna(subset=["ItemCode", "Category"])

    return df[["ItemCode", "ItemName", "Location", "Category", "CustomerName",
               "AvgDisPer", "Demand", "TransactionDate"]]


# ---------------------------------------------------------------------------
# 2. Aggregate to (entity)-daily grain, drop sparse entities, fill calendar
# ---------------------------------------------------------------------------

def aggregate_to_daily(df: pd.DataFrame, grain_col: str) -> pd.DataFrame:
    """Sum Demand to one row per (entity, day). Not yet calendar-filled."""
    return (
        df.groupby([grain_col, pd.Grouper(key="TransactionDate", freq="D")])["Demand"]
        .sum()
        .reset_index()
        .rename(columns={"TransactionDate": "Date"})
    )


def filter_sparse_entities(daily: pd.DataFrame, grain_col: str, min_active_days: int):
    """Drop entities (items or categories) that don't have enough non-zero
    days to model meaningfully. Returns (kept_daily, kept_ids, dropped_ids).
    Filtering happens BEFORE calendar-filling, so a sparse item with 2
    transactions 5 years apart doesn't get blown up into thousands of
    zero-rows for nothing.
    """
    if min_active_days <= 0:
        ids = daily[grain_col].unique().tolist()
        return daily, ids, []

    active = daily[daily["Demand"] != 0]
    counts = active.groupby(grain_col)["Date"].nunique()
    keep_ids = counts[counts >= min_active_days].index.tolist()
    drop_ids = counts[counts < min_active_days].index.tolist()

    # Entities with zero non-zero days at all (never in `counts`) are also dropped
    all_ids = set(daily[grain_col].unique())
    counted_ids = set(counts.index)
    drop_ids += list(all_ids - counted_ids)

    kept = daily[daily[grain_col].isin(keep_ids)]
    return kept, keep_ids, drop_ids


def fill_calendar(daily: pd.DataFrame, grain_col: str) -> pd.DataFrame:
    """Fill in missing calendar days per entity with 0 demand (within that
    entity's own first->last date span), so lag/rolling features don't
    silently skip gaps."""
    filled = []
    for key, g in daily.groupby(grain_col):
        full_range = pd.date_range(g["Date"].min(), g["Date"].max(), freq="D")
        g = g.set_index("Date").reindex(full_range).rename_axis("Date").reset_index()
        g[grain_col] = key
        g["Demand"] = g["Demand"].fillna(0.0)
        filled.append(g)
    return pd.concat(filled, ignore_index=True).sort_values([grain_col, "Date"])


# ---------------------------------------------------------------------------
# 3. Leakage-safe feature engineering
# ---------------------------------------------------------------------------

def get_sri_lanka_holidays(min_year: int, max_year: int):
    """Sri Lanka PUBLIC holidays (includes Poya/full-moon days, which are
    public holidays in Sri Lanka) for the given year range, via the
    `holidays` package. Returns a dict {date: holiday_name}."""
    return holidays.LK(years=range(min_year, max_year + 2), categories="public")


def add_sri_lanka_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds holiday-awareness features so the model can learn holiday /
    pre-holiday / post-holiday demand shifts, not just plain weekday
    patterns."""
    df = df.copy()
    min_year = df["Date"].dt.year.min()
    max_year = df["Date"].dt.year.max()
    lk_holidays = get_sri_lanka_holidays(min_year, max_year)
    holiday_dates = pd.to_datetime(sorted(lk_holidays.keys()))

    df["IsHoliday"] = df["Date"].isin(holiday_dates).astype(int)
    df["IsDayBeforeHoliday"] = (df["Date"] + pd.Timedelta(days=1)).isin(holiday_dates).astype(int)
    df["IsDayAfterHoliday"] = (df["Date"] - pd.Timedelta(days=1)).isin(holiday_dates).astype(int)

    holiday_arr = holiday_dates.values.astype("datetime64[D]")

    def days_to_next(d):
        future = holiday_arr[holiday_arr >= np.datetime64(d, "D")]
        return int((future.min() - np.datetime64(d, "D")).astype(int)) if len(future) else 14

    def days_since_last(d):
        past = holiday_arr[holiday_arr <= np.datetime64(d, "D")]
        return int((np.datetime64(d, "D") - past.max()).astype(int)) if len(past) else 14

    unique_dates = df["Date"].drop_duplicates()
    lookup_next = {d: min(days_to_next(d), 14) for d in unique_dates}
    lookup_since = {d: min(days_since_last(d), 14) for d in unique_dates}
    df["DaysToNextHoliday"] = df["Date"].map(lookup_next)
    df["DaysSinceHoliday"] = df["Date"].map(lookup_since)

    return df


def build_features(daily: pd.DataFrame, grain_col: str) -> pd.DataFrame:
    df = daily.copy()
    df = df.sort_values([grain_col, "Date"])

    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)
    df["IsWeekend"] = df["DayOfWeek"].isin([5, 6]).astype(int)
    df = add_sri_lanka_calendar_features(df)

    grp = df.groupby(grain_col)["Demand"]

    for lag in LAG_DAYS:
        df[f"Lag_{lag}"] = grp.shift(lag)

    for win in ROLLING_WINDOWS:
        df[f"RollMean_{win}"] = grp.shift(1).rolling(win).mean().reset_index(level=0, drop=True)
        df[f"RollStd_{win}"] = grp.shift(1).rolling(win).std().reset_index(level=0, drop=True)

    df["TxnFreq_Expanding"] = (
        df.groupby(grain_col)["Demand"]
        .apply(lambda s: (s.shift(1) > 0).expanding().mean())
        .reset_index(level=0, drop=True)
    )

    # NOTE: Price_Ratio intentionally omitted - this ERP payload carries no
    # price field.

    return df


def make_multi_horizon_targets(df: pd.DataFrame, grain_col: str) -> pd.DataFrame:
    df = df.copy()
    for h in HORIZONS:
        df[f"Target_Day{h}"] = df.groupby(grain_col)["Demand"].shift(-h)
    return df


def preview_prepared_data(payload: dict, grain: str = "category",
                           min_active_days: int = None):
    """Run the full prep pipeline (normalize -> aggregate -> filter sparse
    entities -> calendar-fill -> features -> targets) and return the
    prepared DataFrame WITHOUT training anything, plus the list of any
    entities that got dropped for insufficient history.

    grain: "category" or "item".
    Returns: (with_targets_df, dropped_entity_ids)
    """
    grain_col = GRAIN_COLUMNS[grain]
    if min_active_days is None:
        min_active_days = DEFAULT_MIN_ACTIVE_DAYS[grain]

    raw = normalize_erp_payload(payload)
    daily = aggregate_to_daily(raw, grain_col)
    daily, kept_ids, dropped_ids = filter_sparse_entities(daily, grain_col, min_active_days)
    daily = fill_calendar(daily, grain_col)
    features = build_features(daily, grain_col)
    with_targets = make_multi_horizon_targets(features, grain_col)

    if dropped_ids:
        print(f"[{grain}] Dropped {len(dropped_ids)} of {len(kept_ids) + len(dropped_ids)} "
              f"entities for having < {min_active_days} active days "
              f"(not enough history to model alone).")

    return with_targets, dropped_ids


# ---------------------------------------------------------------------------
# 4. Chronological 70/15/15 split
# ---------------------------------------------------------------------------

def chronological_split(df: pd.DataFrame):
    df = df.sort_values("Date")
    dates = df["Date"].unique()
    n = len(dates)
    train_end = dates[int(n * 0.70) - 1] if n > 0 else None
    val_end = dates[int(n * 0.85) - 1] if n > 0 else None

    train = df[df["Date"] <= train_end]
    val = df[(df["Date"] > train_end) & (df["Date"] <= val_end)]
    test = df[df["Date"] > val_end]
    return train, val, test


# ---------------------------------------------------------------------------
# 5. Metrics
# ---------------------------------------------------------------------------

def wape(y_true, y_pred) -> float:
    denom = np.sum(np.abs(y_true))
    return float(np.sum(np.abs(y_true - y_pred)) / denom) if denom > 0 else float("nan")


def bias_pct(y_true, y_pred) -> float:
    denom = np.sum(y_true)
    return float(np.sum(y_pred - y_true) / denom * 100) if denom > 0 else float("nan")


def get_feature_cols(df: pd.DataFrame, grain_col: str) -> list:
    exclude = {"Date", grain_col} | {f"Target_Day{h}" for h in HORIZONS}
    return [c for c in df.columns if c not in exclude]


def wape_eval(preds: np.ndarray, dtrain: "xgb.DMatrix"):
    """Custom XGBoost eval metric: WAPE instead of MAE.

    Why this matters on intermittent demand data: most days for a given
    item/category have zero demand. Plain MAE lets a model "win" by just
    predicting near-zero every day - it gets punished only on the rare
    days demand actually spikes, and there aren't many of those. WAPE
    measures error as a % of total actual volume, so predicting near-zero
    everywhere scores ~100% (bad), and the search is pushed toward models
    that actually chase the spikes instead of ignoring them.
    """
    labels = dtrain.get_label()
    preds = np.clip(preds, 0, None)  # Tweedie output shouldn't be negative
    denom = np.sum(np.abs(labels))
    value = float(np.sum(np.abs(labels - preds)) / denom) if denom > 0 else 0.0
    return "WAPE", value


# ---------------------------------------------------------------------------
# 6. Hyperparameter tuning (random search, val-scored, early stopping)
# ---------------------------------------------------------------------------

def random_search_params(n_iters: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    keys = list(PARAM_GRID.keys())
    all_combos = list(itertools.product(*PARAM_GRID.values()))
    rng.shuffle(all_combos)
    n_iters = min(n_iters, len(all_combos))
    return [dict(zip(keys, combo)) for combo in all_combos[:n_iters]]


def tune_one_horizon(tr: pd.DataFrame, va: pd.DataFrame, feature_cols: list,
                      target_col: str, n_iters: int):
    """Random search over PARAM_GRID, scored on validation WAPE (via
    wape_eval) with early stopping on that same metric. Returns
    (best_params, best_val_wape, best_num_rounds)."""
    dtrain = xgb.DMatrix(tr[feature_cols], label=tr[target_col])
    dval = xgb.DMatrix(va[feature_cols], label=va[target_col])

    best_wape = np.inf
    best_params = None
    best_rounds = None

    for params in random_search_params(n_iters):
        full_params = {
            **params,
            "objective": "reg:tweedie",
            "seed": 42,
            # Force early stopping / best-iteration selection to use ONLY
            # our custom WAPE metric below, not xgboost's own default
            # metric for the tweedie objective (tweedie-nloglik). Without
            # this, xgboost can silently keep optimizing/selecting on its
            # default metric even though we pass a custom_metric - which
            # is exactly why the earlier fix didn't visibly change
            # anything: the search was still being scored the old way.
            "disable_default_eval_metric": 1,
        }
        model = xgb.train(
            full_params, dtrain, num_boost_round=500,
            evals=[(dval, "val")],
            maximize=False,
            early_stopping_rounds=30, verbose_eval=False,
            **{_CUSTOM_METRIC_KWARG: wape_eval},
        )
        val_wape = model.best_score
        if val_wape < best_wape:
            best_wape = val_wape
            best_params = full_params
            best_rounds = model.best_iteration + 1

    return best_params, best_wape, best_rounds


# ---------------------------------------------------------------------------
# 7. Train + tune all 7 horizons
# ---------------------------------------------------------------------------

def train_all_horizons_tuned(train, val, test, feature_cols, grain_col, client_id,
                              output_dir, n_iters=N_RANDOM_SEARCH_ITERS_DEFAULT):
    os.makedirs(output_dir, exist_ok=True)
    metadata = {
        "client_id": client_id,
        "grain": grain_col,
        "trained_at": datetime.utcnow().isoformat(),
        "feature_cols": feature_cols,
        "n_search_iters": n_iters,
        "horizons": {},
    }

    for h in HORIZONS:
        target_col = f"Target_Day{h}"

        tr = train.dropna(subset=feature_cols + [target_col])
        va = val.dropna(subset=feature_cols + [target_col])
        te = test.dropna(subset=feature_cols + [target_col])

        if tr.empty or va.empty or te.empty:
            print(f"[Day+{h}] Skipped - not enough rows after dropna "
                  f"(train={len(tr)}, val={len(va)}, test={len(te)}).")
            continue

        print(f"[Day+{h}] tuning ({n_iters} random search iterations)...")
        best_params, best_val_wape, best_rounds = tune_one_horizon(
            tr, va, feature_cols, target_col, n_iters
        )
        print(f"[Day+{h}] best params: {best_params}")
        print(f"[Day+{h}] val WAPE: {best_val_wape:.4f} | rounds: {best_rounds}")

        # Retrain final model on train+val combined with the winning
        # params/rounds, then evaluate once on the untouched test split.
        trainval = pd.concat([tr, va], ignore_index=True)
        dtrainval = xgb.DMatrix(trainval[feature_cols], label=trainval[target_col])
        final_model = xgb.train(best_params, dtrainval, num_boost_round=best_rounds)

        dtest = xgb.DMatrix(te[feature_cols], label=te[target_col])
        preds = np.clip(final_model.predict(dtest), 0, None)
        y_true = te[target_col].values

        metrics = {
            "MAE": mean_absolute_error(y_true, preds),
            "RMSE": mean_squared_error(y_true, preds) ** 0.5,
            "WAPE": wape(y_true, preds),
            "Bias_pct": bias_pct(y_true, preds),
            "n_test_rows": int(len(te)),
        }
        print(f"[Day+{h}] TEST {metrics}")

        # Per-entity breakdown - the blended metrics above can hide a model
        # that's great on high-volume entities and bad on low-volume ones,
        # so write out every prediction alongside the actual.
        detail = te[[grain_col, "Date"]].copy()
        detail["Actual"] = y_true
        detail["Predicted"] = preds
        detail["Error"] = detail["Predicted"] - detail["Actual"]
        detail_path = os.path.join(output_dir, f"test_predictions_day{h}.csv")
        detail.to_csv(detail_path, index=False)

        per_entity = (
            detail.groupby(grain_col)
            .apply(lambda g: pd.Series({
                "MAE": mean_absolute_error(g["Actual"], g["Predicted"]),
                "WAPE": wape(g["Actual"].values, g["Predicted"].values),
                "Bias_pct": bias_pct(g["Actual"].values, g["Predicted"].values),
                "n_rows": len(g),
            }))
            .reset_index()
        )
        per_entity_path = os.path.join(output_dir, f"per_{grain_col.lower()}_metrics_day{h}.csv")
        per_entity.to_csv(per_entity_path, index=False)
        print(f"[Day+{h}] per-{grain_col} metrics -> {per_entity_path}\n")

        model_path = os.path.join(output_dir, f"model_day{h}.json")
        final_model.save_model(model_path)
        metadata["horizons"][f"Day+{h}"] = {
            "model_file": model_path,
            "best_params": best_params,
            "best_rounds": best_rounds,
            "val_WAPE": best_val_wape,
            "test_metrics": metrics,
            "test_predictions_file": detail_path,
            "per_entity_metrics_file": per_entity_path,
        }

    with open(os.path.join(output_dir, "feature_list.json"), "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    return metadata


# ---------------------------------------------------------------------------
# 8. Forward-looking prediction (the actual forecast, per entity)
# ---------------------------------------------------------------------------

def predict_next_horizons(payload: dict, client_id: str, output_root: str = "SPnPOP_outputs",
                           grain: str = "category", recency_days: int = 90) -> pd.DataFrame:
    """Load the 7 saved per-horizon models for this client and produce an
    actual forward forecast: for every entity (category or item), Day+1 ...
    Day+7 predicted demand, dated from that entity's own most recent date.

    recency_days: an entity must have its most recent complete-feature row
    within this many days of the freshest date seen anywhere in the live
    pull, or it's excluded from the forecast (but NOT from training - a
    model that's gone quiet stays trained in case it starts moving again).
    Without this, an item whose last real transaction was years ago would
    still get a Day+1..7 forecast, which reads as "here's what's about to
    happen" when it's really "here's what would have happened years ago" -
    misleading on a live dashboard.

    This is the step that turns 'trained models sitting on disk' into
    'a number your boss can act on'.
    """
    grain_col = GRAIN_COLUMNS[grain]
    output_dir = os.path.join(output_root, client_id)
    with open(os.path.join(output_dir, "feature_list.json")) as f:
        feature_cols = json.load(f)

    prepared, _dropped = preview_prepared_data(payload, grain=grain)

    # For each entity, the most recent row that has every feature populated
    # (no NaNs from lags/rolling windows) is what we feed the models - it
    # represents "today" from that entity's point of view.
    latest_rows = (
        prepared.dropna(subset=feature_cols)
        .sort_values("Date")
        .groupby(grain_col)
        .tail(1)
    )

    if latest_rows.empty:
        raise ValueError(f"No {grain} has a complete feature row to predict from - "
                          f"check that there's enough history per {grain}.")

    # Recency filter: exclude entities whose own latest data point is stale
    # relative to the freshest date anywhere in this live pull.
    global_latest_date = prepared["Date"].max()
    cutoff_date = global_latest_date - pd.Timedelta(days=recency_days)
    is_recent = latest_rows["Date"] >= cutoff_date
    stale_entities = latest_rows.loc[~is_recent, grain_col].tolist()
    latest_rows = latest_rows.loc[is_recent]

    if stale_entities:
        print(f"[{grain}] Excluded {len(stale_entities)} entities from the live forecast "
              f"for having no data in the last {recency_days} days (still trained, "
              f"just not currently active): {stale_entities}")

    if latest_rows.empty:
        raise ValueError(f"No {grain} has data within the last {recency_days} days - "
                          f"nothing currently active to forecast.")

    results = []
    for h in HORIZONS:
        model_path = os.path.join(output_dir, f"model_day{h}.json")
        if not os.path.exists(model_path):
            print(f"[Day+{h}] model file missing, skipping - did you train first?")
            continue
        model = xgb.Booster()
        model.load_model(model_path)

        dmat = xgb.DMatrix(latest_rows[feature_cols])
        preds = model.predict(dmat)

        for entity_id, base_date, pred in zip(latest_rows[grain_col], latest_rows["Date"], preds):
            results.append({
                grain_col: entity_id,
                "AsOfDate": base_date,
                "ForecastDate": base_date + timedelta(days=h),
                "Horizon": f"Day+{h}",
                "PredictedDemand": max(0.0, float(pred)),  # demand can't be negative
            })

    forecast_df = pd.DataFrame(results).sort_values([grain_col, "ForecastDate"])
    out_path = os.path.join(output_dir, f"forecast_next_7_days_{grain}.csv")
    forecast_df.to_csv(out_path, index=False)
    print(f"Forecast written to: {out_path}")
    return forecast_df


# ---------------------------------------------------------------------------
# 8b. Customer-wise allocation (post-processing on top of the entity forecast)
# ---------------------------------------------------------------------------
#
# Rather than retraining at item x customer grain (which would badly worsen
# the sparsity problem item-level forecasting already has), this splits the
# existing entity-level forecast across customers using each customer's
# historical share of that entity's demand - the same idea as the sparse-item
# share method, one level down. AvgDisPer can optionally nudge the split.

def compute_customer_shares(payload: dict, grain: str = "item",
                             discount_weight: float = 0.0) -> pd.DataFrame:
    """For each entity (item/category), compute what share of its historical
    demand came from each customer.

    discount_weight: 0.0 (default) = pure historical-volume share. Set e.g.
    0.2 to let a customer's average discount level nudge its share up/down
    slightly - a lightweight proxy until there's enough data to justify a
    full customer-aware model.
    """
    grain_col = GRAIN_COLUMNS[grain]
    raw = normalize_erp_payload(payload)
    raw = raw.dropna(subset=["CustomerName"])

    grp = raw.groupby([grain_col, "CustomerName"]).agg(
        CustomerDemand=("Demand", "sum"),
        AvgDisPer=("AvgDisPer", "mean"),
    ).reset_index()

    entity_totals = grp.groupby(grain_col)["CustomerDemand"].transform("sum")
    grp["CustomerShare"] = grp["CustomerDemand"] / entity_totals

    if discount_weight > 0:
        weight = 1 + discount_weight * (grp["AvgDisPer"].fillna(0) / 100)
        grp["CustomerShare"] = grp["CustomerShare"] * weight
        grp["CustomerShare"] = (
            grp["CustomerShare"] / grp.groupby(grain_col)["CustomerShare"].transform("sum")
        )

    return grp[[grain_col, "CustomerName", "CustomerShare", "AvgDisPer"]]


def allocate_forecast_by_customer(forecast_df: pd.DataFrame, shares_df: pd.DataFrame,
                                   grain_col: str) -> pd.DataFrame:
    """Split an existing entity-level forecast (from predict_next_horizons)
    across customers using their historical share of that entity's demand."""
    merged = forecast_df.merge(shares_df, on=grain_col, how="inner")
    merged["PredictedDemand_Customer"] = merged["PredictedDemand"] * merged["CustomerShare"]
    return merged.sort_values([grain_col, "CustomerName", "ForecastDate"])


def get_customer_forecast(payload: dict, client_id: str, output_root: str = "SPnPOP_outputs",
                           grain: str = "item", recency_days: int = 90,
                           discount_weight: float = 0.0) -> pd.DataFrame:
    """Convenience wrapper: entity-level forecast + customer-share allocation
    in one call. This is what api_server.py's /forecast/{client_id}/by-customer
    endpoint calls."""
    forecast_df = predict_next_horizons(payload, client_id, output_root,
                                         grain=grain, recency_days=recency_days)
    shares_df = compute_customer_shares(payload, grain=grain, discount_weight=discount_weight)
    grain_col = GRAIN_COLUMNS[grain]
    return allocate_forecast_by_customer(forecast_df, shares_df, grain_col)


# ---------------------------------------------------------------------------
# 9. Orchestration
# ---------------------------------------------------------------------------

def run_training_job(payload: dict, client_id: str, output_root: str = "SPnPOP_outputs",
                      n_iters: int = N_RANDOM_SEARCH_ITERS_DEFAULT, grain: str = "category",
                      min_active_days: int = None):
    grain_col = GRAIN_COLUMNS[grain]
    output_dir = os.path.join(output_root, client_id)

    with_targets, dropped_ids = preview_prepared_data(payload, grain=grain,
                                                        min_active_days=min_active_days)
    train, val, test = chronological_split(with_targets)
    feature_cols = get_feature_cols(with_targets, grain_col)

    print(f"[{grain}] entities trained on: {with_targets[grain_col].nunique()} "
          f"(dropped: {len(dropped_ids)})")
    print(f"Rows -> train: {len(train)}, val: {len(val)}, test: {len(test)}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    metadata = train_all_horizons_tuned(train, val, test, feature_cols, grain_col, client_id,
                                         output_dir, n_iters=n_iters)
    metadata["n_entities_trained"] = int(with_targets[grain_col].nunique())
    metadata["dropped_entities"] = dropped_ids
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"\nArtifacts written to: {output_dir}")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to ERP JSON response file")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--output-root", default="SPnPOP_outputs")
    parser.add_argument("--grain", choices=["category", "item"], default="category",
                         help="Forecast grain: pooled category-level, or product-wise (item-level).")
    parser.add_argument("--min-active-days", type=int, default=None,
                         help="Override the auto sparse-entity filter threshold "
                              "(default: 0 for category, 30 for item).")
    parser.add_argument("--preview", action="store_true",
                         help="Only run data prep and print the prepared DataFrame - no training.")
    parser.add_argument("--n-iters", type=int, default=N_RANDOM_SEARCH_ITERS_DEFAULT,
                         help="Random search iterations per horizon during tuning.")
    parser.add_argument("--predict", action="store_true",
                         help="Skip training - load already-trained models for this client "
                              "and produce a forward forecast (Day+1..Day+7 per entity).")
    args = parser.parse_args()

    payload = load_erp_payload(args.input)
    grain_col = GRAIN_COLUMNS[args.grain]

    if args.preview:
        df, dropped = preview_prepared_data(payload, grain=args.grain,
                                             min_active_days=args.min_active_days)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(df.head(20))
        print(f"\nShape: {df.shape}")
        print(f"Date range: {df['Date'].min()} -> {df['Date'].max()}")
        print(f"{grain_col}s kept: {df[grain_col].nunique()} | dropped: {len(dropped)}")
    elif args.predict:
        forecast_df = predict_next_horizons(payload, args.client_id, args.output_root, grain=args.grain)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        print(forecast_df)
    else:
        run_training_job(payload, args.client_id, args.output_root,
                          n_iters=args.n_iters, grain=args.grain,
                          min_active_days=args.min_active_days)