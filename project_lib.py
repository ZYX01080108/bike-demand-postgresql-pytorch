"""PostgreSQL data pipeline, MLP regression, and SQL-verified evaluation."""

from __future__ import annotations

import copy
import getpass
import hashlib
import io
import json
import os
from pathlib import Path
import urllib.request
import uuid
import zipfile

import numpy as np
import pandas as pd
import psycopg
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts"
SOURCE_URL = "https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip"
CSV_SHA256 = "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"
NUMERIC = ["temperature", "humidity", "windspeed", "workingday", "holiday", "year_index"]
# Values and column order are fixed in advance, not learned from the test set.
CATEGORIES = {"hour": list(range(24)), "weekday": list(range(7)),
              "month": list(range(1, 13)), "weather_code": list(range(1, 5))}
SEED = 7


def connect():
    """Use standard PG* environment variables; never save a password in code."""
    password = os.environ.get("PGPASSWORD")
    if password is None:
        password = getpass.getpass("PostgreSQL password (not displayed): ")
    return psycopg.connect(
        host=os.environ.get("PGHOST", "127.0.0.1"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "bike_demand"),
        user=os.environ.get("PGUSER", "postgres"),
        password=password, connect_timeout=5,
    )


def query_frame(conn, query, params=None):
    """Fetch SQL output as a DataFrame, including the column names."""
    cursor = conn.execute(query, params)
    return pd.DataFrame(cursor.fetchall(), columns=[c.name for c in cursor.description])


def read_sql(name):
    return (ROOT / "sql" / name).read_text(encoding="utf-8")


def download_data():
    DATA.mkdir(exist_ok=True)
    path = DATA / "hour.csv"
    if not path.exists():
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        path.write_bytes(archive.read("hour.csv"))
        (DATA / "Readme_uci.txt").write_bytes(archive.read("Readme.txt"))
    if hashlib.sha256(path.read_bytes()).hexdigest() != CSV_SHA256:
        raise ValueError("hour.csv checksum differs from the documented UCI file.")
    return path


def validate_raw(raw):
    """Fail visibly on bad input. Do not silently delete or impute records."""
    required = ["dteday", "hr", "weekday", "mnth", "yr", "holiday", "workingday",
                "weathersit", "temp", "hum", "windspeed", "cnt", "casual", "registered"]
    if not set(required).issubset(raw.columns):
        raise ValueError("Required UCI columns are missing.")
    if raw[required].isna().any().any():
        raise ValueError("Unexpected missing values: investigate their source first.")
    bounds = {"hr": (0, 23), "weekday": (0, 6), "mnth": (1, 12), "yr": (0, 1),
              "holiday": (0, 1), "workingday": (0, 1), "weathersit": (1, 4),
              "temp": (0, 1), "hum": (0, 1), "windspeed": (0, 1)}
    for column, (low, high) in bounds.items():
        values = pd.to_numeric(raw[column], errors="raise")
        if not np.isfinite(values).all() or not values.between(low, high).all():
            raise ValueError(f"Invalid range in {column}.")
        if column not in ("temp", "hum", "windspeed") and not (values % 1 == 0).all():
            raise ValueError(f"Expected integer codes in {column}.")
    for column in ("cnt", "casual", "registered"):
        values = pd.to_numeric(raw[column], errors="raise")
        if not np.isfinite(values).all() or not ((values >= 0) & (values % 1 == 0)).all():
            raise ValueError(f"Expected nonnegative integer counts in {column}.")
    if not (raw["casual"] + raw["registered"] == raw["cnt"]).all():
        raise ValueError("casual + registered must equal cnt.")
    timestamps = pd.to_datetime(raw["dteday"], errors="raise") + pd.to_timedelta(raw["hr"], unit="h")
    if timestamps.duplicated().any():
        raise ValueError("Duplicate hourly timestamps: do not count them as new observations.")
    return timestamps


def load_data(conn):
    raw = pd.read_csv(download_data())
    timestamps = validate_raw(raw)
    conditions, demands = [], []
    for i, row in enumerate(raw.itertuples(index=False)):
        when = timestamps.iloc[i].to_pydatetime()
        conditions.append((when, int(row.hr), int(row.weekday), int(row.mnth), int(row.yr),
                           int(row.holiday), int(row.workingday), int(row.weathersit),
                           float(row.temp), float(row.hum), float(row.windspeed)))
        demands.append((when, int(row.cnt)))
    # Schema and both table imports commit together, or all roll back on an error.
    with conn.transaction():
        conn.execute(read_sql("01_schema.sql"))
        with conn.cursor() as cursor:
            cursor.executemany(
                "INSERT INTO bike_project.hourly_conditions VALUES "
                "(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                "ON CONFLICT (observed_at) DO NOTHING", conditions)
            cursor.executemany(
                "INSERT INTO bike_project.hourly_demand VALUES (%s,%s) "
                "ON CONFLICT (observed_at) DO NOTHING", demands)
        actual = conn.execute("SELECT COUNT(*) FROM bike_project.hourly_conditions").fetchone()[0]
        labels = conn.execute("SELECT COUNT(*) FROM bike_project.hourly_demand").fetchone()[0]
        if actual != len(raw) or labels != len(raw):
            raise ValueError("Existing project tables contain unexpected rows; inspect before training.")
    print(f"Validated and loaded {len(raw):,} hourly records; rerunning does not duplicate rows.")


def load_training_data(conn):
    frame = query_frame(conn, read_sql("04_training_data.sql"))
    if frame.empty or frame["observed_at"].duplicated().any() or frame.isna().any().any():
        raise ValueError("Training JOIN produced empty, duplicate, or missing rows.")
    expected = conn.execute("SELECT COUNT(*) FROM bike_project.hourly_conditions").fetchone()[0]
    if len(frame) != expected:
        raise ValueError("Training JOIN lost records.")
    train_end = frame.loc[frame["split"] == "train", "observed_at"].max()
    val_start = frame.loc[frame["split"] == "validation", "observed_at"].min()
    val_end = frame.loc[frame["split"] == "validation", "observed_at"].max()
    test_start = frame.loc[frame["split"] == "test", "observed_at"].min()
    if not train_end < val_start <= val_end < test_start:
        raise ValueError("Time split boundaries are invalid.")
    return frame


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def explore(conn):
    ARTIFACTS.mkdir(exist_ok=True)
    hourly = query_frame(conn, read_sql("03_hourly_analysis.sql"))
    hourly.to_csv(ARTIFACTS / "hourly_analysis.csv", index=False)
    frame = load_training_data(conn)
    calendar_hours = int((frame.observed_at.max() - frame.observed_at.min()).total_seconds() / 3600) + 1
    quality = {"observed_rows": len(frame), "unrecorded_hours": calendar_hours - len(frame),
               "first_hour": str(frame.observed_at.min()), "last_hour": str(frame.observed_at.max()),
               "split_counts": {str(k): int(v) for k, v in frame["split"].value_counts().items()},
               "missing_values": int(frame.isna().sum().sum())}
    (ARTIFACTS / "data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    plt = pyplot()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for flag, group in hourly.groupby("workingday"):
        ax.plot(group["hour"], group["mean_rentals"], marker=".",
                label="Working day" if flag else "Non-working day")
    ax.set(xlabel="Hour of day", ylabel="Mean rentals per observed hour", title="Hourly rental patterns (descriptive, all dates)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "hourly_patterns.png", dpi=150)
    plt.close(fig)
    print(json.dumps(quality, indent=2))


def prepare_features(frame):
    """6 numeric inputs + fixed one-hot categories = 53 inputs; no target inputs."""
    arrays = [frame[NUMERIC].to_numpy(dtype=np.float32)]
    names = NUMERIC.copy()
    for column, levels in CATEGORIES.items():
        if not frame[column].isin(levels).all():
            raise ValueError(f"Unknown category in {column}.")
        arrays.append(np.column_stack([(frame[column] == level).to_numpy(dtype=np.float32) for level in levels]))
        names.extend(f"{column}_{level}" for level in levels)
    return np.concatenate(arrays, axis=1), names


def build_model(input_size):
    # Fully connected regression network: 53 inputs, two ReLU hidden layers, one output.
    return nn.Sequential(nn.Linear(input_size, 64), nn.ReLU(),
                         nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))


def metrics(actual, predicted):
    errors = np.asarray(predicted) - np.asarray(actual)
    return {"mae": float(np.abs(errors).mean()), "rmse": float(np.sqrt(np.square(errors).mean()))}


def train(conn, max_epochs=180, patience=20):
    ARTIFACTS.mkdir(exist_ok=True)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    frame = load_training_data(conn)
    train_rows = frame[frame["split"] == "train"].copy()
    val_rows = frame[frame["split"] == "validation"].copy()
    test_rows = frame[frame["split"] == "test"].copy()
    x_train, feature_names = prepare_features(train_rows)
    x_val, _ = prepare_features(val_rows)
    x_test, _ = prepare_features(test_rows)

    # Fit all preprocessing parameters on the training partition only.
    x_mean = x_train.mean(axis=0)
    x_std = x_train.std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_train = train_rows["rental_count"].to_numpy(dtype=np.float32)
    y_val = val_rows["rental_count"].to_numpy(dtype=np.float32)
    y_test = test_rows["rental_count"].to_numpy(dtype=np.float32)
    y_mean, y_std = float(y_train.mean()), float(y_train.std())
    if y_std <= 0:
        raise ValueError("A constant target cannot be standardized for this experiment.")
    train_x = torch.from_numpy((x_train - x_mean) / x_std)
    val_x = torch.from_numpy((x_val - x_mean) / x_std)
    test_x = torch.from_numpy((x_test - x_mean) / x_std)
    train_y = torch.from_numpy((y_train - y_mean) / y_std).reshape(-1, 1)
    val_y = torch.from_numpy((y_val - y_mean) / y_std).reshape(-1, 1)

    model = build_model(len(feature_names))
    loss_function = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    history, best_state = [], None
    best_mae, best_epoch = float("inf"), 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_x))
        for indices in order.split(256):
            optimizer.zero_grad()
            prediction = model(train_x[indices])
            loss = loss_function(prediction, train_y[indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_mse = float(loss_function(model(train_x), train_y))
            val_output = model(val_x)
            val_mse = float(loss_function(val_output, val_y))
            val_pred = np.maximum(val_output.numpy().ravel() * y_std + y_mean, 0)
            val_mae = metrics(y_val, val_pred)["mae"]
        history.append({"epoch": epoch, "train_mse": train_mse,
                        "validation_mse": val_mse, "validation_mae": val_mae})
        if val_mae < best_mae:
            best_mae, best_epoch = val_mae, epoch
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 1 or epoch % 20 == 0:
            print(f"Epoch {epoch:3d}: train MSE={train_mse:.4f}; validation MAE={val_mae:.2f}")
        if epoch - best_epoch >= patience:
            break
    if best_state is None:
        raise ValueError("Training did not produce a finite validation score.")
    model.load_state_dict(best_state)
    model.eval()

    # The test partition is evaluated only after selecting a checkpoint by validation MAE.
    with torch.no_grad():
        predicted = np.maximum(model(test_x).numpy().ravel() * y_std + y_mean, 0)
    global_mean = float(train_rows["rental_count"].mean())
    grouped_means = train_rows.groupby(["workingday", "hour"])["rental_count"].mean()
    baseline = np.array([grouped_means.get((int(w), int(h)), global_mean)
                         for w, h in zip(test_rows.workingday, test_rows.hour)])
    scores = {"training_mean": metrics(y_test, np.full(len(y_test), global_mean)),
              "workingday_hour_mean": metrics(y_test, baseline),
              "mlp": metrics(y_test, predicted)}
    run_id = uuid.uuid4().hex
    preprocessing = {"run_id": run_id, "feature_names": feature_names,
                     "x_mean": x_mean.tolist(), "x_std": x_std.tolist(),
                     "y_mean": y_mean, "y_std": y_std}
    torch.save(model.state_dict(), ARTIFACTS / "model.pt")
    (ARTIFACTS / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2), encoding="utf-8")
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(ARTIFACTS / "training_history.csv", index=False)
    output = test_rows[["observed_at", "hour", "workingday", "rental_count"]].copy()
    output["predicted_count"] = predicted
    output["baseline_count"] = baseline
    output["absolute_error"] = np.abs(predicted - y_test)
    output.to_csv(ARTIFACTS / "test_predictions.csv", index=False)

    with conn.transaction():
        conn.execute(
            "INSERT INTO bike_project.model_runs "
            "(run_id,seed,train_rows,validation_rows,test_rows,feature_count,epochs_run,best_epoch,"
            "test_mae,test_rmse,baseline_mae) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
            (run_id, SEED, len(train_rows), len(val_rows), len(test_rows), len(feature_names),
             len(history), best_epoch, scores["mlp"]["mae"], scores["mlp"]["rmse"],
             scores["workingday_hour_mean"]["mae"]))
        records = [(run_id, when.to_pydatetime(), float(p), float(b))
                   for when, p, b in zip(output.observed_at, predicted, baseline)]
        with conn.cursor() as cursor:
            cursor.executemany("INSERT INTO bike_project.predictions VALUES (%s,%s,%s,%s)", records)
        checked = conn.execute(
            "SELECT COUNT(*), AVG(ABS(p.predicted_count-d.rental_count)) "
            "FROM bike_project.predictions p JOIN bike_project.hourly_demand d "
            "ON p.observed_at=d.observed_at WHERE p.run_id=%s", (run_id,)).fetchone()
        if checked[0] != len(test_rows) or not np.isclose(checked[1], scores["mlp"]["mae"], atol=1e-4):
            raise AssertionError("PostgreSQL and Python evaluation disagree.")

    improvement = 100 * (1 - scores["mlp"]["mae"] / scores["workingday_hour_mean"]["mae"])
    report = {"run_id": run_id, "seed": SEED, "train_rows": len(train_rows),
              "validation_rows": len(val_rows), "test_rows": len(test_rows),
              "feature_count": len(feature_names), "parameter_count": sum(p.numel() for p in model.parameters()),
              "epochs_run": len(history), "best_epoch": best_epoch,
              "validation_mae_at_best_epoch": best_mae, "test_scores": scores,
              "test_mae_improvement_vs_group_baseline_percent": improvement,
              "sql_mae_verified": True,
              "split": {"train": "before 2012-07-01", "validation": "2012-07-01 to 2012-08-31",
                        "test": "2012-09-01 to 2012-12-31"},
              "weather_assumption": "Observed same-hour weather; retrospective conditional prediction, not a deployed future forecast.",
              "versions": {"python": __import__("platform").python_version(), "torch": torch.__version__,
                           "numpy": np.__version__, "pandas": pd.__version__, "psycopg": psycopg.__version__}}
    (ARTIFACTS / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_results(report)
    plot_training(history_frame, output)
    print(json.dumps(report, indent=2))
    return report


def write_results(report):
    scores = report["test_scores"]
    content = f"""# Evaluation report

Run ID: `{report['run_id']}`. Generated by `train.py`.

## Protocol

- Training: {report['train_rows']:,} records before 2012-07-01.
- Validation: {report['validation_rows']:,} records from July-August 2012.
- Test: {report['test_rows']:,} records from September-December 2012.
- MLP: {report['feature_count']} inputs, 64 and 32 ReLU hidden units, one output.
- Parameters: {report['parameter_count']:,}; seed: {report['seed']}.
- Epochs: {report['epochs_run']}; checkpoint selected by validation MAE at epoch {report['best_epoch']}.

## Holdout metrics

| Model | MAE (rentals/hour) | RMSE (rentals/hour) |
|---|---:|---:|
| Training mean | {scores['training_mean']['mae']:.2f} | {scores['training_mean']['rmse']:.2f} |
| Working-day/hour mean | {scores['workingday_hour_mean']['mae']:.2f} | {scores['workingday_hour_mean']['rmse']:.2f} |
| MLP | {scores['mlp']['mae']:.2f} | {scores['mlp']['rmse']:.2f} |

MAE reduction relative to the working-day/hour baseline: {report['test_mae_improvement_vs_group_baseline_percent']:.2f}%.
SQL/Python MAE verification: {report['sql_mae_verified']}.

## Scope

Features use observed same-hour weather. This is retrospective conditional prediction, not an advance-weather forecasting service. Scaling and baseline statistics use training records only; model selection uses validation records only. The test set is evaluated after checkpoint selection.

The grouped baseline does not use weather or annual growth features. This comparison does not establish that neural networks outperform linear or tree models. Results apply to one temporal holdout, not an independent-observation confidence interval or evidence of deployment impact.

Full metrics and package versions: `metrics.json`. Per-hour predictions: `test_predictions.csv`.
"""
    (ARTIFACTS / "RESULTS.md").write_text(content, encoding="utf-8")


def plot_training(history, output):
    plt = pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history.epoch, history.train_mse, label="Train")
    axes[0].plot(history.epoch, history.validation_mse, label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Standardized-target MSE", title="Training curve")
    axes[0].legend()
    first = output.iloc[:168]
    axes[1].plot(first.observed_at, first.rental_count, label="Actual", linewidth=1)
    axes[1].plot(first.observed_at, first.predicted_count, label="MLP", linewidth=1)
    axes[1].set(xlabel="Time", ylabel="Rentals per hour", title="First 168 recorded test hours")
    axes[1].tick_params(axis="x", labelrotation=30)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "model_results.png", dpi=150)
    plt.close(fig)


def predict_one(conn, timestamp):
    # Parameter binding: the timestamp is a value, never interpolated into SQL.
    # Only conditions are read; the real rental count is not provided to the model.
    frame = query_frame(conn,
        "SELECT * FROM bike_project.hourly_conditions WHERE observed_at=%s", (timestamp,))
    if frame.empty:
        raise ValueError("This hour is not in the dataset; use an observed hour between 2011 and 2012.")
    saved = json.loads((ARTIFACTS / "preprocessing.json").read_text(encoding="utf-8"))
    values, names = prepare_features(frame)
    if names != saved["feature_names"]:
        raise ValueError("Feature order differs from the trained model.")
    values = (values - np.array(saved["x_mean"], dtype=np.float32)) / np.array(saved["x_std"], dtype=np.float32)
    model = build_model(len(names))
    model.load_state_dict(torch.load(ARTIFACTS / "model.pt", map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        count = max(0.0, model(torch.from_numpy(values)).item() * saved["y_std"] + saved["y_mean"])
    print(f"Observed weather scenario: {timestamp}; predicted rentals = {count:.1f}")
    return count
