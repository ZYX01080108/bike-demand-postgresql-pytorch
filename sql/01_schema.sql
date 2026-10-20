-- 项目表统一存放在 bike_project schema。
CREATE SCHEMA IF NOT EXISTS bike_project;

-- 一行 = 一个实际记录到的小时；这些是当小时观测天气，不是天气预报。
CREATE TABLE IF NOT EXISTS bike_project.hourly_conditions (
    observed_at TIMESTAMP PRIMARY KEY,
    hour SMALLINT NOT NULL CHECK (hour BETWEEN 0 AND 23),
    weekday SMALLINT NOT NULL CHECK (weekday BETWEEN 0 AND 6),
    month SMALLINT NOT NULL CHECK (month BETWEEN 1 AND 12),
    year_index SMALLINT NOT NULL CHECK (year_index IN (0, 1)),
    holiday SMALLINT NOT NULL CHECK (holiday IN (0, 1)),
    workingday SMALLINT NOT NULL CHECK (workingday IN (0, 1)),
    weather_code SMALLINT NOT NULL CHECK (weather_code BETWEEN 1 AND 4),
    temperature DOUBLE PRECISION NOT NULL CHECK (temperature BETWEEN 0 AND 1),
    humidity DOUBLE PRECISION NOT NULL CHECK (humidity BETWEEN 0 AND 1),
    windspeed DOUBLE PRECISION NOT NULL CHECK (windspeed BETWEEN 0 AND 1)
);

-- 需求标签表；模型推理不读取标签列。
CREATE TABLE IF NOT EXISTS bike_project.hourly_demand (
    observed_at TIMESTAMP PRIMARY KEY
        REFERENCES bike_project.hourly_conditions(observed_at),
    rental_count INTEGER NOT NULL CHECK (rental_count >= 0)
);

CREATE TABLE IF NOT EXISTS bike_project.model_runs (
    run_id TEXT PRIMARY KEY,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    seed INTEGER NOT NULL,
    train_rows INTEGER NOT NULL,
    validation_rows INTEGER NOT NULL,
    test_rows INTEGER NOT NULL,
    feature_count INTEGER NOT NULL,
    epochs_run INTEGER NOT NULL,
    best_epoch INTEGER NOT NULL,
    test_mae DOUBLE PRECISION NOT NULL,
    test_rmse DOUBLE PRECISION NOT NULL,
    baseline_mae DOUBLE PRECISION NOT NULL
);

-- 只保存测试集预测，真实值通过 JOIN 标签表取得。
CREATE TABLE IF NOT EXISTS bike_project.predictions (
    run_id TEXT NOT NULL REFERENCES bike_project.model_runs(run_id),
    observed_at TIMESTAMP NOT NULL REFERENCES bike_project.hourly_demand(observed_at),
    predicted_count DOUBLE PRECISION NOT NULL CHECK (predicted_count >= 0),
    baseline_count DOUBLE PRECISION NOT NULL CHECK (baseline_count >= 0),
    PRIMARY KEY (run_id, observed_at)
);
