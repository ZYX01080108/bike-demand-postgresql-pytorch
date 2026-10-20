-- 实验汇总、SQL 误差复核与最近一次实验的小时分组误差。
SELECT run_id, created_at, train_rows, validation_rows, test_rows,
       feature_count, best_epoch, test_mae, test_rmse, baseline_mae
FROM bike_project.model_runs
ORDER BY created_at DESC;

-- 总体 MAE：用 SQL 独立核对 Python 的计算。
SELECT p.run_id, COUNT(*) AS test_hours,
       AVG(ABS(p.predicted_count - d.rental_count)) AS mlp_mae,
       AVG(ABS(p.baseline_count - d.rental_count)) AS baseline_mae
FROM bike_project.predictions AS p
JOIN bike_project.hourly_demand AS d ON p.observed_at = d.observed_at
GROUP BY p.run_id;

-- 哪些小时更难预测？每次只比较同一次实验。
SELECT c.hour, COUNT(*) AS test_hours,
       AVG(ABS(p.predicted_count - d.rental_count)) AS mlp_mae
FROM bike_project.predictions AS p
JOIN bike_project.hourly_demand AS d ON p.observed_at = d.observed_at
JOIN bike_project.hourly_conditions AS c ON p.observed_at = c.observed_at
WHERE p.run_id = (
    SELECT run_id FROM bike_project.model_runs
    ORDER BY created_at DESC, run_id DESC LIMIT 1
)
GROUP BY c.hour
ORDER BY mlp_mae DESC;
