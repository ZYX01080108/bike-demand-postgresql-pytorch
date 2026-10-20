-- 1. 两张数据表的行数是否一致？
SELECT 'conditions' AS table_name, COUNT(*) AS row_count
FROM bike_project.hourly_conditions
UNION ALL
SELECT 'demand', COUNT(*) FROM bike_project.hourly_demand;

-- 2. JOIN 后行数不能膨胀；本项目通过两侧主键约束一对一关系。
SELECT COUNT(*) AS joined_rows
FROM bike_project.hourly_conditions AS c
JOIN bike_project.hourly_demand AS d ON c.observed_at = d.observed_at;

-- 3. 只有天气、没有需求标签的小时，应该为 0。
SELECT COUNT(*) AS conditions_without_demand
FROM bike_project.hourly_conditions AS c
LEFT JOIN bike_project.hourly_demand AS d ON c.observed_at = d.observed_at
WHERE d.observed_at IS NULL;

-- 4. 检查实际覆盖时间。原始数据并非完整连续的小时序列；不擅自补为 0。
SELECT MIN(observed_at) AS first_hour, MAX(observed_at) AS last_hour,
       COUNT(*) AS observed_hours,
       (EXTRACT(EPOCH FROM MAX(observed_at) - MIN(observed_at)) / 3600 + 1)::INTEGER
           - COUNT(*) AS unrecorded_hours
FROM bike_project.hourly_conditions;
