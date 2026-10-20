-- 显式列名，避免 SELECT * 意外把标签衍生变量作为输入。
-- 日期边界在看测试指标前固定；按时间排序，不能随机打散再切分。
SELECT c.observed_at, c.hour, c.weekday, c.month, c.year_index,
       c.holiday, c.workingday, c.weather_code,
       c.temperature, c.humidity, c.windspeed,
       d.rental_count,
       CASE
           WHEN c.observed_at < TIMESTAMP '2012-07-01' THEN 'train'
           WHEN c.observed_at < TIMESTAMP '2012-09-01' THEN 'validation'
           ELSE 'test'
       END AS split
FROM bike_project.hourly_conditions AS c
JOIN bike_project.hourly_demand AS d ON c.observed_at = d.observed_at
ORDER BY c.observed_at;
