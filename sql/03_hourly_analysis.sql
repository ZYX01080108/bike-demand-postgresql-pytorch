-- PostgreSQL 负责关联和分组，Python 读取结果画图。
-- 单位：每个有记录小时的平均租车次数，不是每辆车的周转次数。
SELECT c.workingday, c.hour,
       COUNT(*) AS observed_hours,
       AVG(d.rental_count) AS mean_rentals
FROM bike_project.hourly_conditions AS c
JOIN bike_project.hourly_demand AS d ON c.observed_at = d.observed_at
GROUP BY c.workingday, c.hour
ORDER BY c.workingday, c.hour;
