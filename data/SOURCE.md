# 数据来源和使用范围

- 名称：UCI Bike Sharing，使用 `hour.csv`，2011—2012 年 Capital Bikeshare 小时记录。
- 来源：https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
- 下载：https://archive.ics.uci.edu/static/public/275/bike%2Bsharing%2Bdataset.zip
- 引用：Fanaee-T, H. (2013). Bike Sharing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.
- 许可：CC BY 4.0，https://creativecommons.org/licenses/by/4.0/ 。本项目保留原始数据与来源署名。
- 实际读取 `hour.csv`：17,379 行、17 列。UCI 页面元数据的实例数与文件不同，本项目以下载文件实际计数为准。
- 原始 CSV SHA-256：`e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f`。
- 时间戳使用数据原有日期与小时，不推断时区转换。
- `temp`、`hum`、`windspeed` 已按原始说明归一化；它们不是原始摄氏温度、湿度百分数和风速。
- `cnt = casual + registered`。后两列仅在导入校验时使用，不存入输入表、不输入网络。
- 该数据集没有逐车位置、库存或站点 ID；模型输出为系统级需求，不涉及站点级调度优化。
- 天气是当小时的实测值。本项目是历史条件下的需求估计与离线评估；若做提前预测，需要另外接入当时可用的天气预报，重新回测。
