### Introduction

**Given Benchmark contains 64 instances stored in different folders. Each folder includes the order and vehicle data.**

#### Orders (订单数据) ：

| Column                    | Description                                                  | Example                          |
| ------------------------- | ------------------------------------------------------------ | -------------------------------- |
| order_id                  | id of order                                                  | 0003480001                       |
| q_standard                | standard pallet amount                                       | 1                                |
| q_small                   | small pallet amount                                          | 2                                |
| q_box                     | box amount                                                   | 1                                |
| demand                    | total standard pallet amount $demand = q\_standard + 0.5 \times q\_small + 0.25 \times q\_box$ | 1.75                             |
| creation_time             | creation time ('%H:%M:%S')                                   | 00:03:48                         |
| committed_completion_time | committed completion time ('%H:%M:%S')                       | 04:03:48                         |
| load_time                 | loading time (unit: second)                                  | 120                              |
| unload_time               | unloading time (unit: second)                                | 120                              |
| pickup_id                 | id of pickup factory                                         | 2445d4bd004c457d95957d6ecf77f759 |
| delivery_id               | id of delivery factory                                       | b6dd694ae05541dba369a2a759d2c2b9 |

#### Vehicles (车辆数据)：

| Column         | Description                                                 | Example |
| -------------- | ----------------------------------------------------------- | ------- |
| car_num        | id of vehicle                                               | V_1     |
| capacity       | capacity of vehicle (unit: standard pallet, 单位: 标准栈板) | 15      |
| operation_time | operation time of vehicle(unit: hour), 车辆的运营时长       | 24      |
| gps_id         | id of gps equipment                                         | G_1     |

#### Route Map (地图数据) （route_info.csv）：

| Column           | Description                                     | Example                              |
| ---------------- | ----------------------------------------------- | ------------------------------------ |
| route_code       | id of route                                     | e7eeb0e4-a7c7-11eb-8344-84a93e824626 |
| start_factory_id | start factory id of the route                   | 7782ed919d8f4dd6a1fb220dacd73445     |
| end_factory_id   | end factory id of the route                     | 43a4215be06543c1985c1e9460dec52d     |
| distance         | distance of the route (unit: km)                | 76.0                                 |
| time             | transportation time of the route (unit: second) | 10140                                |

#### Factory (工厂数据) （factory_info.csv）： 

| Column     | Description                                                  | Example                          |
| ---------- | ------------------------------------------------------------ | -------------------------------- |
| factory_id | id of factory                                                | 9829a9e1f6874f28b33b57a7a42bb49f |
| longitude  | longitude                                                    | 116.6259                         |
| latitude   | latitude                                                     | 40.2204                          |
| port_num   | the number of ports used for loading and unloading of vehicle cargos (工厂装卸货物的货口数量) | 6                                |

