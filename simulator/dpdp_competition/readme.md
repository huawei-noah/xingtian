## Introduction 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**The directory structure of simulator (模拟器的目录结构)**

```
dpdp_competition
	│  main.py  模拟器的主程序
	│  main_algorithm.py  算法的主程序
	│  readme.md  文档说明
	│  
	│─algorithm  存放算法代码的文件夹
	│  │  algorithm_demo.py  算法示例代码
	│  │  
	│  └─data_interaction  算法与模拟器交互数据的文件夹 
	│      
	├─benchmark  测试集
	│
	└─src 存放模拟器代码的文件夹 
		├─common  常用类
		│      dispatch_result.py
		│      factory.py
		│      input_info.py
		│      node.py
		│      order.py
		│      route.py
		│      stack.py
		│      vehicle.py
		│      
		├─conf  配置文件
		│      configs.py
		│      
		├─simulator  模拟器
		│      history.py
		│      simulate_api.py
		│      simulate_environment.py
		│      vehicle_simulator.py
		│      
		└─utils  常用工具
		       checker.py
		       evaluator.py
		       input_utils.py
		       json_tools.py
		       logging_engine.py
		       log_utils.py
		       tools.py
```



**Run the simulator（启动模拟器）**

```
# cd dpdp_competition
python main.py
```



## Requirements

* python >= 3.6
* simpy



## Quick Start
### Interaction Between algorithm and simulator

Firstly, the simulator reads the selected test instance which can be modified in Configs.py. Then, the simulator performs simulation at a fixed interval of 10 minutes until all orders in the instance are completed.

In each round of simulation, the simulator outputs the vehicle and order data required by the algorithm to the algorithm/data_interaction folder in JSON format. Next, it calls the main program of the algorithm prefixed with "main_algorithm", e.g. main_algorithm.py, main_algorithm.java, e.t.c. When the algorithm runs, it starts to read JSON files, dispatch orders, output the dispatch result to the algorithm/data_interaction folder in JSON format and print string "SUCCESS" to the console as the flag for the simulator to determine whether the algorithm is successfully returned. If the successful returned flag of the algorithm is obtained, the simulator would read the output JSON files and do the verification. After passing the verification, it continues to simulate.

The running time of the algorithm is limited to 10 minutes. If the algorithm times out, the simulator will exit.

Note: Pathes of folders and files mentioned above are illustrated in the Introduction section.



启动模拟器后，模拟器首先读取选定测试例，测试例的选择可在Configs.py中修改，接着按照固定时间间隔10min来进行模拟，直到完成测试例中的所有订单。

每轮次模拟中，模拟器把算法需要的数据通过JSON文件的形式输出至algorithm/data_interaction文件夹中，接着调用算法主程序（以main_algorithm作为前缀的文件，例如main_algorithm.py, main_algorithm.java等）。当算法成功启动后，会读取JSON文件、派单、把派单结果通过json文件的形式输出至algorithm/data_interaction文件夹中，并且输出"SUCCESS"字符串至**控制台【非日志】**，作为模拟器判断算法是否成功返回的标识。如果成功获取到算法成功返回的标识，模拟器会读取算法输出的JSON文件，并进行校验，通过校验后，继续进行模拟仿真。

**当前算法每次运行时长限定为10分钟，超时模拟器将主动退出。**

注：上面提到的路径和文件名详见Introduction部分的“模拟器的目录结构”



### Main Process (from algorithm perspective)

+ [Read the input json files](#Read the input json files)
+ [Dispatch orders](#Dispatch orders) 
+ [Output required json files](#Output required json files) 




### Read the input json files

The simulator outputs the latest vehicle information in "vehicle_info.json", order items to be allocated in "unallocated_order_items.json" and ongoing order items in "ongoing_order_items.json". The details of these files are in the following.

Three concepts are explained here:

1. Item, which is the smallest unit in an order. For example, if an order contains 2 standard pallets and 1 small pallets, there are 3 items included: item 1 of one standard pallet, item 2 of one standard pallet and item 3 of 1 small pallet.
2. Item status, 0 means initialization; 1 means the item is generated; 3 means the item has been loaded; 4 means the item is delivered.
3. Order status, which is the minimum value of all corresponding items statuses. For example, an order contains two items, namely, item 1 of 1 small pallet and item 2 of 1 box. The status of item 1 is 3 and item 2 is 2, the order status is 2.



模拟器会输出当前最新的车辆信息"vehicle_info.json"、未分配物料信息"unallocated_order_items.json"和已装货物料信息"ongoing_order_items.json"至algorithm/data_interaction文件夹中，下面具体介绍这三个JSON文件。

这里解释三个概念：
 	1. 物料，物料是订单不可切分的最小单元，例如一个订单含有2个标准板, 1个小板，那么其对应的物料有3个, 分别是标准板物料1，标准板物料2和小板物料3；
 	2. 物料的状态，0：初始化状态，1：物料已经生成，2：物料已经装车，3：物料已经送达；
 	3. 订单的状态，取其包含所有物料状态的最小值，例如一个订单包含两个物料，分别是小板物料1和箱子物料2，其中小板物料1的状态为3，箱子物料2的状态为2，那么订单的状态为2



+ **vehicle_info.json**

  Example: 

```
[
  {
    "id": "V_1",
    "operation_time": 24,
    "capacity": 15,
    "gps_id": "G_1",
    "update_time": 1621471200.0,
    "cur_factory_id": "f6faef4b36e743328800b961aced4a2c",
    "arrive_time_at_current_factory": 1621471008.0,
    "leave_time_at_current_factory": 1621473288.0,
    "carrying_items": [
      "2218470047-1",
      "2218470047-2"
    ],
    "destination": {
      "factory_id": "b6dd694ae05541dba369a2a759d2c2b9",
      "delivery_item_list": [
        "2218470047-2",
        "2218470047-1"
      ],
      "pickup_item_list": [],
      "arrive_time": 1621473396.0,
      "leave_time": 1621475676.0
    }
  },
  {
    "id": "V_2",
    "operation_time": 24,
    "capacity": 15,
    "gps_id": "G_2",
    "update_time": 1621471200.0,
    "cur_factory_id": "5e2e9efa5ade4984bb18af66028bc0c8",
    "arrive_time_at_current_factory": 1621464180.0,
    "leave_time_at_current_factory": 1621470600.0,
    "carrying_items": [],
    "destination": null
  },
  ...
]
```

| Column                         | Description                                                  | Type              |
| ------------------------------ | ------------------------------------------------------------ | ----------------- |
| id                             | id of vehicle, 车牌号                                        | str               |
| operation_time                 | operation time of vehicle(unit: hour), 车辆的运营时长        | int               |
| capacity                       | capacity of vehicle (unit: standard pallet, 单位: 标准栈板)  | int               |
| update_time                    | Update time of the current position and status of the vehicle(unit: unix timestamp).<br/>车辆当前位置和状态的更新时间 | int               |
| cur_factory_id                 | the factory id where the vehicle is currently located. If the vehicle currently is not in any factory, the value is "".  <br/>车辆当前所在的工厂, 如果当前不在任何工厂则为""。 | str               |
| arrive_time_at_current_factory | The time when the vehicle arrives at the current factory(unit: unix timestamp). <br/>车辆到达当前所在工厂的时间 | int               |
| leave_time_at_current_factory  | The time when the vehicle leaves the current factory(unit: unix timestamp). <br/>车辆离开当前所在工厂的时间 | int               |
| carrying_items                 | List of items loaded on the vehicle, in the order of loading.<br/>车辆当前载的物料编号列表，顺序为装货顺序 | [str1, str2, ...] |
| destination                    | Current destination of the vehicle. Once determined, the destination cannot be changed until the vehicle has arrived. The destination is None when the vehicle is parked.<br/>当前车辆的目的地, 目的地一旦确定不可更改, 直到车辆到达目的地节点，如果当前车辆处于停车状态，目的地为空 | dict              |

Notes： if vehicle v is in the factory f1, the destination is f2, then the route plan {f3, f2} violates the destination invariant constraint, while {f2, f3} does not



+ **unallocated_order_items.json**
  Example：

```
[
  {
    "id": "1436011625-1",
    "type": "HALF_PALLET",
    "order_id": "1436011625",
    "demand": 0.5,
    "pickup_factory_id": "7b84670cf4164cccba22ebb17a2c290a",
    "delivery_factory_id": "7fb0acc4e2634440ba26a7ebc0040dc2",
    "creation_time": 1621406161,
    "committed_completion_time": 1621420561,
    "load_time": 120,
    "unload_time": 120,
    "delivery_state": 1
  },
  {
    "id": "1446161669-1",
    "type": "PALLET",
    "order_id": "1446161669",
    "demand": 1,
    "pickup_factory_id": "2445d4bd004c457d95957d6ecf77f759",
    "delivery_factory_id": "9f1a09c368584eba9e7f10a53d55caae",
    "creation_time": 1621406776,
    "committed_completion_time": 1621421176,
    "load_time": 240,
    "unload_time": 240,
    "delivery_state": 1
  },
```

| Column                    | Description                                                  | Type   |
| ------------------------- | ------------------------------------------------------------ | ------ |
| id                        | id of item, 物料编号                                         | str    |
| type                      | pallet type, e,g, standard pallet, small pallet and box</br>物料对应托盘类型 | str    |
| order_id                  | id of order, 订单编号                                        | str    |
| demand                    | total standard pallet amount                                 | double |
| pickup_factory_id         | id of pickup factory, 取货地                                 | str    |
| delivery_factory_id       | id of delivery factory, 送货地                               | str    |
| creation_time             | creation time of the corresponding order(unit: unix timestamp), 所属订单的创建时间 | int    |
| committed_completion_time | committed completion time of the corresponding order(unit: unix timestamp), 所属订单的承诺送达时间 | int    |
| load_time                 | loading time of item(unit: second), 该物料的装载时间         | int    |
| unload_time               | unloading time of item(unit: second), 该物料的卸货时间       | int    |
| delivery_state            | Item status, 0 means initialization; 1 means the item is generated; 3 means the item has been loaded; 4 means the item is delivered. | int    |



+ **ongoing_order_items.json**
  Example：

```
[
  {
    "id": "0003480001-1",
    "type": "HALF_PALLET",
    "order_id": "0003480001",
    "demand": 0.5,
    "pickup_factory_id": "2445d4bd004c457d95957d6ecf77f759",
    "delivery_factory_id": "b6dd694ae05541dba369a2a759d2c2b9",
    "creation_time": 1621267428,
    "committed_completion_time": 1621281828,
    "load_time": 120,
    "unload_time": 120,
    "delivery_state": 2
  },
  {
    "id": "0012230002-1",
    "type": "PALLET",
    "order_id": "0012230002",
    "demand": 1,
    "pickup_factory_id": "2445d4bd004c457d95957d6ecf77f759",
    "delivery_factory_id": "9f1a09c368584eba9e7f10a53d55caae",
    "creation_time": 1621267943,
    "committed_completion_time": 1621282343,
    "load_time": 240,
    "unload_time": 240,
    "delivery_state": 2
  },
  ...
]

```




### Dispatch orders

1. When the vehicle $v$ arrives at the factory $f$, the pickup and delivery list of $v$ in $f$ will be generated immediately. The vehicle can only be loaded and unloaded according to the list. We can only change the pickup and delivery items of $v$ in $f$ when the corresponding pickup and delivery list is not generated.
2. Algorithm can reallocate the item to different vehicles as long as this item is not not displayed on the  pickup and delivery list.
3. Considering that the input is the items, the algorithm needs to pay attention to the order splitting constraint. If the order does not exceed the vehicle's capacity (vehicles are homogeneous), the order cannot be split.
4. The algorithm can control the order release. For example, if order A is generated at $t_1$, the algorithm can delay the allocation until $t_2 <= t_1 + 4h$. Note: If an order has been generated for more than 4 hours but is still not dispatched, the simulator will exit.
5. The traveling distance and time required for the algorithm can be obtained only from the distance and time matrix between factories in the benchmark. Do not calculate the distance and time based on the longitude and latitude. If the vehicle $v$ is in transit,  $v$ must have a destination factory $f$, and the simulator will give the estimated time for $v$ to arrive at $f$. The algorithm can plan the route of vehicle $v$ based on the destination $f$.
6. Assume that the simulator sends the latest status information of all vehicles to the algorithm at $t_1$, and the algorithm returns the dispatching result at $t_2$. If the algorithm runs for a long time, the status of vehicles will change greatly after pasing  the time interval $[t_1, t_2]$. As a result, the dispatching result is inconsistent with the actual situation, which affects the running of vehicles. Currently, the running time of the algorithm is limited to 10 minutes.



1. 车辆$v$到达工厂$f$, 就会立刻生成$v$在$f$的取送货清单，此时车辆只能按照清单进行装卸操作，不可以加塞和减少；如果清单未生成，即$v$未到达$f$, 则$v$在$f$的装卸物料可以改变
2. 物料只要未在取送货清单上出现都可以重新分配车辆
3. 考虑到输入的是订单包含物料的信息，算法需要注意订单切分约束，订单未超过车辆载重的(车辆是同质的)，不可以进行切分
4. 算法可以控制订单的释放，例如，$t_1$时刻生成订单A，算法可以不用立即分配，延迟到$t_2 <= t_1 + 4h$再分配。注意如果一个订单已经生成超过4小时，但是算法依旧未分配，模拟器会自动退出
5. 算法计算需要的距离和时间，仅可通过benchmark中工厂之间的距离和时间矩阵进行获取，不要通过经纬度自行计算。如果车辆$v$在运行途中，那么该车一定会有一个目的地工厂$f$，而且模拟器会给出$v$预计到达$f$的时间，后续算法规划$v$的路径可以基于目的地$f$进行计算
6. 假设模拟器将$t_1$时刻的最新车辆位置和装载情况传给算法，算法在$t_2$时刻回传派单结果，如果算法运行时间过长，那么经过$[t_1, t_2]$车辆的状态会发生很大改变，导致派单结果与实际不符，从而影响实际车辆的运行，目前模拟器限制算法的运行时长为10min



### Output required json files

The algorithm needs to output two json files including "output_destination.json" and "output_route.json". The details of these files are in the following.

+ **output_destination.json**

  Example

```
{
  "V_1": {
    "factory_id": "2445d4bd004c457d95957d6ecf77f759",
    "lng": 116.5841,
    "lat": 40.2869,
    "delivery_item_list": [],
    "pickup_item_list": [
      "0003480001-1"
    ],
    "arrive_time": 1621445928.0,
    "leave_time": 0
  },
  "V_2": {
    "factory_id": "9f1a09c368584eba9e7f10a53d55caae",
    "lng": 116.6309,
    "lat": 40.2304,
    "delivery_item_list": [
      "0013570003-1"
    ],
    "pickup_item_list": [],
    "arrive_time": 1621443624.0,
    "leave_time": 0
  },
  "V_3": null,
  ...
} 
```



+ **output_route.json**

  Example

```
{
  "V_1": [
    {
      "factory_id": "b6dd694ae05541dba369a2a759d2c2b9",
      "lng": 116.6264,
      "lat": 40.2253,
      "delivery_item_list": [
        "0003480001-1"
      ],
      "pickup_item_list": [],
      "arrive_time": 0,
      "leave_time": 0
    },
    {
      "factory_id": "2445d4bd004c457d95957d6ecf77f759",
      "lng": 116.5841,
      "lat": 40.2869,
      "delivery_item_list": [],
      "pickup_item_list": [
        "0012230002-1"
      ],
      "arrive_time": 0,
      "leave_time": 0
    },
    {
      "factory_id": "9f1a09c368584eba9e7f10a53d55caae",
      "lng": 116.6309,
      "lat": 40.2304,
      "delivery_item_list": [
        "0012230002-1"
      ],
      "pickup_item_list": [],
      "arrive_time": 0,
      "leave_time": 0
    }
  ],
  "V_2": [
    {
      "factory_id": "2445d4bd004c457d95957d6ecf77f759",
      "lng": 116.5841,
      "lat": 40.2869,
      "delivery_item_list": [],
      "pickup_item_list": [
        "0033520004-1",
        "0033520004-2",
        "0033520004-3"
      ],
      "arrive_time": 0,
      "leave_time": 0
    },
    {
      "factory_id": "b6dd694ae05541dba369a2a759d2c2b9",
      "lng": 116.6264,
      "lat": 40.2253,
      "delivery_item_list": [
        "0033520004-3",
        "0033520004-2",
        "0033520004-1"
      ],
      "pickup_item_list": [],
      "arrive_time": 0,
      "leave_time": 0
    }
  ],
  "V_3": [],
  ...
}
```

| Column             | Description                                                  | Type              |
| ------------------ | ------------------------------------------------------------ | ----------------- |
| factory_id         | id of factory, 节点对应的工厂id                              | str               |
| lng                | longitude, 节点对应工厂的经度                                | double            |
| lat                | latitude, 节点对应工厂的纬度                                 | double            |
| delivery_item_list | List of items unloaded from the vehicle, (Given the length of delivery_item_list is n, the unloading sequence is delivery_item_list[0], delivery_item_list[1], ..., delivery_item_list[n-1]</br>送货物料清单 | [str1, str2, ...] |
| pickup_item_list   | list of items loaded on the vehicle (Given the length of pickup_item_list is n, the loading sequence is pickup_item_list[0], pickup_item_list[1], ..., pickup_item_list[n-1])</br> 取货物料清单 | [str3, str4, ...] |
| arrive_time        | time to reach the node(unit: unix timestamp), 到达节点的时间 | int               |
| leave_time         | time to leave the node(unit: unix timestamp), 离开节点的时间 | int               |




## Submission

**To be continued**

After the online submission of the HUAWEI CLOUD competition platform is open, we will complete the submission process document as soon as possible



## License

The MIT License(MIT)







