#### 1.投递单数据处理

1. 1 保留如下字段

   ```python
   '投递单编号', '投递量_大板', '投递量_小板', '投递量_箱', 'demand',
   'start_time', 'end_time', 'load_time', 'unload_time',
   'pickup_id', 'delivery_id'
   ```

1.2 将原字段用相应字段进行替换

```python
'投递单编号': 'order_id', 
'投递量_大板': 'q_standard', 
'投递量_小板': 'q_small', 
'投递量_箱': 'q_box',
'start_time': 'creation_time', 
'end_time': 'committed_completion_time'
```

1.3  creation_time和 committed_completion_time 保留到秒

1.4 保证 如下公式

```python
demand=q_standard+q_small*0.5+q_box*0.25
```

1.5 保证 load_time和unload_time保持一致；并保证如下公式

```python
#单位是秒
load_time=4*60*demand
```

1.6 保证pickup_id 和 delivery_id 在 site_info.csv 里

1.7 订单按照 'creation_time'+四位数index 生成，并按照’create_time'进行排序

#### 2.车辆数据处理

2.1  保留如下字段

```python
'car_num', 'vehicle_type', 'operation_mode'，'gps_id'
```

2.2 将原字段用相应字段进行替换

```python
'vehicle_type':'capacity'
'operation_mode':'operation_time'
```

2.3 gps_id 此字段用 G_1，G_2,..... 表示

2.4 car_num 此字段用 V_1，V_2,..... 表示

#### 3.route_info数据处理

3.1保留如下字段

```python
'route_code','start_site_id','end_site_id','distance','time'
```

3.2 将原字段用相应字段进行替换

```python
'start_site_id':'start_factory_id'
'end_site_id' : 'end_factory_id'
```

3.3 如果站点之间的距离（distance）是0，distance默认为 0.1 ，time默认为 300s

3.4 route_code用uuid替代

3.5 对route_info.csv根据起始站点（start_factory_id，end_factory_id）进行去重处理（去重保留第一个）

3.6 route_info.csv 保留起始站点（start_factory_id，end_factory_id）在factory_info.csv文件里的时空矩阵

3.7 factory_info.csv文件两两组合（组合个数为n*（n-1）），对于原始route_info.csv没有的起始点组合进行补充，距离计算（根据经纬度进行计算），时间根据distance/(30km/h）进行计算

#### 4.站点数据处理

4.1 保留如下字段 ('load_time','unload_time' 是否需要待定)

```python
'site_id','longitude','latitude'
```

4.3 将原字段用相应字段进行替换

```python
'site_id':'factory_id'
```

4.2  经纬度 小数点后保留几位小数（目前暂定保留4位，精确到10m，目前待测试）

4.3 增加 port_num （货口数量），暂定 port_num 均为 6

#### 5.班次数据处理

4.1 保留如下字段

```python
'site_id','shift_name','shift_list'
```

4.2 shift_list 用 string表示