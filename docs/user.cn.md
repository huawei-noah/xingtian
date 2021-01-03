## 用户使用手册


### 快速入门的例子

```shell
# 刑天主目录
cd rl

# 训练任务
xt_main -f examples/cartpole_ppo.yaml

# 评估任务
xt_main -f examples/cartpole_ppo.yaml -t evaluate

# 显示帮助
>>> xt_main --help

usage: xt_main [-h] -f CONFIG_FILE [-s3 SAVE_TO_S3]
               [-t {train,evaluate,train_with_evaluate}] [-v VERBOSITY]

XingTian Usage.

optional arguments:
  -h, --help            show this help message and exit
  -f CONFIG_FILE, --config_file CONFIG_FILE
                        config file with yaml
  -s3 SAVE_TO_S3, --save_to_s3 SAVE_TO_S3
                        save model/records into s3 bucket.
  -t {train,evaluate,train_with_evaluate}, --task {train,evaluate,train_with_evaluate}
                        task choice to run xingtian.
  -v VERBOSITY, --verbosity VERBOSITY
                        logging.set_verbosity
```


训练启动的时候会打印工作目录（workspace），模型和相关评估的结果会保存在该目录下

```zsh
INFO Dec 26 11:08:22: **********
workspace:
        /home/User/xt_archive/xt_CartPole-v0_PPO+201226110822T0
```



刑天训练任务的参数由`agent, algorithm, model, environment ` 四个模块组成. 使用者可通过设置四个模块的参数，来自行组合已定义的模块. 如用户需要自定义模块，请参考[开发者指导](./docs/developer.cn.md))。 Benchmark 参数用于配置结果相关数据的记录，并进行对比。其主要参数的含义示下。

### yaml配置文件的含义

```yaml
alg_para:                                               # 算法模块的参数
  alg_name: PPO                                         # 系统注册的算法名称，默认为类名

  alg_config:                                           
    process_num: 1                                      # 训练是否启用多进程(完善中)
    only_save_best_model: True                          # 保存模型的策略(完善中)

env_para:                                               # 环境模块的参数
  env_name: GymEnv                                      # 系统注册的环境名称，默认为类名
  env_info:                                             # 仿真器的具体map/游戏名称
    name: CartPole-v0
    vision: False 

agent_para:                                             # Agent的参数
  agent_name: PPO                                       # 系统注册的Agent名称，默认为类名
  agent_num : 1                                         # 生存在同一环境下的agent数量
  agent_config:
    max_steps: 200                                      # 每个episode的交互步数
    complete_step: 50000                                # 整个训练探索的最大步数
    complete_episode: 3550                              # 整个训练交互的最大episode数

model_para:                                             # 模型模块的参数
  actor:                                                # 算法默认包含一个名为actor的模型
    model_name:  PpoMlp                                 # 系统注册的模型名称，默认为类名
    state_dim: [4]                                      # 模型的输入空间维度
    action_dim: 2                                       # 模型的输出空间维度
    summary: False                                      # 是否打印模型结构信息

env_num: 10                                             # 每个节点下并行多实例explorer的数量

# 设置节点的账号信息，支持列表设置多个节点，进行分布式的训练任务
# 1.如果用户使用本地节点实验，可不用设置node_config信息，系统会自动配置该信息
# 2.如果用户需要使用非本地节点进行实验，必须设置所有的节点信息，包含本地节点账户信息
# node_config: [["127.0.0.1", "username", "passwd"]]      # 各actor运行的节点信息

# test_node_config: [["127.0.0.1", "user", "passwd"]]    # 评估节点信息，可支持同时训练与评估

# remote_env:                                           # 支持远端环境
#  conda: /home/user_test/anaconda2/envs/xt_qmix        # 远端conda环境
#  env:                                                 # 支持设置远端环境变量
#    SC2PATH: /home/user_test/user-proj/marl_sim/StarCraftII
#    no_proxy: "192.168.1.*,127.0.0.1,10.*"

#benchmark:                                             # benchmark 信息
## ‘+’ 是ID中的连接符，如果字符中包含该字符， 系统将直接使用该ID，不会添加时间戳等信息。
#  id: xt_cartpole            # default: default_ENV_ALG ('+'.join([ID, START_time]))
#  archive_root: ./xt_archive # default: ~/xt_archive   # 评估信息归档的根目录，会自动分配
#  eval:
#    model_path: /xt_archive/model_data/cartpole_0      # 需进行评估的模型路径
#    gap: 20                                            # 每训练多少次进行一次评估，并归档
#    model_divided_freq: 1                              # 把同一模型分发到多少个节点并行测试
#    episodes_per_eval: 2                               # 每次评估跑多少轮episode	 
#    evaluator_num: 1 	                                # 支持并行评估的实例数量设置
#    max_step_per_episode: 2000                         # 每次评估最大步数

```



默认使用 TensorboardX 展示训练状态信息，并且将任务相关的records信息保存在`workspace` 目录下。

其中，benchmark目录下保存了该次训练任务的参数配置，训练/评估的回报奖励等关键信息；

```zsh
/home/User/xt_archive/xt_CartPole-v0_PPO+201226110822T0
|-- benchmark
|   |-- records.csv
|   `-- train_config.yaml
|-- events.out.tfevents.1608952102.SZXXXXXXXXXX
|-- models
|   |-- actor_00000.npz
|   |-- actor_00100.npz
|   |-- actor_00200.npz
|   |-- actor_00300.npz
|   |-- actor_00400.npz
|   `-- actor_00500.npz
`-- train_records.json
```
