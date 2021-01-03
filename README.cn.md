
<div align="center">
<img width="300px" height="auto" src="./docs/.images/xingtian-logo.png">
</div>

[English](./README.md)

## 简介 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**刑天 (XingTian)** 是一个组件化强化学习库，用于开发、验证强化学习算法。它目前已支持包括DQN、DDPG、PPO和IMPALA等系列算法，可以在多种环境中训练智能体，如Gym、Atari、Torcs、StarCraftII等。 为了满足用户快速验证和解决RL问题的需求，刑天抽象出了四个模块：`Algorithm`,`Model`,`Agent`,`Environment`。它们的工作方式类似于"乐高"积木的组合。更详细的内容请[阅读架构介绍](./docs/basic_arch.cn.md). 

## 系统依赖

```shell
# ubuntu 18.04
sudo apt-get install python3-pip libopencv-dev -y
pip3 install opencv-python

# Run with tensorflow 1.15.0 or tensorflow 2.3.1
pip3 install zmq h5py gym[atari] tqdm imageio matplotlib==3.0.3 Ipython pyyaml tensorflow==1.15.0 pyarrow lz4 fabric2 absl-py psutil tensorboardX setproctitle
```

也可使用pip 进行依赖安装 `pip3 install -r requirements.txt`

如果需要使用Pytorch 作为后端引擎，请自行安装.  [Ref Pytorch](https://pytorch.org/get-started/locally/)




## 安装
```zsh
# cd PATH/TO/XingTian 
pip3 install -e .
```

可通过 `import xt; print(xt.__Version__)`  来确认是否已正常安装. 

```python
In [1]: import xt

In [2]: xt.__version__
Out[2]: '0.3.0'
```



## 快速开始

---------
#### 参数配置
下面是一个有关 [倒立摆](https://gym.openai.com/envs/CartPole-v0/) 简单任务的参数示例，我们通过配置系统中已注册的算法，环境信息来组合训练任务。有关不同参数更详细的描述可以在[用户指导](./docs/user.cn.md) 中找到。


```yaml
alg_para:
  alg_name: PPO
  alg_config:
    process_num: 1
    save_model: True  # default False
    save_interval: 100

env_para:
  env_name: GymEnv
  env_info:
    name: CartPole-v0
    vision: False

agent_para:
  agent_name: PPO
  agent_num : 1
  agent_config:
    max_steps: 200
    complete_step: 1000000
    complete_episode: 3550

model_para:
  actor:
    model_name: PpoMlp
    state_dim: [4]
    action_dim: 2
    input_dtype: float32
    model_config:
      BATCH_SIZE: 200
      CRITIC_LOSS_COEF: 1.0
      ENTROPY_LOSS: 0.01
      LR: 0.0003
      LOSS_CLIPPING: 0.2
      MAX_GRAD_NORM: 5.0
      NUM_SGD_ITER: 8
      SUMMARY: False
      VF_SHARE_LAYERS: False
      activation: tanh
      hidden_sizes: [64, 64]

env_num: 10
```

另外在 [examples](./examples) 目录下，可以找到更加丰富的训练配置示例。

#### 开始训练任务

```python3 xt/main.py -f examples/cartpole_ppo.yaml -t train```

![img](./docs/.images/cartpole.gif)  



#### 评估本机模型

在你的`.yaml`文件中设置 `benchmark.eval.model_path`  参数，然后通过 `-t evaluate` 运行评估任务。

```
benchmark:
  eval:
    model_path: /YOUR/PATH/TO/EVAL/models
    gap: 10           # 目录下需评估模型的间隔
    evaluator_num: 1  # 启动评估实例的数量，可支持并行评估

# 运行命令
python3 xt/main.py -f examples/cartpole_ppo.yaml -t evaluate
```

> 系统默认启动训练任务，即 -t 的默认选项是 train

#### 使用命令行

```zsh
# 在终端中，可直接使用xt_main 替换 python3 xt/main.py 执行命令
xt_main -f examples/cartpole_ppo.yaml -t train

# train with evaluate
xt_main -f examples/cartpole_ppo.yaml -t train_with_evaluate
```

## 自定义任务的开发

1. 编写自定义模块，并注册。 具体可参考 [开发指导](./docs/developer.cn.md)
2. 在配置文件 `your_train_configure.yaml`中，配置自定义的模块名字
3.  启动训练  `xt_main -f path/to/your_train_configure.yaml` :)

## 实验结果参考

#### 平均的训练回报

1. 10M step 之后的**DQN** 收敛回报 (**40M frames**).

   | env           | XingTian Basic DQN | RLlib Basic DQN | Hessel et al. DQN |
   | ------------- | ------------------ | --------------- | ----------------- |
   | BeamRider     | 6706               | 2869            | ~2000             |
   | Breakout      | 352                | 287             | ~150              |
   | QBert         | 14087              | 3921            | ~4000             |
   | SpaceInvaders | 947                | 650             | ~500              |

2. 10M step 之后的**PPO** 收敛回报 (**40M frames**).

   | env           | XingTian PPO | RLlib PPO | Baselines PPO |
   | ------------- | ------------ | --------- | ------------- |
   | BeamRider     | 4877         | 2807      | ~1800         |
   | Breakout      | 341          | 104       | ~250          |
   | QBert         | 14771        | 11085     | ~14000        |
   | SpaceInvaders | 1025         | 671       | ~800          |

3. 10M step 之后的**IMPALA** 收敛回报 (**40M frames**).

   | env           | XingTian IMPALA | RLlib IMPALA |
   | ------------- | --------------- | ------------ |
   | BeamRider     | 2313            | 2071         |
   | Breakout      | 334             | 385          |
   | QBert         | 12205           | 4068         |
   | SpaceInvaders | 742             | 719          |



#### 吞吐量

1. **DQN**

   | env           | XingTian Basic DQN | RLlib Basic DQN |
   | ------------- | ------------------ | --------------- |
   | BeamRider     | 129                | 109             |
   | Breakout      | 117                | 113             |
   | QBert         | 111                | 90              |
   | SpaceInvaders | 115                | 100             |


2. **PPO**

   | env           | XingTian PPO | RLlib PPO |
   | ------------- | ------------ | --------- |
   | BeamRider     | 2422         | 1618      |
   | Breakout      | 2497         | 1535      |
   | QBert         | 2436         | 1617      |
   | SpaceInvaders | 2438         | 1608      |

3. **IMPALA**

   | env           | XingTian IMPALA | RLlib IMPALA |
   | ------------- | --------------- | ------------ |
   | BeamRider     | 8756            | 3637         |
   | Breakout      | 8814            | 3525         |
   | QBert         | 8249            | 3471         |
   | SpaceInvaders | 8463            | 3555         |

> 实验硬件环境： 72  Intel(R) Xeon(R) Gold 6154 CPU @ 3.00GHz with single Tesla V100
>
> Ray reward数据来自 [https://github.com/ray-project/rl-experiments](https://github.com/ray-project/rl-experiments), 吞吐量来自以上硬件设备的测试数据

## 致谢

刑天参考了以下项目: [DeepMind/scalable_agent](https://github.com/deepmind/scalable_agent), [baselines](https://github.com/openai/baselines), [ray](https://github.com/ray-project/ray).

## 许可证

The MIT License(MIT)
