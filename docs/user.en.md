## User Guide



### Quick Start

```shell
# sourecode of XingTian
cd XingTian

# training task
xt_main -f examples/cartpole_ppo.yaml

# evaluate task
xt_main -f examples/cartpole_ppo.yaml -t evaluate

# show help
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



When the starting, the workspace path will be printed. The model and related evaluation results are saved in the workspace.

```zsh
INFO Dec 26 11:08:22: **********
workspace:
        /home/User/xt_archive/xt_CartPole-v0_PPO+201226110822T0
```



The parameters of the XingTian's training task consist of four modules: agent, algorithm, model, and environment.  Users can combine defined modules by setting the parameters of the four modules.  For details about how to customize modules, please see the [developer guide](./docs/developer.en.md).  The Benchmark parameter is used to configure and compare result data.

The parameters are described as follows:



### Description of the .yaml configuration file

```yaml
alg_para:                                               # parameter for algorithm module
  alg_name: PPO                                         # Algorithm name registed, class name as default

  alg_config:                                           
    process_num: 1                                      # train with multiprocess(coming soon)
    only_save_best_model: True                          # model save strategy (coming soon)

env_para:                                               # parameter for Environment module
  env_name: GymEnv                                      # Environment name registed, class name as default
  env_info:                                             # game name
    name: CartPole-v0
    vision: False 

agent_para:                                             # Agent Parameter
  agent_name: PPO                                       # Agent name registed, class name as default
  agent_num : 1                                         # agent number live under the same environment
  agent_config:
    max_steps: 200                                      # max interaction step within each episode
    complete_step: 50000                                # whole explore steps in once train task

model_para:                                             # Model Parameter
  actor:                                                # By default, contains a model named actor.
    model_name:  PpoMlp                                 # Model name registed, class name as default
    state_dim: [4]                                      # dimensions of state
    action_dim: 2                                       # dimensions of action
    summary: False                                      # summary model to log

env_num: 10                                             # The number of explorer to start up in each node

# set the IP, account, and password of your machine,
# If you want to configure a distributed training network,
# just list all of IP, account, and password for each machine here.
# NOTE:
# 1. use a local node, User do not need to set 'node_config' or 'test_node_config'. The system automatically configures it.
# 2. use a non-local node, User must set all node information, including the local node account information.

# node_config: [["127.0.0.1", "username", "passwd"]]    # account for each node

# test_node_config: [["127.0.0.1", "user", "passwd"]]   # evaluate node account

# remote_env:                                           # remote env
#  conda: /home/test_user/anaconda2/envs/xt_qmix        # remove conda env set
#  env:                                                 # remote env set
#    SC2PATH: /home/test_user/user-proj/marl_sim/StarCraftII
#    no_proxy: "192.168.1.*,127.0.0.1,10.*"

# benchmark:                                            # benchmark info
# The plus sign (+) is a connector in the ID. If the character contains the plus sign (+), the system directly uses the ID and does not add information such as the timestamp.
#  id: xt_cartpole            # default: default_ENV_ALG ('+'.join([ID, START_time]))
#  archive_root: ./xt_archive # default: ~/xt_archive   # Root directory for archiving evaluation information, which is automatically allocated.
#  eval:
#    model_path: /xt_archive/model_data/cartpole_0      # model file to evaluate
#    gap: 20                                            # evaluate interval after training
#    model_divided_freq: 1                              # how many times to divided of each model
#    episodes_per_eval: 2                               # Number of episodes run for each evaluation 	 
#    evaluator_num: 1 	                                # Setting the number of instances that support concurrent evaluation
#    max_step_per_episode: 2000                         # Maximum number of steps for each evaluation

```



By default, `TensorboardX` is used to display the training status information, and the task-related records are stored in the `workspace` directory.

The benchmark directory stores key information such as the parameter configuration and train/evaluate reward information of the training task.

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
