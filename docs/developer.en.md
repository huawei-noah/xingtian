## Developer Guide

Based on the distributed `Broker` design, XingTian divides the reinforcement learning process into two parts: `Learner` and `Explorer`. As shown in the following figure, `Learner` obtains the trajectory data through the broker for iterative update of model policies. While, the Explorer uses Broker to update their exploration models and collect exploration trajectory data.

<div align="center">
<img width="auto" height="480px" src="./.images/broker_arch.png">
</div>



The XingTian reinforcement learning library exposes the interfaces of the Algorithm, Model, Agent, and Environment application modules to developers. The following figure shows the relationship between the four application modules. Algorithm is instantiated in the training process and is used to iteratively update the model weight. The updated iteration weight is distributed to the Agent through the distributed background for interactive exploration in the environment. The Agent is instantiated only in the data sampling process. Developers can add classes based on their requirements and register the classes with the system. Then the classes can be combined in the YAML configuration file. 

<div align="center">
<img width="auto" height="240px" src="./.images/four_opening_module.png">
</div>


### Add Algorithm

The working directory of the Algorithm  module is `xt/algorithm`. 

The system provides the Algorithm base class to abstract phased operations in the algorithm training pipeline, including ：

- `prepare_data`： data preparing for training
- `train_ready`：whether call train operation
- `update_weights_map`：Updates the policy relationship between agents and weights in the multi-agent module. 
- `checkpoint_ready`：check whether need checkpoint
- `predict`：inference with state
- `save`：save model
- `restore`：restore model

Create a target folder in the `xt/algorithm` directory and implement `YOUR_ALGORITHM.py` in the folder.  After inheriting the base class, Developer only need to implement the `prepare_data` and `train` interfaces. and register with the system through `@Registers.algorithm`. The following is an example: 

```python
from xt.algorithm import Algorithm
from zeus.common.util.register import Registers


@Registers.algorithm
class NewAlgorithm(Algorithm):
    def train(self):
        # your train processing, to update the weights of models
    
    def prepare_data(self, train_data, **kwargs):
        # train_data point to the total trajectory of all agents
```



### Add Model

The working directory of the Model module is `xt/model`. 

The Model module is used to define the architecture of the deep network to perform the inference and training processes of the network. Considering the differences between the backends of different deep learning frameworks, the system abstracts the `XtModel` base class of Tensorflow. Users who use the Tensorflow backend can directly inherit this base class. The PyTorch backend users can inherit the `torch.nn.Module` base class based on the traditional deep learning method to implement logic such as model definition, prediction, and training. Register with the system through `@Registers.model`. The following is an example: 

```python
import torch
from zeus.common.util.register import Registers

# Pytorch 
@Registers.model
class NewPyTorchModel(torch.nn.Module):
    def __init(self, model_info):
        # init model architecture
    def forward(self, inputs):
        # inference 
        
# Tensorflow 
# tf_compat cover compactibility among different versions tensorflow
from xt.model.tf_compat import K, Dense, Input, Model, Adam  
from xt.model import XTModel

@Registers.model
class NewTFModel(XTModel):
    def create_model(self, model_info):
        # create model architecture
        
    def train(self, state, label):
        # train process
```

### Add Agent

The Agent module is responsible for the interaction logic between the algorithm and the environment, and integrates data of different trajectory in the multiagent. The working directory is `xt/Agent`. Generally, developers only need to inherit the Agent base class and implement the `infer_action` and `handle_env_feedback` interfaces. 

```python
from zeus.common.util.register import Registers
from xt.agent import Agent

@Registers.agent
class NewAgent(Agent):
    def infer_action(self, state, use_explore):
        """
        Infer an action with the `state`
        :param state:
        :param use_explore: Used True, in train, False in evaluate
        :return: action value
        """
        
    def handle_env_feedback(self, next_raw_state, reward, done, info, use_explore):
        # do process on the environment's feedback
```



### Add Environment

The environment module encapsulates the differences between different environments and provides APIs compatible with `Gym`. Developers only need to implement the `init_env`, `reset`, and `step` methods. Its working directory is `xt/environment`. 

```python
from xt.environment.environment import Environment
from zeus.common.util.register import Registers


@Registers.env
class NewEnv(Environment):
    """It encapsulates an openai gym environment."""

    def init_env(self, env_info):
        """
        create a atari environment instance

        :param: the config information of environment.
        :return: the instance of environment
        """

    def reset(self):
        """
        reset the environment, if done is true, must clear obs array

        :return: the observation of gym environment
        """
        return state

    def step(self, action, agent_index=0):
        """
        Run one timestep of the environment's dynamics.
        Accepts an action and returns a tuple (state, reward, done, info).

        :param action: action
        :param agent_index: the index of agent
        :return: state, reward, done, info
        """
        return state, reward, done, info
```

