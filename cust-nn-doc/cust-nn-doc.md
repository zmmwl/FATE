# FATE-Homo-NN升级方案-v2

## 项目概况
| 模块名称 | Homo Cust NN |
| ----: | :----: |
| 系统 | FATE-1.10 |
| 模块负责人 | 陈伟敬 |
| 项目负责人 | 范涛 |
| 文档日期 | 2022-9-20 |

## 背景
Homo-NN目前与Hetero-NN一样，存在定制化困难的问题，在FATE-1.10版本，计划对Homo-NN进行一次升级，提高Homo-NN的定制化能力，使用户能大体按照Homo-NN的范式，基于pytorch编写训练脚本，并让pytorch模型在Fate框架运行起来。这一切都仅仅需要比较低的学习成本。

## 升级简介

提供一个Custom-Homo-NN框架，用户基于基类编写训练脚本，并自行完成参数聚合逻辑，将脚本存在指定目录下。
运行任务时候，Custom-Homo-NN框架会自动导入训练脚本，执行训练与用户定义的聚合逻辑，以达到横向联邦的效果
	本次工作，包含的主要内容：
	
1. 提供用户开发homo模型的基类，接口
	
2. 提供直接可用的Aggregator，Aggregator应该为一些业界的标准，如FedAVG, FedAVGM, FedProx, FedOpt 等
	
3. 提供用户开发自定义Aggregator的基类，接口
          
4. 整理framework下的homo模块，主要整理aggregator部分，让其他算法（如homo-sbt)能够直接使用这次升级开发的aggregator(如homo-sbt)

## Homo-Cust-NN 模块整体设计

本次升级期望达到以下效果，对于一个用户，想要开发一个pytorch homo-nn模型，需要三步

1. 继承NNModuel类，在train接口种像开发pytorch本地一样开发一个训练脚本（其中，数据IO，参数IO可由用户自行读取和写死在脚本里，或者通过train接口的参数拿到），

    脚本必须放在指定位置（nn文件夹下）
    
2. import aggregator类，初始化aggregator类，调用aggregate接口聚合模型

3. 在conf/dsl里指定自己开发的NNModule类，提交任务

**新增内容**

+ **更新框架设计，增加了Aggregator模块相关**
+ **框架中取消了guest，仅仅只有host与arbiter**

### 目录结构
此处展示关联到项目的代码目录结构，其中有cust_nn以及整合到homo framework下的 aggregator
```
./fate/python/federatedml/nn/custom_nn/
├── homo_nn_arbiter.py
├── homo_nn_client.py
├── homo_nn_param.py
|
├── nn
│ ├── dense_layer.py
│ ├── alexnet.py
│ ├── BERT_model.py
│ ├── partial_aggregate.py
|
├── nn_base_module.py

./fate/python/federatedml/framework/homo/ (partial)
├── aggregator
│ ├── agg_base.py
│ ├── fedavg_aggregator.py 
| ├── fedavgm_aggregator.py 
| ├── fedprox_aggregator.py
```

其中，

+ nn：该模块存放用户自己开发的NN脚本

+ nn_base_module.py:
提供基础类NNBaseModule，NNBaseModule为用户继承开发的基类，NNBaseModule中，用户需要自己实现train, predict接口，
在train中，用户自己定义训练流程，并可import 各种aggregator完成模型聚合
同时用户根据需要，调用export等IO接口

+ homo_nn_arbiter/client/param .py：
这三个文件为Fate运行算法时，需要的类; 首先，用户通过param类传入参数，即需要import的模块，与用户自己开发的NNModule类
在homo_nn_client运行时执行用户自定义类的train, predict接口
homo_nn_arbiter执行聚合

+ /fate/python/federatedml/framework/homo/aggregator: 该模块存放可直接使用的aggregator, agg_base为aggregator基类


### 工作流程

![图片alt](./structure.jpg '工作流程')


### 样例代码 

```python
import torch as t
import pandas as pd
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, Dataset
from federatedml.custom_nn.nn_base_module import NNBaseModule
from federatedml.util import LOGGER
from federatedml.util import consts


class TestNet(nn.Module):

    def __init__(self, input_size):
        super(TestNet, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.activation = nn.Sigmoid()

    def forward(self, input_data):
        out = self.seq(input_data)
        return self.activation(out)


class TestDataset(Dataset):

    def __init__(self, feat, label):
        self.feat = feat
        self.label = label

    def __getitem__(self, item):
        return self.feat[item], self.label[item]

    def __len__(self):
        return len(self.label)


class TestNetModule(NNBaseModule):

    def __init__(self):
        super(TestNetModule, self).__init__()
    
    # step 1 用户自己实现训练流程 数据IO 模型初始化 等等 与本地pytorch编写一致
    def train(self, cpn_input, **kwargs):

        LOGGER.debug('input data is {}'.format(cpn_input))

        epochs = kwargs['epochs']
        lr = kwargs['lr']
        test_batch_size = kwargs['batch_size']

        LOGGER.debug('party id is {}'.format(self.party_id))

        if self.party_id == 9999:
            df_train = pd.read_csv('/examples/data/epsilon_5k_homo_guest.csv')
        elif self.party_id == 9998:
            df_train = pd.read_csv('/examples/data/epsilon_5k_homo_host.csv')

        label = np.array(df_train['y']).astype(np.float32)
        id_ = df_train['id']
        features = df_train.drop(columns=['id', 'y']).values
        features = np.array(features).astype(np.float32)
        dataset = TestDataset(features, label)
        dl = DataLoader(dataset, batch_size=test_batch_size)
        
        # step 2 用户自己导入Aggregator
        from federatedml.framework.homo.aggregator import FedAVGClient
        fedavg = FedAVGClient(aggregate_round=epochs, secure_aggregate=False, early_stop=False,
                              aggregate_type='mean')

        self.model = TestNet(100)
        optimizer = t.optim.Adam(self.model.parameters(), lr=lr)
        loss_func = t.nn.BCELoss()

        for i in range(epochs):
            LOGGER.debug('epoch is {}'.format(i))
            epoch_loss = 0
            batch_idx = 0
            for batch_data, batch_label in dl:
                optimizer.zero_grad()
                pred = self.model(batch_data)
                batch_loss = loss_func(pred.flatten(), batch_label.flatten())
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.detach().numpy()
                batch_idx += 1
            epoch_loss = epoch_loss / batch_idx

            # federation proces  step 3 用户使用聚合模型
            fedavg.aggregate_with_optimizer(optimizer, epoch_loss, model_weight=1, loss_weight=1)
            LOGGER.debug('epoch loss is {}'.format(epoch_loss))

        from sklearn.metrics import roc_auc_score
        train_pred = self.model(t.Tensor(features)).detach().numpy()
        LOGGER.debug('final train auc is {}'.format(roc_auc_score(label, train_pred)))

    def predict(self, cpn_input, **kwargs):
        pass

```

### conf 样例

```
{
    "dsl_version": 2,
    "initiator": {
        "role": "host",
        "party_id": 9999
    },
    "role": {
        "host": [
            9998, 9999
        ],
        "arbiter": [9999]
    },
    "component_parameters": {
        "common": {
            "cust_nn_0": {
                "class_file_name": "dense_layer",
                "class_name": "TestNetModule",
                "nn_params": {
                    "epochs": 30，
                    "lr": 0.01,
                    "batch_size": 512
                }
            }
        }
    }
}
```

## Aggregator模块整体设计

除了神经网络部分，还提供一个Aggregator框架，该Aggregator框架经过简化，基类仅仅提供收发接口，
+ 对于无开发需求用户，直接import fate开发好的aggregator使用
+ 对于有开发需求的用户，遵循基类实现aggregate接口，开发聚合逻辑即可。完成后，使用装饰器注册aggregator，在开发client端脚本时
  import即可使用，server会自动根据aggregator client调起aggregator server
  
### Aggregator基类设计

为了方便开发用户，aggregator尽量进行了简化，并不会暴露transfer-variable,
提供get, send两类接口，以及必须实现的aggregate接口，
对于Server端 aggregate的接口必须实现，对于Client端，实现可以随意一些

```python
class AggregatorBaseClient(object):

    def __init__(self, aggregate_round: int, inform_server_aggregator_type: bool = True):
        # 发送消息去server, 同步聚合轮数，并告知server对应的aggregator server是什么
        pass

    def send_to_server(self, obj, suffix):
        pass

    def get_from_server(self, suffix):
        pass

    def aggregate(self, *args, **kwargs):
        raise NotImplementedError('This function need to be implemented')


class AggregatorBaseServer(object):

    def __init__(self):
        # 同步聚合轮数
        pass

    def get_agg_round(self):
        pass

    def send_to_clients(self, obj, suffix, client_idx=-1):
        pass

    def get_from_clients(self, suffix, client_idx=-1):
        pass

    def aggregate(self):
        raise NotImplementedError("This function need to be implemented")
```

### 注册机制

对于基于AggregatorBase基类实现的Aggregator类，需要使用aggregator_client与aggregator_server
两个修饰器对Aggregator进行注册，原理与component模块的ComponentMeta相同，在Server端，会import
aggregator目录下的所有模块，这样修饰器里的代码就会被调用，AggregatorPair（哪个Client对应哪个Server）
就会被注册到一个Dict中

下列代码展示了修饰器的使用方式
```python
from torch.optim import Optimizer
from typing import List

@aggregator_client('fedavg')  #register fedavg client
class FedAVGClient(AggregatorBaseClient):

    def __init__(self, aggregate_round: int, secure_aggregate=True, early_stop=False, tol=0.0001,
                 aggregate_type: str = 'mean', inform_server_aggregator_type=False):
        pass

    def _inc_agg_round(self):
        pass

    def aggregate(self, model: List[np.ndarray], loss, model_weight=1, loss_weight=1):
        pass

    def aggregate_with_optimizer(self, optimizer: Optimizer, loss, model_weight=1, loss_weight=1):
        pass

@aggregator_server('fedavg')  # register fedavg server
class FedAVGServer(AggregatorBaseServer):

    def __init__(self):
        pass

    def _inc_agg_round(self):
        pass

    def aggregate(self):
        pass
```

下列为修饰器的部分代码
```python

# add client class to global class dict
def _add_client(cls, aggregator_name):

    if aggregator_name not in _AGGREGATOR_PAIR_DICT:
        _AGGREGATOR_PAIR_DICT[aggregator_name] = AggregatorPair()
    _AGGREGATOR_PAIR_DICT[aggregator_name].set_client_class(cls)


# add server class to global class dict
def _add_server(cls, aggregator_name):

    if aggregator_name not in _AGGREGATOR_PAIR_DICT:
        _AGGREGATOR_PAIR_DICT[aggregator_name] = AggregatorPair()
    _AGGREGATOR_PAIR_DICT[aggregator_name].set_server_class(cls)


# A decorator that registers new aggregator client
def aggregator_client(aggregator_name):

    def aggregator_decorator(cls):
        _add_client(cls, aggregator_name)
        return cls

    return aggregator_decorator


# A decorator that registers new aggregator server
def aggregator_server(aggregator_name):

    def aggregator_decorator(cls):
        _add_server(cls, aggregator_name)
        return cls

    return aggregator_decorator
```

### 自动调起机制流程

![图片alt](./agg.jpg 'agg')

## Homo Framework调整

除了以上工作外， 计划对homo框架进行调整，提升代码利用效率，

经过了解，计划更新原框架与aggregator相关的模块，其他保持不变

```
./fate/python/federatedml/framework/homo/ (partial)
|
├── aggregator
│ ├── agg_base.py
│ ├── fedavg_aggregator.py
|
├── blocks
│ ├──   aggregator.py <-- remove
│ ├── base.py
│ ├── diffie_hellman.py
│ ├──   has_converged.py <-- remove
│ ├──   loss_scatter.py <-- remove
│ ├──   model_broadcaster.py <-- remove
│ ├──   model_scatter.py <--remove
│ ├── paillier_cipher.py
│ ├── random_padding_cipher.py
│ ├──   secure_aggregator.py  <-- remove
│ ├──   secure_mean_aggregator.py <-- remove
│ ├──   secure_sum_aggregator.py  <-- remove
│ └── uuid_generator.py
```

## 待讨论

+ 目前的框架是否满足需要
  + 对于普通只有模型开发的用户
  + 对于有定制需求的科研人群（选择性聚合等）
+ 安全性问题
  + Aggregator调起
  + 脚本编写