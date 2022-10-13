import json
import torch
import torch as t
import numpy as np
import random
import tempfile
from federatedml.model_base import ModelBase
from federatedml.nn_.dataset.base import get_dataset_class, Dataset
from federatedml.nn_.homo.trainer.trainer_base import get_trainer_class, TrainerBase
from federatedml.nn_.dataset.table import TableDataset
from federatedml.nn_.dataset.image import ImageDataset
from federatedml.param.homo_cust_nn_param import HomoCustNNParam
from federatedml.nn_.backend.torch import serialization as s
from federatedml.nn_.backend.torch.base import FateTorchOptimizer
from federatedml.util import LOGGER
from federatedml.util import consts

MODELMETA = "HomoNNMeta"
MODELPARAM = "HomoNNParam"


def global_seed(seed):
    # set all random seeds
    # set random seed of torch
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    # np & random
    np.random.seed(seed)
    random.seed(seed)


class HomoCustNNClient(ModelBase):

    def __init__(self):
        super(HomoCustNNClient, self).__init__()
        self.model_param = HomoCustNNParam()
        self.trainer = consts.FEDAVG_TRAINER
        self.trainer_param = {}
        self.dataset_module = None
        self.dataset = None
        self.dataset_param = {}
        self.torch_seed = None
        self.loss = None
        self.optimizer = None
        self.validation_freq = None
        self.nn_define = None

        # running varialbles
        self.trainer_inst = None

        # export model
        self.model_loaded = False
        self.model = None

        # cache dataset
        self.cache_dataset = {}

    def _init_model(self, param: HomoCustNNParam):

        train_param = param.trainer.to_dict()
        dataset_param = param.dataset.to_dict()
        self.trainer = train_param['trainer_name']
        self.dataset = dataset_param['dataset_name']
        self.trainer_param = train_param['param']
        self.dataset_param = dataset_param['param']
        self.torch_seed = param.torch_seed
        self.validation_freq = param.validation_freq
        self.nn_define = param.nn_define
        self.loss = param.loss
        self.optimizer = param.optimizer

    def try_dataset_class(self, dataset_class, path):
        # try default dataset
        try:
            dataset_inst: Dataset = dataset_class(**self.dataset_param)
            dataset_inst.load(path)
            return dataset_inst
        except Exception as e:
            LOGGER.debug('try to load dataset failed, exception :{}'.format(e))
            return None

    def load_dataset(self, data_path_or_dtable):

        # load dataset class
        if isinstance(data_path_or_dtable, str):
            cached_id = data_path_or_dtable
        else:
            cached_id = id(data_path_or_dtable)

        if cached_id in self.cache_dataset:
            LOGGER.debug('use cached dataset, cached id {}'.format(cached_id))
            return self.cache_dataset[cached_id]

        if self.dataset is None or self.dataset == '':
            # automatically match default dataset
            for ds_class in [TableDataset, ImageDataset]:
                dataset_inst = self.try_dataset_class(ds_class, data_path_or_dtable)
                if dataset_inst is not None:
                    break
            if dataset_inst is None:
                raise ValueError('cannot find default dataset that can successfully load data from path {}'.
                                 format(data_path_or_dtable))
        else:
            # load specified dataset
            dataset_class = get_dataset_class(self.dataset)
            dataset_inst = dataset_class(**self.dataset_param)
            dataset_inst.load(data_path_or_dtable)

        if isinstance(data_path_or_dtable, str):
            self.cache_dataset[data_path_or_dtable] = dataset_inst
        else:
            self.cache_dataset[id(data_path_or_dtable)] = dataset_inst

        return dataset_inst

    # read model from model bytes
    @staticmethod
    def recover_model_bytes(model_bytes):
        with tempfile.TemporaryFile() as f:
            f.write(model_bytes)
            f.seek(0)
            model_dict = torch.load(f)
        return model_dict

    def init(self):

        # set random seed
        global_seed(self.torch_seed)

        # load trainer class
        trainer_class = get_trainer_class(self.trainer)
        LOGGER.info('trainer class is {}'.format(trainer_class))

        # recover model from model config / or recover from saved model param
        loaded_model_dict = None
        if self.model_loaded:
            param, meta = self.model
            self.nn_define = json.loads(meta.nn_define[0])
            self.loss = json.loads(meta.loss_func_define[0])
            self.optimizer = json.loads(meta.optimzer_define[0])
            loaded_model_dict = self.recover_model_bytes(param.model_bytes)

        # get model from nn define
        model = s.recover_sequential_from_dict(self.nn_define)
        if loaded_model_dict:
            model.load_state_dict(loaded_model_dict['model'])
            LOGGER.info('load model parameters from check point')

        LOGGER.info('model structure is {}'.format(model))
        LOGGER.debug('init model param is {}'.format(list(model.parameters())))
        # init optimizer
        if self.optimizer is not None:
            optimizer_: FateTorchOptimizer = s.recover_optimizer_from_dict(self.optimizer)
            optimizer = optimizer_.to_torch_instance(model.parameters())
            if loaded_model_dict:
                optimizer.load_state_dict(loaded_model_dict['optimizer'])
            LOGGER.info('optimizer is {}'.format(optimizer))
        else:
            optimizer = None
            LOGGER.debug('optimizer is not specified')

        # init loss
        if self.loss is not None:
            loss_fn = s.recover_loss_fn_from_dict(self.loss)
            LOGGER.info('loss function is {}'.format(loss_fn))
        else:
            loss_fn = None
            LOGGER.info('loss function is not specified')

        # init trainer
        trainer_inst: TrainerBase = trainer_class(**self.trainer_param)
        trainer_inst.set_nn_config(self.nn_define, self.optimizer, self.loss)

        return trainer_inst, model, optimizer, loss_fn

    def fit(self, cpn_input):

        # set random seed
        global_seed(self.torch_seed)

        if self.component_properties.local_partyid == 9999:
            # test_path = '/home/cwj/standalone_fate_install_1.9.0_release/examples/data/breast_homo_guest.csv'
            test_path = '/home/cwj/standalone_fate_install_1.9.0_release/examples/data/movielens_host_0.csv'
        else:
            # test_path = '/home/cwj/standalone_fate_install_1.9.0_release/examples/data/breast_homo_host.csv'
            test_path = '/home/cwj/standalone_fate_install_1.9.0_release/examples/data/movielens_host_1.csv'

        # load dataset class
        dataset_inst = self.load_dataset(test_path)
        LOGGER.info('dataset instance is {}'.format(dataset_inst))

        self.trainer_inst, model, optimizer, loss_fn = self.init()
        self.trainer_inst.set_model(model)
        dataset_inst.set_type('train')
        self.trainer_inst.train(dataset_inst, None, optimizer, loss_fn)

        # training is done, get exported model
        self.model = self.trainer_inst.get_cached_model()

    def predict(self, cpn_input):

        LOGGER.debug('running predict')
        if self.trainer_inst is None:
            LOGGER.debug('skip')
        else:
            # test_path = '/home/cwj/standalone_fate_install_1.9.0_release/examples/data/breast_homo_guest.csv'
            test_path = '/home/cwj/standalone_fate_install_1.9.0_release/examples/data/movielens_host_1.csv'
            dataset_inst = self.load_dataset(test_path)
            if not dataset_inst.has_dataset_type():
                dataset_inst.set_type('predict')
            ids, pred_rs, label = self.trainer_inst.predict(dataset_inst)
            LOGGER.debug('pred rs is {} {} {}'.format(ids, pred_rs, label))

    def export_model(self):

        if self.model is None:
            return

        return {MODELPARAM: self.model[0],  # param
                MODELMETA: self.model[1]}  # meta

    def load_model(self, model_dict):

        model_dict = list(model_dict["model"].values())[0]
        param = model_dict.get(MODELPARAM)
        meta = model_dict.get(MODELMETA)
        self.model = (param, meta)
        self.model_loaded = True
