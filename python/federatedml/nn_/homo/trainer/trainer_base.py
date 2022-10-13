import abc
import json
import torch.optim
from federatedml.nn_.backend.util import ML_PATH
from torch.nn import Module
from torch.utils.data import Dataset
import importlib
import tempfile
from federatedml.protobuf.generated.homo_nn_model_param_pb2 import HomoNNParam
from federatedml.protobuf.generated.homo_nn_model_meta_pb2 import HomoNNMeta


class TrainerBase(object):

    def __init__(self, **kwargs):
        self.run_local_mode = False
        self.role = None
        self.party_id = None
        self.party_id_list = None
        self._flowid = None
        self._cache_model = None
        self.model = None

        # nn config
        self.nn_define, self.opt_define, self.loss_define = {}, {}, {}

    def set_nn_config(self, nn_define, optimizer_define, loss_define):
        self.nn_define = nn_define
        self.opt_define = optimizer_define
        self.loss_define = loss_define

    def set_flowid(self, flowid):
        """
        Set flow id, and initialize transfer variable
        """
        self._flowid = flowid

    def set_role(self, role):
        """
        set self role
        """
        self.role = role

    def set_party_id(self, party_id):
        self.party_id = party_id

    def set_party_id_list(self, party_id_list):
        self.party_id_list = party_id_list

    def get_formatted_predict_result(self, pred_rs, task_type='binary'):
        pass

    def local_mode(self):
        self.run_local_mode = True

    def fed_mode(self):
        self.run_local_mode = False

    def set_model(self, model: Module):
        if not issubclass(type(model), Module):
            raise ValueError('model must be a subclass of pytorch nn.Module')
        self.model = model

    def _get_model_param_and_meta(self, model, optimizer=None, epoch_idx=-1):

        if issubclass(type(model), Module):
            self._cache_model = model
            opt_state_dict = None
            if optimizer is not None:
                assert isinstance(optimizer, torch.optim.Optimizer), \
                    'optimizer must be an instance of torch.optim.Optimizer'
                opt_state_dict = optimizer.state_dict()

            model_status = {
                'model': model.state_dict(),
                'optimizer': opt_state_dict,
            }

            with tempfile.TemporaryFile() as f:
                torch.save(model_status, f)
                f.seek(0)
                model_saved_bytes = f.read()

            param = HomoNNParam()
            meta = HomoNNMeta()

            param.model_bytes = model_saved_bytes
            meta.nn_define.append(json.dumps(self.nn_define))
            meta.optimizer_define.append(json.dumps(self.opt_define))
            meta.loss_func_define.append(json.dumps(self.loss_define))

            return param, meta

        else:
            raise ValueError('export model must be a subclass of torch nn.Module, however got {}'.format(type(model)))

    def export_model(self, model, optimizer=None, epoch_idx=-1):

        param, meta = self._get_model_param_and_meta(model, optimizer, epoch_idx)
        self._cache_model = (param, meta)

    def get_cached_model(self):
        return self._cache_model

    def set_checkpoint(self, model, optimizer):
        pass

    @abc.abstractmethod
    def train(self, train_set, validate_set=None, optimizer=None, loss=None):
        """
            train_set : A Dataset Instance, must be a instance of subclass of Dataset (federatedml.nn.dataset.base),
                      for example, TableDataset() (from federatedml.nn.dataset.table)

            validate_set : A Dataset Instance, but optional must be a instance of subclass of Dataset
                    (federatedml.nn.dataset.base), for example, TableDataset() (from federatedml.nn.dataset.table)

            optimizer : A pytorch optimizer class instance, for example, t.optim.Adam(), t.optim.SGD()

            loss : A pytorch Loss class, for example, nn.BECLoss(), nn.CrossEntropyLoss()
        """
        pass

    @abc.abstractmethod
    def predict(self, dataset: Dataset):
        pass


def get_trainer_class(trainer_module_name: str):

    ds_modules = importlib.import_module('{}.homo.trainer.{}'.format(ML_PATH, trainer_module_name))
    try:

        for k, v in ds_modules.__dict__.items():
            if isinstance(v, type):
                if issubclass(v, TrainerBase) and v is not TrainerBase:
                    return v
        raise ValueError('Did not find any class in {}.py that is the subclass of Trainer class'.
                         format(trainer_module_name))
    except ValueError as e:
        raise e
