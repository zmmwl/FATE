from typing import List, Type

from fate.components.runner.procedure import Situation, dispatch
from fate.core.context import Context
from fate.interface import Module
from federatedml.callbacks.callback_list import CallbackList
from federatedml.param.base_param import BaseParam
from federatedml.util import LOGGER

from .parser.data import Datasets
from .parser.model import Models, serialize_models
from .parser.tracker import Tracker


class Runner:
    def __init__(self, cpn_class: Type[Module], cpn_param: BaseParam):
        self.cpn_class = cpn_class
        self.cpn_param = cpn_param

    def run(self, cpn_input):

        # update params
        params = self.cpn_param
        params.update(cpn_input.parameters)

        # instance cpn
        cpn = self.cpn_class(params)

        # parse datasets
        datasets = Datasets.parse(cpn_input.datasets, cpn)

        # deserialize models
        models = Models.parse(cpn_input.models)

        # emmm, should move to ctx in future
        cpn.checkpoint_manager = cpn_input.checkpoint_manager
        role = cpn_input.roles["local"]["role"]
        party_id = cpn_input.roles["local"]["party_id"]
        tracker = Tracker.parse(cpn_input.tracker)

        # init context
        ctx = Context()
        ctx.role = role
        ctx.party_id = party_id
        ctx.tracker = tracker
        cpn.callback_list = CallbackList(role, cpn.mode, cpn)

        # dispatch to concrate procedure according to situations
        situations = Situation(
            need_run=params.get("need_run", True),
            need_cv=params.get("cv_param.need_cv", False),
            need_stepwise=params.get("stepwise_param.need_stepwise", False),
            has_model=models.has_model,
            has_isometric_model=models.has_isometric_model,
            has_train_data=datasets.has_train_data,
            has_test_data=datasets.has_test_data,
            has_validate_data=datasets.has_validate_data,
            has_data=datasets.has_data,
        )
        LOGGER.debug(f"situations: {situations}")
        procedure = dispatch(situations)
        output = procedure.run(ctx, cpn, params, datasets, models)

        cpn.save_summary()
        return ComponentOutput(output, cpn._export(), cpn.save_cache())


class ComponentOutput:
    def __init__(self, data, models, cache: List[tuple]) -> None:
        self._data = data
        if not isinstance(self._data, list):
            self._data = [data]

        self._models = models
        if self._models is None:
            self._models = {}

        self._cache = cache
        if not isinstance(self._cache, list):
            self._cache = [cache]

    @property
    def data(self) -> list:
        return self._data

    @property
    def model(self):
        return serialize_models(self._models)

    @property
    def cache(self):
        return self._cache
