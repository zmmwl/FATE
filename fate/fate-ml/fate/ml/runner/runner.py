import typing
from typing import Any, Optional, Type

from fate.ml.context import Context
from fate.ml.context.tracker import WarpedTrackerClient
from fate.ml.module.module import Module
from fate.ml.runner.data import Datasets
from fate.ml.runner.model import Models, serialize_models
from federatedml.callbacks.callback_list import CallbackList
from federatedml.model_selection.stepwise.hetero_stepwise import HeteroStepwise
from federatedml.param.base_param import BaseParam
from federatedml.util import LOGGER, consts


class Runner:
    def __init__(self, cpn_class: Type[Module], cpn_param: BaseParam):
        self.cpn_class = cpn_class
        self.cpn_param = cpn_param

    def run(self, cpn_input):
        cpn, params, datasets, models = prepare(
            self.cpn_class, self.cpn_param, cpn_input
        )
        ctx = Context()
        role = cpn_input.roles["local"]["role"]
        party_id = cpn_input.roles["local"]["party_id"]
        ctx.set_party(role=role, party_id=party_id)
        ctx.set_tracker(WarpedTrackerClient(cpn_input.tracker))
        cpn.callback_list = CallbackList(role, getattr(cpn, "mode", None), cpn)
        output = dispatch_run(ctx, cpn, params, datasets, models)
        cpn.save_summary()
        return ComponentOutput(output, cpn._export(), cpn.save_cache())


class ComponentOutput:
    def __init__(self, data, models, cache: typing.List[tuple]) -> None:
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


def prepare(cpn_class, cpn_param, cpn_input):
    # update params
    cpn_param.update(cpn_input.parameters)

    # instance cpn
    cpn = cpn_class(cpn_param)

    # parse datasets
    datasets = Datasets.parse(cpn_input.datasets, cpn)

    # deserialize models
    models = Models.parse(cpn_input.models)

    # emmm, should move to ctx in future
    cpn.tracker = cpn_input.tracker
    cpn.checkpoint_manager = cpn_input.checkpoint_manager

    # init_model here to make the old module happy
    cpn._init_model(cpn_param)

    return cpn, cpn_param, datasets, models


def dispatch_run(
    ctx: Context, cpn: "Module", params: BaseParam, datasets: "Datasets", models: dict
) -> Optional[Any]:
    # situation: what we have
    has_model = models.get("model") is not None
    has_isometric_model = models.get("isometric_model") is not None
    has_train_data = datasets.has_train_data
    has_test_data = datasets.has_test_data
    has_validate_data = datasets.has_validate_data
    has_data = datasets.has_data
    LOGGER.debug(
        f"{has_model=}, {has_isometric_model=}, {has_train_data=}, {has_test_data=}, {has_validate_data=}, {has_data=}"
    )

    # switches: guess what you want?
    is_skip_run = not params.get("need_run", True)
    is_cv = params.get("cv_param.need_cv", False)
    is_stepwise = params.get("stepwise_param.need_stepwise", False)
    is_warm_start = has_model and has_train_data
    is_train_with_validate = has_train_data and has_validate_data
    is_train_without_validate = has_train_data and (not has_validate_data)
    is_test = (not has_test_data) and has_test_data
    is_transform_fit = has_data and (not has_model)
    is_transform = has_data and has_model
    # TODO: one vs rest?

    LOGGER.debug(
        f"{is_skip_run=}, {is_cv=}, {is_stepwise=}, {is_warm_start=}, {is_train_with_validate=}, {is_train_without_validate=}, {is_test=}, {is_transform_fit=}, {is_transform=}"
    )

    # case 1: not need run, pass data
    if is_skip_run:
        if isinstance(datasets.data, dict) and len(datasets.data) >= 1:
            return list(datasets.data.values())[0]

    # case 2: cross validate
    if is_cv:
        # TODO: need advice
        # should be something like
        # for fold_id in range(n_Fold):
        #     with ctx.namespace(f"cv_{fold_id}") as fold_ctx:
        #         cpn.xxx(fold_ctx, ...)
        from federatedml.model_selection.k_fold import KFold

        kflod_obj = KFold()

        def _get_cv_param(model):
            model.model_param.cv_param.role = model.role
            model.model_param.cv_param.mode = model.mode
            return model.model_param.cv_param

        cv_param = _get_cv_param(cpn)
        output_data = kflod_obj.run(cv_param, datasets.train_data, cpn, False)
        return output_data

    # case 3: stepwise
    if is_stepwise:
        cpn.disable_callback_loss()  # TODO: cpn need aware about stepwise
        if cpn.mode == consts.HETERO:
            step_obj = HeteroStepwise()
        else:
            raise ValueError("stepwise currently only support Hetero mode.")

        def _get_stepwise_param(model):
            model.model_param.stepwise_param.role = model.role
            model.model_param.stepwise_param.mode = model.mode
            return model.model_param.stepwise_param

        stepwise_param = _get_stepwise_param(cpn)
        step_obj.run(stepwise_param, datasets.train_data, datasets.validate_data, cpn)
        pred_result = HeteroStepwise.predict(datasets.train_data, cpn)
        union_output = union_data([pred_result], ["train"])
        return cpn.set_predict_data_schema(union_output, datasets.schema)

    # maybe load model
    if has_model or has_isometric_model:
        cpn.load_model(models)

    if is_warm_start:
        # FIXME: no bandwidth to take care of it
        return _warm_start_run(
            datasets.train_data, datasets.validate_data, datasets.schema
        )

    if is_train_with_validate:
        with ctx.namespace("fit") as subctx:
            cpn.fit(subctx, datasets.train_data, datasets.validate_data)
        with ctx.namespace("predict") as subctx:
            predict_on_validate_data = cpn.predict(subctx, datasets.train_data)
        with ctx.namespace("validate") as subctx:
            predict_on_train_data = cpn.predict(subctx, datasets.validate_data)
        union_output = union_data(
            [predict_on_validate_data, predict_on_train_data], ["train", "validate"]
        )
        return cpn.set_predict_data_schema(union_output, datasets.schema)

    if is_train_without_validate:
        with ctx.namespace("fit") as subctx:
            cpn.fit(subctx,datasets.train_data)
        with ctx.namespace("predict") as subctx:
            predict_on_train_data = cpn.predict(subctx, datasets.validate_data)
        union_output = union_data([predict_on_train_data], ["train"])
        return cpn.set_predict_data_schema(union_output, datasets.schema)

    if is_test:
        with ctx.namespace("predict") as subctx:
            predict_on_test_data = cpn.predict(subctx, datasets.test_data)
        union_output = union_data([predict_on_test_data], ["predict"])
        return cpn.set_predict_data_schema(union_output, datasets.schema)

    if is_transform_fit:
        data = cpn.extract_data(datasets.data)
        with ctx.namespace("fit") as subctx:
            return cpn.fit(subctx, data)

    if is_transform:
        data = cpn.extract_data(datasets.data)
        with ctx.namespace("transform") as subctx:
            return cpn.transform(subctx, data)


def union_data(previews_data, name_list):
    if len(previews_data) == 0:
        return None

    if any([x is None for x in previews_data]):
        return None

    assert len(previews_data) == len(name_list)

    def _append_name(value, name):
        inst = copy.deepcopy(value)
        if isinstance(inst.features, list):
            inst.features.append(name)
        else:
            inst.features = np.append(inst.features, name)
        return inst

    result_data = None
    for data, name in zip(previews_data, name_list):
        # LOGGER.debug("before mapValues, one data: {}".format(data.first()))
        f = functools.partial(_append_name, name=name)
        data = data.mapValues(f)
        # LOGGER.debug("after mapValues, one data: {}".format(data.first()))

        if result_data is None:
            result_data = data
        else:
            LOGGER.debug(
                f"Before union, t1 count: {result_data.count()}, t2 count: {data.count()}"
            )
            result_data = result_data.union(data)
            LOGGER.debug(f"After union, result count: {result_data.count()}")
        # LOGGER.debug("before out loop, one data: {}".format(result_data.first()))

    return result_data
