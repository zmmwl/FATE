from typing import Any, Optional, Type

from fate.ml.context import Context
from fate.ml.modules.data import Datasets
from fate.ml.modules.module import ComponentOutput, Module, WarpedTrackerClient
from federatedml.callbacks.callback_list import CallbackList
from federatedml.model_selection.stepwise.hetero_stepwise import HeteroStepwise
from federatedml.param.base_param import BaseParam
from federatedml.protobuf import deserialize_models
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


def prepare(cpn_class, cpn_param, cpn_input):
    # update params
    cpn_param.update(cpn_input.parameters)

    # instance cpn
    cpn = cpn_class(cpn_param)

    # parse datasets
    datasets = Datasets.parse(cpn_input.datasets, cpn)

    # deserialize models
    models = cpn_input.models
    deserialize_models(models)

    # emmm...
    cpn.tracker = cpn_input.tracker
    cpn.checkpoint_manager = cpn_input.checkpoint_manager

    # init_model here to make the old module happy
    cpn._init_model(cpn_param)

    return cpn, cpn_param, datasets, models


def dispatch_run(
    ctx, cpn: "Module", params: BaseParam, datasets: "Datasets", models: dict
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
        return pred_result

        union_output = union_data([output], ["train"])
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
        cpn.set_flowid("fit")
        cpn.fit(datasets.train_data, datasets.validate_data)
        cpn.set_flowid("predict")
        predict_on_validate_data = cpn.predict(datasets.train_data)
        cpn.set_flowid("validate")
        predict_on_train_data = cpn.predict(datasets.validate_data)
        union_output = union_data(
            [predict_on_validate_data, predict_on_train_data], ["train", "validate"]
        )
        return cpn.set_predict_data_schema(union_output, datasets.schema)

    if is_train_without_validate:
        cpn.set_flowid("fit")
        cpn.fit(datasets.train_data)
        cpn.set_flowid("predict")
        predict_on_train_data = cpn.predict(datasets.validate_data)
        union_output = union_data([predict_on_train_data], ["train"])
        return cpn.set_predict_data_schema(union_output, datasets.schema)

    if is_test:
        cpn.set_flowid("predict")
        predict_on_test_data = cpn.predict(datasets.test_data)
        union_output = union_data([predict_on_test_data], ["predict"])
        return cpn.set_predict_data_schema(union_output, datasets.schema)

    if is_transform_fit:
        data = cpn.extract_data(datasets.data)
        cpn.set_flowid("fit")
        return cpn.fit(ctx, data)

    if is_transform:
        data = cpn.extract_data(datasets.data)
        cpn.set_flowid("transform")
        return cpn.transform(data)


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
