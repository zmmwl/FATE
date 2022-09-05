from .components import ComponentMeta

custom_nn_cpn_meta = ComponentMeta("Custom NN")


@custom_nn_cpn_meta.bind_param
def nn_param():
    from federatedml.custom_nn.param import NNParam

    return NNParam


@custom_nn_cpn_meta.bind_runner.on_guest.on_host
def nn_client_runner():
    from federatedml.custom_nn.client import NNClient

    return NNClient


@custom_nn_cpn_meta.bind_runner.on_arbiter
def nn_aggregator_runner():
    from federatedml.custom_nn.aggregator import NNAggregator

    return NNAggregator
