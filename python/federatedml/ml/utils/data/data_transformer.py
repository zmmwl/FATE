from federatedml.ml.data.data_transformer_param import DataTransformerParam
from federatedml.model_base import ComponentOutput, ModelBase
from fate_arch.tensor import Context


class DenseDataTransformer(object):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self._params = DataTransformerParam(**kwargs)

    def transform_to_frame(self, ctx, data):
        schema = _process_schema(data.schema, self._params)

        transform_ret = data.mapPartitions(self._data_to_tensor, schema)
        tensor_dict = {}

        # TODO: data type
        if self._params.with_label:
            tensor_dict["label"] = ctx.to_tensor(transform_ret.mapValues(lambda value: value["values"]))

        if self._params.weight_name:
            tensor_dict["weight"] = ctx.to_tensor(transform_ret.mapValues(lambda value: value["weight"]))

        tensor_dict["index"] = ctx.to_tensor(transform_ret.mapValues(lambda value: value["ids"]))
        tensor_dict["values"] = ctx.to_tensor(transform_ret.mapValues(lambda value: value["values"]))

        return tensor_dict

    def _data_to_tensor(self, kvs, schema):
        ids = []
        values = []
        y = []
        weights = []

        block_key = None
        delimiter = schema.get("delimiter", -1)
        label_index = schema.get("label_index", -1)
        weight_index = schema.get("weight_index", -1)
        match_id_index = schema.get("match_id_index", -1)
        feature_indexes = schema.get("feature_indexes", [])

        for key, value in kvs:
            block_key = block_key if block_key is None else key
            row = value.split(self._params.delimiter, delimiter)

            if match_id_index >= 0:
                ids.append(row[match_id_index])
            else:
                ids.append(key)

            if label_index >= 0:
                y.append(row[label_index])

            if weight_index >= 0:
                weights.append(row[weight_index])

            feature = [row[idx] for idx in feature_indexes]
            values.append(feature)

        return block_key, {"ids": ids, "label": y, "weight": weights, "values": values}


class DataTransformer(ModelBase):
    def __init__(self):
        super(DataTransformer, self).__init__()
        self.model_param = DataTransformerParam()

    def _init_runtime_parameters(self, cpn_input):
        self.model_param.update(cpn_input.parameters)

    def run(self, cpn_input):
        ctx = Context.from_cpn_input(cpn_input)
        self._init_runtime_parameters(cpn_input)

        data = cpn_input["data"]["data"]
        transformer = get_data_transformer(**cpn_input.parameters)
        return transformer.transform_to_frame(ctx, data)


def get_data_transformer(**kwargs):
    input_format = kwargs.pop("input_format", "dense")
    return DenseDataTransformer(**kwargs)


def _process_schema(schema, params):
    post_schema = dict()
    post_schema["sid"] = schema["sid"]
    post_schema["delimiter"] = schema.get("delimiter", ",")
    header = schema.get("header", {}).split(post_schema["delimiter"], -1)

    filter_indexes = []
    if params.with_match_id:
        post_schema["match_id_index"] = header.index(params.match_id_name)
        filter_indexes.append(post_schema["match_id_index"])

    if params.with_label:
        post_schema["label_index"] = header.index(params.label_name)
        filter_indexes.append(post_schema["label_index"])

    if params.with_weight:
        post_schema["weight_index"] = header.index(params.weight_name)
        filter_indexes.append(post_schema["weight_index"])

    if header:
        post_schema["feature_indexes"] = list(filter(lambda _id: _id not in filter_indexes, range(len(header))))

    return post_schema
