import functools
import numpy as np
from federatedml.model_base import ComponentOutput, ModelBase
from fate_arch.data.dataframe import DataFrame, StorageMeta


class DenseFormatTableLoader(object):
    def __init__(self,
                 with_match_id=False,
                 match_id_name=None,
                 with_label=False,
                 label_name="y",
                 with_weight=False,
                 weight_name="weight",
                 data_type="float64"):
        self._with_match_id = with_match_id
        self._match_id_name = match_id_name
        self._with_label = with_label
        self._label_name = label_name
        self._with_weight = with_weight
        self._weight_name = weight_name
        self._data_type = data_type

    def to_frame(self, ctx, data):
        schema = _process_schema(data.schema,
                                 self._with_match_id,
                                 self._match_id_name,
                                 self._with_label,
                                 self._label_name,
                                 self._with_weight,
                                 self._weight_name)

        data_trans = data.mapValues(lambda value: value.split(schema["delimiter"], -1))
        data_dict = {}
        if schema.get("match_id_index") is not None:
            data_dict["match_id"] = data_trans.mapValues(lambda value: value[schema["match_id_index"]])

        if schema.get("label_index") is not None:
            data_dict["label"] = data_trans.mapValues(lambda value: value[schema["label_index"]])

        if schema.get("weight_index") is not None:
            data_dict["weight_index"] = data_trans.mapValues(lambda value: value[schema["weight_index"]])

        if schema.get("values") is not None:
            data_dict["values"] = data_trans.mapValues(lambda value: np.array(value)[schema["feature_indexes"]].tolist())

        data_dict["index"] = data.mapValues(lambda value: None)

        return DataFrame(ctx,
                         schema,
                         **data_dict,
                         storage_meta=StorageMeta(value_storage_type="row"))


def load(ctx, data, **kwargs):
    input_format = kwargs.pop("input_format", "dense")
    dataset_loader = DenseFormatTableLoader(**kwargs)
    return dataset_loader.to_frame(ctx, data)


def _process_schema(schema, with_match_id, match_id_name, with_label, label_name, with_weight, weight_name):
    post_schema = dict()
    post_schema["sid"] = schema["sid"]
    post_schema["delimiter"] = schema.get("delimiter", ",")
    header = schema.get("header", {}).split(post_schema["delimiter"], -1)

    filter_indexes = []
    if with_match_id:
        post_schema["match_id_index"] = header.index(match_id_name)
        filter_indexes.append(post_schema["match_id_index"])

    if with_label:
        post_schema["label_index"] = header.index(label_name)
        filter_indexes.append(post_schema["label_index"])

    if with_weight:
        post_schema["weight_index"] = header.index(weight_name)
        filter_indexes.append(post_schema["weight_index"])

    if header:
        post_schema["feature_indexes"] = list(filter(lambda _id: _id not in filter_indexes, range(len(header))))

    return post_schema


