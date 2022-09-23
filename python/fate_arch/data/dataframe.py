import copy

import functools
import pandas as pd
import numpy as np
import operator
import types
from fate_arch.tensor import FPTensor
from fate_arch.tensor.impl.tensor.distributed import FPTensorDistributed
from fate_arch.data import ops


# TODO: record data type, support multiple data types
class DataFrame(object):
    def __init__(self,
                 ctx,
                 schema,
                 index=None,
                 match_id=None,
                 values=None,
                 label=None,
                 weight=None,
                 storage_meta=None):
        self._ctx = ctx
        self._index = index
        self._match_id = match_id
        self._values = values
        self._label = label
        self._weight = weight
        self._schema = schema

        self.__shape = None
        self._columns = None

        self._tensor_label = None

        self._index_with_block_id = None
        self._full_data = None

        self._storage_meta = storage_meta

    @property
    def index(self):
        return self._index

    @property
    def values(self):
        if self._storage_meta.value_storage_type != "block":
            self._values = self._convert_to_ndarray_block(self._values)
            self._storage_meta.value_storage_type = "block"

        return FPTensor(self._ctx,
                        FPTensorDistributed(self._values))

    @property
    def label(self):
        return self._label

    @property
    def weight(self):
        return self._weight

    @property
    def match_id(self):
        return self._match_id

    @property
    def shape(self):
        if self.__shape:
            return self.__shape

        if self._values is None:
            self.__shape = (self._index.count(), 0)
        else:
            self.__shape = (self._index.count(), len(self._schema["header"]))

        return self.__shape

    @property
    def columns(self) -> "ColumnObject":
        if not self._columns:
            self._columns = ColumnObject(self._schema["header"])
        else:
            return self._columns

    def max(self, *args, **kwargs) -> "DataFrame":
        self._values = self._convert_to_ndarray_block(self._values)
        max_ret = ops.max(*args, **kwargs)
        return self._process_stat_func(max_ret, *args, **kwargs)

    def min(self, *args, **kwargs) -> "DataFrame":
        self._values = self._convert_to_ndarray_block(self._values)
        min_ret = ops.min(*args, **kwargs)
        return self._process_stat_func(min_ret, *args, **kwargs)

    def mean(self, *args, **kwargs) -> "DataFrame":
        self._values = self._convert_to_ndarray_block(self._values)
        mean_ret = ops.mean(*args, **kwargs)
        return self._process_stat_func(mean_ret, *args, **kwargs)

    def sum(self, *args, **kwargs) -> "DataFrame":
        self._values = self._convert_to_ndarray_block(self._values)
        sum_ret = ops.sum(*args, **kwargs)
        return self._process_stat_func(sum_ret, *args, **kwargs)

    def std(self, *args, **kwargs) -> "DataFrame":
        self._values = self._convert_to_ndarray_block(self._values)
        std_ret = ops.std(*args, **kwargs)
        return self._process_stat_func(std_ret, *args, **kwargs)

    def count(self) -> "int":
        return self.shape[0]

    def _process_stat_func(self, stat_ret, *args, **kwargs) -> "pd.Series":
        if not kwargs.get("axis", 0):
            return pd.Series(stat_ret, index=self._schema["header"])
        else:
            return pd.Series(stat_ret)

    def __add__(self, other):
        return self._arithmetic_operate(np.add, other)

    def __sub__(self, other):
        return self._arithmetic_operate(np.sub, other)

    def __mul__(self, other):
        return self._arithmetic_operate(np.mul, other)

    def __truediv__(self, other):
        return self._arithmetic_operate(np.truediv, other)

    def _arithmetic_operate(self, op, other):
        if isinstance(other, pd.Series):
            other = np.array(pd.Series)
        elif isinstance(other, DataFrame):
            other.value_to_block()
            other = other._values
        elif isinstance(other, (int, float, np.int, np.int32, np.int64, np.float, np.float32, np.float64)):
            pass
        else:
            raise ValueError(f"{op.__name__} between {DataFrame} and {type(other)} is not supported")

        ret_value = self._values.join(other, lambda blk1, blk2: op(blk1, blk2))
        return DataFrame(self._ctx, index=self._index, values=ret_value, label=self._label, weight=self._weight,
                         schema=self._schema, storage_meta=self._storage_meta)

    def __getitem__(self, items):
        indexes = self.__get_index_by_column_names(items)
        ret_tensor = self._values[:, indexes]

        header_mapping = dict(zip(self._schema["header"], range(len(self._schema["header"]))))
        new_schema = copy.deepcopy(self._schema)
        new_header = items if isinstance(items, list) else [items]
        new_anonymous_header = []

        for item in items:
            index = header_mapping[item]
            new_anonymous_header.append(self._schema["anonymous_header"][index])

        new_schema["header"] = new_header
        new_schema["anonymous__header"] = new_anonymous_header

        return DataFrame(self._ctx, index=self._index, values=ret_tensor, label=self._label, weight=self._weight,
                         schema=new_schema, storage_meta=self._storage_meta)

    def __setitem__(self, keys, item):
        if not isinstance(item, DataFrame):
            raise ValueError("Using syntax df[[col1, col2...]] = rhs, rhs should be a dataframe")

        indexes = self.__get_index_by_column_names(keys)
        self._values[:, indexes] = item._values

        return self

    def __len__(self):
        return self.count()

    """
    def __iter__(self):
        return (col for col in self._schema["header"])
    """

    def __get_index_by_column_names(self, column_names):
        if isinstance(column_names, str):
            column_names = [column_names]

        indexes = []
        header_mapping = dict(zip(self._schema["header"], range(len(self._schema["header"]))))
        for col in column_names:
            index = header_mapping.get(col, None)
            if index is None:
                raise ValueError(f"Can not find column: {col}")
            indexes.append(index)

        return indexes

    def _convert_to_order_indexes(self):
        def _get_block_summary(kvs):
            key = next(kvs)
            block_size = 1 + len(kvs)
            return {key : block_size}

        block_summary = self._index.mapPartitions(_get_block_summary).reduce(lambda blk1, blk2: {**blk1, **blk2})

        start_index, block_id = 0, 0
        block_keys_mapping = dict()
        for blk_key, blk_size in block_summary.items():
            block_keys_mapping[blk_key] = dict(start_index=start_index,
                                               end_index=start_index + blk_size - 1,
                                               block_id=block_id)
            start_index += blk_size
            block_id += 1

        self._storage_meta.block_keys_mapping = block_keys_mapping

    def _convert_to_ndarray_block(self, _values):
        """
        convert order indexes to make sure that using of indexes, values, label, weight can be match.
        """
        if self._storage_meta.value_storage_type == "block":
            return

        if not self._index_with_block_id:
            self._convert_to_order_indexes()

        def _to_ndarray(kvs, order_maps=None, dtypes="float64"):
            block_id = None
            ret = []
            for key, value in kvs:
                if not block_id:
                    block_id = order_maps[key]["block_id"]

                ret.append(value)

            return block_id, np.array(ret, dtypes=dtypes)

        block_table = functools.partial(_to_ndarray,
                                        order_maps=self._storage_meta.block_keys_info)

        return block_table

    def take(self, indices):
        if set(indices) != len(indices):
            raise ValueError("Only support to take indices of ")

        ...

    def serialize(self):
        ...

    def deserialize(self):
        ...


class ColumnObject(object):
    def __init__(self, col_names):
        self._col_names = col_names

    def __getitem__(self, items):
        if isinstance(items, int):
            return self._col_names[items]
        else:
            ret_cols = []
            for item in items:
                ret_cols.append(self._col_names[item])

            return ColumnObject(ret_cols)

    def tolist(self):
        return self._col_names

    def __iter__(self):
        return (col_name for col_name in self._col_names)


class StorageMeta(object):
    def __init__(self,
                 value_storage_type="row",
                 label_storage_type="row",
                 weight_storage_type="row",
                 block_keys_info=None):
        self._value_storage_type = value_storage_type
        self._label_storage_type = label_storage_type
        self._weight_storage_type = weight_storage_type
        self._block_keys_info = block_keys_info

    @property
    def value_storage_type(self):
        return self._value_storage_type

    @value_storage_type.setter
    def value_storage_type(self, storage_type):
        self._value_storage_type = storage_type

    @property
    def label_storage_type(self):
        return self._label_storage_type

    @label_storage_type.setter
    def label_storage_type(self, storage_type):
        self._label_storage_type = storage_type

    @property
    def weight_storage_type(self):
        return self._weight_storage_type

    @weight_storage_type.setter
    def weight_storage_type(self, storage_type):
        self._weight_storage_type = storage_type

    @property
    def block_keys_info(self):
        return self._block_keys_info

    @block_keys_info.setter
    def block_keys_info(self, info_dict):
        self._block_keys_info = info_dict
