import numpy as np


class StatOperators(object):
    @classmethod
    def min(cls, table, *args, **kwargs):
        return table.mapValues(lambda block: np.min(block, *args, **kwargs)).\
            reduce(lambda blk1, blk2: np.min(np.vstack(blk1, blk2), *args, **kwargs))

    @classmethod
    def max(cls, table, *args, **kwargs):
        return table.mapValues(lambda block: np.max(block, *args, **kwargs)). \
            reduce(lambda blk1, blk2: np.max(np.vstack(blk1, blk2)))

    @classmethod
    def sum(cls, table, *args, **kwargs):
        return table.mapValues(lambda block: np.sum(block, *args, **kwargs)). \
            reduce(lambda blk1, blk2: np.sum(np.vstack(blk1, blk2), *args, **kwargs))

    @classmethod
    def mean(cls, table, *args, **kwargs):
        return cls.sum(table, *args, **kwargs) / table.count()

    @classmethod
    def var(cls, table, *args, **kwargs):
        sum_avg = table.mapValues(lambda block: np.sum(block ** 2, *args, **kwargs)).\
            reduce(lambda blk1, blk2: np.sum(np.vstack(blk1, blk2), *args, **kwargs))
        table_mean = cls.mean(table, *args, **kwargs)

        return sum_avg / table.count() - table_mean

    @classmethod
    def std(cls, table, *args, **kwargs):
        return np.sqrt(cls.var(table, *args, **kwargs))

