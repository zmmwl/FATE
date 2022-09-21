from fate_arch.data.dataframe import DataFrame
from fate_arch.tensor import GUEST, HOST, Arbiter


class DataLoader(object):
    def __init__(self, dataset, ctx=None, mode="homo", need_align=False, batch_size=-1, shuffle=False, batch_strategy="full", random_seed=None):
        self._ctx = ctx
        self._dataset = dataset
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._batch_strategy = batch_strategy
        self._random_seed = random_seed
        self._need_align = need_align
        self._mode = mode

        self._init_settings()

    def _init_settings(self):
        if self._batch_strategy == "full":
            self._batch_generator = FullBatchDataLoader(self._dataset,
                                                        self._ctx,
                                                        mode=self._mode,
                                                        batch_size=self._batch_size,
                                                        shuffle=self._shuffle,
                                                        random_seed=self._random_seed,
                                                        need_align=self._need_align)
        else:
            raise ValueError(f"batch strategy {self._batch_strategy} is not support")

    def next_batch(self, with_index=True):
        batch = next(self._batch_generator)
        if with_index:
            return batch
        else:
            return batc[1:]

    @staticmethod
    def batch_num(self):
        return self._batch_generator.batch_num


class FullBatchDataLoader(object):
    def __init__(self, dataset, ctx, mode, batch_size, shuffle, random_seed, need_align):
        self._dataset = dataset
        self._ctx = ctx
        self._mode = mode
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._random_seed = random_seed
        self._need_align = need_align

        self._batch_num = None
        self._prepare()

    def _prepare(self):
        if self._mode == "homo":
            if self._ctx.is_arbiter:
                batch_info = self._ctx.pull(HOST, "batch_info")
                self._batch_size = batch_info["batch_size"]
                self._batch_num = batch_info["batch_num"]
            else:
                self._batch_num = (len(self._dataset) + self._batch_size - 1) // self._batch_size
                self._ctx.push(Arbiter, "batch_info", dict(batch_num=self._batch_num, batch_size=self._batch_size))
        elif self._mode == "local":
            self._batch_num = (len(self._dataset) + self._batch_size - 1) // self._batch_size
        elif self._mode == "hetero":
            # TODO: index should be align first
            self._batch_num = (len(self._dataset) + self._batch_size - 1) // self._batch_size

    def __next__(self):
        # TODO: generate a batch of data
        if self._dataset.label and self._dataset.weight:
            return self._dataset.values, self._dataset.label, self._dataset.weight
        elif self._dataset.label:
            return self._dataset.values, self._dataset.label
        else:
            return self._dataset.values

    @property
    def batch_num(self):
        return self._batch_num
