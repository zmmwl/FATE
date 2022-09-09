import typing
from fate.ml.modules.module import Metric
from federatedml.callbacks.callback_list import CallbackList
from federatedml.util.anonymous_generator_util import Anonymous


class Context:
    def __init__(self) -> None:
        self._anonymous_generator = None
        self._role = None
        self._party_id = None
        self._tracker = None


    def callback_list(self):
        self._callback_list = CallbackList(self.role, self.mode, self)

    @property
    def anonymous_generator(self):
        if self._anonymous_generator is None:
            self._anonymous_generator = Anonymous(self.role, self.party_id)
        return self._anonymous_generator

    def set_party(self, role, party_id):
        self._role = role
        self._party_id = party_id


    @property
    def role(self):
        if self._role is None:
            raise RuntimeError(f"role used before set")
        return self._role

    @property
    def party_id(self):
        if self._party_id is None:
            raise RuntimeError(f"party_id used before set")
        return self._party_id

    def set_tracker(self, tracker):
        self._tracker = tracker

    @property
    def tracker(self):
        if self._tracker is None:
            raise RuntimeError(f"tracker used before set")
        return self._tracker

    def callback_metric(
        self, metric_name, metric_namespace, metric_data: typing.List[Metric]
    ):
        # TODO: namespace for tracker used in cv/stepwise
        # if self.need_cv:
        #     metric_name = ".".join([metric_name, str(self.cv_fold)])

        self.tracker.log_metric_data(
            metric_name=metric_name,
            metric_namespace=metric_namespace,
            metrics=metric_data,
        )
