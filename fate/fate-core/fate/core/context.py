import typing
from contextlib import contextmanager

from fate.interface import Context as ContextInterface
from federatedml.callbacks.callback_list import CallbackList
from federatedml.util.anonymous_generator_util import Anonymous

from .tracker import Metric, WarpedTrackerClient


def _add_propery(name):
    attr_name = f"_{name}"

    def fget(self):
        if not hasattr(self, attr_name):
            raise AttributeError(f"{name} used before set")
        return getattr(self, attr_name)

    def fset(self, value):
        if hasattr(self, attr_name):
            raise AttributeError(f"{name} already set")
        setattr(self, attr_name, value)

    return property(fget, fset)


class Context(ContextInterface):
    """
    implement fate.interface.ContextInterface
    """
    role = _add_propery("role")
    party_id = _add_propery("party_id")
    tracker: property = _add_propery("tracker")

    def __init__(self, namespace=None) -> None:
        self._namespace = namespace
        self._anonymous_generator = None

    @contextmanager
    def namespace(self, name):
        try:
            yield self.with_namespace(name)
        finally:
            # set global context as self
            ...

    def with_namespace(self, namespace) -> "Context":
        if self._namespace is None:
            namespace = namespace
        else:
            namespace = f"{self._namespace}.{namespace}"
        subctx = Context(namespace)
        subctx._anonymous_generator = self._anonymous_generator
        subctx._role = self._role
        subctx._party_id = self._party_id
        subctx._tracker = self._tracker
        return subctx

    def callback_list(self):
        self._callback_list = CallbackList(self.role, self.mode, self)

    @property
    def anonymous_generator(self):
        if self._anonymous_generator is None:
            self._anonymous_generator = Anonymous(self.role, self.party_id)
        return self._anonymous_generator

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
