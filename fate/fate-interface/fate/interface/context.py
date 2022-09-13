from contextlib import contextmanager


class Context:
    role: property
    party_id: property
    tracker: Tracker = property

    @contextmanager
    def namespace(self, name):
        try:
            yield self.with_namespace(name)
        finally:
            # set global context as self
            ...

    def with_namespace(self, name) -> "Context":
        # return context with namespace changed
        ...

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
