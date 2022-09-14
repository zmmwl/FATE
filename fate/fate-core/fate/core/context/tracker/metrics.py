
def callback_meta(self, metric_name, metric_namespace, metric_meta: MetricMeta):
    if self.need_cv:
        metric_name = ".".join([metric_name, str(self.cv_fold)])
        flow_id_list = self.flowid.split(".")
        LOGGER.debug(
            "Need cv, change callback_meta, flow_id_list: {}".format(flow_id_list)
        )
        if len(flow_id_list) > 1:
            curve_name = ".".join(flow_id_list[1:])
            metric_meta.update_metas({"curve_name": curve_name})
    else:
        metric_meta.update_metas({"curve_name": metric_name})

    self.tracker.set_metric_meta(
        metric_name=metric_name,
        metric_namespace=metric_namespace,
        metric_meta=metric_meta,
    )

def callback_metric(
    self, metric_name, metric_namespace, metric_data: typing.List[Metric]
):
    if self.need_cv:
        metric_name = ".".join([metric_name, str(self.cv_fold)])

    self.tracker.log_metric_data(
        metric_name=metric_name,
        metric_namespace=metric_namespace,
        metrics=metric_data,
    )

def callback_warm_start_init_iter(self, iter_num):
    metric_meta = MetricMeta(
        name="train",
        metric_type="init_iter",
        extra_metas={
            "unit_name": "iters",
        },
    )

    self.callback_meta(
        metric_name="init_iter", metric_namespace="train", metric_meta=metric_meta
    )
    self.callback_metric(
        metric_name="init_iter",
        metric_namespace="train",
        metric_data=[Metric("init_iter", iter_num)],
    )
