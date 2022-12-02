#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import argparse
import time

from pipeline.backend.pipeline import PipeLine
from pipeline.component import DataTransform
from pipeline.component import HeteroSecureBoost, HeteroLR
from pipeline.component import Intersection
from pipeline.component import Reader
from pipeline.interface import Data
from pipeline.component import Evaluation


# data sets
GUEST_FAST_DATA = {"name": "breast_hetero_guest", "namespace": f"experiment"}
HOST_FAST_DATA = {"name": "breast_hetero_host", "namespace": f"experiment"}

GUEST_NORMAL_DATA = {"name": "default_credit_hetero_guest", "namespace": f"experiment"}
HOST_NORMAL_DATA = {"name": "default_credit_hetero_host", "namespace": f"experiment"}

def main(file_type, task_type, run_predict, gid, hid, aid):
    if file_type == "fast":
        guest_train_data, host_train_data = GUEST_FAST_DATA, HOST_FAST_DATA
        auc_base = 0.98
    elif file_type == "normal":
        guest_train_data, host_train_data = GUEST_NORMAL_DATA, HOST_NORMAL_DATA
        auc_base = 0.69
    else:
        raise ValueError(f"Unknown file type {file_type}, please check.")
    guest = gid
    host = hid
    arbiter = aid
    if task_type == "lr":
        pipeline = lr_train_pipeline(guest, host, arbiter, guest_train_data, host_train_data)
    elif task_type == "sbt":
        pipeline = sbt_train_pipeline(guest, host, guest_train_data, host_train_data)
    else:
        raise ValueError(f"unknown task type: {task_type}")
    if task_type == "lr":
        model_auc = get_auc(pipeline, "hetero_lr_0")
        if model_auc < auc_base:
            time_print(f"[Warning]  The auc: {model_auc} is lower than expect value: {auc_base}")
    else:
        model_auc = get_auc(pipeline, "hetero_secureboost_0")
        if model_auc < auc_base:
            time_print(f"[Warning]  The auc: {model_auc} is lower than expect value: {auc_base}")
    if run_predict:
        cpn_list = pipeline.get_component_list()[1:]
        pipeline.deploy_component(cpn_list)
        predict_pipeline = PipeLine()
        reader_0 = Reader(name="reader_0")
        reader_0.get_party_instance(role="guest", party_id=guest).component_param(table=guest_train_data)
        reader_0.get_party_instance(role="host", party_id=host).component_param(table=host_train_data)
        predict_pipeline.add_component(reader_0)
        predict_pipeline.add_component(pipeline,
                                       data=Data(
                                           predict_input={pipeline.data_transform_0.input.data: reader_0.output.data}))
        # run predict model
        predict_pipeline.predict()


def lr_train_pipeline(guest, host, arbiter, guest_train_data, host_train_data):
    # init pipeline
    pipeline = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest, host=host, arbiter=arbiter)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role="guest", party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role="host", party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(
        role="guest", party_id=guest).component_param(
        with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role="host", party_id=host).component_param(with_label=False)

    # data intersect component
    intersection_0 = Intersection(name="intersection_0")

    lr_param = {
        "penalty": "L2",
        "tol": 0.0001,
        "alpha": 0.01,
        "optimizer": "rmsprop",
        "batch_size": -1,
        "learning_rate": 0.15,
        "init_param": {
            "init_method": "zeros",
            "fit_intercept": True
        },
        "max_iter": 30,
        "early_stop": "diff",
        "encrypt_param": {
            "key_length": 1024
        },
        "cv_param": {
            "n_splits": 5,
            "shuffle": False,
            "random_seed": 103,
            "need_cv": False
        },
        "validation_freqs": 3
    }
    hetero_lr_0 = HeteroLR(name="hetero_lr_0", **lr_param)

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_lr_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_lr_0.output.data))

    pipeline.compile()
    pipeline.fit()

    return pipeline


def sbt_train_pipeline(guest, host, guest_train_data, host_train_data):
    # init pipeline
    pipeline = PipeLine().set_initiator(role="guest", party_id=guest).set_roles(guest=guest, host=host)

    reader_0 = Reader(name="reader_0")
    reader_0.get_party_instance(role="guest", party_id=guest).component_param(table=guest_train_data)
    reader_0.get_party_instance(role="host", party_id=host).component_param(table=host_train_data)

    data_transform_0 = DataTransform(name="data_transform_0")
    data_transform_0.get_party_instance(
        role="guest", party_id=guest).component_param(
        with_label=True, output_format="dense")
    data_transform_0.get_party_instance(role="host", party_id=host).component_param(with_label=False)

    intersection_0 = Intersection(name="intersection_0")

    sbt_param = {
        "task_type": "classification",
        "objective_param": {
            "objective": "cross_entropy"
        },
        "num_trees": 3,
        "validation_freqs": 1,
        "encrypt_param": {
            "method": "paillier"
        },
        "tree_param": {
            "max_depth": 3
        }
    }
    hetero_secure_boost_0 = HeteroSecureBoost(name="hetero_secureboost_0", **sbt_param)

    evaluation_0 = Evaluation(name="evaluation_0", eval_type="binary")

    pipeline.add_component(reader_0)
    pipeline.add_component(data_transform_0, data=Data(data=reader_0.output.data))
    pipeline.add_component(intersection_0, data=Data(data=data_transform_0.output.data))
    pipeline.add_component(hetero_secure_boost_0, data=Data(train_data=intersection_0.output.data))
    pipeline.add_component(evaluation_0, data=Data(data=hetero_secure_boost_0.output.data))

    pipeline.compile()
    pipeline.fit()

    return pipeline


def get_auc(pipeline, component_name):
    cpn_summary = pipeline.get_component(component_name).get_summary()
    auc = cpn_summary.get("validation_metrics").get("train").get("auc")[-1]
    return auc


def time_print(msg):
    print(f"[{time.strftime('%Y-%m-%d %X')}] {msg}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("MIN TEST")
    parser.add_argument("-gid", type=int, required=True,
                        help="host id")
    parser.add_argument("-hid", type=int, required=True,
                        help="host id")
    parser.add_argument("-aid", type=int,
                        help="host id")
    parser.add_argument("-t", "--task-type", type=str, default="lr",
                        choices=["lr", "sbt"],
                        help="choose from {sbt, lr}")
    parser.add_argument("-f", "--file_type", type=str,
                        help="file_type, "
                             "'fast' means breast data "
                             "'normal' means default credit data",
                        choices=["fast", "normal"],
                        default="fast")
    parser.add_argument("--run-predict", type=bool, default=True,
                        help="whether to run predict task")
    args = parser.parse_args()
    main(args.file_type, args.task_type, args.run_predict, args.gid, args.hid, args.aid)
