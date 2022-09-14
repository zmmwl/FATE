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

from fate.ml.module.model import ModelLoader, ModelSaver
from fate.interface import ContextInterface, ModuleInterface
from federatedml.util import consts


class Module(ModuleInterface):
    mode = None

    def fit(self, ctx: ContextInterface, train_data, validate_data=None):
        ...

    def transform(self, ctx: ContextInterface, transform_data):
        ...

    def predict(self, ctx: ContextInterface, predict_data):
        ...

    @classmethod
    def load_model(
        cls, ctx: ContextInterface, loader: ModelLoader
    ) -> "ModuleInterface":
        ...

    def save_model(self, ctx: ContextInterface, saver: ModelSaver):
        ...


class HeteroModule(Module):
    mode = consts.HETERO


class HomoModule(Module):
    mode = consts.HOMO
