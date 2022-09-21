from federatedml.param.base_param import BaseParam


class DataTransformParam(BaseParam):
    def __init__(self, input_format="dense", delimiter=",",
                 with_label=False, label_name="y", with_weight=False, weight_name=None,
                 with_match_id=False, match_id_name=None):
        self.input_format = input_format
        self.delimiter = delimiter
        self.with_label = with_label
        self.label_name = label_name
        self.with_weight = with_weight
        self.weight_name = weight_name
        self.with_match_id = with_match_id
        self.match_id_name = match_id_name

    def check(self):
        assert isinstance(self.with_match_id, bool), f"with_match_id is bool, not {type(self.with_match_id)}"
