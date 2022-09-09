import typing
from federatedml.protobuf import get_proto_buffer_class

class Models:
    @classmethod
    def parse(cls, model_input):
        for model_type, models in model_input.items():
            for cpn_name, cpn_models in models.items():
                for model_name, (pb_name, pb_buffer) in cpn_models.items():
                    pb_object = get_proto_buffer_class(pb_name)()
                    pb_object.ParseFromString(pb_buffer)
                    model_input[model_type][cpn_name][model_name] = pb_object
        return model_input

def serialize_models(models):
    from google.protobuf import json_format

    serialized_models: typing.Dict[str, typing.Tuple[str, bytes, dict]] = {}

    for model_name, buffer_object in models.items():
        serialized_string = buffer_object.SerializeToString()
        pb_name = type(buffer_object).__name__
        json_format_dict = json_format.MessageToDict(
            buffer_object, including_default_value_fields=True
        )

        serialized_models[model_name] = (
            pb_name,
            serialized_string,
            json_format_dict,
        )

    return serialized_models
