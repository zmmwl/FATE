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
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: inference_service.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='inference_service.proto',
  package='com.webank.ai.fate.api.serving',
  syntax='proto3',
  serialized_options=_b('B\025InferenceServiceProto'),
  serialized_pb=_b('\n\x17inference_service.proto\x12\x1e\x63om.webank.ai.fate.api.serving\" \n\x10InferenceMessage\x12\x0c\n\x04\x64\x61ta\x18\x01 \x01(\x0c\x32\xfd\x01\n\x10InferenceService\x12o\n\tinference\x12\x30.com.webank.ai.fate.api.serving.InferenceMessage\x1a\x30.com.webank.ai.fate.api.serving.InferenceMessage\x12x\n\x12getInferenceResult\x12\x30.com.webank.ai.fate.api.serving.InferenceMessage\x1a\x30.com.webank.ai.fate.api.serving.InferenceMessageB\x17\x42\x15InferenceServiceProtob\x06proto3')
)




_INFERENCEMESSAGE = _descriptor.Descriptor(
  name='InferenceMessage',
  full_name='com.webank.ai.fate.api.serving.InferenceMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='com.webank.ai.fate.api.serving.InferenceMessage.data', index=0,
      number=1, type=12, cpp_type=9, label=1,
      has_default_value=False, default_value=_b(""),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=59,
  serialized_end=91,
)

DESCRIPTOR.message_types_by_name['InferenceMessage'] = _INFERENCEMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

InferenceMessage = _reflection.GeneratedProtocolMessageType('InferenceMessage', (_message.Message,), dict(
  DESCRIPTOR = _INFERENCEMESSAGE,
  __module__ = 'inference_service_pb2'
  # @@protoc_insertion_point(class_scope:com.webank.ai.fate.api.serving.InferenceMessage)
  ))
_sym_db.RegisterMessage(InferenceMessage)


DESCRIPTOR._options = None

_INFERENCESERVICE = _descriptor.ServiceDescriptor(
  name='InferenceService',
  full_name='com.webank.ai.fate.api.serving.InferenceService',
  file=DESCRIPTOR,
  index=0,
  serialized_options=None,
  serialized_start=94,
  serialized_end=347,
  methods=[
  _descriptor.MethodDescriptor(
    name='inference',
    full_name='com.webank.ai.fate.api.serving.InferenceService.inference',
    index=0,
    containing_service=None,
    input_type=_INFERENCEMESSAGE,
    output_type=_INFERENCEMESSAGE,
    serialized_options=None,
  ),
  _descriptor.MethodDescriptor(
    name='getInferenceResult',
    full_name='com.webank.ai.fate.api.serving.InferenceService.getInferenceResult',
    index=1,
    containing_service=None,
    input_type=_INFERENCEMESSAGE,
    output_type=_INFERENCEMESSAGE,
    serialized_options=None,
  ),
])
_sym_db.RegisterServiceDescriptor(_INFERENCESERVICE)

DESCRIPTOR.services_by_name['InferenceService'] = _INFERENCESERVICE

# @@protoc_insertion_point(module_scope)
