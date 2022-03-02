# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/npsv2/npsv2.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='src/npsv2/npsv2.proto',
  package='npsv2',
  syntax='proto3',
  serialized_pb=_b('\n\x15src/npsv2/npsv2.proto\x12\x05npsv2\"\xb2\x01\n\x11StructuralVariant\x12\x0e\n\x06\x63ontig\x18\x01 \x01(\t\x12\r\n\x05start\x18\x02 \x01(\x03\x12\x0b\n\x03\x65nd\x18\x03 \x01(\x03\x12\r\n\x05svlen\x18\x04 \x03(\x03\x12-\n\x06svtype\x18\x05 \x01(\x0e\x32\x1d.npsv2.StructuralVariant.Type\"3\n\x04Type\x12\x07\n\x03\x44\x45L\x10\x00\x12\x07\n\x03INS\x10\x01\x12\x07\n\x03\x44UP\x10\x02\x12\x07\n\x03INV\x10\x03\x12\x07\n\x03SUB\x10\x04\x62\x06proto3')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)



_STRUCTURALVARIANT_TYPE = _descriptor.EnumDescriptor(
  name='Type',
  full_name='npsv2.StructuralVariant.Type',
  filename=None,
  file=DESCRIPTOR,
  values=[
    _descriptor.EnumValueDescriptor(
      name='DEL', index=0, number=0,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INS', index=1, number=1,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='DUP', index=2, number=2,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='INV', index=3, number=3,
      options=None,
      type=None),
    _descriptor.EnumValueDescriptor(
      name='SUB', index=4, number=4,
      options=None,
      type=None),
  ],
  containing_type=None,
  options=None,
  serialized_start=160,
  serialized_end=211,
)
_sym_db.RegisterEnumDescriptor(_STRUCTURALVARIANT_TYPE)


_STRUCTURALVARIANT = _descriptor.Descriptor(
  name='StructuralVariant',
  full_name='npsv2.StructuralVariant',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='contig', full_name='npsv2.StructuralVariant.contig', index=0,
      number=1, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='start', full_name='npsv2.StructuralVariant.start', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='end', full_name='npsv2.StructuralVariant.end', index=2,
      number=3, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='svlen', full_name='npsv2.StructuralVariant.svlen', index=3,
      number=4, type=3, cpp_type=2, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
    _descriptor.FieldDescriptor(
      name='svtype', full_name='npsv2.StructuralVariant.svtype', index=4,
      number=5, type=14, cpp_type=8, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
    _STRUCTURALVARIANT_TYPE,
  ],
  options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=33,
  serialized_end=211,
)

_STRUCTURALVARIANT.fields_by_name['svtype'].enum_type = _STRUCTURALVARIANT_TYPE
_STRUCTURALVARIANT_TYPE.containing_type = _STRUCTURALVARIANT
DESCRIPTOR.message_types_by_name['StructuralVariant'] = _STRUCTURALVARIANT

StructuralVariant = _reflection.GeneratedProtocolMessageType('StructuralVariant', (_message.Message,), dict(
  DESCRIPTOR = _STRUCTURALVARIANT,
  __module__ = 'src.npsv2.npsv2_pb2'
  # @@protoc_insertion_point(class_scope:npsv2.StructuralVariant)
  ))
_sym_db.RegisterMessage(StructuralVariant)


# @@protoc_insertion_point(module_scope)
