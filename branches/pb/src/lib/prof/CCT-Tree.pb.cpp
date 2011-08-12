// Generated by the protocol buffer compiler.  DO NOT EDIT!

#define INTERNAL_SUPPRESS_PROTOBUF_FIELD_DEPRECATION
#include "CCT-Tree.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/once.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/wire_format_lite_inl.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)

namespace Nodes {

namespace {

const ::google::protobuf::Descriptor* GenNode_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  GenNode_reflection_ = NULL;
const ::google::protobuf::Descriptor* Metric_descriptor_ = NULL;
const ::google::protobuf::internal::GeneratedMessageReflection*
  Metric_reflection_ = NULL;

}  // namespace


void protobuf_AssignDesc_CCT_2dTree_2eproto() {
  protobuf_AddDesc_CCT_2dTree_2eproto();
  const ::google::protobuf::FileDescriptor* file =
    ::google::protobuf::DescriptorPool::generated_pool()->FindFileByName(
      "CCT-Tree.proto");
  GOOGLE_CHECK(file != NULL);
  GenNode_descriptor_ = file->message_type(0);
  static const int GenNode_offsets_[10] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(GenNode, id_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(GenNode, static_scope_id_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(GenNode, parent_id_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(GenNode, name_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(GenNode, metric_values_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(GenNode, type_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(GenNode, trace_id_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(GenNode, depth_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(GenNode, file_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(GenNode, line_range_),
  };
  GenNode_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      GenNode_descriptor_,
      GenNode::default_instance_,
      GenNode_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(GenNode, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(GenNode, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(GenNode));
  Metric_descriptor_ = file->message_type(1);
  static const int Metric_offsets_[2] = {
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Metric, name_),
    GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Metric, value_),
  };
  Metric_reflection_ =
    new ::google::protobuf::internal::GeneratedMessageReflection(
      Metric_descriptor_,
      Metric::default_instance_,
      Metric_offsets_,
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Metric, _has_bits_[0]),
      GOOGLE_PROTOBUF_GENERATED_MESSAGE_FIELD_OFFSET(Metric, _unknown_fields_),
      -1,
      ::google::protobuf::DescriptorPool::generated_pool(),
      ::google::protobuf::MessageFactory::generated_factory(),
      sizeof(Metric));
}

namespace {

GOOGLE_PROTOBUF_DECLARE_ONCE(protobuf_AssignDescriptors_once_);
inline void protobuf_AssignDescriptorsOnce() {
  ::google::protobuf::GoogleOnceInit(&protobuf_AssignDescriptors_once_,
                 &protobuf_AssignDesc_CCT_2dTree_2eproto);
}

void protobuf_RegisterTypes(const ::std::string&) {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    GenNode_descriptor_, &GenNode::default_instance());
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedMessage(
    Metric_descriptor_, &Metric::default_instance());
}

}  // namespace

void protobuf_ShutdownFile_CCT_2dTree_2eproto() {
  delete GenNode::default_instance_;
  delete GenNode_reflection_;
  delete Metric::default_instance_;
  delete Metric_reflection_;
}

void protobuf_AddDesc_CCT_2dTree_2eproto() {
  static bool already_here = false;
  if (already_here) return;
  already_here = true;
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  ::google::protobuf::DescriptorPool::InternalAddGeneratedFile(
    "\n\016CCT-Tree.proto\022\005Nodes\"\306\001\n\007GenNode\022\n\n\002i"
    "d\030\001 \001(\005\022\027\n\017static_scope_id\030\002 \001(\005\022\021\n\tpare"
    "nt_id\030\003 \002(\005\022\014\n\004name\030\004 \001(\005\022$\n\rmetric_valu"
    "es\030\005 \003(\0132\r.Nodes.Metric\022\014\n\004type\030\006 \002(\005\022\020\n"
    "\010trace_id\030\007 \001(\005\022\r\n\005depth\030\010 \001(\005\022\014\n\004file\030\t"
    " \001(\005\022\022\n\nline_range\030\n \001(\005\"%\n\006Metric\022\014\n\004na"
    "me\030\001 \002(\005\022\r\n\005value\030\002 \002(\001B\025\n\010ProtobufB\tCCT"
    "TreePB", 286);
  ::google::protobuf::MessageFactory::InternalRegisterGeneratedFile(
    "CCT-Tree.proto", &protobuf_RegisterTypes);
  GenNode::default_instance_ = new GenNode();
  Metric::default_instance_ = new Metric();
  GenNode::default_instance_->InitAsDefaultInstance();
  Metric::default_instance_->InitAsDefaultInstance();
  ::google::protobuf::internal::OnShutdown(&protobuf_ShutdownFile_CCT_2dTree_2eproto);
}

// Force AddDescriptors() to be called at static initialization time.
struct StaticDescriptorInitializer_CCT_2dTree_2eproto {
  StaticDescriptorInitializer_CCT_2dTree_2eproto() {
    protobuf_AddDesc_CCT_2dTree_2eproto();
  }
} static_descriptor_initializer_CCT_2dTree_2eproto_;


// ===================================================================

#ifndef _MSC_VER
const int GenNode::kIdFieldNumber;
const int GenNode::kStaticScopeIdFieldNumber;
const int GenNode::kParentIdFieldNumber;
const int GenNode::kNameFieldNumber;
const int GenNode::kMetricValuesFieldNumber;
const int GenNode::kTypeFieldNumber;
const int GenNode::kTraceIdFieldNumber;
const int GenNode::kDepthFieldNumber;
const int GenNode::kFileFieldNumber;
const int GenNode::kLineRangeFieldNumber;
#endif  // !_MSC_VER

GenNode::GenNode()
  : ::google::protobuf::Message() {
  SharedCtor();
}

void GenNode::InitAsDefaultInstance() {
}

GenNode::GenNode(const GenNode& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
}

void GenNode::SharedCtor() {
  _cached_size_ = 0;
  id_ = 0;
  static_scope_id_ = 0;
  parent_id_ = 0;
  name_ = 0;
  type_ = 0;
  trace_id_ = 0;
  depth_ = 0;
  file_ = 0;
  line_range_ = 0;
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

GenNode::~GenNode() {
  SharedDtor();
}

void GenNode::SharedDtor() {
  if (this != default_instance_) {
  }
}

void GenNode::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* GenNode::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return GenNode_descriptor_;
}

const GenNode& GenNode::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_CCT_2dTree_2eproto();  return *default_instance_;
}

GenNode* GenNode::default_instance_ = NULL;

GenNode* GenNode::New() const {
  return new GenNode;
}

void GenNode::Clear() {
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    id_ = 0;
    static_scope_id_ = 0;
    parent_id_ = 0;
    name_ = 0;
    type_ = 0;
    trace_id_ = 0;
    depth_ = 0;
  }
  if (_has_bits_[8 / 32] & (0xffu << (8 % 32))) {
    file_ = 0;
    line_range_ = 0;
  }
  metric_values_.Clear();
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool GenNode::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) return false
  ::google::protobuf::uint32 tag;
  while ((tag = input->ReadTag()) != 0) {
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // optional int32 id = 1;
      case 1: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &id_)));
          set_has_id();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(16)) goto parse_static_scope_id;
        break;
      }
      
      // optional int32 static_scope_id = 2;
      case 2: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_static_scope_id:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &static_scope_id_)));
          set_has_static_scope_id();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(24)) goto parse_parent_id;
        break;
      }
      
      // required int32 parent_id = 3;
      case 3: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_parent_id:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &parent_id_)));
          set_has_parent_id();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(32)) goto parse_name;
        break;
      }
      
      // optional int32 name = 4;
      case 4: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_name:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &name_)));
          set_has_name();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(42)) goto parse_metric_values;
        break;
      }
      
      // repeated .Nodes.Metric metric_values = 5;
      case 5: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED) {
         parse_metric_values:
          DO_(::google::protobuf::internal::WireFormatLite::ReadMessageNoVirtual(
                input, add_metric_values()));
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(42)) goto parse_metric_values;
        if (input->ExpectTag(48)) goto parse_type;
        break;
      }
      
      // required int32 type = 6;
      case 6: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_type:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &type_)));
          set_has_type();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(56)) goto parse_trace_id;
        break;
      }
      
      // optional int32 trace_id = 7;
      case 7: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_trace_id:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &trace_id_)));
          set_has_trace_id();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(64)) goto parse_depth;
        break;
      }
      
      // optional int32 depth = 8;
      case 8: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_depth:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &depth_)));
          set_has_depth();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(72)) goto parse_file;
        break;
      }
      
      // optional int32 file = 9;
      case 9: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_file:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &file_)));
          set_has_file();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(80)) goto parse_line_range;
        break;
      }
      
      // optional int32 line_range = 10;
      case 10: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
         parse_line_range:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &line_range_)));
          set_has_line_range();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectAtEnd()) return true;
        break;
      }
      
      default: {
      handle_uninterpreted:
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          return true;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
  return true;
#undef DO_
}

void GenNode::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // optional int32 id = 1;
  if (has_id()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(1, this->id(), output);
  }
  
  // optional int32 static_scope_id = 2;
  if (has_static_scope_id()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(2, this->static_scope_id(), output);
  }
  
  // required int32 parent_id = 3;
  if (has_parent_id()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(3, this->parent_id(), output);
  }
  
  // optional int32 name = 4;
  if (has_name()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(4, this->name(), output);
  }
  
  // repeated .Nodes.Metric metric_values = 5;
  for (int i = 0; i < this->metric_values_size(); i++) {
    ::google::protobuf::internal::WireFormatLite::WriteMessageMaybeToArray(
      5, this->metric_values(i), output);
  }
  
  // required int32 type = 6;
  if (has_type()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(6, this->type(), output);
  }
  
  // optional int32 trace_id = 7;
  if (has_trace_id()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(7, this->trace_id(), output);
  }
  
  // optional int32 depth = 8;
  if (has_depth()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(8, this->depth(), output);
  }
  
  // optional int32 file = 9;
  if (has_file()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(9, this->file(), output);
  }
  
  // optional int32 line_range = 10;
  if (has_line_range()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(10, this->line_range(), output);
  }
  
  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
}

::google::protobuf::uint8* GenNode::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // optional int32 id = 1;
  if (has_id()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(1, this->id(), target);
  }
  
  // optional int32 static_scope_id = 2;
  if (has_static_scope_id()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(2, this->static_scope_id(), target);
  }
  
  // required int32 parent_id = 3;
  if (has_parent_id()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(3, this->parent_id(), target);
  }
  
  // optional int32 name = 4;
  if (has_name()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(4, this->name(), target);
  }
  
  // repeated .Nodes.Metric metric_values = 5;
  for (int i = 0; i < this->metric_values_size(); i++) {
    target = ::google::protobuf::internal::WireFormatLite::
      WriteMessageNoVirtualToArray(
        5, this->metric_values(i), target);
  }
  
  // required int32 type = 6;
  if (has_type()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(6, this->type(), target);
  }
  
  // optional int32 trace_id = 7;
  if (has_trace_id()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(7, this->trace_id(), target);
  }
  
  // optional int32 depth = 8;
  if (has_depth()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(8, this->depth(), target);
  }
  
  // optional int32 file = 9;
  if (has_file()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(9, this->file(), target);
  }
  
  // optional int32 line_range = 10;
  if (has_line_range()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(10, this->line_range(), target);
  }
  
  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  return target;
}

int GenNode::ByteSize() const {
  int total_size = 0;
  
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    // optional int32 id = 1;
    if (has_id()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
          this->id());
    }
    
    // optional int32 static_scope_id = 2;
    if (has_static_scope_id()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
          this->static_scope_id());
    }
    
    // required int32 parent_id = 3;
    if (has_parent_id()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
          this->parent_id());
    }
    
    // optional int32 name = 4;
    if (has_name()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
          this->name());
    }
    
    // required int32 type = 6;
    if (has_type()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
          this->type());
    }
    
    // optional int32 trace_id = 7;
    if (has_trace_id()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
          this->trace_id());
    }
    
    // optional int32 depth = 8;
    if (has_depth()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
          this->depth());
    }
    
  }
  if (_has_bits_[8 / 32] & (0xffu << (8 % 32))) {
    // optional int32 file = 9;
    if (has_file()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
          this->file());
    }
    
    // optional int32 line_range = 10;
    if (has_line_range()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
          this->line_range());
    }
    
  }
  // repeated .Nodes.Metric metric_values = 5;
  total_size += 1 * this->metric_values_size();
  for (int i = 0; i < this->metric_values_size(); i++) {
    total_size +=
      ::google::protobuf::internal::WireFormatLite::MessageSizeNoVirtual(
        this->metric_values(i));
  }
  
  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void GenNode::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const GenNode* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const GenNode*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void GenNode::MergeFrom(const GenNode& from) {
  GOOGLE_CHECK_NE(&from, this);
  metric_values_.MergeFrom(from.metric_values_);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from.has_id()) {
      set_id(from.id());
    }
    if (from.has_static_scope_id()) {
      set_static_scope_id(from.static_scope_id());
    }
    if (from.has_parent_id()) {
      set_parent_id(from.parent_id());
    }
    if (from.has_name()) {
      set_name(from.name());
    }
    if (from.has_type()) {
      set_type(from.type());
    }
    if (from.has_trace_id()) {
      set_trace_id(from.trace_id());
    }
    if (from.has_depth()) {
      set_depth(from.depth());
    }
  }
  if (from._has_bits_[8 / 32] & (0xffu << (8 % 32))) {
    if (from.has_file()) {
      set_file(from.file());
    }
    if (from.has_line_range()) {
      set_line_range(from.line_range());
    }
  }
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void GenNode::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void GenNode::CopyFrom(const GenNode& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool GenNode::IsInitialized() const {
  if ((_has_bits_[0] & 0x00000024) != 0x00000024) return false;
  
  for (int i = 0; i < metric_values_size(); i++) {
    if (!this->metric_values(i).IsInitialized()) return false;
  }
  return true;
}

void GenNode::Swap(GenNode* other) {
  if (other != this) {
    std::swap(id_, other->id_);
    std::swap(static_scope_id_, other->static_scope_id_);
    std::swap(parent_id_, other->parent_id_);
    std::swap(name_, other->name_);
    metric_values_.Swap(&other->metric_values_);
    std::swap(type_, other->type_);
    std::swap(trace_id_, other->trace_id_);
    std::swap(depth_, other->depth_);
    std::swap(file_, other->file_);
    std::swap(line_range_, other->line_range_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata GenNode::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = GenNode_descriptor_;
  metadata.reflection = GenNode_reflection_;
  return metadata;
}


// ===================================================================

#ifndef _MSC_VER
const int Metric::kNameFieldNumber;
const int Metric::kValueFieldNumber;
#endif  // !_MSC_VER

Metric::Metric()
  : ::google::protobuf::Message() {
  SharedCtor();
}

void Metric::InitAsDefaultInstance() {
}

Metric::Metric(const Metric& from)
  : ::google::protobuf::Message() {
  SharedCtor();
  MergeFrom(from);
}

void Metric::SharedCtor() {
  _cached_size_ = 0;
  name_ = 0;
  value_ = 0;
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
}

Metric::~Metric() {
  SharedDtor();
}

void Metric::SharedDtor() {
  if (this != default_instance_) {
  }
}

void Metric::SetCachedSize(int size) const {
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
}
const ::google::protobuf::Descriptor* Metric::descriptor() {
  protobuf_AssignDescriptorsOnce();
  return Metric_descriptor_;
}

const Metric& Metric::default_instance() {
  if (default_instance_ == NULL) protobuf_AddDesc_CCT_2dTree_2eproto();  return *default_instance_;
}

Metric* Metric::default_instance_ = NULL;

Metric* Metric::New() const {
  return new Metric;
}

void Metric::Clear() {
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    name_ = 0;
    value_ = 0;
  }
  ::memset(_has_bits_, 0, sizeof(_has_bits_));
  mutable_unknown_fields()->Clear();
}

bool Metric::MergePartialFromCodedStream(
    ::google::protobuf::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!(EXPRESSION)) return false
  ::google::protobuf::uint32 tag;
  while ((tag = input->ReadTag()) != 0) {
    switch (::google::protobuf::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // required int32 name = 1;
      case 1: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_VARINT) {
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   ::google::protobuf::int32, ::google::protobuf::internal::WireFormatLite::TYPE_INT32>(
                 input, &name_)));
          set_has_name();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectTag(17)) goto parse_value;
        break;
      }
      
      // required double value = 2;
      case 2: {
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_FIXED64) {
         parse_value:
          DO_((::google::protobuf::internal::WireFormatLite::ReadPrimitive<
                   double, ::google::protobuf::internal::WireFormatLite::TYPE_DOUBLE>(
                 input, &value_)));
          set_has_value();
        } else {
          goto handle_uninterpreted;
        }
        if (input->ExpectAtEnd()) return true;
        break;
      }
      
      default: {
      handle_uninterpreted:
        if (::google::protobuf::internal::WireFormatLite::GetTagWireType(tag) ==
            ::google::protobuf::internal::WireFormatLite::WIRETYPE_END_GROUP) {
          return true;
        }
        DO_(::google::protobuf::internal::WireFormat::SkipField(
              input, tag, mutable_unknown_fields()));
        break;
      }
    }
  }
  return true;
#undef DO_
}

void Metric::SerializeWithCachedSizes(
    ::google::protobuf::io::CodedOutputStream* output) const {
  // required int32 name = 1;
  if (has_name()) {
    ::google::protobuf::internal::WireFormatLite::WriteInt32(1, this->name(), output);
  }
  
  // required double value = 2;
  if (has_value()) {
    ::google::protobuf::internal::WireFormatLite::WriteDouble(2, this->value(), output);
  }
  
  if (!unknown_fields().empty()) {
    ::google::protobuf::internal::WireFormat::SerializeUnknownFields(
        unknown_fields(), output);
  }
}

::google::protobuf::uint8* Metric::SerializeWithCachedSizesToArray(
    ::google::protobuf::uint8* target) const {
  // required int32 name = 1;
  if (has_name()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteInt32ToArray(1, this->name(), target);
  }
  
  // required double value = 2;
  if (has_value()) {
    target = ::google::protobuf::internal::WireFormatLite::WriteDoubleToArray(2, this->value(), target);
  }
  
  if (!unknown_fields().empty()) {
    target = ::google::protobuf::internal::WireFormat::SerializeUnknownFieldsToArray(
        unknown_fields(), target);
  }
  return target;
}

int Metric::ByteSize() const {
  int total_size = 0;
  
  if (_has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    // required int32 name = 1;
    if (has_name()) {
      total_size += 1 +
        ::google::protobuf::internal::WireFormatLite::Int32Size(
          this->name());
    }
    
    // required double value = 2;
    if (has_value()) {
      total_size += 1 + 8;
    }
    
  }
  if (!unknown_fields().empty()) {
    total_size +=
      ::google::protobuf::internal::WireFormat::ComputeUnknownFieldsSize(
        unknown_fields());
  }
  GOOGLE_SAFE_CONCURRENT_WRITES_BEGIN();
  _cached_size_ = total_size;
  GOOGLE_SAFE_CONCURRENT_WRITES_END();
  return total_size;
}

void Metric::MergeFrom(const ::google::protobuf::Message& from) {
  GOOGLE_CHECK_NE(&from, this);
  const Metric* source =
    ::google::protobuf::internal::dynamic_cast_if_available<const Metric*>(
      &from);
  if (source == NULL) {
    ::google::protobuf::internal::ReflectionOps::Merge(from, this);
  } else {
    MergeFrom(*source);
  }
}

void Metric::MergeFrom(const Metric& from) {
  GOOGLE_CHECK_NE(&from, this);
  if (from._has_bits_[0 / 32] & (0xffu << (0 % 32))) {
    if (from.has_name()) {
      set_name(from.name());
    }
    if (from.has_value()) {
      set_value(from.value());
    }
  }
  mutable_unknown_fields()->MergeFrom(from.unknown_fields());
}

void Metric::CopyFrom(const ::google::protobuf::Message& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void Metric::CopyFrom(const Metric& from) {
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool Metric::IsInitialized() const {
  if ((_has_bits_[0] & 0x00000003) != 0x00000003) return false;
  
  return true;
}

void Metric::Swap(Metric* other) {
  if (other != this) {
    std::swap(name_, other->name_);
    std::swap(value_, other->value_);
    std::swap(_has_bits_[0], other->_has_bits_[0]);
    _unknown_fields_.Swap(&other->_unknown_fields_);
    std::swap(_cached_size_, other->_cached_size_);
  }
}

::google::protobuf::Metadata Metric::GetMetadata() const {
  protobuf_AssignDescriptorsOnce();
  ::google::protobuf::Metadata metadata;
  metadata.descriptor = Metric_descriptor_;
  metadata.reflection = Metric_reflection_;
  return metadata;
}


// @@protoc_insertion_point(namespace_scope)

}  // namespace Nodes

// @@protoc_insertion_point(global_scope)
