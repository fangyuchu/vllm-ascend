#include <acl/acl.h>

#include "catlass/layout/layout.hpp"
template <aclDataType T>
struct AclType2Type;

template <>
struct AclType2Type<ACL_FLOAT> {
  using type = float;
};

template <>
struct AclType2Type<ACL_INT32> {
  using type = int32_t;
};

template <>
struct AclType2Type<ACL_INT8> {
  using type = int8_t;
};

template <>
struct AclType2Type<ACL_FLOAT16> {
  using type = half;
};

template <>
struct AclType2Type<ACL_BF16> {
  using type = bfloat16_t;
};

template <bool IS_TRANSPOSE>
struct Transpose2Layout;

template <>
struct Transpose2Layout<false> {
  using layout = Catlass::layout::RowMajor;
};
template <>
struct Transpose2Layout<true> {
  using layout = Catlass::layout::ColumnMajor;
};