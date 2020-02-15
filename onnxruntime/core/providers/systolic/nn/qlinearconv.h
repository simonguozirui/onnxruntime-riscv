// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/op_kernel.h"
#include "core/providers/cpu/nn/conv_attributes.h"
#include "core/mlas/inc/mlas.h"

namespace onnxruntime {
namespace systolic {

template <typename T1, typename T2, typename T3>
class QLinearConv : public OpKernel {
 public:
  explicit QLinearConv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_(info) {
  }

  Status Compute(OpKernelContext* context) const override;
  ConvAttributes conv_attrs_;
  bool fused_relu_ = false;
};

template <typename T1, typename T2, typename T3>
class FusedQLinearConvRelu : public QLinearConv<T1, T2, T3> {
 public:
  explicit FusedQLinearConvRelu(const OpKernelInfo& info) : QLinearConv<T1, T2, T3>(info) {
    this->fused_relu_ = true;
  }
};

} // namespace systolic
}  // namespace onnxruntime
