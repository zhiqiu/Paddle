//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/fluid/operators/auto_increment_op.h"
#include <memory>
#include <string>

namespace paddle {
namespace operators {

class AutoIncrementOp : public framework::OperatorWithKernel {
 public:
  AutoIncrementOp(const std::string &type,
                  const framework::VariableNameMap &inputs,
                  const framework::VariableNameMap &outputs,
                  const framework::AttributeMap &attrs)
      : OperatorWithKernel(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of AutoIncrementOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of AutoIncrementOp should not be null.");
    ctx->SetOutputDim("Out", ctx->GetInputDim("X"));
    ctx->ShareLoD("X", "Out");
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext &ctx) const override {
    framework::OpKernelType kt = OperatorWithKernel::GetExpectedKernelType(ctx);
    // AutoIncrementOp kernel's device type is decided by input tensor place
    kt.place_ = ctx.Input<framework::LoDTensor>("X")->place();
    return kt;
  }
};

class AutoIncrementOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) The input tensor of auto_increment operator");
    AddOutput("Out", "(Tensor) The output tensor of auto_increment operator.");
    AddComment(R"DOC(
AutoIncrement Operator.

The equation is: 
$$Out = X + 1$$

)DOC");
  }
};

class AutoIncrementGradOpMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

  std::unique_ptr<framework::OpDesc> Apply() const override {
    auto *grad_op = new framework::OpDesc();
    grad_op->SetType("auto_increment");
    grad_op->SetInput("X", Output("Out"));
    grad_op->SetOutput("Out", Input("X"));
    return std::unique_ptr<framework::OpDesc>(grad_op);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(auto_increment, ops::AutoIncrementOp,
                  ops::AutoIncrementOpMaker, ops::AutoIncrementGradOpMaker);
REGISTER_OP_CPU_KERNEL(
    auto_increment,
    ops::AutoIncrementKernel<paddle::platform::CPUDeviceContext, float>,
    ops::AutoIncrementKernel<paddle::platform::CPUDeviceContext, double>,
    ops::AutoIncrementKernel<paddle::platform::CPUDeviceContext, int>,
    ops::AutoIncrementKernel<paddle::platform::CPUDeviceContext, int64_t>);
