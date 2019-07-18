# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
import paddle.fluid.core as core
from op_test import OpTest


class TestAutoIncrementOp(OpTest):
    def setUp(self):
        self.op_type = "auto_increment"
        self.inputs = {'X': np.random.random(size=[1])}
        self.outputs = {'Out': np.add(self.inputs['X'], 1)}

    def test_check_output(self):
        self.check_output()

    def test_check_grad_normal(self):
        self.check_grad(['X'], 'Out', max_relative_error=0.1)


@unittest.skipIf(not core.is_compiled_with_cuda(),
                 "core is not compiled with CUDA")
class TestFP16AutoIncrementOp(TestAutoIncrementOp):
    def init_dtype_type(self):
        self.dtype = np.float16

    def test_check_output(self):
        place = core.CUDAPlace(0)
        if core.is_float16_supported(place):
            self.check_output_with_place(place, atol=le - 1)

    def test_check_grad_normal(self):
        place = core.CUDAPlace(0)
        self.check_grad_with_place(place, ['X'], 'Out', max_relative_error=0.1)


if __name__ == "__main__":
    unittest.main()
