# -------------------------------------------------------------------------------
# Copyright 2025 Advanced Micro Devices, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------------

from enums import gemm_index

class gemm_params:
    def __init__(self):
        # Assume GEMV MxKxN
        self.gemm_model_id = "test_model"
        self.gemm_source_id = "test_gemm"
        self.gemm_m = 4096
        self.gemm_k = 4096
        self.gemm_n = 1
        self.gemm_bs = 1    # For BGEMMs

    # Extract the GEMV params of current run
    def extract_gemm_params(self, in_gemm):
        self.gemm_model_id = in_gemm[gemm_index.MODEL_ID.value]
        self.gemm_source_id = in_gemm[gemm_index.GEMM_ID.value]
        self.gemm_m = in_gemm[gemm_index.GEMM_M.value]
        self.gemm_k = in_gemm[gemm_index.GEMM_K.value]
        self.gemm_n = in_gemm[gemm_index.GEMM_N_LIST.value]
        self.gemm_bs = in_gemm[gemm_index.BGEMM_BATCH_SIZE.value]    # For BGEMMs