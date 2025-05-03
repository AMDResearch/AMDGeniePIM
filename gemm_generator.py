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

from enums import gemm_gen_mode_enum, gemm_index, llm_input_index
from options_parser import ARGS

DEBUG = ARGS.debug

GEMM_LIST = []
# Check the GEMV generator option
if ARGS.gemm_gen_mode == gemm_gen_mode_enum.GEN_GEMM_IN.value:  # Read GEMVs to evaluate from gemm.in file
    input_file = "Inputs/GEMMs/" + ARGS.gemm_gen_input
    separator = ","
    header_line_flag = True
    with open(input_file, 'r') as f:
        for line in f:
            if header_line_flag == False:   # File must have a header line 
                line_parts = line.strip().split(separator)
                assert(len(line_parts) == 6)    # Six entires per line

                model_id = line_parts[gemm_index.MODEL_ID.value]
                gemm_id = line_parts[gemm_index.GEMM_ID.value]
                gemm_m = int(line_parts[gemm_index.GEMM_M.value])
                gemm_k = int(line_parts[gemm_index.GEMM_K.value])
                bgemm_batch_size = int(line_parts[gemm_index.BGEMM_BATCH_SIZE.value])
                gemm_n_list = line_parts[gemm_index.GEMM_N_LIST.value]
                gemm_n_list = gemm_n_list.strip().split("/")

                # Loop over gemm_ values 
                for gemm_n in gemm_n_list:
                    gemm_n = int(gemm_n)
                    GEMM_LIST.append((model_id, gemm_id, gemm_m, gemm_k, bgemm_batch_size, gemm_n))
            else:
                header_line_flag = False

elif ARGS.gemm_gen_mode == gemm_gen_mode_enum.GEN_LLM_IN.value: # Generate GEMVs to evaluate based on LLMs in models.in file
    input_file = "Inputs/LLMs/" + ARGS.gemm_gen_input
    separator = ","
    header_line_flag = True
    with open(input_file, 'r') as f:
        for line in f:
            if header_line_flag == False:   # File must have a header line 
                line_parts = line.strip().split(separator)
                assert(len(line_parts) == 8)    # 8 entires per line

                llm_id = line_parts[llm_input_index.LLM_ID.value]
                llm_h = int(line_parts[llm_input_index.LLM_H.value])
                llm_i = int(line_parts[llm_input_index.LLM_I.value])
                llm_a = int(line_parts[llm_input_index.LLM_A.value])
                llm_sl_list = line_parts[llm_input_index.LLM_SL.value]
                llm_b_list = line_parts[llm_input_index.LLM_B.value]
                llm_prompt_size_list = line_parts[llm_input_index.LLM_PROMPT_SIZE.value]
                llm_t_list = line_parts[llm_input_index.LLM_T.value]

                SL_LIST = []
                # Process LLM SL list
                llm_sl_list = llm_sl_list.strip().split("/")
                for llm_sl in llm_sl_list:
                    SL_LIST.append(int(llm_sl))
                
                PROMPT_SIZE_LIST = []
                if llm_prompt_size_list != "x":
                    # Process LLM SL list
                    llm_prompt_size_list = llm_prompt_size_list.strip().split("/")
                    for llm_prompt_size in llm_prompt_size_list:
                        PROMPT_SIZE_LIST.append(int(llm_prompt_size))

                T_LIST = []
                if llm_t_list != "x":
                    # Process LLM SL list
                    llm_t_list = llm_t_list.strip().split("/")
                    for llm_t in llm_t_list:
                        T_LIST.append(int(llm_t))

                B_LIST = []
                # Process LLM B list
                llm_b_list = llm_b_list.strip().split("/")
                B_LIST = llm_b_list

                # Loop over B values 
                for B_TUPLE in B_LIST:
                    B = int(B_TUPLE)

                    # First GEMV sizes 
                    # ------------------
                    # Loop over SL values
                    for SL in SL_LIST:
                        # Input projection (fused QKV) GEMV - i.e, batch-size = 3 but a fused kernel is assumed (so M -> 3*M)
                        weight_matrix_id = "ip-proj"
                        weight_matrix_m = 3 * llm_h   # 3 for fusing Q, K, and V
                        weight_matrix_k = llm_h
                        weight_matrix_n = SL * B
                        batch_size = 1
                        GEMM_LIST.append((llm_id, weight_matrix_id, weight_matrix_m, weight_matrix_k, batch_size, weight_matrix_n))

                        # Output projection GEMV
                        weight_matrix_id = "op-proj"
                        weight_matrix_m = llm_h
                        weight_matrix_k = llm_h
                        weight_matrix_n = SL * B
                        batch_size = 1
                        GEMM_LIST.append((llm_id, weight_matrix_id, weight_matrix_m, weight_matrix_k, batch_size, weight_matrix_n))

                        # FF-1 GEMV
                        weight_matrix_id = "linear1"
                        weight_matrix_m = llm_i
                        weight_matrix_k = llm_h
                        weight_matrix_n = SL * B
                        batch_size = 1
                        GEMM_LIST.append((llm_id, weight_matrix_id, weight_matrix_m, weight_matrix_k, batch_size, weight_matrix_n))

                        # FF-2 GEMV
                        weight_matrix_id = "linear2"
                        weight_matrix_m = llm_h
                        weight_matrix_k = llm_i
                        weight_matrix_n = SL * B
                        batch_size = 1
                        GEMM_LIST.append((llm_id, weight_matrix_id, weight_matrix_m, weight_matrix_k, batch_size, weight_matrix_n))

                    # Second BGEMM/V sizes 
                    # --------------------
                    # Loop over prompt size values 
                    for PROMPT_SIZE in PROMPT_SIZE_LIST:
                        # Loop over T values (max token count)
                        for T in T_LIST:
                            # Loop from 0 to T-1
                            for T_CURRENT in range(0, T):
                                # QK BGEMM/V
                                weight_matrix_id = "qk"
                                weight_matrix_m = PROMPT_SIZE + T_CURRENT
                                assert(llm_h % llm_a == 0)  # H/A == integer
                                weight_matrix_k = int(llm_h / llm_a)
                                weight_matrix_n = 1
                                batch_size = B * llm_a
                                GEMM_LIST.append((llm_id, weight_matrix_id, weight_matrix_m, weight_matrix_k, batch_size, weight_matrix_n))

                                # xV BGEMM/V
                                weight_matrix_id = "xv"
                                weight_matrix_m = int(llm_h / llm_a)
                                weight_matrix_k = PROMPT_SIZE + T_CURRENT
                                weight_matrix_n = 1
                                batch_size = B * llm_a
                                GEMM_LIST.append((llm_id, weight_matrix_id, weight_matrix_m, weight_matrix_k, batch_size, weight_matrix_n))

            else:
                header_line_flag = False
else:
    assert(ARGS.gemm_gen_mode == gemm_gen_mode_enum.GEN_CUSTOM_IN.value)

if DEBUG: print("GEMV count = {}".format(len(GEMM_LIST)))
