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

import itertools

from options_parser import ARGS
from config_parser import config_parser

DEBUG = ARGS.debug

# Process the input config file 
input_file = "Inputs/Configs/" + ARGS.config_input_file
if DEBUG: print("Input file = {}".format(input_file))

# Parse the input config file
parser = config_parser(input_file)
parser.parse_config_file(False)

# Generate all parameters combinations 
# The order below matches the order in the configuration file (e.g., check Inputs/Configs/config.in).
COMBINATIONS = itertools.product(
    parser.config_dict["STACK_COUNT_LIST"], 
    parser.config_dict["CHANNEL_PER_STACK_LIST"], 
    parser.config_dict["BANKS_PER_CHANNEL_LIST"], 
    parser.config_dict["DRAM_ROW_SIZE_LIST"],
    parser.config_dict["BANKS_PER_PIM_UNIT_LIST"],
    parser.config_dict["PIM_ALU_PER_PIM_UNIT_LIST"],
    parser.config_dict["ORF_REGISTERS_PER_PIM_ALU_LIST"], 
    parser.config_dict["IRF_REGISTERS_PER_PIM_ALU_LIST"],
    parser.config_dict["SIMD_WIDTH_LIST"], 
    parser.config_dict["PIM_COMPUTE_RATE_LIST"],
    parser.config_dict["PIM_ORF_REGISTER_SIZE_LIST"],
    parser.config_dict["PIM_IRF_REGISTER_SIZE_LIST"],
    parser.config_dict["MEMORY_INTERLEAVING_GRANULARITY_SIZE_LIST"],
    parser.config_dict["IGNORE_HOST_VECTOR_WRITE_OVERHEAD_LIST"],
    parser.config_dict["HOST_PEAK_MEM_COMPUTE_LIST"], 
    parser.config_dict["DRAM_PARAMTERS_LIST"], 
    parser.config_dict["OPERAND_SIZE_LIST"],
    parser.config_dict["ACT_OPERAND_SIZE_LIST"],
    parser.config_dict["BLOCK_SIZE_LIST"], 
    parser.config_dict["PROCESS_SCALE_FACTORS_AT_HOST_FLAG_LIST"], 
    parser.config_dict["BLOCKED_INPUTS_FLAG_LIST"],
    parser.config_dict["SCALE_FACTOR_OPERAND_SIZE_LIST"], 
    parser.config_dict["ASSUME_OPTIMIZED_SCALE_FACTORS_INTO_REGISTER_FLAG_LIST"], 
    parser.config_dict["ACT_SCALE_FACTOR_OPERAND_SIZE_LIST"], 
    parser.config_dict["ACCUMULATION_OPERAND_SIZE_LIST"], 
    parser.config_dict["SCALE_FACTOR_EXTRA_PIM_COMMANDS_OVERHEAD_LIST"],
    parser.config_dict["REGISTERS_MULT_REQUIRED_FOR_ACCUMULATION_LIST"],
    parser.config_dict["UPCASTING_PIM_COMMANDS_OVERHEAD_LIST"],
    parser.config_dict["HIDE_ROW_OPEN_OVERHEAD_FLAG_LIST"], 
    parser.config_dict["IGNORE_HOST_READ_INPUT_FLAG_FOR_PIM_EXEC_LIST"], 
    parser.config_dict["IGNORE_HOST_READ_OUTPUT_FLAG_FOR_PIM_EXEC_LIST"], 
    parser.config_dict["PIM_EXECUTION_MODE_LIST"], 
    parser.config_dict["IGNORE_HOST_READ_INPUT_FLAG_FOR_HOST_EXEC_LIST"], 
    parser.config_dict["IGNORE_HOST_WRITE_OUTPUT_FLAG_FOR_HOST_EXEC_LIST"], 
    parser.config_dict["ACTIVATIONS_PIM_COMMANDS_OVERHEAD_LIST"],
    parser.config_dict["GEMM_MATRIX_INPUT_SRC_LIST"],
    parser.config_dict["GEMM_VECTOR_INPUT_SRC_LIST"],
    parser.config_dict["GEMM_VECTOR_OUTPUT_DEST_LIST"],
    parser.config_dict["REGISTERS_REQUIRED_FOR_VECTOR_SCALAR_DATA_LIST"],
    parser.config_dict["REGISTERS_REQUIRED_FOR_VECTOR_SCALE_FACTOR_DATA_LIST"],
    parser.config_dict["REGISTERS_REQUIRED_FOR_MATRIX_SCALE_FACTOR_DATA_LIST"],
    parser.config_dict["IGNORE_OUTPUT_REGISTER_PRESSURE_FLAG_LIST"],
    parser.config_dict["IGNORE_INPUT_REGISTER_PRESSURE_FLAG_LIST"],
    parser.config_dict["PIM_TILE_SHAPE_DEGREE_LIST"],
    parser.config_dict["PIM_TILE_ORDER_DEGREE_LIST"],
    parser.config_dict["SPLIT_K_DEGREE_LIST"],
    parser.config_dict["SHIFT_LANE_MODE_LIST"],
    parser.config_dict["ASSUME_FULL_REGISTER_BEFORE_WRITE_TO_MEM_LIST"],
    parser.config_dict["HOST_INDUCED_TURNAROUND_OVERHEAD_LIST"],
    parser.config_dict["ACCUMULATION_REGISTER_LOAD_PIM_COMMANDS_OVERHEAD_LIST"],
    parser.config_dict["ACCUMULATION_REGISTER_SPILL_PIM_COMMANDS_OVERHEAD_LIST"],
    parser.config_dict["MEM_SPILL_SIZE_LIST"],
    parser.config_dict["MAC_UNIT_OUTPUT_SIZE_LIST"],
    parser.config_dict["ASSUME_MATRIX_SCALE_FACTORS_SMART_PACKING_LIST"],
    parser.config_dict["OUTPUT_COMPACT_EXTRA_PIM_COMMANDS_OVERHEAD_LIST"],
    parser.config_dict["ASSUME_FREE_CROSS_SIMD_REDUCTION_LIST"],
    parser.config_dict["ASSUME_SINGLE_INPUT_REGISTER_LIST"]
)
COMBINATIONS = list(COMBINATIONS)   # Convert to list to iterate over multiple times. 

