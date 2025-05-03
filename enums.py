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

from enum import Enum, auto

# Host memory and compute parameters
class host_param_index(Enum):
    HOST_ID = 0
    HOST_PEAK_MEM_BW_UTIL = auto()
    HOST_PEAK_COMPUTE_THROUGHPUT_UTIL = auto()

# Memory bandwidth & compute dict index
class peak_util_index(Enum):
    PEAK = 0
    UTIL = auto()

# DRAM 
class dram_param_index(Enum):
    DRAM_MEM_ID = 0
    DRAM_T_RP = auto()
    DRAM_T_RCDRD = auto()
    DRAM_T_CCDL = auto()
    DRAM_T_RAS = auto()
    DRAM_T_RTW = auto()
    DRAM_T_WTR = auto()

# PIM mode
class pim_mode_index(Enum):
    OPTIM = "OPTIM"             # PIM = GPU in terms of the input matrix size
    REAL = "REAL"               # PIM >= GPU in terms of the input matrix size -> PIM input matrix size is, possibly, padded to align with banks count, etc. 

# Combinations list index
class combination_list_index(Enum):
    STACK_COUNT = 0
    CHANNEL_PER_STACK = auto() 
    BANKS_PER_CHANNEL = auto()
    DRAM_ROW_SIZE = auto()
    BANKS_PER_PIM_UNIT = auto()
    PIM_ALU_PER_PIM_UNIT = auto()
    ORF_REGISTERS_PER_PIM_ALU = auto()
    IRF_REGISTERS_PER_PIM_ALU = auto()
    SIMD_WIDTH = auto()
    PIM_COMPUTE_RATE = auto()
    PIM_ORF_REGISTER_SIZE = auto()
    PIM_IRF_REGISTER_SIZE = auto()
    MEMORY_INTERLEAVING_GRANULARITY_SIZE = auto()
    IGNORE_HOST_VECTOR_WRITE_OVERHEAD = auto()
    HOST_PEAK_MEM_COMPUTE = auto()
    DRAM_PARAMTERS = auto()
    OPERAND_SIZE = auto()
    ACT_OPERAND_SIZE = auto()
    BLOCK_SIZE = auto() 
    PROCESS_SCALE_FACTORS_AT_HOST_FLAG = auto() 
    BLOCKED_INPUTS_FLAG = auto()
    SCALE_FACTOR_OPERAND_SIZE = auto() 
    ASSUME_OPTIMIZED_SCALE_FACTORS_INTO_REGISTER_FLAG = auto() 
    ACT_SCALE_FACTOR_OPERAND_SIZE = auto() 
    ACCUMULATION_OPERAND_SIZE = auto() 
    SCALE_FACTOR_EXTRA_PIM_COMMANDS_OVERHEAD = auto()
    REGISTERS_MULT_REQUIRED_FOR_ACCUMULATION = auto()
    UPCASTING_PIM_COMMANDS_OVERHEAD = auto()
    HIDE_ROW_OPEN_OVERHEAD_FLAG = auto()
    IGNORE_HOST_READ_INPUT_FLAG_FOR_PIM_EXEC = auto()
    IGNORE_HOST_READ_OUTPUT_FLAG_FOR_PIM_EXEC = auto()
    PIM_EXECUTION_MODE = auto()
    IGNORE_HOST_READ_INPUT_FLAG_FOR_HOST_EXEC = auto()
    IGNORE_HOST_WRITE_OUTPUT_FLAG_FOR_HOST_EXEC = auto()
    ACTIVATIONS_PIM_COMMANDS_OVERHEAD = auto()
    GEMM_MATRIX_INPUT_SRC = auto()
    GEMM_VECTOR_INPUT_SRC = auto()
    GEMM_VECTOR_OUTPUT_DEST = auto()
    REGISTERS_REQUIRED_FOR_VECTOR_SCALAR_DATA = auto()
    REGISTERS_REQUIRED_FOR_VECTOR_SCALE_FACTOR_DATA = auto()
    REGISTERS_REQUIRED_FOR_MATRIX_SCALE_FACTOR_DATA = auto()
    IGNORE_OUTPUT_REGISTER_PRESSURE_FLAG = auto()
    IGNORE_INPUT_REGISTER_PRESSURE_FLAG = auto()
    PIM_TILE_SHAPE_DEGREE = auto()
    PIM_TILE_ORDER_DEGREE = auto()
    SPLIT_K_DEGREE = auto()
    SHIFT_LANE_MODE = auto()
    ASSUME_FULL_REGISTER_BEFORE_WRITE_TO_MEM = auto()
    HOST_INDUCED_TURNAROUND_OVERHEAD = auto()
    ACCUMULATION_REGISTER_LOAD_PIM_COMMANDS_OVERHEAD = auto()
    ACCUMULATION_REGISTER_SPILL_RESET_PIM_COMMANDS_OVERHEAD = auto()
    MEM_SPILL_SIZE = auto()
    MAC_UNIT_OUTPUT_SIZE = auto()
    ASSUME_MATRIX_SCALE_FACTORS_SMART_PACKING = auto()
    OUTPUT_COMPACT_EXTRA_PIM_COMMANDS_OVERHEAD = auto()
    ASSUME_FREE_CROSS_SIMD_REDUCTION = auto()
    ASSUME_SINGLE_INPUT_REGISTER = auto()

# Output format index 
class output_format_index_enum(Enum):
    OUT_ALL = 0
    OUT_CONDENSED = auto()
    OUT_CUSTOM = auto()

# GEMV source files 
class gemm_gen_mode_enum(Enum):
    GEN_GEMM_IN = "gemm"
    GEN_LLM_IN = "models"
    GEN_CUSTOM_IN = "custom"

# GEMV list index (also gemm.in header index)
class gemm_index(Enum):
    MODEL_ID = 0          
    GEMM_ID = auto()  
    GEMM_M = auto()
    GEMM_K = auto()
    BGEMM_BATCH_SIZE = auto()  
    GEMM_N_LIST = auto()   

# LLM input file (models.in) header index
class llm_input_index(Enum):
    LLM_ID = 0
    LLM_H = auto() 
    LLM_I = auto() 
    LLM_A = auto() 
    LLM_SL = auto() 
    LLM_B = auto() 
    LLM_PROMPT_SIZE = auto() 
    LLM_T = auto() 

# Registers required for host vector data and their type
class host_vector_reg_count_type(Enum):
    REG_COUNT = 0
    REG_TYPE = auto()  

# Host writing vector data (then the register type) or host sending the vector data as part of PIM command.
class data_src_dest(Enum):
    PART_OF_PIM_COMMAND = -1    # Host is sending vector data as part of PIM command
    ORF = 0                     # Host is writing the vector data into one (or more) registers in ORF
    IRF = 1                     # Host is writing the vector data into one (or more) registers in IRF
    BANK = 2
    HOST = 3
    IGNORE = 4
    AR = 5

# Special modes for number of matrix rows for PIM tile.
class special_pim_tile_shape_mode(Enum):
    LANE_COUNT_MODE = 0         # AMD GeniePIM assumes a PIM tile in which the number of rows per tile is the number of SIMD lanes (depends on operand_size)
    AUTO_NO_PADDING_MODE = -1   # AMD GeniePIM automatically picks the number of rows per tile so that the need for padding is eliminated.

# Special modes for split-K.
class special_split_k_mode(Enum):
    AUTO_NO_PADDING_MODE = -1   # AMD GeniePIM automatically picks the split-K degree so that the need for padding is eliminated.

# Special modes for tile order.
class special_pim_tile_order_mode(Enum):
    CRO_MAX = 0                 # AMD GeniePIM assumes CRO-MAX for the PIM tile order 
    CRO_AUTO = -1               # AMD GeniePIM automatically picks the tile order degree so that uses the remaining available registers.

# Lane-shift modes for cross-SIMD reductions.
class lane_shift_mode(Enum):
    SINGLE_LANE_SHIFT_MODE = 0
    MIN_LANE_SHIFT_MODE = auto()

# Blocked formats for GEMV inputs
class blocked_format_inputs(Enum):
    IN_WEIGHT = 0
    IN_ACT = auto()

# Special values for the scale factors register count 
class special_scale_factor_reg_count(Enum):
    USE_FREE = -1