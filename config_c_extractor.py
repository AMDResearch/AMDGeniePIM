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

from enums import combination_list_index, blocked_format_inputs, host_param_index, peak_util_index, dram_param_index, host_vector_reg_count_type, data_src_dest, pim_mode_index, lane_shift_mode

class run_core_config_params:
    def __init__(self):
        # Input & output data types
        self.block_size = 1  # Block size of the blocked format
        self.process_scale_factors_at_host_flag = False  # Flag indicating processing scaling of blocked formats at host
        self.matrix_blocked_flag = False # Matrix MxK is not blocked (non-blocked data format) 
        self.vector_blocked_flag = False    # Vector KxN is not blocked 
        self.operand_size = 16   # Operand size of matrix (bits)
        self.vector_operand_size = 16    # Operand size of vector (bits)
        self.scale_factor_operand_size = 16  # Operand size of the matrix's scale factors (bits)
        self.vector_scale_factor_operand_size = 16 # Operand size of the vector's scale factors (bits)
        self.accum_operand_size = 32  # Accumulation size (bits)  
        self.compute_operand_size = max(self.operand_size, self.vector_operand_size)   # PIM compute operand size

        # Host compute params
        self.host_peak_compute_throughput = 32   # Peak compute throughput (TOPS)
        self.host_compute_eff = 1    # Compute efficiency (1 = 100%)

        # Host memory bandwidth params for the matrix source
        self.matrix_src = "MEM"
        self.matrix_host_peak_mem_bw = 120   # Peak memory bandwidth (GB/sec) - Default: https://www.amd.com/en/product/13041
        self.matrix_host_mem_bw_util = 1 # Memory bandwidth utilization
        self.matrix_host_mem_bw = self.matrix_host_peak_mem_bw * self.matrix_host_mem_bw_util  # Available memory bandwidth

        # Host memory bandwidth params for the input vector source 
        self.host_ignore_read_input_flag = True  # Host read input data flag 
        self.ip_vec_src = "MEM"
        self.ip_vec_host_peak_mem_bw = 120   # Peak memory bandwidth (GB/sec) - Default: https://www.amd.com/en/product/13041
        self.ip_vec_host_mem_bw_util = 1 # Memory bandwidth utilization
        self.ip_vec_host_mem_bw = self.ip_vec_host_peak_mem_bw * self.ip_vec_host_mem_bw_util  # Available memory bandwidth

        # Host memory bandwidth params for the output vector source      
        self.host_ignore_write_output_flag = True    # Host read output data flag 
        self.op_vec_dest = "MEM"
        self.op_vec_host_peak_mem_bw = 120   # Peak memory bandwidth (GB/sec) - Default: https://www.amd.com/en/product/13041
        self.op_vec_host_mem_bw_util = 1 # Memory bandwidth utilization
        self.op_vec_host_mem_bw = self.op_vec_host_peak_mem_bw * self.op_vec_host_mem_bw_util  # Available memory bandwidth
        
        # Host memory bandwidth params for PIM-induced operations
        self.mem_src = "MEM" # This must be MEM as PIM generates GEMV output in memory
        self.host_peak_mem_bw = 120  # Peak memory bandwidth (GB/sec) - Default: https://www.amd.com/en/product/13041
        self.host_mem_bw_util = 1    # Memory bandwidth utilization
        self.host_mem_bw = self.host_peak_mem_bw * self.host_mem_bw_util   # Available memory bandwidth

        # PIM architecture/execution params - Default LPDDR5X-7500
        self.pim_exec_mode = pim_mode_index.REAL.value   # PIM GEMV execution mode
        self.stack_count = 1         # Number of stacks
        self.channels_per_stack = 8  # Number of pseudo channels per stack
        self.banks_per_channel = 16  # Number of banks per pseudo channel
        self.simd_width = 256        # DRAM word width (bits)
        self.mac_compute_rate = 1    # MAC execution rate in PIM unit
        self.dram_row_size = 2048    # Row buffer size (bytes)
        self.orf_reg_per_pim_alu = 8 # Number of ORFs in PIM ALU
        self.irf_reg_per_pim_alu = 8 # Number of IRFs in PIM ALU
        self.banks_per_pim_unit = 1  # Number of banks sharing a PIM unit 
        self.alus_per_pim_unit = 1   # Number of PIM ALU per PIM unit 
        self.orf_reg_size = 256 # Size of registers in ORF (bits)
        self.irf_reg_size = 256 # Size of registers in IRF (bits)
        self.memory_interleaving_granularity_size = 256  # Interleaving granularity (bytes) 
        self.reg_spill_mem_size = 0
        self.mac_unit_output_size = 0
        self.pim_host_induced_turnaround_overhead = 0    # Extra turnaround time induced by host
        self.shift_lane_mode = lane_shift_mode.SINGLE_LANE_SHIFT_MODE.value  # Lane shift modes for cross-SIMD reduction
        self.pim_hide_row_open_overhead_flag = False # Ignore row overhead flag for PIM
        self.pim_ignore_host_vector_write_overhead_flag = False  # Ignore bus turnaround overhead when writing input vector self.data from host to PIM 
        self.pim_assume_full_reg_before_write_to_mem_flag = False # Flag to enable writing full register to memory
        self.pim_host_ignore_read_input_flag = False     # Ignore time for host to read input vector during PIM execution
        self.pim_host_ignore_read_output_flag = False    # Ignore time for host to read output vector during PIM execution
        self.pim_matrix_scale_factors_smart_pack_flag = True # Assume smart packing of matrix elements and scale factors
        self.pim_free_cross_simd_reduction_flag = False # Assume free cross-SIMD reduction (free pim-SHIFTs/ADDs)
        self.pim_single_input_reg_flag = False  # Assume a minimum of a single input register that will be overwritten. This is from a register requirement perspective and the user can allocate more register to input to reduce turnaround time.

        # Compute number of registers per bank
        self.orf_reg_per_bank = self.orf_reg_per_pim_alu / self.banks_per_pim_unit
        self.irf_reg_per_bank = self.irf_reg_per_pim_alu / self.banks_per_pim_unit

        # DRAM params - Default: https://www.jedec.org/standards-documents/docs/jesd209-5c
        self.dram_t_rp = 21             # ns
        self.dram_t_rcdrd = 18          # ns
        self.dram_t_ccdl = 4.266667     # ns
        self.dram_t_ras = 42            # ns
        self.dram_t_rtw = 18.13333333   # ns
        self.dram_t_wtr = 12            # ns

        # PIM registers multiplier & flags
        self.reg_mult_required_for_blocked_format = 1  # Register multiplier required for blocked data formats
        self.pim_ignore_output_reg_pressure_flag = True # Flag indicating unlimited output register in PIM 
        self.pim_ignore_input_reg_pressure_flag = True  # Flag indicating unlimited input register in PIM
        self.assume_optimized_scale_factors_into_reg_flag = True    # Optimized loading of matrix's scale factors into the registers

        # Extract information related to register use for vector's scalar data
        self.vector_scalar_data_reg_count = 2
        self.vector_scalar_data_reg_type = data_src_dest.IRF.value

        # Extract information related to register use for vector's scale factors
        self.vector_scale_factor_data_reg_count = 2
        self.vector_scale_factor_data_reg_type = data_src_dest.IRF.value

        # Extract information related to register use for matrix's scale factors
        self.matrix_scale_factor_data_reg_count = 4
        self.matrix_scale_factor_data_reg_type = data_src_dest.IRF.value

        # PIM commands overhead
        self.upcasting_pim_commands_overhead = 0     # Number of PIM commands to perform upcasting 
        self.activations_pim_commands_overhead = 0   # Number of PIM commands to perform activation after GEMV (e.g., ReLU)
        self.scale_factor_pim_commands_overhead = 0  # Number of non-MUL/MAC PIM commands to process scale factors 
        self.accum_reg_load_pim_commands_overhead = 0  # Number of PIM commands to load accum. reg.
        self.accum_reg_spill_reset_pim_commands_overhead = 0  # Number of PIM commands to spill/reset accum. reg.
        self.output_compact_pim_commands_overhead = 0  # Number of PIM commands to compact output before spilling to memory

        # PIM data placement params
        self.pim_tile_shape_degree = 0   # PIM tile shape degree - number of matrix rows in the PIM tile
        self.pim_tile_order_degree = 1   # PIM tile order degree - number of active row blocks (row block interleaving)
        self.pim_split_k_degree = 1      # Split-K degree - number of channel groups after the vertical splits

    # Extract the configuration of current run
    def extract_config_params(self, in_combination):
        blocked_inputs_flags = in_combination[combination_list_index.BLOCKED_INPUTS_FLAG.value]
        dram_paramters = in_combination[combination_list_index.DRAM_PARAMTERS.value]
        registers_required_for_vector_scalar_data = in_combination[combination_list_index.REGISTERS_REQUIRED_FOR_VECTOR_SCALAR_DATA.value]
        registers_required_for_vector_scale_factor_data = in_combination[combination_list_index.REGISTERS_REQUIRED_FOR_VECTOR_SCALE_FACTOR_DATA.value]
        registers_required_for_matrix_scale_factor_data = in_combination[combination_list_index.REGISTERS_REQUIRED_FOR_MATRIX_SCALE_FACTOR_DATA.value]

        self.block_size = int(in_combination[combination_list_index.BLOCK_SIZE.value])  
        self.process_scale_factors_at_host_flag = eval(in_combination[combination_list_index.PROCESS_SCALE_FACTORS_AT_HOST_FLAG.value])  
        self.matrix_blocked_flag = eval(blocked_inputs_flags[blocked_format_inputs.IN_WEIGHT.value]) 
        self.vector_blocked_flag = eval(blocked_inputs_flags[blocked_format_inputs.IN_ACT.value])
        self.operand_size = int(in_combination[combination_list_index.OPERAND_SIZE.value])
        self.vector_operand_size = int(in_combination[combination_list_index.ACT_OPERAND_SIZE.value])
        self.scale_factor_operand_size = int(in_combination[combination_list_index.SCALE_FACTOR_OPERAND_SIZE.value])
        self.vector_scale_factor_operand_size = int(in_combination[combination_list_index.ACT_SCALE_FACTOR_OPERAND_SIZE.value])
        self.accum_operand_size = int(in_combination[combination_list_index.ACCUMULATION_OPERAND_SIZE.value])  
        self.compute_operand_size = max(self.operand_size, self.vector_operand_size)

        # Host compute params
        self.host_peak_compute_throughput = float(in_combination[combination_list_index.HOST_PEAK_MEM_COMPUTE.value][host_param_index.HOST_PEAK_COMPUTE_THROUGHPUT_UTIL.value][str(self.compute_operand_size)][peak_util_index.PEAK.value])
        self.host_compute_eff = float(in_combination[combination_list_index.HOST_PEAK_MEM_COMPUTE.value][host_param_index.HOST_PEAK_COMPUTE_THROUGHPUT_UTIL.value][str(self.compute_operand_size)][peak_util_index.UTIL.value])

        # Host memory bandwidth params for the matrix source
        self.matrix_src = in_combination[combination_list_index.GEMM_MATRIX_INPUT_SRC.value]
        self.matrix_host_peak_mem_bw = float(in_combination[combination_list_index.HOST_PEAK_MEM_COMPUTE.value][host_param_index.HOST_PEAK_MEM_BW_UTIL.value][self.matrix_src][peak_util_index.PEAK.value])
        self.matrix_host_mem_bw_util = float(in_combination[combination_list_index.HOST_PEAK_MEM_COMPUTE.value][host_param_index.HOST_PEAK_MEM_BW_UTIL.value][self.matrix_src][peak_util_index.UTIL.value])
        self.matrix_host_mem_bw = self.matrix_host_peak_mem_bw * self.matrix_host_mem_bw_util

        # Host memory bandwidth params for the input vector source 
        self.host_ignore_read_input_flag = eval(in_combination[combination_list_index.IGNORE_HOST_READ_INPUT_FLAG_FOR_HOST_EXEC.value])
        self.ip_vec_src = in_combination[combination_list_index.GEMM_VECTOR_INPUT_SRC.value]
        self.ip_vec_host_peak_mem_bw = float(in_combination[combination_list_index.HOST_PEAK_MEM_COMPUTE.value][host_param_index.HOST_PEAK_MEM_BW_UTIL.value][self.ip_vec_src][peak_util_index.PEAK.value])
        self.ip_vec_host_mem_bw_util = float(in_combination[combination_list_index.HOST_PEAK_MEM_COMPUTE.value][host_param_index.HOST_PEAK_MEM_BW_UTIL.value][self.ip_vec_src][peak_util_index.UTIL.value])
        self.ip_vec_host_mem_bw = self.ip_vec_host_peak_mem_bw * self.ip_vec_host_mem_bw_util

        # Host memory bandwidth params for the output vector source      
        self.host_ignore_write_output_flag = eval(in_combination[combination_list_index.IGNORE_HOST_WRITE_OUTPUT_FLAG_FOR_HOST_EXEC.value])
        self.op_vec_dest = in_combination[combination_list_index.GEMM_VECTOR_OUTPUT_DEST.value]
        self.op_vec_host_peak_mem_bw = float(in_combination[combination_list_index.HOST_PEAK_MEM_COMPUTE.value][host_param_index.HOST_PEAK_MEM_BW_UTIL.value][self.op_vec_dest][peak_util_index.PEAK.value])
        self.op_vec_host_mem_bw_util = float(in_combination[combination_list_index.HOST_PEAK_MEM_COMPUTE.value][host_param_index.HOST_PEAK_MEM_BW_UTIL.value][self.op_vec_dest][peak_util_index.UTIL.value])
        self.op_vec_host_mem_bw = self.op_vec_host_peak_mem_bw * self.op_vec_host_mem_bw_util 
        
        # Host memory bandwidth params for PIM-induced operations
        self.mem_src = "MEM" # This must be MEM as PIM generates GEMV output in memory
        self.host_peak_mem_bw = float(in_combination[combination_list_index.HOST_PEAK_MEM_COMPUTE.value][host_param_index.HOST_PEAK_MEM_BW_UTIL.value][self.mem_src][peak_util_index.PEAK.value])
        self.host_mem_bw_util = float(in_combination[combination_list_index.HOST_PEAK_MEM_COMPUTE.value][host_param_index.HOST_PEAK_MEM_BW_UTIL.value][self.mem_src][peak_util_index.UTIL.value])
        self.host_mem_bw = self.host_peak_mem_bw * self.host_mem_bw_util

        # PIM architecture/execution params
        self.pim_exec_mode = in_combination[combination_list_index.PIM_EXECUTION_MODE.value]
        self.stack_count = int(in_combination[combination_list_index.STACK_COUNT.value])
        self.channels_per_stack = int(in_combination[combination_list_index.CHANNEL_PER_STACK.value])
        self.banks_per_channel = int(in_combination[combination_list_index.BANKS_PER_CHANNEL.value])
        self.simd_width = int(in_combination[combination_list_index.SIMD_WIDTH.value])
        self.mac_compute_rate = float(in_combination[combination_list_index.PIM_COMPUTE_RATE.value])
        self.dram_row_size = int(in_combination[combination_list_index.DRAM_ROW_SIZE.value])
        self.orf_reg_per_pim_alu = int(in_combination[combination_list_index.ORF_REGISTERS_PER_PIM_ALU.value])
        self.irf_reg_per_pim_alu = int(in_combination[combination_list_index.IRF_REGISTERS_PER_PIM_ALU.value])
        self.banks_per_pim_unit = int(in_combination[combination_list_index.BANKS_PER_PIM_UNIT.value])
        self.alus_per_pim_unit = int(in_combination[combination_list_index.PIM_ALU_PER_PIM_UNIT.value])
        self.orf_reg_size = int(in_combination[combination_list_index.PIM_ORF_REGISTER_SIZE.value])
        self.irf_reg_size = int(in_combination[combination_list_index.PIM_IRF_REGISTER_SIZE.value])
        self.memory_interleaving_granularity_size = int(in_combination[combination_list_index.MEMORY_INTERLEAVING_GRANULARITY_SIZE.value]) 
        self.reg_spill_mem_size = int(in_combination[combination_list_index.MEM_SPILL_SIZE.value])
        self.mac_unit_output_size = int(in_combination[combination_list_index.MAC_UNIT_OUTPUT_SIZE.value])
        self.pim_host_induced_turnaround_overhead = int(in_combination[combination_list_index.HOST_INDUCED_TURNAROUND_OVERHEAD.value])
        self.shift_lane_mode = int(in_combination[combination_list_index.SHIFT_LANE_MODE.value])
        self.pim_hide_row_open_overhead_flag = eval(in_combination[combination_list_index.HIDE_ROW_OPEN_OVERHEAD_FLAG.value])
        self.pim_ignore_host_vector_write_overhead_flag = eval(in_combination[combination_list_index.IGNORE_HOST_VECTOR_WRITE_OVERHEAD.value]) 
        self.pim_assume_full_reg_before_write_to_mem_flag = eval(in_combination[combination_list_index.ASSUME_FULL_REGISTER_BEFORE_WRITE_TO_MEM.value])
        self.pim_host_ignore_read_input_flag = eval(in_combination[combination_list_index.IGNORE_HOST_READ_INPUT_FLAG_FOR_PIM_EXEC.value])
        self.pim_host_ignore_read_output_flag = eval(in_combination[combination_list_index.IGNORE_HOST_READ_OUTPUT_FLAG_FOR_PIM_EXEC.value])
        self.pim_matrix_scale_factors_smart_pack_flag = eval(in_combination[combination_list_index.ASSUME_MATRIX_SCALE_FACTORS_SMART_PACKING.value])
        self.pim_free_cross_simd_reduction_flag = eval(in_combination[combination_list_index.ASSUME_FREE_CROSS_SIMD_REDUCTION.value])
        self.pim_single_input_reg_flag = eval(in_combination[combination_list_index.ASSUME_SINGLE_INPUT_REGISTER.value])
        
        # Compute number of registers per bank
        self.orf_reg_per_bank = self.orf_reg_per_pim_alu / self.banks_per_pim_unit
        self.irf_reg_per_bank = self.irf_reg_per_pim_alu / self.banks_per_pim_unit

        # DRAM params
        self.dram_t_rp = float(dram_paramters[dram_param_index.DRAM_T_RP.value])
        self.dram_t_rcdrd = float(dram_paramters[dram_param_index.DRAM_T_RCDRD.value])
        self.dram_t_ccdl = float(dram_paramters[dram_param_index.DRAM_T_CCDL.value])
        self.dram_t_ras = float(dram_paramters[dram_param_index.DRAM_T_RAS.value])
        self.dram_t_rtw = float(dram_paramters[dram_param_index.DRAM_T_RTW.value])
        self.dram_t_wtr = float(dram_paramters[dram_param_index.DRAM_T_WTR.value])

        # PIM registers multiplier & flags
        self.reg_mult_required_for_blocked_format = int(in_combination[combination_list_index.REGISTERS_MULT_REQUIRED_FOR_ACCUMULATION.value])
        self.pim_ignore_output_reg_pressure_flag = eval(in_combination[combination_list_index.IGNORE_OUTPUT_REGISTER_PRESSURE_FLAG.value]) 
        self.pim_ignore_input_reg_pressure_flag = eval(in_combination[combination_list_index.IGNORE_INPUT_REGISTER_PRESSURE_FLAG.value])
        self.assume_optimized_scale_factors_into_reg_flag = eval(in_combination[combination_list_index.ASSUME_OPTIMIZED_SCALE_FACTORS_INTO_REGISTER_FLAG.value])

        # Extract information related to register use for vector's scalar data
        self.vector_scalar_data_reg_count = int(registers_required_for_vector_scalar_data[host_vector_reg_count_type.REG_COUNT.value])
        self.vector_scalar_data_reg_type = int(registers_required_for_vector_scalar_data[host_vector_reg_count_type.REG_TYPE.value])

        if self.vector_scalar_data_reg_type == data_src_dest.PART_OF_PIM_COMMAND.value:
            self.vector_scalar_data_reg_count = 0  # Set to ZERO as there is no need to store the vector data from host

        # Extract information related to register use for vector's scale factors
        self.vector_scale_factor_data_reg_count = int(registers_required_for_vector_scale_factor_data[host_vector_reg_count_type.REG_COUNT.value])
        self.vector_scale_factor_data_reg_type = int(registers_required_for_vector_scale_factor_data[host_vector_reg_count_type.REG_TYPE.value])

        if self.vector_scale_factor_data_reg_type == data_src_dest.PART_OF_PIM_COMMAND.value:
            self.vector_scale_factor_data_reg_count = 0  # Set to ZERO as there is no need to store the vector data from host
        
        # Extract information related to register use for matrix's scale factors
        self.matrix_scale_factor_data_reg_count = int(registers_required_for_matrix_scale_factor_data[host_vector_reg_count_type.REG_COUNT.value])
        self.matrix_scale_factor_data_reg_type = int(registers_required_for_matrix_scale_factor_data[host_vector_reg_count_type.REG_TYPE.value])

        # PIM commands overhead
        self.upcasting_pim_commands_overhead = int(in_combination[combination_list_index.UPCASTING_PIM_COMMANDS_OVERHEAD.value])
        self.activations_pim_commands_overhead = int(in_combination[combination_list_index.ACTIVATIONS_PIM_COMMANDS_OVERHEAD.value])
        self.scale_factor_pim_commands_overhead = int(in_combination[combination_list_index.SCALE_FACTOR_EXTRA_PIM_COMMANDS_OVERHEAD.value])
        self.accum_reg_load_pim_commands_overhead = int(in_combination[combination_list_index.ACCUMULATION_REGISTER_LOAD_PIM_COMMANDS_OVERHEAD.value])
        self.accum_reg_spill_reset_pim_commands_overhead = int(in_combination[combination_list_index.ACCUMULATION_REGISTER_SPILL_RESET_PIM_COMMANDS_OVERHEAD.value])
        self.output_compact_pim_commands_overhead = int(in_combination[combination_list_index.OUTPUT_COMPACT_EXTRA_PIM_COMMANDS_OVERHEAD.value])

        # PIM data placement params
        self.pim_tile_shape_degree = int(in_combination[combination_list_index.PIM_TILE_SHAPE_DEGREE.value])
        self.pim_tile_order_degree = int(in_combination[combination_list_index.PIM_TILE_ORDER_DEGREE.value])
        self.pim_split_k_degree = int(in_combination[combination_list_index.SPLIT_K_DEGREE.value])

        # Asserts 
        assert(self.block_size >= 1)
        assert(self.simd_width % self.operand_size == 0)
        assert((self.dram_row_size * 8) % self.operand_size == 0)
        assert((self.dram_row_size * 8) % self.simd_width == 0)
        assert(self.orf_reg_size % self.operand_size == 0) # Will not work with 256b register and 6b inputs
        assert(self.irf_reg_size % self.operand_size == 0)
        assert(self.orf_reg_per_pim_alu % self.banks_per_pim_unit == 0)
        assert(self.irf_reg_per_pim_alu % self.banks_per_pim_unit == 0)
        assert((self.memory_interleaving_granularity_size * 8) >= self.operand_size)
        assert((self.memory_interleaving_granularity_size * 8) % self.operand_size == 0)
        assert(self.reg_spill_mem_size >= 0)   # not negative
        assert(self.mac_unit_output_size >= 0)   # not negative
        assert(self.vector_scalar_data_reg_count > 0)   # not negative
        assert(self.vector_scale_factor_data_reg_count > 0) # not negative
        assert(self.matrix_scale_factor_data_reg_count > 0) # not negative
        assert(self.upcasting_pim_commands_overhead >= 0)
        assert(self.activations_pim_commands_overhead >= 0)
        assert(self.scale_factor_pim_commands_overhead >= 0)
        assert(self.accum_reg_load_pim_commands_overhead >= 0)
        assert(self.accum_reg_spill_reset_pim_commands_overhead >= 0)
        assert(self.output_compact_pim_commands_overhead >= 0)
        assert((self.pim_split_k_degree == 1) or (self.pim_split_k_degree % 2 == 0))
        assert(self.pim_split_k_degree >= 1)
        assert(self.pim_split_k_degree <= self.channels_per_stack)  # Number of split cannot exceed the number of channels

        assert(self.accum_operand_size >= max(self.operand_size, self.vector_operand_size))   # Assume accumulation precision is equal to or bigger compared to the input/output precisions.

        # Ensure that the host is sending the vector data (both scalar and scale factors) the same way.
        # In other words, if the host sends the scalar data as part of the PIM command, then same applies to scale factor data.
        # If the host writes data scalar data into registers (does not matter IRF or ORF), then scale factors are also written into registers (does not have to be the same register type as scalar data).    
        if self.vector_scalar_data_reg_type == data_src_dest.PART_OF_PIM_COMMAND.value:
            assert(self.vector_scale_factor_data_reg_type == data_src_dest.PART_OF_PIM_COMMAND.value)

        # Ensure that host does not send the scale factors of the matrix.
        assert self.matrix_scale_factor_data_reg_type != data_src_dest.PART_OF_PIM_COMMAND.value, "Host does not send the scale factors of the matrix to PIM. PIM reads the scale factors of the matrix from DRAM rows and, if possible, load it into ORF or IRF registers to be used."