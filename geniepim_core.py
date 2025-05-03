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

import math

from config_c_extractor import run_core_config_params
from gemm_extractor import gemm_params
from geniepim_output import geniepim_output
from options_parser import ARGS
from enums import data_src_dest, special_pim_tile_shape_mode, pim_mode_index, special_pim_tile_order_mode, special_scale_factor_reg_count, lane_shift_mode, special_split_k_mode

DEBUG = ARGS.debug

class geniepim_core:
    def __init__(self, in_gemm, in_config):
        # Extract run self.config params
        self.config = run_core_config_params()
        self.config.extract_config_params(in_config)

        # Extract GEMV info for host
        self.host_gemm = gemm_params()
        self.host_gemm.extract_gemm_params(in_gemm)

        # Extract GEMV info for PIM
        self.pim_gemm = gemm_params()
        self.pim_gemm.extract_gemm_params(in_gemm)

        # Output structure 
        self.output = geniepim_output()

        # Track the number of remaining registers (IRF & ORF)
        self.free_orf_reg_per_bank = self.config.orf_reg_per_bank
        self.free_irf_reg_per_bank = self.config.irf_reg_per_bank 
        
        # Track number of registers used for input & output
        self.pim_output_req_reg = 0  
        self.pim_vector_scalar_req_reg = 0    
        self.pim_vector_scale_factor_req_reg = 0
        self.pim_matrix_scale_factor_req_reg = 0
        self.pim_orf_req_reg = 0
        self.pim_irf_req_reg = 0

        # PIM tile shape & order 
        self.pim_tile_shape_m_dim = self.config.pim_tile_shape_degree
        self.pim_tile_shape_k_dim = 0
        self.pim_tile_order = 1

    # Compute multiply count for blocked formats
    def compute_blocked_mul_count(self):
        # Check if blocked format 
        mul_count = 0
        if self.config.matrix_blocked_flag:
            mul_count += 1
        if self.config.vector_blocked_flag:
            mul_count += 1

        return mul_count
    
    # Get register size per input type
    def get_reg_size(self, reg_type):
        reg_size = self.config.orf_reg_size
        if reg_type == data_src_dest.IRF.value:
            reg_size = self.config.irf_reg_size
        
        return reg_size

    # Track number of registers required 
    def track_and_update_reg(self, req_reg_type, req_reg_count, ignore_reg_pressure_flag, test_flag=False, tmp_free_orf=0, tmp_free_irf=0):
        if req_reg_type != data_src_dest.PART_OF_PIM_COMMAND.value:
            # Check if required registers exceed the available registers
            if req_reg_type == data_src_dest.ORF.value:
                if ignore_reg_pressure_flag == False:
                    assert req_reg_count <= self.free_orf_reg_per_bank, "Not enough ORF registers (remaining = {}, required = {}).".format(self.free_orf_reg_per_bank, req_reg_count)
                    
                    # Update the available ORF registers
                    if test_flag == False:
                        self.free_orf_reg_per_bank -= req_reg_count
                    else:
                        tmp_free_orf -= req_reg_count
                
                if test_flag == False: self.pim_orf_req_reg += req_reg_count
            
            elif req_reg_type == data_src_dest.IRF.value:
                if ignore_reg_pressure_flag == False:
                    assert req_reg_count <= self.free_irf_reg_per_bank, "Not enough IRF registers (remaining = {}, required = {}).".format(self.free_irf_reg_per_bank, req_reg_count)
                    
                    # Update the available IRF registers
                    if test_flag == False:
                        self.free_irf_reg_per_bank -= req_reg_count
                    else:
                        tmp_free_irf -= req_reg_count
                
                if test_flag == False: self.pim_irf_req_reg += req_reg_count
        
        return tmp_free_orf, tmp_free_irf
    
    # Compute PIM tile shape 
    # This function handles the bank locality test.
    def estimate_pim_tile_shape(self, total_banks_count, lanes_per_simd, matrix_elements_per_interleaving_block, elements_per_accum_reg, pim_split_k_degree):
        self.pim_gemm.gemm_m = self.host_gemm.gemm_m
        keep_searching = True
        pim_tile_factor = 1
        while(keep_searching):
            keep_searching = False  # Stop searching by default 

            # Check if LANE_COUNT_MODE is used.
            if self.config.pim_tile_shape_degree == special_pim_tile_shape_mode.LANE_COUNT_MODE.value:
                # Assume a row count in PIM tile that is equal to SIMD lane count. In other words, number of matrix rows in a row block (computed next) is equal to number of lanes in PIM SIMD.
                self.pim_tile_shape_m_dim = lanes_per_simd  # m = L
            
            # Check if AUTO_NO_PADDING_MODE is used.
            elif self.config.pim_tile_shape_degree == special_pim_tile_shape_mode.AUTO_NO_PADDING_MODE.value:
                # Do not use optimistic PIM execution mode with auto pick of tile shape. 
                assert(self.config.pim_exec_mode != pim_mode_index.OPTIM.value)

                # Automatically figure out the row count in PIM tile ('m') that result in zero padding.
                assert(matrix_elements_per_interleaving_block % pim_tile_factor == 0)
                # Check if a fixed accumulation register size is assumed
                if self.config.mac_unit_output_size > 0:
                    # Fixed, then start from the maximum number of elements in accumulation register.
                    self.pim_tile_shape_m_dim = math.ceil(elements_per_accum_reg / pim_tile_factor)
                else:
                    # No limit, then start from the number of elements per interleaving granularity. 
                    self.pim_tile_shape_m_dim = math.ceil(matrix_elements_per_interleaving_block / pim_tile_factor)
                
                pim_tile_factor *= 2
            
            # self.pim_tile_shape_m_dim must be >= 1 and power-of-two.
            assert(self.pim_tile_shape_m_dim >= 1)
            assert(not(self.pim_tile_shape_m_dim & (self.pim_tile_shape_m_dim - 1)))

            # Compute the row blocks count based on the row count in PIM tile ('m').
            # Multiply by the split-K degree to effectively increase the M dimension of the GEMV. 
            # The idea of split-K is to avail more row blocks to distribute across banks. 
            # X = M / m -> Row block = block of matrix rows, in which number of matrix rows in a block is equal to self.pim_tile_shape_m_dim.
            total_row_blocks = (pim_split_k_degree * self.host_gemm.gemm_m) / self.pim_tile_shape_m_dim  

            # Compute number of row blocks per bank (across all channels).
            row_blocks_per_bank = total_row_blocks / total_banks_count  # Y = X / B

            # Compute updated GEMV size (M dimension) for PIM execution 
            if self.config.pim_exec_mode == pim_mode_index.REAL.value:
                # Compute number of full row blocks per bank (no partial row block)
                full_row_blocks_per_bank = math.ceil(row_blocks_per_bank)   # Q = CEIL(Y)
                padding_mult = full_row_blocks_per_bank - row_blocks_per_bank
                self.pim_gemm.gemm_m = self.host_gemm.gemm_m + (padding_mult * self.pim_tile_shape_m_dim * total_banks_count)
                
                # Compute padding overhead (PIM extra work) 
                padding_overhead = self.pim_gemm.gemm_m / self.host_gemm.gemm_m

                # Check if auto pick of PIM tile shape is set
                if self.config.pim_tile_shape_degree == special_pim_tile_shape_mode.AUTO_NO_PADDING_MODE.value:
                    # Check if padding overhead is eliminated or if a self.pim_tile_shape_m_dim of 1 is already reached.
                    if (padding_overhead != 1) and (self.pim_tile_shape_m_dim != 1):
                        # Continue searching
                        keep_searching = True
    
    # Refine PIM tile shape 
    # This function handles the register pressure test for PIM tile shape in our placement methodology. 
    def refine_pim_tile_shape(self, lanes_per_simd):
        keep_refining = True
        while(keep_refining):
            keep_refining = False  # Stop refining by default

            # Update number of output registers required based on PIM tile shape
            # Output must be in ORF
            # Check if free cross-SIMD reduction is assumed. 
            if self.config.pim_free_cross_simd_reduction_flag == True:
                num_output_reg_per_row_blk = math.ceil((self.pim_tile_shape_m_dim * self.config.accum_operand_size) / self.config.orf_reg_size)
            else:
                num_output_reg_per_row_blk = math.ceil((lanes_per_simd * self.config.accum_operand_size) / self.config.orf_reg_size)
            
            # Check if blocked format is used
            if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == False):  
                # More output registers are required for local and global accumulation
                num_output_reg_per_row_blk *= self.config.reg_mult_required_for_blocked_format

            # Check if unlimited output registers is assumed
            if self.config.pim_ignore_output_reg_pressure_flag == False:
                # Assume available ORF registers count

                # If PIM tile shape mode is AUTO_NO_PADDING_MODE, then exit if no available registers
                if (self.config.pim_tile_shape_degree == special_pim_tile_shape_mode.AUTO_NO_PADDING_MODE.value) and (self.pim_tile_shape_m_dim != 1):
                    # Check if remaining registers are enough
                    if num_output_reg_per_row_blk > self.free_orf_reg_per_bank:
                        # Not enough
                        self.pim_tile_shape_m_dim /= 2

                        # Continue refining
                        keep_refining = True
                
                # Check registers and decrement 
                if keep_refining == False:
                    assert num_output_reg_per_row_blk <= self.free_orf_reg_per_bank, "Not enough ORF registers (remaining = {}, required = {}).".format(self.free_orf_reg_per_bank, num_output_reg_per_row_blk)

                    # Update the available ORF registers
                    self.free_orf_reg_per_bank -= num_output_reg_per_row_blk

        # Track how many output registers are needed.
        self.pim_output_req_reg += num_output_reg_per_row_blk
        self.pim_orf_req_reg += num_output_reg_per_row_blk

        return num_output_reg_per_row_blk
    
    # Compute PIM tile order 
    # This function handles the register pressure test for PIM tile order in our placement methodology. 
    def estime_pim_tile_order(self, num_row_blocks_per_bank, num_output_reg_per_row_blk, num_scalar_input_reg_per_row_blk, num_scale_factor_input_reg_per_row_blk):
        keep_searching = True
        current_decrement = 0
        while(keep_searching):
            keep_searching = False  # Stop searching by default
            
            self.pim_tile_order = self.config.pim_tile_order_degree  
            
            # Check if AUTO_NO_PADDING_MODE is used.
            if self.config.pim_tile_order_degree == special_pim_tile_order_mode.CRO_MAX.value:
                # PIM tile order = CRO-MAX - process all row blocks per each input vector data (full reuse of input registers)
                self.pim_tile_order = num_row_blocks_per_bank 
            
            elif self.config.pim_tile_order_degree == special_pim_tile_order_mode.CRO_AUTO.value:
                assert(self.config.pim_ignore_output_reg_pressure_flag == False)    # Do not assume unlimited ORF registers in CRO_AUTO mode
                assert(self.config.pim_ignore_input_reg_pressure_flag == False)     # Do not assume unlimited input registers (IRF/ORF) in CRO_AUTO mode
                self.pim_tile_order = num_row_blocks_per_bank - current_decrement   # Start from CRO-MAX (the number of row blocks per bank)
                current_decrement += 1

            else:
                # Fixed CRO-degree 
                # Check if the user set CRO-degree is more than the number of row blocks (> CRO-MAX)
                # If yes, then set to CRO-MAX.
                if self.config.pim_tile_order_degree > num_row_blocks_per_bank:
                    self.pim_tile_order = num_row_blocks_per_bank

            # Update number of output registers required based on PIM tile order
            extra_out_reg = (self.pim_tile_order - 1) * num_output_reg_per_row_blk
            # If PIM tile order mode is AUTO_NO_PADDING_MODE, then exit if no available registers
            if (self.config.pim_tile_order_degree == special_pim_tile_order_mode.CRO_AUTO.value) and (self.pim_tile_order != 1):
                tmp_free_orf = self.free_orf_reg_per_bank
                tmp_free_irf = self.free_irf_reg_per_bank

                # Check if remaining registers are enough
                if extra_out_reg > tmp_free_orf:
                    # Not enough, keep searching
                    keep_searching = True
                else:
                    tmp_free_orf -= extra_out_reg
                
                # Enough registers for output, check for scalar input. 
                # Check if required registers exceed the available registers
                if keep_searching == False:
                    if self.config.vector_scalar_data_reg_type == data_src_dest.ORF.value:
                        # Check if remaining ORF registers are enough
                        # Check for (num_scalar_input_reg_per_row_blk - 1). A single input register is already checked. 
                        if (num_scalar_input_reg_per_row_blk - 1) > tmp_free_orf:
                            # Not enough, keep searching
                            keep_searching = True
                        else:
                            tmp_free_orf -= (num_scalar_input_reg_per_row_blk - 1)
                    
                    elif self.config.vector_scalar_data_reg_type == data_src_dest.IRF.value:
                        # Check if remaining IRF registers are enough
                        # Check for (num_scalar_input_reg_per_row_blk - 1). A single input register is already checked. 
                        if (num_scalar_input_reg_per_row_blk - 1) > tmp_free_irf:
                            # Not enough, keep searching
                            keep_searching = True
                        else:
                            tmp_free_irf -= (num_scalar_input_reg_per_row_blk - 1)

                # Enough registers for scalar inputs, check for scale factors input. 
                # Check if required registers exceed the available registers
                if keep_searching == False:
                    if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == False) and (self.config.vector_blocked_flag == True):
                        if self.config.vector_scale_factor_data_reg_type == data_src_dest.ORF.value:
                            # Check if remaining registers are enough
                            # Check for (num_scale_factor_input_reg_per_row_blk - 1). A single input register is already checked. 
                            if (num_scale_factor_input_reg_per_row_blk - 1) > tmp_free_orf:
                                # Not enough, keep searching
                                keep_searching = True
                        
                        elif self.config.vector_scalar_data_reg_type == data_src_dest.IRF.value:
                            # Check if remaining registers are enough
                            # Check for (num_scale_factor_input_reg_per_row_blk - 1). A single input register is already checked.
                            if (num_scale_factor_input_reg_per_row_blk - 1) > tmp_free_irf:
                                # Not enough, keep searching
                                keep_searching = True
            else:
                tmp_free_orf = self.free_orf_reg_per_bank
                tmp_free_irf = self.free_irf_reg_per_bank

                if self.config.pim_ignore_output_reg_pressure_flag == False:
                    assert extra_out_reg <= tmp_free_orf, "Not enough ORF registers (remaining = {}, required = {})".format(tmp_free_orf, extra_out_reg)
                    
                    tmp_free_orf -= extra_out_reg

                if self.pim_tile_order > 1:  # Do not check again for PIM tile order = 1 (CRO-1)
                    # Vector scalar data 
                    tmp_free_orf, tmp_free_irf = self.track_and_update_reg(self.config.vector_scalar_data_reg_type, (num_scalar_input_reg_per_row_blk - 1), self.config.pim_ignore_input_reg_pressure_flag, True, tmp_free_orf, tmp_free_irf)
                    
                    # Vector scale factors data
                    if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == False) and (self.config.vector_blocked_flag == True):
                        tmp_free_orf, tmp_free_irf = self.track_and_update_reg(self.config.vector_scale_factor_data_reg_type, (num_scale_factor_input_reg_per_row_blk - 1), self.config.pim_ignore_input_reg_pressure_flag, True, tmp_free_orf, tmp_free_irf)
                                            
            if keep_searching == False:  # Register requirement of the current PIM tile order is met 
                # Update the available ORF registers
                self.pim_output_req_reg += extra_out_reg
                self.pim_orf_req_reg += extra_out_reg
                self.free_orf_reg_per_bank -= extra_out_reg
                
                # Update the available ORF/IRF registers for vector scalar data
                self.pim_vector_scalar_req_reg += (num_scalar_input_reg_per_row_blk - 1)
                if self.config.vector_scalar_data_reg_type != data_src_dest.PART_OF_PIM_COMMAND.value:
                    if self.config.pim_ignore_input_reg_pressure_flag == False:
                        assert self.pim_vector_scalar_req_reg <= self.config.vector_scalar_data_reg_count, "Required registers for input is less than what the user set in the self.config file (required = {}, self.config = {}).".format(self.pim_vector_scalar_req_reg, self.config.vector_scalar_data_reg_count)
                
                # Check register availability for the scalar data from the host
                self.track_and_update_reg(self.config.vector_scalar_data_reg_type, (num_scalar_input_reg_per_row_blk - 1), self.config.pim_ignore_input_reg_pressure_flag)

                # Update the available ORF/IRF registers for vector scale factors data
                if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == False) and (self.config.vector_blocked_flag == True):
                    self.pim_vector_scale_factor_req_reg += (num_scale_factor_input_reg_per_row_blk - 1)
                    if self.config.vector_scale_factor_data_reg_type != data_src_dest.PART_OF_PIM_COMMAND.value:
                        if self.config.pim_ignore_input_reg_pressure_flag == False:
                            assert self.pim_vector_scale_factor_req_reg <= self.config.vector_scale_factor_data_reg_count, "Required registers for input is less than what the user set in the self.config file (required = {}, self.config = {}).".format(self.pim_vector_scale_factor_req_reg, self.config.vector_scale_factor_data_reg_count)
                        
                    # Check register availability for the scale factors from the host
                    self.track_and_update_reg(self.config.vector_scale_factor_data_reg_type, (num_scale_factor_input_reg_per_row_blk - 1), self.config.pim_ignore_input_reg_pressure_flag)

    # Estimate the host compute time & ops
    def estimate_host_compute_stats(self, matrix_number_of_blocks):
        host_compute_throughput = (self.config.host_peak_compute_throughput * 1000) * self.config.host_compute_eff   # Available compute throughput. Multiply by 1000 as compute throughput is in TOPS.
        host_gemm_ops_count = 2 * self.host_gemm.gemm_m * self.host_gemm.gemm_k * self.host_gemm.gemm_n  # GEMV ops

        mul_count = self.compute_blocked_mul_count()
        if self.config.block_size > 1:
            extra_ops_count = (matrix_number_of_blocks * mul_count) * self.host_gemm.gemm_n  # Extra multiplications per block (to multiply the scale factors) + multiply by number of vectors (N dimension) 
            host_gemm_ops_count += extra_ops_count

        # Estimate compute time
        host_gemm_compute_time_ns = host_gemm_ops_count / host_compute_throughput   
        
        # Multiply by BGEMM batch size
        host_gemm_ops_count *= self.host_gemm.gemm_bs
        host_gemm_compute_time_ns *= self.host_gemm.gemm_bs
        self.output.host_compute_ops = host_gemm_ops_count
        self.output.host_compute_time_ns = host_gemm_compute_time_ns

        if DEBUG: print("Host GEMV Compute Time (ns) = {:.2f} ({} Ops)".format(host_gemm_compute_time_ns, host_gemm_ops_count))

        return host_gemm_compute_time_ns

    # Estimate the host memory time & bytes
    def estimate_host_mem_stats(self, matrix_number_of_blocks):
        gemm_bytes = 0
        host_gemm_mem_time_ns = 0
        # 1 - Read matrix 
        tmp_bytes = self.host_gemm.gemm_m * self.host_gemm.gemm_k * (self.config.operand_size / 8)
        gemm_bytes += tmp_bytes

        # Check if blocked format to read scale factors of input matrix
        if (self.config.block_size > 1) and (self.config.matrix_blocked_flag == True):
            # Calcuate the number of bytes to read for the scale factors
            scale_factors_bytes = matrix_number_of_blocks * (self.config.scale_factor_operand_size / 8)
            tmp_bytes += scale_factors_bytes  # Accumulate to the matrix bytes (tmp_bytes)
            gemm_bytes += scale_factors_bytes # Accumulate to the total bytes (gemm_bytes)
        
        # Estimate time to read matrix 
        host_gemm_mem_time_ns += tmp_bytes / self.config.matrix_host_mem_bw # Estimate memory time

        # 2 - Read input vector 
        # Check if read second GEMV input for host
        if self.config.host_ignore_read_input_flag == False:
            tmp_bytes = (self.host_gemm.gemm_k * self.host_gemm.gemm_n) * (self.config.vector_operand_size / 8)
            gemm_bytes += tmp_bytes

            # Check if blocked format to read scale factors of input vector 
            input_vector_number_of_blocks = 0
            if (self.config.block_size > 1) and (self.config.vector_blocked_flag == True):
                # Calculate number of blocks in the input vector
                input_vector_number_of_blocks = math.ceil((self.host_gemm.gemm_k * self.host_gemm.gemm_n) / self.config.block_size)  # Ceil ~ pad to not have partial block
                # Calcuate the number of bytes to read for the scale factors of the input vector
                scale_factors_bytes = input_vector_number_of_blocks * (self.config.vector_scale_factor_operand_size / 8)
                tmp_bytes += scale_factors_bytes  # Accumulate to the input vector bytes (tmp_bytes)
                gemm_bytes += scale_factors_bytes # Accumulate to the total bytes (gemm_bytes)
            
            # Estimate time to read second GEMV input 
            host_gemm_mem_time_ns += tmp_bytes / self.config.ip_vec_host_mem_bw    # Estimate memory time

        # 3 - Write output vector
        # Check if write GEMV output for host
        # Assume output activations are using the same data format as input activations (same operand size and same block size).
        if self.config.host_ignore_write_output_flag == False:
            tmp_bytes = (self.host_gemm.gemm_m * self.host_gemm.gemm_n) * (self.config.vector_operand_size / 8)
            gemm_bytes += tmp_bytes

            # Check if blocked format to write scale factors of output vector 
            output_vector_number_of_blocks = 0
            if (self.config.block_size > 1) and (self.config.vector_blocked_flag == True):
                # Calculate number of blocks in the output vector
                output_vector_number_of_blocks = math.ceil((self.host_gemm.gemm_m * self.host_gemm.gemm_n) / self.config.block_size)  # Pad to not have partial block
                # Calcuate the number of bytes to read for the scale factors of the input vector
                scale_factors_bytes = output_vector_number_of_blocks * (self.config.vector_scale_factor_operand_size / 8)
                tmp_bytes += scale_factors_bytes  # Accumulate to the output vector bytes (tmp_bytes)
                gemm_bytes += scale_factors_bytes # Accumulate to the total bytes (gemm_bytes)

            # Estimate time to write GEMV output 
            host_gemm_mem_time_ns += tmp_bytes / self.config.op_vec_host_mem_bw    # Estimate memory time

        # Multiply by BGEMM batch size
        gemm_bytes *= self.host_gemm.gemm_bs   
        host_gemm_mem_time_ns *= self.host_gemm.gemm_bs
        self.output.host_memory_bytes = gemm_bytes
        self.output.host_memory_time_ns = host_gemm_mem_time_ns

        if DEBUG: print("Host GEMV Memory Time (ns) = {:.2f} ({} Bytes)".format(host_gemm_mem_time_ns, gemm_bytes))

        return host_gemm_mem_time_ns

    # Compute the host execution time, ops, and bytes
    def compute_host_exec_stats(self):
        matrix_number_of_blocks = 0
        if self.config.block_size > 1:
            # Calculate number of blocks in the matrix
            matrix_number_of_blocks = math.ceil((self.host_gemm.gemm_m * self.host_gemm.gemm_k) / self.config.block_size)  # Ceil ~ pad to not have partial block

        # Compute time 
        host_gemm_compute_time_ns = self.estimate_host_compute_stats(matrix_number_of_blocks)

        # Memory time 
        host_gemm_mem_time_ns = self.estimate_host_mem_stats(matrix_number_of_blocks)
        
        # Host time = max(compute, memory)
        host_gemm_time_ns = max(host_gemm_compute_time_ns, host_gemm_mem_time_ns)
        self.output.host_gemm_time_ns = host_gemm_time_ns
        
        if DEBUG: print("Host GEMV Time (ns) = {:.2f}".format(host_gemm_time_ns))

    # Compute the PIM execution time and bytes
    def compute_pim_exec_stats(self):
        # Number of SIMD lanes based on operand size.
        lanes_per_simd = int(self.config.simd_width / self.config.operand_size)

        # Number of matrix elements in the interleaving granularity.
        # memory_interleaving_granularity_size is in bytes, hence the multiplication by eight.
        matrix_elements_per_interleaving_block = math.ceil((self.config.memory_interleaving_granularity_size * 8) / self.config.operand_size)

        # Number of lanes (elements) in accumumation register 
        elements_per_accum_reg = math.ceil(self.config.mac_unit_output_size / self.config.accum_operand_size)
        
        # Total number of banks acorss channels and stacks.
        total_banks_count = self.config.stack_count * self.config.channels_per_stack * self.config.banks_per_channel
        
        # PIM accumulation precision multiplier 
        # Accumulate in higher precision and possibly store accumulated data in >1 register. Write more registers to memory at the end.
        pim_accum_precision_mult = math.ceil(self.config.accum_operand_size / self.config.compute_operand_size)
        if DEBUG: print("PIM - Accumulation Precision Multiplier = {}".format(pim_accum_precision_mult)) 

        # PIM compuation multiplier
        # Due to heterogeneous input precisions. The extra ALUs per PIM unit can be used to compensate the lost PIM compute throughput.
        compute_ratio = self.config.compute_operand_size / (self.config.operand_size * self.config.mac_compute_rate)
        pim_hetero_compute_mult = math.ceil(compute_ratio / self.config.alus_per_pim_unit)
        if DEBUG: print("PIM - Heterogeneous Compute Multiplier = {}".format(pim_hetero_compute_mult)) 

        # PIM batch multiplier
        # Due to the use of accumulation register in the MAC unit which accumulates in higher precision, either spill the higher precision accumlated result before reusing the weights or process a single batch at a time. Do the latter (as done in blocked format).
        # Compute the effective number of ALU "groups" that can process vectors concurrently. For example, if there are four ALUs per PIM unit and hetero fp4xfp8 execution (compute_ratio = 2), then two ALUs are used to handle the hetero execution, and the remaining two ALUs can process another hetero GEMV execution in parallel.
        pim_alu_hetero_groups = math.ceil(self.config.alus_per_pim_unit / compute_ratio)
        
        # Then, the extra ALUs per PIM unit can be used to process other vectors in a given batch concurrently.
        pim_batch_size_mult = math.ceil(self.pim_gemm.gemm_n / pim_alu_hetero_groups) # It will not help batch size of one. 
        if DEBUG: print("PIM - Batch Multiplier = {}".format(pim_batch_size_mult))

        # Concurrent vector multiplier 
        # To consider the number of concurrent vector processed (for writing vector data to registers or writing outputs to memory).
        pim_concurrent_vector_mult = math.ceil(self.pim_gemm.gemm_n / pim_batch_size_mult)
        if DEBUG: print("PIM - Concurrent Batch Multiplier = {}".format(pim_concurrent_vector_mult))

        # Spill to memory multiplier
        pim_spill_mem_mult = 1 
        if self.config.reg_spill_mem_size > 0:
            pim_spill_mem_mult = math.ceil(self.config.orf_reg_size / self.config.reg_spill_mem_size)

        # Fixed MAC output multiplier 
        pim_fixed_mac_output_mult = 1 
        if self.config.mac_unit_output_size > 0:
            pim_fixed_mac_output_mult = self.config.mac_unit_output_size / (lanes_per_simd * self.config.accum_operand_size)
        
        # Check register availability for the scalar data from the host
        self.pim_vector_scalar_req_reg += 1 # Initially assume a single register to be used to hold the vector's scalar data
        self.track_and_update_reg(self.config.vector_scalar_data_reg_type, self.pim_vector_scalar_req_reg, self.config.pim_ignore_input_reg_pressure_flag)
        
        # Check register availability for the scale factors from the host
        if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == False) and (self.config.vector_blocked_flag == True):
            self.pim_vector_scale_factor_req_reg += 1 # Initially assume a single register to be used to hold the vector's scale factor data
            self.track_and_update_reg(self.config.vector_scale_factor_data_reg_type, self.pim_vector_scale_factor_req_reg, self.config.pim_ignore_input_reg_pressure_flag)

        # Check register availability for the matrix's scale factors 
        if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == False) and (self.config.matrix_blocked_flag == True):
            self.pim_matrix_scale_factor_req_reg += 1 # At least a single register to be used to hold the matrix's scale factor data
            self.track_and_update_reg(self.config.matrix_scale_factor_data_reg_type, self.pim_matrix_scale_factor_req_reg, self.config.pim_ignore_input_reg_pressure_flag)

        # Compute the 'm' dimension of the PIM tile shape (for mapping purposes).
        self.estimate_pim_tile_shape(total_banks_count, lanes_per_simd, matrix_elements_per_interleaving_block, elements_per_accum_reg, self.config.pim_split_k_degree)

        # Refine PIM tile shape based on available registers
        num_output_reg_per_row_blk = self.refine_pim_tile_shape(lanes_per_simd)

        # Compute the PIM tile shape ('k' dimension) based on the interleaving granularity and operand size.
        assert(matrix_elements_per_interleaving_block % self.pim_tile_shape_m_dim == 0)
        self.pim_tile_shape_k_dim = math.ceil(matrix_elements_per_interleaving_block / self.pim_tile_shape_m_dim)
        pim_tile_shape = "{}x{}".format(self.pim_tile_shape_m_dim, self.pim_tile_shape_k_dim)
        self.output.pim_tile_shape = pim_tile_shape
        if DEBUG: print("PIM Tile Shape = {}".format(pim_tile_shape))

        if DEBUG: print("Host GEMV (M = {}, K = {}, N = {}, BS = {}) vs. PIM GEMV (M = {}, K = {}, N = {}, BS = {})".format(self.host_gemm.gemm_m, self.host_gemm.gemm_k, self.host_gemm.gemm_n, self.host_gemm.gemm_bs, self.pim_gemm.gemm_m, self.pim_gemm.gemm_k, self.pim_gemm.gemm_n, self.pim_gemm.gemm_bs))

        # Number of input registers required based on PIM tile shape
        # Check if minimum of a single input register is assumed. 
        num_scalar_input_reg_per_row_blk = 0
        if self.config.pim_single_input_reg_flag == False:
            if self.config.vector_scalar_data_reg_type != data_src_dest.PART_OF_PIM_COMMAND.value:
                reg_size = self.get_reg_size(self.config.vector_scalar_data_reg_type)
                # No need to multiply by pim_concurrent_vector_mult as each PIM ALU has its own set of registers.
                num_scalar_input_reg_per_row_blk = math.ceil(self.pim_tile_shape_k_dim / (int(reg_size / self.config.vector_operand_size))) 
            else:
                num_scalar_input_reg_per_row_blk = 1    # Minimum of a single input register 

        # Number of scale factors input registers required based on PIM tile shape
        num_scale_factor_input_reg_per_row_blk = 0
        if self.config.vector_scale_factor_data_reg_type != data_src_dest.PART_OF_PIM_COMMAND.value:
            reg_size = self.get_reg_size(self.config.vector_scale_factor_data_reg_type)
            # No need to multiply by pim_concurrent_vector_mult as each PIM ALU has its own set of registers.
            num_scale_factor_input_reg_per_row_blk = math.ceil((self.pim_tile_shape_k_dim / self.config.block_size) / (int(reg_size / self.config.vector_scale_factor_operand_size))) 
        
        # PIM tile multiplier based on the shape of the PIM tile.
        assert(max(lanes_per_simd, self.pim_tile_shape_m_dim) % min(lanes_per_simd, self.pim_tile_shape_m_dim) == 0)
        # Do not take the ceil as the pim_tile_mult is needed as a fraction in a subset of the computations. 
        pim_tile_mult = lanes_per_simd / self.pim_tile_shape_m_dim    
        if DEBUG: print("PIM - PIM Tile Multiplier = {}".format(pim_tile_mult))

        # Adjust pim_tile_mult in case MAC unit output is fixed
        pim_tile_mult *= pim_fixed_mac_output_mult

        num_row_blocks_per_bank = int(math.ceil((self.pim_gemm.gemm_m / self.pim_tile_shape_m_dim) / total_banks_count))
        if DEBUG: print("PIM - Number of Matrix Row Blocks per Bank = {}".format(num_row_blocks_per_bank))
        self.output.pim_num_row_blocks_per_bank = num_row_blocks_per_bank

        # Compute the PIM tile order (CRO-1 -> CRO-MAX). So far, CRO-1 is assumed, which has least output register pressure but worst input reuse.
        self.estime_pim_tile_order(num_row_blocks_per_bank, num_output_reg_per_row_blk, num_scalar_input_reg_per_row_blk, num_scale_factor_input_reg_per_row_blk)
        if DEBUG: print("PIM - PIM Tile Order - Config = {}, CRO-deg={}".format(self.config.pim_tile_order_degree, self.pim_tile_order))
                
        # Check register availability for the matrix's scale factor
        if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == False) and (self.config.matrix_blocked_flag == True):
            self.pim_matrix_scale_factor_req_reg += self.config.matrix_scale_factor_data_reg_count - 1
            if self.config.matrix_scale_factor_data_reg_count == special_scale_factor_reg_count.USE_FREE.value:
                if self.config.matrix_scale_factor_data_reg_type == data_src_dest.ORF.value:
                    self.pim_matrix_scale_factor_req_reg += max(0, self.free_orf_reg_per_bank)
                elif self.config.matrix_scale_factor_data_reg_type == data_src_dest.IRF.value:
                    self.pim_matrix_scale_factor_req_reg += max(0, self.free_irf_reg_per_bank)
            
            # Check register availability for the matrix's scale factors
            self.track_and_update_reg(self.config.matrix_scale_factor_data_reg_type, (self.pim_matrix_scale_factor_req_reg - 1), self.config.pim_ignore_input_reg_pressure_flag)
        
            assert(self.pim_matrix_scale_factor_req_reg > 0)
        
        self.output.pim_tile_order = self.pim_tile_order
        self.output.pim_vector_scalar_req_reg = self.pim_vector_scalar_req_reg
        self.output.pim_vector_scale_factor_req_reg = self.pim_vector_scale_factor_req_reg
        self.output.pim_output_req_reg = self.pim_output_req_reg
        self.output.pim_matrix_scale_factor_req_reg = self.pim_matrix_scale_factor_req_reg
        self.output.pim_orf_req_reg = self.pim_orf_req_reg
        self.output.pim_irf_req_reg = self.pim_irf_req_reg
        
        # Compute the total number of row blocks (after possible resizing) from the matrix. 
        assert(self.pim_gemm.gemm_m % self.pim_tile_shape_m_dim == 0)
        num_row_blocks = min(int(math.ceil(self.pim_gemm.gemm_m / self.pim_tile_shape_m_dim)), total_banks_count)
        
        # Compute the number of groups
        # This is the split-K groups in case the matrix is small compared to total number of banks.
        if self.config.pim_split_k_degree != special_split_k_mode.AUTO_NO_PADDING_MODE.value:
            num_groups = self.config.pim_split_k_degree
        else:
            num_groups = int(math.ceil(total_banks_count / num_row_blocks))
        assert(num_groups >= 1)
        if DEBUG: print("PIM - Total Number of Groups = {}".format(num_groups))
        self.output.pim_split_k_deg = num_groups    # This is the split-K degree

        # Compute number of elements per bank
        assert((self.pim_gemm.gemm_m * self.pim_gemm.gemm_k) % total_banks_count == 0)
        num_matrix_elements_per_bank = int(math.ceil((self.pim_gemm.gemm_m * self.pim_gemm.gemm_k) / total_banks_count))
        self.output.pim_num_matrix_elements_per_bank = num_matrix_elements_per_bank
        if DEBUG: print("PIM - Total Matrix Elements per Bank = {}".format(num_matrix_elements_per_bank))

        # Compute number of blocks per bank in case of blocked format 
        num_matrix_scale_factors_per_bank = 0
        if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == False):
            # Calculate number of blocks in the matrix per bank
            num_matrix_scale_factors_per_bank = math.ceil(num_matrix_elements_per_bank / self.config.block_size)  # Ceil ~ pad to not have partial block
        self.output.pim_num_matrix_scale_factors_per_bank = num_matrix_scale_factors_per_bank
        if DEBUG: print("PIM - Total Matrix Scale Factors per Bank = {}".format(num_matrix_scale_factors_per_bank))

        # Compute number of occupied DRAM rows per bank
        max_elements_per_dram_row = int((self.config.dram_row_size * 8) / self.config.operand_size)
        num_occupied_dram_rows_per_bank = num_matrix_elements_per_bank / max_elements_per_dram_row
        self.output.pim_matrix_scalar_num_occupied_dram_rows_per_bank = num_occupied_dram_rows_per_bank
        if DEBUG: print("PIM - Total Number of Occupied DRAM Rows per Bank = {:.2f}".format(num_occupied_dram_rows_per_bank))

        # Compute number of occupied DRAM rows for the input matrix scale factors per bank
        num_occupied_dram_rows_for_scale_factors_per_bank = 0
        if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == False) and (self.config.matrix_blocked_flag == True):
            max_scale_factors_per_dram_row = int((self.config.dram_row_size * 8) / self.config.scale_factor_operand_size)
            num_occupied_dram_rows_for_scale_factors_per_bank = num_matrix_scale_factors_per_bank / max_scale_factors_per_dram_row
            if DEBUG: print("PIM - Total Number of Occupied DRAM Rows for Scale Factors per Bank = {:.2f}".format(num_occupied_dram_rows_for_scale_factors_per_bank))
        self.output.pim_matrix_scale_factors_num_occupied_dram_rows_per_bank = num_occupied_dram_rows_for_scale_factors_per_bank

        # Compute PIM time (WITHOUT PIM_induced HOST time)
        
        # a- Row overhead time (ns) to read the scale factors input of the input matrix
        # This can be read repeatedly based on how many registers are available (and therefore how many scale factors are processed when a DRAM row is opened).
        pim_extra_row_overhead_time_ns = 0
        if self.config.pim_hide_row_open_overhead_flag == False:
            if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == False) and (self.config.matrix_blocked_flag == True):
                # Check if smart packing of matrix scale factors is used. 
                if self.config.pim_matrix_scale_factors_smart_pack_flag == False:
                    # No smart packing
                    reg_size = self.get_reg_size(self.config.matrix_scale_factor_data_reg_type)
                    
                    # Check if optimized scale factors loading into registers is assumed
                    tmp_scale_factor_operand_size = self.config.scale_factor_operand_size
                    if self.config.assume_optimized_scale_factors_into_reg_flag == False:
                        tmp_scale_factor_operand_size = self.config.accum_operand_size
                    
                    assert(reg_size % tmp_scale_factor_operand_size == 0)
                    weight_scale_factor_per_reg = int(reg_size / tmp_scale_factor_operand_size) # Number of scale factors per register

                    # Total number of scale factors per allocated registers
                    weight_scale_factor_per_total_reg = weight_scale_factor_per_reg * self.pim_matrix_scale_factor_req_reg 
                        
                    # Number of DRAM rows containing the matrix's scale factors
                    num_extra_dram_row_opens_per_bank = math.ceil(num_matrix_scale_factors_per_bank / weight_scale_factor_per_total_reg) 
                    
                    # The tCCDLs here are for the pim-MOV.
                    # No need to mulyiply by the number of concurrent vectors as the same PIM commands will run on all ALUs per PIM unit.
                    pim_extra_row_overhead_time_ns = self.config.dram_t_rp + max(self.config.dram_t_ras, self.config.dram_t_rcdrd + (self.pim_matrix_scale_factor_req_reg * self.config.dram_t_ccdl * self.config.banks_per_pim_unit))
                    pim_extra_row_overhead_time_ns *= math.ceil(num_extra_dram_row_opens_per_bank)
                else:
                    # Smart packing 
                    # This may be more than what is actually needed. 
                    # A more accurate way is to count the total bytes for the matrix elements and scale factors, then divide by DRAM row size.
                    pim_extra_row_overhead_time_ns = math.ceil(num_occupied_dram_rows_for_scale_factors_per_bank) * (self.config.dram_t_rp + self.config.dram_t_rcdrd)

        pim_extra_row_overhead_time_ns *= pim_batch_size_mult
        
        # b- Row overhead time (ns) to read input matrix 
        pim_row_overhead_time_ns = 0
        matrix_reopen_dram_rows_factor = 1
        simd_chunks_per_dram_row = (self.config.dram_row_size * 8) / self.config.simd_width
        # In case of blocked formats, check if reopening the DRAM rows of the input matrix (MxK) is needed due to the limited registers
        if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == False) and (self.config.matrix_blocked_flag == True) and (self.config.pim_matrix_scale_factors_smart_pack_flag == False):
            # Calculate the number of blocks (in blocked formats) per DRAM row
            assert(max_elements_per_dram_row % self.config.block_size == 0)
            max_blocks_per_dram_row = max_elements_per_dram_row / self.config.block_size # This is independent of the tile shape as the total number of blocks processed per DRAM row is the same.
            
            # Number of times to reopen the DRAM rows containing the scalar values of the input matrix (MxK) due to the limited registers
            matrix_reopen_dram_rows_factor = math.ceil(max_blocks_per_dram_row / weight_scale_factor_per_total_reg) 
            
        if self.config.pim_hide_row_open_overhead_flag == False:
            pim_row_overhead_time_ns = math.ceil(num_occupied_dram_rows_per_bank) * matrix_reopen_dram_rows_factor * (self.config.dram_t_rp + self.config.dram_t_rcdrd)
        
        pim_row_overhead_time_ns *= pim_batch_size_mult

        # c- If needed in case of heterogeneous computation, upcasting time (ns) of input matrix (MxK) elements precision to input vector (KxN) precision.
        # Assume upcasting operation that works on elements of input matrix (MxK) in parallel assuming their precision in the PIM ALU. Then, spills the outputs in more registers.
        pim_upcast_time_ns = 0
        # Check if upcasting is required
        if self.config.operand_size < self.config.vector_operand_size: 
            pim_upcast_time_ns = self.config.upcasting_pim_commands_overhead * num_occupied_dram_rows_per_bank * simd_chunks_per_dram_row * self.config.dram_t_ccdl * self.config.banks_per_pim_unit
        pim_upcast_time_ns *= pim_batch_size_mult
        
        # d- MAC PIM time (ns)
        pim_gemm_time_ns = num_occupied_dram_rows_per_bank * simd_chunks_per_dram_row * self.config.dram_t_ccdl * self.config.banks_per_pim_unit * pim_hetero_compute_mult
        pim_gemm_time_ns *= pim_batch_size_mult

        # e- Write to memory time (ns) 
        pim_write_mem_time_ns = 0
        
        # Compute number of spills to memory in terms of *full* row blocks
        num_of_writes = num_row_blocks_per_bank
        # Check if a full register is needed before writing to memory.
        if self.config.pim_assume_full_reg_before_write_to_mem_flag == True:
            num_of_writes = math.ceil(num_of_writes / math.ceil(pim_tile_mult))
        adjusted_num_of_writes = math.ceil(num_of_writes / self.pim_tile_order)
        
        # Compute number of output registers to write per row block
        adjusted_num_output_reg_per_row_blk = num_output_reg_per_row_blk
        if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == False):
            adjusted_num_output_reg_per_row_blk = math.ceil(num_output_reg_per_row_blk / self.config.reg_mult_required_for_blocked_format)
        
        # Optimize output spill in case of processing scale factors at host 
        num_reg_to_write_per_spill = 1
        if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == True):
            # assert(self.config.orf_reg_per_bank % adjusted_num_output_reg_per_row_blk == 0)
            num_reg_to_write_per_spill = math.ceil(self.config.orf_reg_per_bank / (adjusted_num_output_reg_per_row_blk * self.pim_tile_order))
            adjusted_num_of_writes /= num_reg_to_write_per_spill

        extra_compact_overhead = 0
        if self.config.pim_assume_full_reg_before_write_to_mem_flag == False:
            if pim_tile_mult > 1:
                extra_compact_overhead = self.config.output_compact_pim_commands_overhead
        
        for single_simd_wide_w_rows_index in range(math.ceil(adjusted_num_of_writes)):
            tmp_mult = adjusted_num_of_writes - single_simd_wide_w_rows_index 
            if tmp_mult > 1: 
                tmp_mult = 1

            pim_write_mem_time_ns += self.config.dram_t_rtw # Add read-to-write delay
            pim_write_mem_time_ns += self.config.dram_t_rp + max(self.config.dram_t_ras, self.config.dram_t_rcdrd + (self.config.dram_t_ccdl * tmp_mult * num_reg_to_write_per_spill * adjusted_num_output_reg_per_row_blk * (pim_spill_mem_mult + self.config.accum_reg_spill_reset_pim_commands_overhead + extra_compact_overhead) * self.pim_tile_order * pim_concurrent_vector_mult * self.config.banks_per_pim_unit)) # Add time to write the accumulated (final or partial in case of split-K) to memory
            pim_write_mem_time_ns += self.config.dram_t_wtr # Add read-to-write delay
        pim_write_mem_time_ns *= pim_batch_size_mult

        # f- PIM-induced time in case the host handles the scaling multiplication of the scale factors. 
        # There are three parts: 
        # First, the spilling of local block accumulation from PIM registers to memory. 
        # Second the overhead of the host reading the local block accumulation from memory. 
        # Third, the overhead of the host reading the matrix's scale factors from memory.
        # Here, the cost of spilling the local block accumulation to memory is estimated.
        pim_extra_write_mem_time_ns = 0
        number_of_extra_mem_spills = 0
        if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == True):
            # Estimate that while taking into consideration the split-K partial output. 
            # Do not count the partial output from split-K and host processing blocked format scaling factors. 
            number_of_extra_mem_spills = math.ceil(self.pim_gemm.gemm_k / self.config.block_size)

            # Update based on the split-K degree
            assert(number_of_extra_mem_spills % num_groups == 0)
            number_of_extra_mem_spills /= num_groups

            # Decrement one spill which is normally done after processing a row block (or part of row block in split-k)
            number_of_extra_mem_spills -= 1 

            pim_extra_write_mem_time_ns = pim_write_mem_time_ns * number_of_extra_mem_spills
        
        # g- Activation function (e.g., ReLU) compute time (ns)  
        pim_act_fn_time_ns = 0 
        if num_groups == 1: # No split-K approach as in split-K Host reads partial results and therefore can perform activations by host
            pim_act_fn_time_ns = self.config.activations_pim_commands_overhead * num_of_writes * self.config.dram_t_ccdl * adjusted_num_output_reg_per_row_blk * self.config.banks_per_pim_unit 
        pim_act_fn_time_ns *= pim_batch_size_mult  

        # h- PIM MUL/MAC time (ns) to process the multiplication of the scale factors in the blocked formats.
        # PIM_MUL(A,B) = A*B = Reg <- 0 then PIM_MAC(A,B,Reg), Reg = A*B + Reg
        pim_mul_time_ns = 0
        if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == False):
            # After the dot product computation of each block, multiply by the scale factors or each block. 
            # Assume moving the output from the accumulation register into one (or more) register using >=0 operations. 
            # Perform one of the multiplies when loading the scale factors of the matrix into the registers. In other words, instead of opening the DRAM rows of the matrix scale factors to move to registers, open the rows and multiply the LIO with the shared exponent of the input vector (from host). Therefore, the code uses (mul_count - 1). 
            # No need to mulyiply by the number of concurrent vectors as the same PIM commands will run on all ALUs per PIM unit.
            mul_count = self.compute_blocked_mul_count()
            mul_pim_commands_per_block = mul_count * adjusted_num_output_reg_per_row_blk

            # Check if smart packing of scale factors is used. 
            if self.config.pim_matrix_scale_factors_smart_pack_flag == False:
                # No smart packing, so assume performing one MUL/MAC (between vector and matrix scale factors) when loading matrix scale factors into IRFs.
                extra_pim_commands_per_block = self.config.scale_factor_pim_commands_overhead + ((mul_count - 1) * adjusted_num_output_reg_per_row_blk)
            else:
                # Smart packing, so must perform the total required MULs.
                extra_pim_commands_per_block = self.config.scale_factor_pim_commands_overhead + (mul_count * adjusted_num_output_reg_per_row_blk)
            
            # Compute the minimum of SIMD chunks and number of MX blocks in case one SIMD chunk fits >1 MX block.
            # For example, with row-major MX4 with 256b SIMD width and 32 elements per MX block), tehre are two MX blocks per SIMD width. Therefore, using num_matrix_scale_factors_per_bank instead of min(num_matrix_scale_factors_per_bank, total_simd_chunks) is an overkill as extra MUL/MAC commands is assumed in this case, although sending one MUL/MAC per SIMD chunk, that processes two MX blocks concurrently, is enough.
            blocks_to_process = num_matrix_scale_factors_per_bank 

            total_extra_pim_commands = extra_pim_commands_per_block * math.ceil(blocks_to_process / self.pim_tile_shape_m_dim)
            # Check if a writing a full register to memory is assumed.
            if self.config.pim_assume_full_reg_before_write_to_mem_flag == True:
                total_extra_pim_commands = math.ceil(total_extra_pim_commands / math.ceil(pim_tile_mult))
            pim_mul_time_ns = total_extra_pim_commands * self.config.dram_t_ccdl * self.config.banks_per_pim_unit
        pim_mul_time_ns *= pim_batch_size_mult

        # i- PIM time (ns) to write vector's scalar data from host to one (or more) registers.
        pim_host_write_scalar_vector_data_time_ns = 0
        # Check if register type (or mode in general) is not PART_OF_PIM_COMMAND
        if self.config.vector_scalar_data_reg_type != data_src_dest.PART_OF_PIM_COMMAND.value:
            # Host is writing vector data to register(s) instead of sending the vector data as part of PIM command.
            reg_size = self.get_reg_size(self.config.vector_scalar_data_reg_type)
            number_of_scalar_vector_data_per_register = int(reg_size / self.config.vector_operand_size)
            
            # Compute number of PIM commands to store the scalar values from the host
            # Use number_of_scalar_vector_data_per_register instead of (number_of_scalar_vector_data_per_register * self.config.vector_scalar_data_reg_count) as a write command from host, per each register written, is needed.
            assert(self.pim_gemm.gemm_k % self.config.pim_split_k_degree == 0)
            num_host_writes_per_row_block = math.ceil((self.pim_gemm.gemm_k /self.config.pim_split_k_degree) / number_of_scalar_vector_data_per_register)

            # Compute PIM time for writing the scalar part of the vector data.
            pim_host_write_scalar_vector_data_time_ns = self.config.dram_t_ccdl * num_host_writes_per_row_block
            
            # As the per vector number of scalar values is more than the number of scale factors (for each block size, there exists a single scale factor), assume the bus turnaround time for only the scalar values. This is because the writes of the scale factors can be orchestrated to be in the same window as the writes of the scalar values. 
            switch_overhead_count = math.ceil(num_host_writes_per_row_block / self.config.vector_scalar_data_reg_count)
            
            # Check if read/write mode switching is required.
            if self.config.pim_ignore_host_vector_write_overhead_flag == False: 
                pim_host_write_scalar_vector_data_time_ns += ((self.config.dram_t_rtw + self.config.dram_t_wtr + self.config.pim_host_induced_turnaround_overhead) * switch_overhead_count)
            
            # Multiply by (number of row blocks / PIM tile order) as it represents the number of times needed to send the input vector to PIM.
            pim_host_write_scalar_vector_data_time_ns *= math.ceil(num_row_blocks_per_bank / self.pim_tile_order)

        pim_host_write_scalar_vector_data_time_ns *= pim_concurrent_vector_mult
        pim_host_write_scalar_vector_data_time_ns *= pim_batch_size_mult

        # j- PIM time (ns) to spill/reset and load accumulation register due to interleaving row blocks execution (if CRO degree > 1). 
        pim_cro_induced_spill_load_accum_time_ns = 0
        # Check if register type (or mode in general) is not PART_OF_PIM_COMMAND
        if (self.config.vector_scalar_data_reg_type != data_src_dest.PART_OF_PIM_COMMAND.value) and (self.pim_tile_order > 1):
            # Higher CRO degree helps in reusing the vector inputs and therefore helps if vector data is written from host to PIM registers (not if they are sent as part of the PIM command).
            # Use switch_overhead_count. Every time a row block fully consumes the scalar vector values in the registers, then before loading a new set of inputs go and process N row blocks (N = CRO degree). This entails spilling and loading N times.
            row_blocks_to_switch = num_row_blocks_per_bank
            if num_row_blocks_per_bank % self.pim_tile_order == 1:
                row_blocks_to_switch -= 1
            pim_cro_induced_spill_load_accum_time_ns = switch_overhead_count * (self.config.accum_reg_load_pim_commands_overhead + self.config.accum_reg_spill_reset_pim_commands_overhead) * row_blocks_to_switch * self.config.dram_t_ccdl * self.config.banks_per_pim_unit  
        pim_cro_induced_spill_load_accum_time_ns *= pim_batch_size_mult
        
        # k- PIM time (ns) to write vector's scale factor data from host to one (or more) registers.
        pim_host_write_scale_factor_vector_data_time_ns = 0
        if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == False) and (self.config.vector_blocked_flag == True):
            # Check if register type (or mode in general) is not PART_OF_PIM_COMMAND
            if self.config.vector_scale_factor_data_reg_type != data_src_dest.PART_OF_PIM_COMMAND.value:
                # Host is writing vector's scale factor data to register(s) instead of sending the vector data as part of PIM command.
                reg_size = self.get_reg_size(self.config.vector_scale_factor_data_reg_type)

                # Check if optimized scale factors loading into registers is assumed
                tmp_scale_factor_operand_size = self.config.vector_scale_factor_operand_size
                if self.config.assume_optimized_scale_factors_into_reg_flag == False:
                    tmp_scale_factor_operand_size = self.config.accum_operand_size
                
                assert(reg_size % tmp_scale_factor_operand_size == 0)
                number_of_scale_factor_vector_data_per_register = int(reg_size / tmp_scale_factor_operand_size)
                
                # Compute number of PIM commands to store the scale factors values from the host
                num_host_scale_factor_writes_per_row_block = math.ceil(((self.pim_gemm.gemm_k / self.config.pim_split_k_degree) / self.config.block_size) / number_of_scale_factor_vector_data_per_register)
                
                # Compute PIM time for writing the scale factors part of the vector data.
                # As the per vector number of scalar values is more than the number of scale factors (for each block size, there exists a single scale factor), assume the bus turnaround time for only the scalar values. This is because the writes of the scale factors can be orchestrated to be in the same window as the writes of the scalar values. 
                pim_host_write_scale_factor_vector_data_time_ns = num_host_scale_factor_writes_per_row_block * self.config.dram_t_ccdl

                # Multiply by (number of row blocks / PIM tile order) as it represents the number of times needed to send the input vector to PIM.
                pim_host_write_scale_factor_vector_data_time_ns *= math.ceil(num_row_blocks_per_bank / self.pim_tile_order)

        pim_host_write_scale_factor_vector_data_time_ns *= pim_concurrent_vector_mult
        pim_host_write_scale_factor_vector_data_time_ns *= pim_batch_size_mult

        # Add up all the time related to writing vector data from host to PIM
        pim_host_write_vector_data_time_ns = pim_host_write_scalar_vector_data_time_ns + pim_host_write_scale_factor_vector_data_time_ns 

        # l- PIM lane-shift time (ns) to perform the required lane alignement in case of using a PIM tile that results in cross-SIMD reduction.
        pim_lane_shift_time_ns = 0
        # Check if free cross-SIMD reduction is assumed. 
        if self.config.pim_free_cross_simd_reduction_flag == False:
            # Adjust the pim_tile_mult to consider accumlation precision.
            adjusted_lanes_per_simd = math.ceil(self.config.orf_reg_size / self.config.accum_operand_size)
            adjusted_pim_tile_mult = adjusted_lanes_per_simd / self.pim_tile_shape_m_dim
            adjusted_pim_fixed_mac_output_mult = 1
            if self.config.mac_unit_output_size > 0:
                adjusted_pim_fixed_mac_output_mult = self.config.mac_unit_output_size / (adjusted_lanes_per_simd * self.config.accum_operand_size)
                adjusted_pim_tile_mult *= adjusted_pim_fixed_mac_output_mult
            adjusted_pim_tile_mult = math.ceil(adjusted_pim_tile_mult)

            # Check lane-shift mode.
            if self.config.shift_lane_mode == lane_shift_mode.MIN_LANE_SHIFT_MODE.value:
                pim_lane_shift_cmds_per_row_block = math.log2(adjusted_pim_tile_mult) 
            else:
                # This should be lane_shift_mode.SINGLE_LANE_SHIFT_MODE. 
                # Default is to assume single-lane shifts. 
                pim_lane_shift_cmds_per_row_block = max(adjusted_lanes_per_simd - self.pim_tile_shape_m_dim, 0)
            
            # Adjust for the number of output registers per row block
            pim_lane_shift_cmds_per_row_block *= adjusted_num_output_reg_per_row_blk

            pim_lane_shift_time_ns = pim_lane_shift_cmds_per_row_block * num_row_blocks_per_bank * self.config.dram_t_ccdl * self.config.banks_per_pim_unit * pim_batch_size_mult

        # m- PIM ADD time (ns) to perform the required reduction in case of using a PIM tile that results in cross-SIMD reduction.
        pim_add_time_ns = 0
        # Check if free cross-SIMD reduction is assumed. 
        if self.config.pim_free_cross_simd_reduction_flag == False:
            homo_reg_groups = math.ceil(adjusted_num_output_reg_per_row_blk / math.ceil(pim_tile_mult))
            pim_add_cmds_per_row_block = adjusted_num_output_reg_per_row_blk - homo_reg_groups + (math.log2(adjusted_pim_tile_mult) * adjusted_num_output_reg_per_row_blk)
            pim_add_time_ns = pim_add_cmds_per_row_block * num_row_blocks_per_bank * self.config.dram_t_ccdl * self.config.banks_per_pim_unit * pim_batch_size_mult 

        # Multiply by BGEMM batch size
        assert(self.pim_gemm.gemm_bs == self.host_gemm.gemm_bs)
        pim_row_overhead_time_ns *= self.pim_gemm.gemm_bs
        pim_upcast_time_ns *= self.pim_gemm.gemm_bs 
        pim_gemm_time_ns *= self.pim_gemm.gemm_bs
        pim_act_fn_time_ns *= self.pim_gemm.gemm_bs
        pim_write_mem_time_ns *= self.pim_gemm.gemm_bs
        pim_extra_write_mem_time_ns *= self.pim_gemm.gemm_bs
        pim_extra_row_overhead_time_ns *= self.pim_gemm.gemm_bs
        pim_mul_time_ns *= self.pim_gemm.gemm_bs
        pim_host_write_vector_data_time_ns *= self.pim_gemm.gemm_bs
        pim_cro_induced_spill_load_accum_time_ns *= self.pim_gemm.gemm_bs
        pim_lane_shift_time_ns *= self.pim_gemm.gemm_bs
        pim_add_time_ns *= self.pim_gemm.gemm_bs
        self.output.pim_matrix_scalar_row_overhead_time_ns = pim_row_overhead_time_ns
        self.output.pim_upcast_time_ns = pim_upcast_time_ns
        self.output.pim_mac_time_ns = pim_gemm_time_ns
        self.output.pim_act_fn_time_ns = pim_act_fn_time_ns
        self.output.pim_write_output_mem_time_ns = pim_write_mem_time_ns
        self.output.pim_write_local_output_mem_time_ns = pim_extra_write_mem_time_ns
        self.output.pim_matrix_scale_factor_row_overhead_time_ns = pim_extra_row_overhead_time_ns
        self.output.pim_mul_time_ns = pim_mul_time_ns
        self.output.pim_write_vector_pim_time_ns = pim_host_write_vector_data_time_ns
        self.output.pim_cro_spill_load_accum_time_ns = pim_cro_induced_spill_load_accum_time_ns
        self.output.pim_lane_shift_time_ns = pim_lane_shift_time_ns
        self.output.pim_add_time_ns = pim_add_time_ns

        # Total PIM time (without PIM-indused host time) 
        pim_only_time_ns = pim_row_overhead_time_ns + pim_upcast_time_ns + pim_gemm_time_ns + pim_act_fn_time_ns + pim_write_mem_time_ns + pim_extra_write_mem_time_ns + pim_extra_row_overhead_time_ns + pim_mul_time_ns + pim_host_write_vector_data_time_ns + pim_cro_induced_spill_load_accum_time_ns + pim_lane_shift_time_ns + pim_add_time_ns
        self.output.pim_only_time_ns = pim_only_time_ns

        if DEBUG: print("PIM - Time (without PIM-induced Host time) = {:.2f} -> Row Overhead = {:.2f}, Upcast Matrix Input = {:.2f}, GEMV = {:.2f}, Act Fn = {:.2f}, Write to Memory = {:.2f}, Extra Row Overhead (Blocked Format Scale Factors) = {:.2f}, Extra GEMV Ops (Blocked Format MUL/MOV) = {:.2f}, Write Vector Data from Host = {:.2f}, PIM SHIFT Ops {:.2f}, PIM ADD Ops = {:.2f}".format(pim_only_time_ns, pim_row_overhead_time_ns, pim_upcast_time_ns, pim_gemm_time_ns, pim_act_fn_time_ns, pim_write_mem_time_ns, pim_extra_row_overhead_time_ns, pim_mul_time_ns, pim_host_write_vector_data_time_ns, pim_lane_shift_time_ns, pim_add_time_ns))

        # Compute PIM-induced Host time
        # Host reading input matrix/vector
        # Host reading PIM matrix/vector output
        # Host reading partial output for split-K
        # Host reading part of the matrix in case of COLLAB mode

        # a- Host reading input matrix/vector
        pim_host_read_input_bytes = 0
        pim_host_read_input_time_ns = 0
        if self.config.pim_host_ignore_read_input_flag == False:
            pim_host_read_input_bytes = (self.host_gemm.gemm_k * self.host_gemm.gemm_n) * (self.config.vector_operand_size / 8)

            # Check if blocked format to read the scale factors of input vector 
            input_vector_number_of_blocks = 0
            if (self.config.block_size > 1) and (self.config.vector_blocked_flag == True):
                # Calculate number of blocks in the input vector
                input_vector_number_of_blocks = math.ceil((self.host_gemm.gemm_k * self.host_gemm.gemm_n) / self.config.block_size)  # Ceil ~ pad to not have partial block
                # Calcuate the number of bytes to read for the scale factors of the input vector
                scale_factors_bytes = input_vector_number_of_blocks * (self.config.vector_scale_factor_operand_size / 8)
                pim_host_read_input_bytes += scale_factors_bytes  # Accumulate to the input vector bytes
            
            pim_host_read_input_time_ns = pim_host_read_input_bytes / self.config.ip_vec_host_mem_bw

        # Multiply by BGEMM batch size
        pim_host_read_input_bytes *= self.pim_gemm.gemm_bs
        pim_host_read_input_time_ns *= self.pim_gemm.gemm_bs  
        self.output.pim_host_read_input_bytes = pim_host_read_input_bytes
        self.output.pim_host_read_input_time_ns = pim_host_read_input_time_ns

        # b- Host reading PIM matrix/vector output
        pim_host_read_output_bytes = 0
        pim_host_read_output_time_ns = 0
        if self.config.pim_host_ignore_read_output_flag == False:
            # Use pim_gemm_m which is >= gemm_m. That said, host should read the based on the original gemm_m not tha padded one. One reason pim_gemm_m is used is to be inline with reading partial results due to split-K or processing blocked formats at host as both use pim_gemm_m for their estimates. 
            pim_host_read_output_bytes = (self.pim_gemm.gemm_m * self.pim_gemm.gemm_n) * (self.config.accum_operand_size / 8)
            pim_host_read_output_time_ns = pim_host_read_output_bytes / self.config.host_mem_bw

        # Multiply by BGEMM batch size
        pim_host_read_output_bytes *= self.pim_gemm.gemm_bs
        pim_host_read_output_time_ns *= self.pim_gemm.gemm_bs
        self.output.pim_host_read_output_bytes = pim_host_read_output_bytes
        self.output.pim_host_read_output_time_ns = pim_host_read_output_time_ns

        # c- Host reading partial output for split-K
        # Compute the number of elements to be read by Host in case of small matrix with split-K.
        partial_w_elements_to_host = 0
        updated_num_groups = num_groups
        if self.config.pim_host_ignore_read_output_flag == False:
            # Decrement by one as the host reading the output from memory is already counted.
            updated_num_groups -= 1 

        if num_groups > 1:
            partial_w_elements_to_host = self.pim_gemm.gemm_m * updated_num_groups  # This is more traffic than needed as the host has to read the original elements and not the padded elements. 
        if DEBUG: print("PIM - Total Partial Output Elements to Host (in case of split-K) = {}".format(partial_w_elements_to_host))
        
        pim_host_read_partial_output_bytes = partial_w_elements_to_host * (self.config.accum_operand_size / 8)
        pim_host_read_partial_output_bytes *= pim_concurrent_vector_mult * pim_batch_size_mult
        pim_host_read_partial_output_time_ns = pim_host_read_partial_output_bytes / self.config.host_mem_bw

        # Multiply by BGEMM batch size
        pim_host_read_partial_output_bytes *= self.pim_gemm.gemm_bs
        pim_host_read_partial_output_time_ns *= self.pim_gemm.gemm_bs
        self.output.pim_host_read_split_k_output_bytes = pim_host_read_partial_output_bytes
        self.output.pim_host_read_split_k_output_time_ns = pim_host_read_partial_output_time_ns

        # d- PIM-induced time in case the host handles the scaling multiplication of the scale factors. 
        # There are three parts: 
        # First, the spilling of local block accumulation from PIM registers to memory. 
        # Second the overhead of the host reading the local block accumulation from memory. 
        # Third, the overhead of the host reading the matrix's scale factors from memory.
        # Here, estimate the cost of host reading the local block accumulation.
        pim_host_read_local_blocks_output_bytes = 0
        pim_host_read_local_blocks_output_time_ns = 0
        if (self.config.block_size > 1) and (self.config.process_scale_factors_at_host_flag == True): 
            number_of_local_partial_outputs = math.ceil(self.pim_gemm.gemm_k / self.config.block_size)

            # Decrement the host reads due to normal output or split-K outputs
            number_of_local_partial_outputs -= num_groups

            pim_host_read_local_blocks_output_bytes = self.pim_gemm.gemm_m * number_of_local_partial_outputs * (self.config.accum_operand_size / 8)
            pim_host_read_local_blocks_output_bytes *= pim_concurrent_vector_mult * pim_batch_size_mult
            pim_host_read_local_blocks_output_time_ns = pim_host_read_local_blocks_output_bytes / self.config.host_mem_bw

        # Multiply by BGEMM batch size
        pim_host_read_local_blocks_output_bytes *= self.pim_gemm.gemm_bs
        pim_host_read_local_blocks_output_time_ns *= self.pim_gemm.gemm_bs
        self.output.pim_host_read_local_output_bytes = pim_host_read_local_blocks_output_bytes
        self.output.pim_host_read_local_output_time_ns = pim_host_read_local_blocks_output_time_ns
        
        # e- Host reading part of the matrix in case of COLLAB mode
        pim_host_collab_mode_bytes = 0
        pim_host_collab_mode_time_ns = 0
        self.output.pim_host_collab_mode_bytes = pim_host_collab_mode_bytes
        self.output.pim_host_collab_mode_time_ns = pim_host_collab_mode_time_ns
            
        # f- Host flushing the matrix to memory if the matrix is sourced from local storage (cache, etc.)
        pim_host_flush_weights_bytes = 0
        pim_host_flush_weights_time_ns = 0
        self.output.pim_host_flush_matrix_bytes = pim_host_flush_weights_bytes
        self.output.pim_host_flush_matrix_time_ns = pim_host_flush_weights_time_ns

        # g- PIM-induced time in case the host handles the scaling multiplication of the scale factors. 
        # There are three parts: 
        # First, the spilling of local block accumulation from PIM registers to memory. 
        # Second the overhead of the host reading the local block accumulation from memory. 
        # Third, the overhead of the host reading the matrix's scale factors from memory.
        # Here, estimate the cost of host reading the matrix's scale factors.
        pim_host_read_matrix_scale_factors_bytes = 0
        pim_host_read_matrix_scale_factors_time_ns = 0
        # Check if blocked format to read the scale factors of the matrix 
        if (self.config.block_size > 1) and (self.config.matrix_blocked_flag == True) and (self.config.process_scale_factors_at_host_flag == True):
            # Calculate number of blocks in the input vector
            matrix_number_of_blocks = math.ceil((self.pim_gemm.gemm_m * self.pim_gemm.gemm_k) / self.config.block_size)  # Ceil ~ pad to not have partial block
            # Calcuate the number of bytes to read for the scale factors of the input matrix
            pim_host_read_matrix_scale_factors_bytes = matrix_number_of_blocks * (self.config.scale_factor_operand_size / 8)
        
            pim_host_read_matrix_scale_factors_time_ns = pim_host_read_matrix_scale_factors_bytes / self.config.host_mem_bw
        pim_host_read_matrix_scale_factors_bytes *= self.pim_gemm.gemm_bs
        pim_host_read_matrix_scale_factors_time_ns *= self.pim_gemm.gemm_bs
        self.output.pim_host_read_matrix_scale_factors_bytes = pim_host_read_matrix_scale_factors_bytes
        self.output.pim_host_read_matrix_scale_factors_time_ns = pim_host_read_matrix_scale_factors_time_ns

        # Total PIM-induced Host time (does not include Host reading input/output in it)
        pim_induced_host_bytes = pim_host_read_partial_output_bytes + pim_host_read_local_blocks_output_bytes + pim_host_collab_mode_bytes + pim_host_flush_weights_bytes + pim_host_read_matrix_scale_factors_bytes
        pim_induced_host_time_ns = pim_host_read_partial_output_time_ns + pim_host_read_local_blocks_output_time_ns + pim_host_collab_mode_time_ns + pim_host_flush_weights_time_ns + pim_host_read_matrix_scale_factors_time_ns
        self.output.pim_induced_host_bytes = pim_induced_host_bytes
        self.output.pim_induced_host_time_ns = pim_induced_host_time_ns

        # Total PIM time (WITH PIM-indused Host time)
        pim_time_ns = pim_only_time_ns + pim_induced_host_time_ns
        self.output.pim_time_ns = pim_time_ns

        if DEBUG: print("PIM - PIM-induced Host Time = {:.2f} ({} Bytes) -> Read Input = {:.2f} ({} Bytes), Read Output = {:.2f} ({} Bytes), Read Partial Output (Split-K) = {:.2f} ({} Bytes), Read W Part (COLLAB) = {:.2f} ({} Bytes), Flush W = {:.2f} ({} Bytes)".format(pim_induced_host_time_ns, pim_induced_host_bytes, pim_host_read_input_time_ns, pim_host_read_input_bytes, pim_host_read_output_time_ns, pim_host_read_output_bytes, pim_host_read_partial_output_time_ns, pim_host_read_partial_output_bytes, pim_host_collab_mode_time_ns, pim_host_collab_mode_bytes, pim_host_flush_weights_time_ns, pim_host_flush_weights_bytes))

        if DEBUG: print("PIM - Total Time = {:.2f} (with BS of {} for BGEMM)".format(pim_time_ns, self.pim_gemm.gemm_bs))

    # Run AMD GeniePIM core to compute host and PIM execution stats
    def run_geniepim_core(self):
        # Check if blocked format 
        if self.config.block_size > 1:
            # Update the K dimension to pad for block size of the blocked data formats
            self.host_gemm.gemm_k = (math.ceil(self.host_gemm.gemm_k / self.config.block_size)) * self.config.block_size

        # Estimate host execution stats
        self.compute_host_exec_stats()
        
        # Estimate PIM execution stats
        self.compute_pim_exec_stats()