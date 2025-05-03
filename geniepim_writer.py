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

import os

from enums import output_format_index_enum

class geniepim_writer:
    def __init__(self, output_file, output_format, flush_threshold=16384):
        self.output_file = output_file
        self.output_format = output_format
        self.flush_output_to_file_threshold = flush_threshold

        self.first_access_flag = True
        self.header_line_flag = False
        self.current_line_count = 0
        self.out_str = ""

    # Function to flush self.out_str to output_file if self.current_line_count exceeds a threshold
    def flush_output_to_file(self, force_flush=False):
        # Check if output directory exists and create if not
        isExist = os.path.exists("Outputs/")
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs("Outputs/")

        flush_flag = False
        # Check if output to file threashold reached 
        if force_flush or (self.current_line_count >= self.flush_output_to_file_threshold):
            if self.first_access_flag == True:
                self.first_access_flag = False
                out_file = open(self.output_file, "w")   # Changed the write mode to overwrite the outfile if exists on the first write
            else:
                out_file = open(self.output_file, "a")   # Changed the write mode to append during the full run

            out_file.write(self.out_str)
            out_file.close()
            
            flush_flag = True
        
        return flush_flag

    # Function to generate output in the desired format
    def write_output_file(self, geniepim, ignore_header=False):
        # Write headers?
        if ignore_header == True:
            # No
            self.header_line_flag = True

        if self.output_format == output_format_index_enum.OUT_ALL.value:
            pass
            
        elif self.output_format == output_format_index_enum.OUT_CONDENSED.value:
            # Basic output showing host and PIM execution time
            # Use with fixed PIM, memory, workload config - only variable is GEMV sizes
            tmp_str = ""
            if self.header_line_flag == False:
                self.header_line_flag = True
                tmp_str += "gemm_model_id,gemm_source_id,gemm_m,gemm_k,gemm_n,host_gemm_time_ns,pim_time_ns,speedup\n"
                self.current_line_count += 1

            speedup = geniepim.output.host_gemm_time_ns / geniepim.output.pim_time_ns

            tmp_str += "{},{},{},{},{},{},{},{}\n".format(
                # Config data
                # GEMV data 
                geniepim.host_gemm.gemm_model_id,
                geniepim.host_gemm.gemm_source_id,
                geniepim.host_gemm.gemm_m,
                geniepim.host_gemm.gemm_k,
                geniepim.host_gemm.gemm_n,
                # Output data 
                geniepim.output.host_gemm_time_ns,
                geniepim.output.pim_time_ns,
                speedup
                )
            
            print("Results: Model = {}, GEMV id = {}, GEMV = {}x{}x{}, IPU time = {:.2f} ns, PIM time = {:.2f} ns, PIM Speedup = {:.2f}x".format(
                geniepim.host_gemm.gemm_model_id,
                geniepim.host_gemm.gemm_source_id,
                geniepim.host_gemm.gemm_m,
                geniepim.host_gemm.gemm_k,
                geniepim.host_gemm.gemm_n,
                # Output data 
                geniepim.output.host_gemm_time_ns,
                geniepim.output.pim_time_ns,
                speedup
            ))

            self.current_line_count += 1 # Increament line count
            self.out_str += tmp_str  # Concatenate line

        elif self.output_format == output_format_index_enum.OUT_CUSTOM.value:
            tmp_str = ""
            if self.header_line_flag == False:
                self.header_line_flag = True
                tmp_str += "banks_per_pim_unit,pim_compute_rate,banks_per_channel,grf_registers_per_pim_alu,srf_registers_per_pim_alu,grf_reg_size,srf_reg_size,operand_size,act_operand_size,block_size,scale_factor_operand_size,act_scale_factor_operand_size,accumulation_operand_size,scale_factor_extra_pim_commands_overhead,pim_tile_shape_degree,pim_tile_order_degree,host_scalar_data_reg_count,host_scalar_data_reg_type,host_scale_factor_data_reg_count,host_scale_factor_data_reg_type,matrix_scale_factor_data_reg_count,matrix_scale_factor_data_reg_type,host_peak_mem_bw,host_mem_bw_util,reg_mult_required_for_blocked_format,pim_matrix_scale_factors_smart_pack_flag,pim_free_cross_simd_reduction_flag,mac_unit_output_size,gemm_model_id,gemm_source_id,gemm_m,gemm_k,gemm_n,pim_gemm_m,host_gemm_time_ns,pim_tile_shape,pim_tile_order,pim_req_scalar_input_reg,pim_req_scale_factor_input_reg,pim_req_output_reg,pim_matrix_scale_factor_input_reg,pim_req_grf_reg,pim_req_srf_reg,pim_row_overhead_time_ns,pim_upcast_time_ns,pim_gemm_time_ns,pim_act_fn_time_ns,pim_write_mem_time_ns,pim_extra_write_mem_time_ns,pim_max_extra_row_overhead_time_ns,pim_extra_gemm_time_ns,pim_host_write_vector_data_time_ns,pim_cro_induced_spill_load_accum_time_ns,pim_lane_shift_time_ns,pim_add_time_ns,pim_only_time_ns,pim_host_read_input_time_ns,pim_host_read_output_time_ns,pim_host_read_local_blocks_output_time_ns,pim_host_read_matrix_scale_factors_time_ns,pim_induced_host_time_ns,pim_time_ns\n"
                self.current_line_count += 1

            tmp_str += "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                # Config data
                geniepim.config.banks_per_pim_unit, 
                geniepim.config.mac_compute_rate,
                geniepim.config.banks_per_channel,
                geniepim.config.orf_reg_per_pim_alu, 
                geniepim.config.irf_reg_per_pim_alu, 
                geniepim.config.orf_reg_size,
                geniepim.config.irf_reg_size,
                geniepim.config.operand_size,
                geniepim.config.vector_operand_size,
                geniepim.config.block_size,
                geniepim.config.scale_factor_operand_size,
                geniepim.config.vector_scale_factor_operand_size,
                geniepim.config.accum_operand_size,
                geniepim.config.scale_factor_pim_commands_overhead,
                geniepim.config.pim_tile_shape_degree,
                geniepim.config.pim_tile_order_degree,
                geniepim.config.vector_scalar_data_reg_count,
                geniepim.config.vector_scalar_data_reg_type,
                geniepim.config.vector_scale_factor_data_reg_count,
                geniepim.config.vector_scale_factor_data_reg_type,
                geniepim.config.matrix_scale_factor_data_reg_count,
                geniepim.config.matrix_scale_factor_data_reg_type,
                geniepim.config.host_peak_mem_bw,
                geniepim.config.host_mem_bw_util,
                geniepim.config.reg_mult_required_for_blocked_format,
                geniepim.config.pim_matrix_scale_factors_smart_pack_flag,
                geniepim.config.pim_free_cross_simd_reduction_flag,
                geniepim.config.mac_unit_output_size,
                
                # GEMV data 
                geniepim.host_gemm.gemm_model_id,
                geniepim.host_gemm.gemm_source_id,
                geniepim.host_gemm.gemm_m,
                geniepim.host_gemm.gemm_k,
                geniepim.host_gemm.gemm_n,
                geniepim.pim_gemm.gemm_m,
                
                # Output data 
                geniepim.output.host_gemm_time_ns,
                geniepim.output.pim_tile_shape,
                geniepim.output.pim_tile_order,
                geniepim.output.pim_vector_scalar_req_reg,
                geniepim.output.pim_vector_scale_factor_req_reg,
                geniepim.output.pim_output_req_reg,
                geniepim.output.pim_matrix_scale_factor_req_reg,
                geniepim.output.pim_orf_req_reg,
                geniepim.output.pim_irf_req_reg,
                geniepim.output.pim_matrix_scalar_row_overhead_time_ns,
                geniepim.output.pim_upcast_time_ns,
                geniepim.output.pim_mac_time_ns,
                geniepim.output.pim_act_fn_time_ns,
                geniepim.output.pim_write_output_mem_time_ns,
                geniepim.output.pim_write_local_output_mem_time_ns,
                geniepim.output.pim_matrix_scale_factor_row_overhead_time_ns,
                geniepim.output.pim_mul_time_ns,
                geniepim.output.pim_write_vector_pim_time_ns,
                geniepim.output.pim_cro_spill_load_accum_time_ns,
                geniepim.output.pim_lane_shift_time_ns,
                geniepim.output.pim_add_time_ns,
                geniepim.output.pim_only_time_ns,
                geniepim.output.pim_host_read_input_time_ns,
                geniepim.output.pim_host_read_output_time_ns,
                geniepim.output.pim_host_read_local_output_time_ns,
                geniepim.output.pim_host_read_matrix_scale_factors_time_ns,
                geniepim.output.pim_induced_host_time_ns,
                geniepim.output.pim_time_ns,
                )

            self.current_line_count += 1 # Increament line count
            self.out_str += tmp_str  # Concatenate line

        # Flush output to file
        if (self.flush_output_to_file()):
            # Reset output string and line count 
            self.current_line_count = 0
            self.out_str = ""