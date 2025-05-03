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

class geniepim_output:
    def __init__(self):
        # Host execution ops & bytes  
        self.host_compute_ops = 0
        self.host_memory_bytes = 0
        
        # Host execution time (ns)
        self.host_compute_time_ns = 0
        self.host_memory_time_ns = 0
        self.host_gemm_time_ns = 0  
        
        # PIM tile shape & order
        self.pim_tile_shape = 0
        self.pim_tile_order = 0
        self.pim_split_k_deg = 0 
        
        # PIM register usage
        self.pim_vector_scalar_req_reg = 0  
        self.pim_vector_scale_factor_req_reg = 0
        self.pim_output_req_reg = 0 
        self.pim_matrix_scale_factor_req_reg = 0 
        self.pim_orf_req_reg = 0 
        self.pim_irf_req_reg = 0 
        
        # PIM matrix distribution stats per bank
        self.pim_num_row_blocks_per_bank = 0
        self.pim_num_matrix_elements_per_bank = 0 
        self.pim_num_matrix_scale_factors_per_bank = 0
        self.pim_matrix_scalar_num_occupied_dram_rows_per_bank = 0
        self.pim_matrix_scale_factors_num_occupied_dram_rows_per_bank = 0
        
        # PIM execution time (ns)
        self.pim_matrix_scalar_row_overhead_time_ns = 0
        self.pim_matrix_scale_factor_row_overhead_time_ns = 0
        self.pim_write_vector_pim_time_ns = 0
        self.pim_write_output_mem_time_ns = 0
        self.pim_write_local_output_mem_time_ns = 0
        self.pim_mac_time_ns = 0
        self.pim_mul_time_ns = 0
        self.pim_upcast_time_ns = 0
        self.pim_lane_shift_time_ns = 0
        self.pim_add_time_ns = 0
        self.pim_act_fn_time_ns = 0
        self.pim_cro_spill_load_accum_time_ns = 0
        self.pim_only_time_ns = 0
        
        # PIM-induced host execution time (ns)
        self.pim_host_read_input_time_ns = 0
        self.pim_host_read_output_time_ns = 0
        self.pim_host_read_split_k_output_time_ns = 0   
        self.pim_host_read_local_output_time_ns = 0     
        self.pim_host_collab_mode_time_ns = 0           
        self.pim_host_flush_matrix_time_ns = 0           
        self.pim_induced_host_time_ns = 0

        # PIM-induced host bytes 
        self.pim_host_read_input_bytes = 0
        self.pim_host_read_output_bytes = 0
        self.pim_host_read_split_k_output_bytes = 0
        self.pim_host_read_local_output_bytes = 0
        self.pim_host_collab_mode_bytes = 0
        self.pim_host_flush_matrix_bytes = 0
        self.pim_host_read_matrix_scale_factors_bytes = 0
        self.pim_host_read_matrix_scale_factors_time_ns = 0
        self.pim_induced_host_bytes = 0

        # PIM total execution time (ns)
        self.pim_time_ns = 0