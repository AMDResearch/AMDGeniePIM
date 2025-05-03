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

from options_parser import ARGS
from geniepim_c_combinations_generator import COMBINATIONS
from gemm_generator import GEMM_LIST
from geniepim_core import geniepim_core
from geniepim_writer import geniepim_writer

DEBUG = ARGS.debug
VERBOSE = ARGS.verbose
IGNORE_HEADER = ARGS.output_ignore_header

# Create an output writer
output_writer = geniepim_writer("Outputs/" + ARGS.output_file, ARGS.output_file_format)

# Estimate GEMV execution time on host vs. PIM 
# Loop over differemt excecution combinations
for combination_index, curr_config in enumerate(COMBINATIONS):
    if VERBOSE or DEBUG: print("\n** Processing PIM/host/workload configuration #{} out of {}".format(combination_index+1, len(COMBINATIONS)))

    if DEBUG: print("\n- Current parameter combination = {}".format(curr_config))
    
    # Loop over GEMV sizes
    for curr_gemm in GEMM_LIST:
        # Create AMD GeniePIM instance
        # ---------------------- 
        geniepim = geniepim_core(curr_gemm, curr_config)

        if VERBOSE or DEBUG: print("\n- Processing GEMV - Model = {}, Source = {}, M = {}, K = {}, N = {}, BS = {}".format(geniepim.host_gemm.gemm_model_id, geniepim.host_gemm.gemm_source_id, geniepim.host_gemm.gemm_m, geniepim.host_gemm.gemm_k, geniepim.host_gemm.gemm_n, geniepim.host_gemm.gemm_bs))

        # Run AMD GeniePIM
        # ----------
        geniepim.run_geniepim_core()

        # Write in required output format
        # -------------------------------   
        if not(DEBUG): output_writer.write_output_file(geniepim, IGNORE_HEADER)

# Flush remaining lines  
if not(DEBUG): output_writer.flush_output_to_file(True)

# Inform user
if VERBOSE or DEBUG: print("\n** Complete - Output is in Outputs/{}".format(ARGS.output_file))