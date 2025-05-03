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

import argparse

parser = argparse.ArgumentParser(description=("GEMV PIM model options"))

parser.add_argument(
    "-of",
    "--output_file",
    default="out_pim.csv",
    type=str,
    help="Output CSV file",
)

parser.add_argument(
    "-off",
    "--output_file_format",
    choices=[0, 1, 2],
    default=1,
    type=int,
    help="CSV output format: 0 (not implemented), 1 (condensed output format - default), 2 (custom)",
)

parser.add_argument(
    "-oih",
    "--output_ignore_header",
    choices=[0, 1],
    default=0,
    type=int,
    help="Ignore header in output file: 0 (add header - default), 1 (ignore header)",
)

parser.add_argument(
    "-ggm",
    "--gemm_gen_mode",
    choices=["gemm", "models", "custom"],
    default="gemm",
    type=str,
    help="GEMV generator mode: 'gemm' (list of GEMV sizes - default), 'models' (list of LLMs hyperparameters), 'custom'",
)

parser.add_argument(
    "-ggi",
    "--gemm_gen_input",
    default="gemm.in",
    type=str,
    help="GEMV generator input file",
)

parser.add_argument(
    "-cif",
    "--config_input_file",
    default="config.in",
    type=str,
    help="GEMV config file (PIM, Host, Workload parameters)",
)

parser.add_argument(
    "-d",
    "--debug",
    choices=[0, 1],
    default=0,
    type=int,
    help="Debug flag: 0 (disable - default), 1 (enable)",
)

parser.add_argument(
    "-v",
    "--verbose",
    choices=[0, 1],
    default=1,
    type=int,
    help="Verbose flag: 0 (disable), 1 (enable - default)",
)

ARGS = parser.parse_args()
