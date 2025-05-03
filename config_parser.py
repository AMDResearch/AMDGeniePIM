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

class config_parser:
    def __init__(self, in_file):
        self.input_file = in_file
        self.config_dict = {}
    
    # Generate dict of combinations 
    def product_dict(self, **kwargs):
        keys = kwargs.keys()
        for instance in itertools.product(*kwargs.values()):
            yield dict(zip(keys, instance))
    
    # Get the list of combinations 
    def get_config_combinations(self):
        return list(self.product_dict(**self.config_dict))
    
    # Get config parameters
    def get_config_params(self):
        return list(self.config_dict.keys())

    # Extract the GEMV params of current run
    def parse_config_file(self, preprocess_flag):
        with open(self.input_file, 'r') as f:
            for line in f:
                line = line.strip()
                # Check for empty line
                if line == "":
                    continue
                elif "#" in line:
                    continue
                else:
                    line_parts = line.split("=")
                    
                    parameter_name = line_parts[0]
                    if preprocess_flag == True:
                        parameter_name = line_parts[0].replace("_LIST", "")
                    parameter_val_str = line_parts[1]

                    # Add parameter in config dictionary 
                    assert(parameter_name not in self.config_dict)
                    self.config_dict[parameter_name] = [] 
                    
                    # Process parameter (first split on ",", then on "/", then on "&")
                    parameter_val_list = parameter_val_str.split(",")
                    for parameter_val in parameter_val_list:
                        sep_a_exsits = "/" in parameter_val
                        sep_b_exsits = "&" in parameter_val
                        sep_c_exsits = ":" in parameter_val

                        if not(sep_a_exsits) and not(sep_b_exsits) and not(sep_c_exsits):
                            self.config_dict[parameter_name].append(parameter_val)
                        else:
                            tmp_list = []
                            parameter_val_parts = parameter_val.split("/")
                            if not(sep_b_exsits) and not(sep_c_exsits):
                                for parameter_val_part in parameter_val_parts:
                                    tmp_list.append(parameter_val_part)
                            else:
                                for parameter_val_part in parameter_val_parts:
                                    if ("&" not in parameter_val_part) and (":" not in parameter_val_part):
                                        tmp_list.append(parameter_val_part)
                                    else:
                                        tmp_parts = parameter_val_part.split("&")
                                        tmp_dict = {}
                                        for tmp_part in tmp_parts:
                                            parts = tmp_part.split(":")
                                            key = parts[0]
                                            peak = parts[1]
                                            util = parts[2]

                                            assert(key not in tmp_dict)
                                            tmp_dict[key] = (peak, util)
                                        tmp_list.append(tmp_dict)
                            self.config_dict[parameter_name].append(tuple(tmp_list))