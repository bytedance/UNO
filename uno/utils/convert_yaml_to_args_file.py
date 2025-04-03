# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument("--yaml", type=str, required=True)
parser.add_argument("--arg", type=str, required=True)
args = parser.parse_args()


with open(args.yaml, "r") as f:
    data = yaml.safe_load(f)

with open(args.arg, "w") as f:
    for k, v in data.items():
        if isinstance(v, list):
            v = list(map(str, v))
            v = " ".join(v)
        if v is None:
            continue
        print(f"--{k} {v}", end=" ", file=f)
