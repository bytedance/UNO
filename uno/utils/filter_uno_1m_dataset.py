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

"""
Simple UNO data filter script.
Filters JSON data based on score_final threshold and converts to specified format.
"""

import json
import argparse


def filter_and_convert_data(input_file, output_file, score_threshold):
    """
    Filter JSON data based on score_final and convert to target format.
    
    Args:
        input_file (str): Path to input JSON file
        output_file (str): Path to output JSON file
        score_threshold (float): Minimum score_final threshold
    """
    print(f"Loading data from {input_file}...")
    
    # Load input data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"Total records: {len(data)}")
    
    # Filter and convert data
    filtered_data = []
    for item in data:
        # Check if score_final meets threshold
        if item['vlm_filter_cot'].get('score_final', 0) >= score_threshold:
            # Convert to target format
            converted_item = {
                "prompt": item['caption'].get('img_path2', ''),
                "image_tgt_path": 'images/' + item.get('img_path2', ''),
                "image_paths": ['images/' + item.get('img_path1', '')]
            }
            filtered_data.append(converted_item)
    
    print(f"Filtered records: {len(filtered_data)}")
    
    # Save output data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_data, f, ensure_ascii=False, indent=2)
    
    print(f"Results saved to {output_file}")


def main():
    """Main function to parse arguments and run the filter."""
    parser = argparse.ArgumentParser(
        description='Filter UNO dataset based on score_final threshold'
    )
    
    parser.add_argument(
        'input_json',
        help='Input JSON file path'
    )
    
    parser.add_argument(
        'output_json', 
        help='Output JSON file path'
    )
    
    parser.add_argument(
        'score_threshold',
        type=float,
        help='score_final threshold (e.g., 3.5)'
    )
    
    args = parser.parse_args()
    
    # Run the filter
    filter_and_convert_data(
        args.input_json,
        args.output_json, 
        args.score_threshold
    )


if __name__ == '__main__':
    main()