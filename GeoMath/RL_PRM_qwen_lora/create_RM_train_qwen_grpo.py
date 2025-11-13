import json
import os
from pathlib import Path
import numpy as np
import argparse
import math
EPISILON = 0.1

def get_parent_states(tree, node_index, states=None):
    """Recursively collect states from the node to the root."""
    if states is None:
        states = []
    
    # Find the node with the given index
    node = next(n for n in tree if n['index'] == node_index)
    
    # Add the current node's state
    states.append(node['state'])
    
    # If the node has a parent, recurse
    if node['parent'] is not None:
        get_parent_states(tree, node['parent'], states)
    
    return states

def find_other_children_values(tree_file, node):
    parent_node = tree_file[node["parent"]]
    children_values = []
    for child_node_index in parent_node["children"]:
        if child_node_index != node["index"]:
            child_node = tree_file[child_node_index]
            if child_node['visits'] != 0:
                children_values.append(child_node['value'] / child_node['visits'])
            else:
                children_values.append(0.0)
    return children_values

def find_first_nonzero_divide_by_two(nums):
    for num in nums:
        if num != 0:
            return num / 2
    return 0.0

def log_boost(x, a=10):
    """对数非线性增强函数，输入x应在[0, 1]范围内"""
    if x <= 0:
        return 0.0
    elif x >= 1:
        return 1.0
    else:
        return math.log(1 + a * x) / math.log(1 + a)
    
def process_json_file(json_data):
    """Process a single JSON file and extract dataset fields."""
    input_sentences = []
    images = []
    labels = []
    
    # Each JSON file may contain multiple data entries
    for entry in json_data:

        tree_file = entry['tree_file']
        image_path = entry['image_path']
        
        # Process each node in the tree_file
        for node in tree_file:
            if node["index"] == 0:
                continue
            # Get all states from the current node to the root
            states = get_parent_states(tree_file, node['index'])
            # Reverse to have root-to-node order and concatenate
            concatenated_state = " ".join(states[::-1])
            if node['visits'] > 0:
            # Calculate label as value/visits (handle division by zero)
                label = node['value'] / node['visits']
            elif node['visits'] == 0:
                other_children_values = find_other_children_values(tree_file, node)
                #print(other_children_values)
                if all(x == 0.0 for x in other_children_values):
                    continue
                label = 0.0
                # sorted_other_children_values = sorted(other_children_values)
                # label = find_first_nonzero_divide_by_two(sorted_other_children_values)
            # Append to lists
            label = log_boost(label)
            input_sentences.append(concatenated_state)
            images.append(image_path)
            labels.append(label)
    
    return input_sentences, images, labels

def create_combined_dataset(folder_path, epoch):
    """Create a combined dataset from all JSON files in the folder."""
    # Initialize lists to store combined data
    all_entries = []
    
    # Iterate through all JSON files in the folder
    folder = Path(folder_path)
    for json_file in folder.glob("*.json"):
        if f"epoch{epoch}" in os.path.basename(json_file):
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                
                # Process the JSON file
                input_sentences, images, labels = process_json_file(json_data)
                
                for input_sentence, image, label in zip(input_sentences, images, labels):
                    entry = {
                        "messages": [
                            {
                                "role": "user",
                                "content": "<image>"+input_sentence
                            },
                        ],
                        "images": [
                            image
                        ],
                        "label": label
                    }
                    all_entries.append(entry)
                
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
        
    # Create and return the dataset
    return all_entries

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--epoch", type=int, required=False, default=0, help="Output JSON file path")
    args = parser.parse_args()
    # Specify the folder containing JSON files
    folder_path = "./tree_data_qwen_grpo"  # Replace with your folder path
    
    # Create the combined dataset
    combined_dataset = create_combined_dataset(folder_path, args.epoch)
    
    print(f"Total entries: {len(combined_dataset)}")
    if len(combined_dataset) > 0:
        print("Example entry:", combined_dataset[0])
    with open(f"./rm_data_qwen_grpo/rm_data_qwen_epoch{args.epoch}.json", 'w') as f:
        json.dump(combined_dataset, f)