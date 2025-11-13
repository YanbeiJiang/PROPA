import json
import os
from pathlib import Path
import numpy as np
import argparse
import sys
EPISILON = 0.1

with open("/data/projects/punim1996/Data/AVR-RL/GeoMath/GeoMath_train_CoT_concate.json", "r") as f:
    concate_data = json.load(f)

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

def check_if_correct_final_answer(node):
    """Check if the response contains the final answer."""
    # This is a simple implementation - in practice, you might need a more robust solution
    is_final = False
    if node["is_terminal"] and node["reward"] >= 1:
        is_final = True
    return is_final

def find_common_ancestor(tree_file, node1_idx, node2_idx):
    """Find the lowest common ancestor of two nodes."""
    # Get paths to root for both nodes
    def get_path_to_root(node_idx):
        path = []
        current = node_idx
        while current is not None:
            path.append(current)
            # Find parent of current node
            parent = None
            for node in tree_file:
                if node['index'] == current:
                    parent = node['parent']
                    break
            current = parent
        return path[::-1]  # Reverse to get root-to-node path
    
    path1 = get_path_to_root(node1_idx)
    path2 = get_path_to_root(node2_idx)
    
    # Find the last common node in both paths
    common_ancestor = None
    for i in range(min(len(path1), len(path2))):
        if path1[i] == path2[i]:
            common_ancestor = path1[i]
        else:
            break
    
    return common_ancestor

def select_best_node_pairs(correct_nodes, tree_file):
    """Select at most 2 nodes with the lowest common ancestor closest to root."""
    if len(correct_nodes) <= 2:
        return correct_nodes
    
    best_pairs = []
    min_ancestor_depth = float('inf')
    
    # Try all combinations of 2 nodes
    from itertools import combinations
    for node1, node2 in combinations(correct_nodes, 2):
        ancestor = find_common_ancestor(tree_file, node1['index'], node2['index'])
        
        # Calculate depth of ancestor (distance from root)
        ancestor_depth = 0
        current = ancestor
        while current is not None:
            # Find parent of current node
            parent = None
            for node in tree_file:
                if node['index'] == current:
                    parent = node['parent']
                    break
            if parent is not None:
                ancestor_depth += 1
            current = parent
        
        # If this pair has a shallower common ancestor, update best pairs
        if ancestor_depth < min_ancestor_depth:
            min_ancestor_depth = ancestor_depth
            best_pairs = [node1, node2]
        elif ancestor_depth == min_ancestor_depth:
            # If same depth, we could keep multiple pairs, but we'll just keep the first one
            pass
    
    return best_pairs

def process_json_file(json_data):

    results_concate = []
    # Each JSON file may contain multiple data entries
    for entry in json_data:

        tree_file = entry['tree_file']
        image_path = entry['image_path']
        final_answer = entry['final_answer']
        # Process each node in the tree_file

        correct_nodes = []
        for node in tree_file:
            if check_if_correct_final_answer(node):
                correct_nodes.append(node)
        
        if not correct_nodes:
            for each in concate_data:
                if image_path == each["images"][0]:
                    results_concate.append(each)
                    
        # Select at most 2 nodes with common ancestor closest to root
        selected_nodes = select_best_node_pairs(correct_nodes, tree_file)
        
        # Process each selected node
        for node in selected_nodes:
            states = get_parent_states(tree_file, node['index'])[::-1]
            
            for i, step in enumerate(states):
                if i == 0:
                    question = step
                    continue
                elif i == 1:
                    accumulated_steps = step
                    entry_data = {
                        "messages": [
                            {
                                "role": "system", 
                                "content": "A conversation between User and Assistant. The input by user may include some existing steps to solve the question and the Assistant should continue to derive only the next step based on these existing steps. If the input does not provide any existing steps, the Assistant need to analyze the problem and then give the first step in solving the problem. If the Assistant think it has reached the final step, provide the final answer number following the format 'The final answer is ...', otherwise continue to the next inference step."
                            },
                            {
                                "role": "user",
                                "content": f"<image>{question} First step:"
                            },
                            {
                                "role": "assistant",
                                "content": step
                            }
                        ],
                        "images": [image_path],
                        "solution": f"<answer>{final_answer}</answer>"
                    }
                    results_concate.append(entry_data)
                else:
                    accumulated_steps += f" {step}"
                    entry_data = {
                        "messages": [
                            {
                                "role": "system", 
                                "content": "A conversation between User and Assistant. The input by user may include some existing steps to solve the question and the Assistant should continue to derive only the next step based on these existing steps. If the input does not provide any existing steps, the Assistant need to analyze the problem and then give the first step in solving the problem. If the Assistant think it has reached the final step, provide the final answer number following the format 'The final answer is ...', otherwise continue to the next inference step."
                            },
                            {
                                "role": "user",
                                "content": f"<image>{question} Existing steps: {accumulated_steps[:-len(step)]}Next step:"
                            },
                            {
                                "role": "assistant",
                                "content": step
                            }
                        ],
                        "images": [image_path],
                        "solution": f"<answer>{final_answer}</answer>"
                    }
                    results_concate.append(entry_data)
    
    return results_concate

def create_combined_dataset(folder_path, epoch, batch_size, batch_num):
    """Create a combined dataset from all JSON files in the folder."""
    # Initialize lists to store combined data
    all_entries = []
    
    # Iterate through all JSON files in the folder
    folder = Path(folder_path)
    for json_file in folder.glob("*.json"):
        index = int(os.path.basename(json_file).split("_")[-1].split(".json")[0])
        if f"epoch{epoch}" in os.path.basename(json_file) and index >= batch_size*batch_num and index < batch_size*(batch_num+1):

            with open(json_file, 'r') as f:
                json_data = json.load(f)
            
            if json_data[0]["tree_file"][0]["value"] == 0:
            # Process the JSON file
                results_concate = process_json_file(json_data)

                all_entries.extend(results_concate)
    
    # Create and return the dataset
    return all_entries

# Example usage
if __name__ == "__main__":
    # Specify the folder containing JSON files
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--batch_size", type=int, default=40, help="batch_size")
    parser.add_argument("--epoch", type=int, default=0, help="epoch")
    parser.add_argument("--batch_num", type=int, default=0, help="batch_num")
    args = parser.parse_args()
    folder_path = "./tree_data_intern_grpo"  # Replace with your folder path
    output = f"./SFT_after_grpo/SFT_data_intern_epoch{args.epoch}_{args.batch_num}.json"

    combined_dataset = create_combined_dataset(folder_path, args.epoch, args.batch_size, args.batch_num)
    
    print(f"Total entries: {len(combined_dataset)}")
    if len(combined_dataset) > 0:
        with open(output, 'w') as f:
            json.dump(combined_dataset, f, indent=2)