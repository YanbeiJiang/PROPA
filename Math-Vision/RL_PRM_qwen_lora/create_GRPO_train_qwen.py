import json
import os
from pathlib import Path
import numpy as np
import argparse
import sys
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

def check_if_final_answer(text):
    """Check if the response contains the final answer."""
    # This is a simple implementation - in practice, you might need a more robust solution
    clean_text = text.strip().lower()
    is_final = False
    if f"the final answer is" in clean_text:
        is_final = True
    return is_final

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
    solutions = []
    final_answers = []
    # Each JSON file may contain multiple data entries
    for entry in json_data:

        tree_file = entry['tree_file']
        image_path = entry['image_path']
        final_answer = entry['final_answer']
        # Process each node in the tree_file
        for node in tree_file:
            if check_if_final_answer(node["state"]):
                continue
            if len(node['children']) == 0:
                continue
            if node["index"] == 0:
                concatenated_state = f"{node['state']}"
            # Get all states from the current node to the root
            else:
                states = get_parent_states(tree_file, node['index'])[::-1]
                accumulated_steps = "\n".join(states[1:])
                concatenated_state = f"{states[0]} Existing steps: {accumulated_steps} Next step:"
            solution = []
            all_labels = []
            for child_index in node['children']:
                child_node = tree_file[child_index]
                if child_node['visits'] > 0:
                # Calculate label as value/visits (handle division by zero)
                    label = child_node['value'] / child_node['visits']
                elif child_node['visits'] == 0:
                    # other_children_values = find_other_children_values(tree_file, node)
                    #print(other_children_values)
                    # if all(x == 0.0 for x in other_children_values):
                    #     continue
                    # other_children_values = find_other_children_values(tree_file, child_node)
                    # #print(other_children_values)
                    # sorted_other_children_values = sorted(other_children_values)
                    # label = find_first_nonzero_divide_by_two(sorted_other_children_values)
                    label = 0.0
                label = log_boost(label)
                solution.append(child_node["state"] + " **Label:** " + str(label))
                all_labels.append(label)
            if all(x == 0 for x in all_labels) or all(x == 1 for x in all_labels) or (max(all_labels) - min(all_labels) < 0.1):
                continue
                # Append to lists
            input_sentences.append(tree_file[node['children'][0]]["prompt"])
            images.append(image_path)
            solutions.append(solution)
            final_answers.append(final_answer)
    return input_sentences, images, solutions, final_answers

def create_combined_dataset(folder_path, epoch, batch_size, batch_num):
    """Create a combined dataset from all JSON files in the folder."""
    # Initialize lists to store combined data
    all_entries = []
    
    # Iterate through all JSON files in the folder
    folder = Path(folder_path)
    for json_file in folder.glob("*.json"):
        index = int(os.path.basename(json_file).split("_")[-1].split(".json")[0])
        if f"epoch{epoch}" in os.path.basename(json_file) and index >= batch_size*batch_num and index < batch_size*(batch_num+1):
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                if json_data[0]["tree_file"][0]["value"] != 0:
                # Process the JSON file
                    input_sentences, images, solutions, final_answers = process_json_file(json_data)

                    
                    for input_sentence, image, solution, final_answer in zip(input_sentences, images, solutions, final_answers):

                        entry = {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "A conversation between User and Assistant. The input by user may include some existing steps to solve the question and the Assistant should continue to derive only the next step based on these existing steps. If the input does not provide any existing steps and ask for the 'First step', the Assistant need to analyze the problem and then give the first step in solving the problem. If the Assistant think it has reached the final step, provide the final answer number following the format 'The final answer is ...', otherwise continue to the next inference step."
                                },
                                {
                                    "role": "user",
                                    "content": "<image>"+input_sentence,
                                },
                                {
                                    "role": "assistant",
                                    "content": final_answer
                                }
                            ],
                            "images": [image],
                            "solution": solution,
                            "model": "Qwen/Qwen2.5-VL-3B-Instruct"
                        }
                        all_entries.append(entry)
                    
            except Exception as e:
                print(f"Error processing {json_file}: {e}")
                continue
    
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
    folder_path = "./tree_data_qwen_grpo"  # Replace with your folder path
    output = f"./grpo_data_qwen/grpo_data_qwen_epoch{args.epoch}_{args.batch_num}.json"
    # if os.path.exists(output):
    #     #print(f"Total entries: {len(combined_dataset)}")
    #     print(f"File {output} already exists. Exiting.")
    #     sys.exit(0)
    # Create the combined dataset
    combined_dataset = create_combined_dataset(folder_path, args.epoch, args.batch_size, args.batch_num)
    
    print(f"Total entries: {len(combined_dataset)}")
    if len(combined_dataset) > 3:
        with open(output, 'w') as f:
            json.dump(combined_dataset, f, indent=2)