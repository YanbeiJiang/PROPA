import json
from pathlib import Path
import argparse
import os 
import re
from sympy import sympify, simplify


def find_best_path_by_model_reward(tree_data, json_file):
    # 构建 index -> node 映射
    index_to_node = {node['index']: node for node in tree_data}

    # 找到所有的 terminal nodes
    terminal_nodes = [node for node in tree_data if node['is_terminal']]

    best_avg_reward = -float('inf')
    best_path = []
    best_terminal_node = None
    # print(terminal_nodes)
    # print(json_file)
    for terminal_node in terminal_nodes:
        path = []
        total_reward = 0
        count = 0

        current = terminal_node.copy()
        while current is not None:
            path.append(current)
            total_reward += current['reward_given_by_model']
            count += 1
            parent_index = current['parent']
            current = index_to_node[parent_index] if parent_index is not None else None

        avg_reward = total_reward / count
        # print(json_file)
        # print(terminal_node["index"])
        # print(best_terminal_node)
        #print(best_avg_reward)
        if avg_reward > best_avg_reward:
            #print("aa")
            best_avg_reward = avg_reward
            best_path = path[::-1]  # Reverse to start from root
            best_terminal_node = terminal_node
        #print(best_terminal_node)
    return best_path, best_terminal_node, best_avg_reward

# def check_if_correct(generated, ground_truth):
#     """Check if the response contains the final answer."""
#     # This is a simple implementation - in practice, you might need a more robust solution
#     if "\\(" in generated or "\\)" in generated:
#         generated = generated.replace("\\(", "").replace("\\)", "")
#     clean_text = generated.strip().lower()
#     clean_answer = ground_truth.lower()
#     # Check if the text ends with the answer or contains it prominently
#     if f"the final answer is {clean_answer}" in clean_text:
#         is_correct = True
#     else:
#         is_correct = False
#     return is_correct
def check_if_correct(generated, ground_truth):
    """Check if the response contains the final answer."""
    # This is a simple implementation - in practice, you might need a more robust solution
    if "\\(" in generated or "\\)" in generated:
        generated = generated.replace("\\(", "").replace("\\)", "")
    clean_text = generated.strip()
    #print(clean_text)
    clean_answer = ground_truth
    try:
        match = re.search(r"the final answer is\s+([\-\$A-Za-z0-9π\$\/\.\*\^\(\)\{\}\\]+)", clean_text, re.IGNORECASE)
        generated = match.group(1)
        #print(generated)
        #generated = generated.replace("π", "*pi") 
        if generated[-1] == ".":
            generated = generated.replace(".", "")
        if any(c.isalpha() for c in generated):
        # Check if the text ends with the answer or contains it prominently
            if generated == clean_answer:
                is_correct = True
            else:
                # print("generated:", generated)
                # print("gt: ", clean_answer)
                is_correct = False
        else:
            generated = generated.replace(",", "")
            if simplify(sympify(generated) - sympify(clean_answer)) == 0:
                is_correct = True
            else:
                # print("generated:", generated)
                # print("gt: ", clean_answer)
                is_correct = False
    except:
        is_correct = False
    return is_correct

parser = argparse.ArgumentParser(description="Run MCTS to generate reasoning trajectories")
parser.add_argument("--epoch", type=int, default=6, help="epoch")
parser.add_argument("--dataset", type=str, default="/data/projects/punim1996/Data/AVR-RL/Math-Vision/DynaMath_test.json", help="epoch")
parser.add_argument("--tree_path", type=str, default="/data/projects/punim1996/Data/AVR-RL/Math-Vision/RL_PRM_qwen_lora/tree_data_qwen_test_OOD/all_epochs_result/checkpoint_9", help="epoch")
args = parser.parse_args()


with open(args.dataset, "r") as f:
    test_data = json.load(f)
# 加载之前保存的 tree 文件
folder_path = args.tree_path

folder = Path(folder_path)
i = 0
correct = 0
def extract_suffix_number(filename):
    """提取文件名末尾 `_数字` 的数字部分"""
    match = re.search(r'_(\d+)\.json$', filename)
    return int(match.group(1)) if match else -1

json_files = sorted(folder.glob("*.json"), key=lambda x: extract_suffix_number(x.name))

for json_file in json_files:
    if f"epoch{args.epoch}" in os.path.basename(json_file):
        
        with open(json_file, "r") as f:
            all_data = json.load(f)
        if not all_data:
            continue
        item = all_data[0]
        image_path = item["image_path"]
        tree_data = item['tree_file']
        best_path, best_terminal_node, best_avg_reward = find_best_path_by_model_reward(tree_data, json_file)
        #print(best_terminal_node)
        # print(f"Question: {item['question']}")
        # print(f"Final Answer: {item['final_answer']}")
        # print(f"Best Avg Reward: {best_avg_reward}")
        # print("Best Path States:")
        # for node in best_path:
        #     print(f"  State: {node['state']}, Reward by model: {node['reward_given_by_model']}")
        # print(f"Best Terminal Node State: {best_terminal_node['state']}")
        # print("-" * 50)
        #print(image_path)
        assert(test_data[i]["images"][0] == image_path)
        if best_terminal_node is None:
            i += 1
            continue
        if check_if_correct(best_terminal_node["state"], test_data[i]["messages"][1]["content"]):
            correct += 1
        i += 1
accuracy = correct/i * 100
print(f"Accuracy: {accuracy:.2f}%")