import json
from pathlib import Path
import argparse
import os 
import re
from collections import deque

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

def extract_items(text):
    pattern = re.compile(r"(\w+)\((\w+),\s*'?(\w+)'?\)")
    matches = pattern.findall(text)
    filtered_matches = list(set(matches))
    return filtered_matches

def get_score_trance(pred, sol):
    reward = 0.0
    pred_list = extract_items(pred)
    sol_list = extract_items(sol)
    if not sol_list:
        return 0.0
    
    item_score = 1.0 / max(len(pred_list), len(sol_list)) if pred_list else 0
    
    pred_queue = deque(pred_list)
    sol_queue = deque(sol_list)
    
    # full mapping
    full_mapping_num = 0
    exact_matches = [(p, s) for p in pred_queue for s in sol_queue if p == s]
    for p, s in exact_matches:
        if p in pred_queue and s in sol_queue:
            full_mapping_num += 1
            pred_queue.remove(p)
            sol_queue.remove(s)
    reward += full_mapping_num * item_score
        # (func, object_id) mapping
    partial_matches_1_num = 0
    partial_matches_1 = [(p, s) for p in pred_queue for s in sol_queue if p[:2] == s[:2]]
    for p, s in partial_matches_1:
        if p in pred_queue and s in sol_queue:
            partial_matches_1_num += 1
            pred_queue.remove(p)
            sol_queue.remove(s)
    reward += partial_matches_1_num * item_score * 0.5
    
    # (func, value) mapping
    partial_matches_2_num = 0
    partial_matches_2 = [(p, s) for p in pred_queue for s in sol_queue if (p[0], p[2]) == (s[0], s[2])]
    for p, s in partial_matches_2:
        if p in pred_queue and s in sol_queue:
            partial_matches_2_num += 1
            pred_queue.remove(p)
            sol_queue.remove(s)
    reward += partial_matches_2_num * item_score * 0.5
    
    # only-func mapping
    func_matches_num = 0
    func_matches = [(p, s) for p in pred_queue for s in sol_queue if p[0] == s[0]]
    for p, s in func_matches:
        if p in pred_queue and s in sol_queue:
            func_matches_num += 1
            pred_queue.remove(p)
            sol_queue.remove(s)
    reward += func_matches_num * item_score * 0.25
    
    return reward


def check_if_final_answer(text, final_answer) -> bool:
    """Check if the response contains the final answer."""
    # This is a simple implementation - in practice, you might need a more robust solution
    clean_text = text.strip().lower()
    clean_answer = final_answer.lower()
    is_final = False
    is_correct = False
    if f"the final answer is" in clean_text:
        is_final = True
    # Check if the text ends with the answer or contains it prominently
    if f"the final answer is" in clean_text:
        clean_text = clean_text.split("the final answer is")[1].strip()
    # print(clean_text)
    # print(clean_answer)
    # get_score_trance(clean_text, clean_answer)
    reward = 0

    if "change_" in clean_text and "change_" in clean_answer:
        pattern = r'\w+\([^)]+\)'
        predictions = re.findall(pattern, clean_text)
        ground_truths = re.findall(pattern, clean_answer)
        # if list(predictions) == list(ground_truths):
        #     reward = 1
        same_count = len(set(list(predictions)) & set(list(ground_truths)))
        reward = same_count/len(ground_truths)
    return reward

parser = argparse.ArgumentParser(description="Run MCTS to generate reasoning trajectories")
parser.add_argument("--epoch", type=int, default=6, help="epoch")
parser.add_argument("--dataset", type=str, default="/data/projects/punim1996/Data/AVR-RL/Trance/Trance_test.json", help="epoch")
parser.add_argument("--tree_path", type=str, default="/data/projects/punim1996/Data/AVR-RL/Trance/RL_PRM_qwen_lora_7b/tree_data_qwen_test/llm_judge_result", help="epoch")
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
        assert(test_data[i]["images"] == image_path)
        if best_terminal_node is None:
            i += 1
            continue
        correct += check_if_final_answer(best_terminal_node["state"], test_data[i]["messages"][1]["content"])
        # print(json_file)
        # print(correct)
        i += 1
        
final_accuracy = correct/i * 100
print(f"Accuracy: {final_accuracy:.2f}%")