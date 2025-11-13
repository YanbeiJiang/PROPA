import json
import argparse
from pathlib import Path
import re
from collections import deque

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

def calculate_accuracy(jsonl_file_path):
    """
    Calculate accuracy by comparing 'response' (predictions) with 'labels' (ground truth)
    from a JSONL file.
    
    Args:
        jsonl_file_path (str): Path to the JSONL file
        
    Returns:
        float: Accuracy score (percentage)
    """
    correct = 0
    total = 0
    
    # target_list = []
    # with open("/data/projects/punim1996/Data/AVR-RL/Clevr-Math/Clevr_Math_test.json", "r") as f:
    #     data = json.load(f)
    # for each in data:
    #     if each["template_filename"] == "subtraction.json":
    #         target_list.append(each["images"][0])
    try:
        with open(jsonl_file_path, 'r') as file:
            for line in file:
                try:
                    # Parse JSON line
                    data = json.loads(line.strip())
                    
                    # Get prediction and ground truth
                    prediction = data.get('response')
                    ground_truth = data.get('labels')
                                     
                    if prediction is None or ground_truth is None:
                        continue
                    # if "change_" in prediction and "change_" in ground_truth:
                    #     pattern = r'\w+\([^)]+\)'
                    #     predictions = re.findall(pattern, prediction)
                    #     ground_truths = re.findall(pattern, ground_truth)
                    #     if list(predictions) == list(ground_truths):
                    #         correct += 1
                    # else:
                    #     correct += 0
                    if "change_" in prediction and "change_" in ground_truth:
                        pattern = r'\w+\([^)]+\)'
                        predictions = re.findall(pattern, prediction)
                        ground_truths = re.findall(pattern, ground_truth)
                        same_count = len(set(list(predictions)) & set(list(ground_truths)))
                        correct += same_count/len(ground_truths)
                        # print("prediction: ", prediction)
                        # print("ground_truth: ", ground_truth)
                        # print("correct: ", correct)
                    else:
                        # Compare and count
                        #if str(prediction) == str(ground_truth):
                        correct += 0
                    
                    total += 1
                    
                except json.JSONDecodeError:
                    print(f"Warning: Skipping invalid JSON line: {line}")
        
        # Calculate accuracy
        if total == 0:
            return 0.0
        
        accuracy = (correct / total) * 100
        return accuracy
    
    except FileNotFoundError:
        print(f"Error: File not found at {jsonl_file_path}")
        return 0.0

if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser(description="Run MCTS to generate reasoning trajectories")
    parser.add_argument("--file_path", type=str, default="/data/gpfs/projects/punim1996/Data/AVR-RL/Math-Vision/SFT/result/InternVL2_5-2B/infer_result/20250818-095242.jsonl", help="epoch")
    args = parser.parse_args()

    file_path = args.file_path  # Update this with your file path
    accuracy = calculate_accuracy(file_path)
    print(f"Accuracy: {accuracy:.2f}%")