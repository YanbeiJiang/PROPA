import json
import re
import argparse
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
    
    try:
        with open(jsonl_file_path, 'r') as file:
            for line in file:
                try:
                    # Parse JSON line
                    data = json.loads(line.strip())
                    
                    # Get prediction and ground truth
                    prediction = data.get('response')
                    ground_truth = data.get('labels')
                    
                    # Skip if either is missing
                    if prediction is None or ground_truth is None:
                        continue
                    prediction_match = re.search(r'<answer>(.*?)</answer>', prediction)
                    prediction_answer = prediction_match.group(1).strip() if prediction_match else prediction.strip()
                    if "change_" in prediction_answer and "change_" in ground_truth:
                        pattern = r'\w+\([^)]+\)'
                        predictions = re.findall(pattern, prediction_answer)
                        ground_truths = re.findall(pattern, ground_truth)
                        same_count = len(set(list(predictions)) & set(list(ground_truths)))
                        correct += same_count/len(ground_truths)
                    # ground_truth_match = re.search(r'<answer>(.*?)</answer>', ground_truth)
                    # ground_truth_answer = ground_truth_match.group(1).strip() if ground_truth_match else ground_truth.strip()
                    # Compare and count
                    else:
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

    file_path = args.file_path     
    accuracy = calculate_accuracy(file_path)
    print(f"Accuracy: {accuracy:.2f}%")