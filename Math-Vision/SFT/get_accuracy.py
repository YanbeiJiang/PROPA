import json
import argparse
from pathlib import Path
from sympy import sympify, simplify

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
                    
                    try:
                        generated = prediction
                        #print(generated)
                        #generated = generated.replace("π", "*pi") 
                        if any(c.isalpha() for c in generated):
                        # Check if the text ends with the answer or contains it prominently
                            if generated == ground_truth:
                                correct += 1
                        else:
                            generated = generated.replace(",", "")
                            if simplify(sympify(generated) - sympify(ground_truth)) == 0:
                                correct += 1
                    except:
                        total += 1
                        continue
                    
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