import json
import re
import argparse

def extract_final_answer(final_step):
    """
    Extract the final answer from the final_step string.
    Looks for patterns like "The final answer is X" and extracts X.
    """
    # Pattern to match "The final answer is [answer]"
    # pattern = r"The final answer is\s+(.+?)\.?"
    # match = re.search(pattern, final_step, re.IGNORECASE)
    
    # if match:
    if "the final answer is" in final_step.lower():
        answer = final_step.lower().split("the final answer is")[1].strip()
        return answer
    
    return None

def calculate_accuracy(data):
    """
    Calculate the match accuracy between final answers and ground truth.
    """
    total_questions = len(data)
    correct_matches = 0
    
    results = []
    
    for i, item in enumerate(data, 1):
        final_step = item.get('final_step', '').lower()
        ground_truth = str(item.get('ground_truth', '')).lower()
        
        # Extract final answer from the final_step
        extracted_answer = extract_final_answer(final_step)
        if ground_truth and extracted_answer:
            if "change_" in extracted_answer and "change_" in ground_truth:
                pattern = r'\w+\([^)]+\)'
                predictions = re.findall(pattern, extracted_answer)
                ground_truths = re.findall(pattern, ground_truth)
                same_count = len(set(list(predictions)) & set(list(ground_truths)))
                correct = same_count/len(ground_truths)
                correct_matches += same_count/len(ground_truths)
        else:
            correct_matches += 0
            correct = 0
            
        results.append({
            'question_num': i,
            'extracted_answer': extracted_answer,
            'ground_truth': ground_truth,
            'match': correct
        })
    
    accuracy = correct_matches / total_questions * 100 if total_questions > 0 else 0
    
    return accuracy, results, correct_matches, total_questions

parser = argparse.ArgumentParser(description="Run MCTS to generate reasoning trajectories")
parser.add_argument("--output", type=str, default="/data/projects/punim1996/Data/AVR-RL/Trance/RL_PRM_qwen_lora/test/concatenate_inference_results_intern_Math_Vista.json", help="output")
args = parser.parse_args()

with open(args.output, 'r') as f:
    data = json.load(f)
    
accuracy, detailed_results, correct_count, total_count = calculate_accuracy(data)

# Print results
print(f"Accuracy: {accuracy:.2f}%")

# print("=== DETAILED BREAKDOWN ===")
# for result in detailed_results:
#     status = "✓ CORRECT" if result['match'] else "✗ INCORRECT"
#     print(f"Question {result['question_num']}: {status}")
#     print(f"  Extracted Answer: '{result['extracted_answer']}'")
#     print(f"  Ground Truth: '{result['ground_truth']}'")