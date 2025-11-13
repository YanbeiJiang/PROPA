import json
import re
import argparse
from sympy import sympify, simplify


def extract_final_answer(final_step):
    """
    Extract the final answer from the final_step string.
    Looks for patterns like "The final answer is X" and extracts X.
    """
    # Pattern to match "The final answer is [answer]"
    # pattern = r"The final answer is\s+(.+?)\.?"
    # match = re.search(pattern, final_step, re.IGNORECASE)
    
    # if match:
    if "The final answer is" in final_step:
        if "\\(" in final_step or "\\)" in final_step:
            final_step = final_step.replace("\\(", "").replace("\\)", "")
        answer = final_step.split("The final answer is")[1].split(".")[0].strip()
        # Remove trailing period if present
        return answer
    
    return None

def check_if_correct(generated, ground_truth):
    """Check if the response contains the final answer."""
    # This is a simple implementation - in practice, you might need a more robust solution
    # if "\\(" in generated or "\\)" in generated:
    #     generated = generated.replace("\\(", "").replace("\\)", "")
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
                print("generated:", generated)
                print("gt: ", clean_answer)
                is_correct = False
    except:
        is_correct = False
    return is_correct

def calculate_accuracy(data):
    """
    Calculate the match accuracy between final answers and ground truth.
    """
    total_questions = len(data)
    correct_matches = 0
    
    results = []
    
    for i, item in enumerate(data, 1):
        is_correct = False
        generated = item.get('final_step', '')
        ground_truth = str(item.get('ground_truth', ''))
        if check_if_correct(generated, ground_truth):
            is_correct = True
            correct_matches += 1
        
        results.append({
            'question_num': i,
            'extracted_answer': generated,
            'ground_truth': ground_truth,
            'match': is_correct
        })
    
    accuracy = correct_matches / total_questions * 100 if total_questions > 0 else 0
    
    return accuracy, results, correct_matches, total_questions

parser = argparse.ArgumentParser(description="Run MCTS to generate reasoning trajectories")
parser.add_argument("--output", type=str, default="/data/projects/punim1996/Data/AVR-RL/GeoMath/RL_PRM_qwen_lora/test/concatenate_inference_results_intern_Math_Vista.json", help="output")
args = parser.parse_args()

with open(args.output, 'r') as f:
    data = json.load(f)
    
accuracy, detailed_results, correct_count, total_count = calculate_accuracy(data)

# Print results
# print("=== MATCH ACCURACY RESULTS ===")
# print(f"Total Questions: {total_count}")
# print(f"Correct Matches: {correct_count}")
print(f"Accuracy: {accuracy:.2f}%")

# print("=== DETAILED BREAKDOWN ===")
# for result in detailed_results:
#     status = "✓ CORRECT" if result['match'] else "✗ INCORRECT"
#     print(f"Question {result['question_num']}: {status}")
#     print(f"  Extracted Answer: '{result['extracted_answer']}'")
#     print(f"  Ground Truth: '{result['ground_truth']}'")