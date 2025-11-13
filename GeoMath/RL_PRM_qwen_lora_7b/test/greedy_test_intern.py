from swift.llm import PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest, BaseArguments
from swift.tuners import Swift
import multiprocessing as mp
import time
import sys
from openai import OpenAI
from swift.llm import InferRequest, InferClient, RequestConfig
from swift.plugin import InferStats
from swift.plugin import InferStats
import json
from tqdm import tqdm
import argparse
def generate_greedy(prompt, image_path, gpu_id):
        """Generate a single response using greedy decoding with online vLLM server."""
        if gpu_id == 0:
            port_num = 30000
        
        # Make request to vLLM server
        attempts = 0
        while attempts < 5:  # Retry up to 5 times
            try:
                engine = InferClient(host='0.0.0.0', port=port_num)
                request_config = RequestConfig(max_tokens=128, seed=42, n=1, temperature=0)
                #print(prompt)
                infer_requests = [
                    InferRequest(messages=[{'role': 'system', 'content': "A conversation between User and Assistant. The input by user may include some existing steps to solve the question and the Assistant should continue to derive only the next step based on these existing steps. If the input does not provide any existing steps, the Assistant need to analyze the problem and then give the first step in solving the problem. If the Assistant think it has reached the final step, provide the final answer number following the format 'The final answer is ...', otherwise continue to the next inference step."}, {"role": "user", "content": "<image>"+prompt}], images=[image_path])
                ]
                resp_list = engine.infer(infer_requests, request_config)

                generated_text = resp_list[0].choices[0].message.content
                #print("generated text for greedy: ", generated_text)
                return generated_text
            except Exception as e:
                # Increment the attempts counter if an error occurs
                attempts += 1
                print("greedy phase")


def extract_answer(text):
    """Extract numerical answer from text containing 'The final answer is...' (case insensitive)"""
    import re
    
    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Check for various forms of "the final answer is"
    patterns = [
        r'the final answer is\s*(.+?)(?:[.。]|$)',
        r'final answer:\s*(.+?)(?:[.。]|$)',
        r'answer:\s*(.+?)(?:[.。]|$)',
        r'the answer is\s*(.+?)(?:[.。]|$)'
    ]
    
    answer_part = None
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            answer_part = match.group(1).strip()
            break
    return answer_part
    # if answer_part is None:
    #     return None
    
    # # Extract numbers from the answer part
    # numbers = re.findall(r'-?\d+\.?\d*', answer_part)
    
    # if numbers:
    #     try:
    #         # Try to convert to float first, then to int if it's a whole number
    #         answer = float(numbers[0])
    #         if answer.is_integer():
    #             return int(answer)
    #         return answer
    #     except ValueError:
    #         return None
    # return None

# Function to check if response contains final answer (case insensitive)
def contains_final_answer(text):
    """Check if text contains any form of final answer indicator (case insensitive)"""
    import re
    text_lower = text.lower()
    
    patterns = [
        r'the final answer is',
        r'final answer:',
        r'the answer is'
    ]
    
    for pattern in patterns:
        if re.search(pattern, text_lower):
            return True
    return False

# Calculate accuracy by comparing with ground truth
def calculate_accuracy(results):
    """Calculate accuracy by comparing predicted answers with ground truth"""
    correct_count = 0
    total_with_answers = 0
    
    for result in results:
        if "error" in result:
            continue
            
        # Extract predicted answer from final step
        predicted_answer = extract_answer(result.get("final_step", ""))
        
        # Extract ground truth answer
        ground_truth_text = result.get("ground_truth", "")
        ground_truth_answer = ground_truth_text
        
        # Only count items where we have both predicted and ground truth answers
        if predicted_answer is not None and ground_truth_answer is not None:
            total_with_answers += 1
            if "\\(" in predicted_answer or "\\)" in predicted_answer:
                predicted_answer = predicted_answer.replace("\\(", "").replace("\\)", "")
            if predicted_answer == ground_truth_answer:
                correct_count += 1
                result["correct"] = True
            else:
                result["correct"] = False
                #(f"Mismatch - Predicted: {predicted_answer}, Ground Truth: {ground_truth_answer}")
        else:
            result["correct"] = None  # Unable to determine
    
    accuracy = correct_count / total_with_answers if total_with_answers > 0 else 0
    return correct_count, total_with_answers, accuracy


parser = argparse.ArgumentParser(description="Run MCTS to generate reasoning trajectories")
parser.add_argument("--input", type=str, default="/data/projects/punim1996/Data/AVR-RL/GeoMath/Math_Vision_test.json", help="input")
parser.add_argument("--output", type=str, default="/data/gpfs/projects/punim1996/Data/AVR-RL/GeoMath/RL_PRM_qwen_lora_7b/test/concatenate_inference_results_intern_without_training.json", help="output")
args = parser.parse_args()

with open(args.input, 'r') as f:
    dataset = json.load(f)

results = []

for item in tqdm(dataset, desc="Processing examples"):
    # Extract question and answer
    messages = item["messages"]
    image_path = item.get("images", [])[0]
    
    #base_question = "Think step by step to solve the problem in the image: " + messages[0]["content"].split("<image>")[1].strip() + " Please ONLY response one step at a time."
    if "Please ONLY respond with a number." in messages[0]["content"]:
        base_question = "Think step by step to solve the problem in the image: " + messages[0]["content"].split("<image>")[1].split(" Please ONLY respond with a number.")[0].strip() + " The final answer should be a number. Please ONLY response one step at a time."
    else:
        base_question = "Think step by step to solve the problem in the image: " + messages[0]["content"].split("<image>")[1].split(" Please ONLY respond with the selected option letter.")[0].strip() + " The final answer should be a option letter. Please ONLY response one step at a time."
    final_answer = messages[1]["content"]
    
    # Initialize variables for concatenate inference
    all_steps = []
    max_steps = 15  # Maximum number of steps to prevent infinite loops
    step_count = 0
    
    # First inference - ask for first step
    first_prompt = base_question + " First step:"
    
    try:
        first_response = generate_greedy(first_prompt, image_path, gpu_id=0)  # You can adjust gpu_id as needed
        all_steps.append(first_response)
        step_count += 1
        
        #print(f"Step {step_count}: {first_response}")
        
        # Check if first response contains final answer
        if "The final answer is" in first_response or "the final answer is" in first_response:
            #print("Final answer reached in first step")
            result = {
                "question": base_question,
                "steps": all_steps,
                "final_step": first_response,
                "ground_truth": final_answer,
                "total_steps": step_count,
                "image_path": image_path
            }
            results.append(result)
            continue
        
        # Continue with subsequent steps
        while step_count < max_steps:
            # Concatenate all previous steps
            existing_steps_text = " ".join(all_steps)
            
            # Create prompt with existing steps
            next_prompt = base_question + f" Existing steps: {existing_steps_text} Next step:"
            
            # Generate next step
            next_response = generate_greedy(next_prompt, image_path, gpu_id=0)
            
            all_steps.append(next_response)
            step_count += 1
            
            #print(f"Step {step_count}: {next_response}")
            
            # Check if this step contains the final answer
            if contains_final_answer(next_response):
                #print(f"Final answer reached in step {step_count}")
                break
                
        # Store the result for this item
        result = {
            "question": base_question,
            "steps": all_steps,
            "final_step": all_steps[-1] if all_steps else "",
            "ground_truth": final_answer,
            "total_steps": step_count,
            "image_path": image_path
        }
        results.append(result)
        
    except Exception as e:
        print(f"Error processing item: {e}")
        # Store error result
        error_result = {
            "question": base_question,
            "steps": [],
            "final_step": "",
            "ground_truth": final_answer,
            "total_steps": 0,
            "image_path": image_path,
            "error": str(e)
        }
        results.append(error_result)

# Save results to file
output_file = args.output
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

# print(f"Processed {len(results)} items")


# # Print statistics
# successful_items = [r for r in results if "error" not in r]
# avg_steps = sum(r["total_steps"] for r in successful_items) / len(successful_items) if successful_items else 0
# print(f"Average steps per successful item: {avg_steps:.2f}")

# final_answer_items = [r for r in successful_items if "The final answer is" in r.get("final_step", "")]
# print(f"Items that reached final answer: {len(final_answer_items)}/{len(successful_items)}")

# with open("/data/gpfs/projects/punim1996/Data/AVR-RL/GeoMath/RL_PRM/test/concatenate_inference_results_intern.json", 'r') as f:
#     results = json.load(f)
# # Calculate and print accuracy
# correct_count, total_with_answers, accuracy = calculate_accuracy(results)
# print(f"\nAccuracy Results:")
# print(f"Correct answers: {correct_count}")
# print(f"Total answerable questions: {total_with_answers}")
# print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
# print(f"Total processed items: {len(results)}")
# print(f"Items with errors: {len([r for r in results if 'error' in r])}")