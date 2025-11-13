import json
import numpy as np
import os
import argparse
from typing import List, Dict, Any, Tuple, Optional
#from vllm import LLM, SamplingParams
import math
from tqdm import tqdm
import random
from PIL import Image
from transformers import AutoTokenizer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, set_seed
from qwen_vl_utils import process_vision_info
from transformers import GenerationConfig
import torch
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from multiprocessing import Pool
from vllm import LLM, SamplingParams, EngineArgs
from vllm.sampling_params import BeamSearchParams
from swift.llm import PtEngine, RequestConfig, safe_snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, AutoModel
from peft import PeftModel
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from qwen_vl_utils import process_vision_info
import base64
import requests
from typing import NamedTuple, Optional

from swift.llm import PtEngine, RequestConfig, safe_snapshot_download, get_model_tokenizer, get_template, InferRequest, BaseArguments
from swift.tuners import Swift
import multiprocessing as mp
import time
import sys
from openai import OpenAI
from swift.llm import InferRequest, InferClient, RequestConfig
from swift.plugin import InferStats
from swift.plugin import InferStats
from concurrent.futures import ProcessPoolExecutor, as_completed
import queue

# Use spawn instead of fork for better isolation
#mp.set_start_method('spawn', force=True)
# #from MCTS_run_save_qwen_vllm import mcts_main
# def worker_init():
#     # Set CUDA device in subprocess
#     torch.cuda.set_device(0)  # or appropriate device
#     # Initialize your model here

class MCTSNode:
    def __init__(self, state: str, parent=None, reward=0, is_terminal=False, prompt=None, reward_given_by_model=0):
        self.state = state  # The reasoning step as text
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0
        self.reward = reward  # Direct reward for this node (1 if final answer, 0 otherwise)
        self.is_terminal_node = is_terminal 
        self.prompt = prompt
        self.reward_given_by_model = reward_given_by_model
    
    def add_child(self, child_node):
        self.children.append(child_node)
        
    def update(self, reward):
        self.visits += 1
        self.value += reward
    
    def is_fully_expanded(self):
        # We consider a node fully expanded if it has any children
        # In practice, we'll expand it once with top-k candidates from the LLM
        return len(self.children) > 0
    
    def is_terminal(self):
        # A node is terminal if it has any final answer (correct or incorrect)
        return self.is_terminal_node

    def get_value(self):
        if self.visits == 0:
            return float('inf')  # UCB formula gives infinite value to unvisited nodes
        return self.value / self.visits

class MCTS:
    def __init__(self, final_answer: str, num_sentences=4, exploration_weight=1.0, 
                 top_k=5, top_p=0.9, temperature=1.1, max_new_tokens=2048, device="cuda"):
        self.device = device
        self.final_answer = final_answer.strip()
        self.exploration_weight = exploration_weight
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.num_sentences=num_sentences
        self.max_new_tokens=max_new_tokens

    def generate_responses(self, prompt, image_path, port_num):
        """Generate multiple responses with their probabilities using online vLLM server."""
        responses = []
        
        engine = InferClient(host='0.0.0.0', port=port_num)
        request_config = RequestConfig(max_tokens=128, seed=42, n=4, temperature=1.2, top_k=50, top_p=0.95)
        
        infer_requests = [
            InferRequest(messages=[
                {'role': 'system', 'content': "A conversation between User and Assistant. The input by user may include some existing steps to solve the question and the Assistant should continue to derive only the next step based on these existing steps. If the input does not provide any existing steps, the Assistant need to analyze the problem and then give the first step in solving the problem. If the Assistant think it has reached the final step, provide the final answer number following the format 'The final answer is ...', otherwise continue to the next inference step."}, 
                {"role": "user", "content": "<image>"+prompt}
            ], images=[image_path])
        ]
        resp_list = engine.infer(infer_requests, request_config)
        
        for each in resp_list[0].choices:
            generated_text = each.message.content
            responses.append(generated_text)
        return responses
                

    
    def generate_greedy(self, prompt, image_path, port_num):
        """Generate a single response using greedy decoding with online vLLM server."""

        engine = InferClient(host='0.0.0.0', port=port_num)
        request_config = RequestConfig(max_tokens=128, seed=42, n=1, temperature=0)
        
        infer_requests = [
            InferRequest(messages=[
                {'role': 'system', 'content': "A conversation between User and Assistant. The input by user may include some existing steps to solve the question and the Assistant should continue to derive only the next step based on these existing steps. If the input does not provide any existing steps, the Assistant need to analyze the problem and then give the first step in solving the problem. If the Assistant think it has reached the final step, provide the final answer number following the format 'The final answer is ...', otherwise continue to the next inference step."}, 
                {"role": "user", "content": "<image>"+prompt}
            ], images=[image_path])
        ]
        resp_list = engine.infer(infer_requests, request_config)
        
        generated_text = resp_list[0].choices[0].message.content
        return generated_text

    def ucb_score(self, parent_visits, child_value, child_visits, reward_given_by_model, epoch):
        """Calculate the UCB score for a child node."""
        if child_visits == 0:
            return float('inf')
        alpha = epoch * 0.2
        #exploitation = (1-alpha) * child_value / child_visits + alpha * reward_given_by_model
        exploitation = child_value / child_visits
        exploration = self.exploration_weight * math.sqrt(2*math.log(parent_visits) / child_visits)
        return exploitation + exploration
    
    def select(self, node: MCTSNode, epoch):
        """Select a node to expand using UCB."""
        if not node.is_fully_expanded() or node.is_terminal():
            return node
        
        # Select child with highest UCB score
        best_score = -float('inf')
        best_child = None
        
        for child in node.children:
            ucb = self.ucb_score(node.visits, child.value, child.visits, child.reward_given_by_model, epoch)
            if ucb > best_score:
                best_score = ucb
                best_child = child
        
        # Recursively select from the best child
        return self.select(best_child, epoch)
    
    def expand(self, node: MCTSNode, question: str, image_path: str, port_num, epoch):
        """Expand a node by adding possible next steps from the LLM."""
        # Construct prompt for the next step
        if node.parent is None:  # Root node (question)
            prompt = f"{question} First step:"
        else:
            # Construct the full trajectory up to this point
            trajectory = self.get_trajectory(node)
            prompt = f"{question} Existing steps: {trajectory} Next step:"
        #print(prompt)
        #print("the expand prompt: ", prompt)
        outputs = self.generate_responses(prompt, image_path, port_num)
        # Extract responses and their probabilities
        responses = []
        for generated_text in outputs:
            is_final_answer, is_correct = self.check_if_final_answer(generated_text)
            reward = 1 if is_correct else 0 
            responses.append((generated_text, reward, is_final_answer))

        #print("expand: ", responses)
        # Create child nodes for each response
        for text, reward, is_final in responses:
            child = MCTSNode(state=text, parent=node, reward=reward, is_terminal=is_final, prompt=prompt)
            node.add_child(child)
            
        for child in node.children:
            if child.is_terminal():
                return node.children
        # Return child with highest probability
        try:
            return [node.children[0]]
        except Exception as e:
            print("responses: ", responses)
            print(node.children)
            print(len(node.children))

    def check_if_final_answer(self, text: str) -> bool:
        """Check if the response contains the final answer."""
        # This is a simple implementation - in practice, you might need a more robust solution
        if "\\(" in text or "\\)" in text:
            text = text.replace("\\(", "").replace("\\)", "")
            
        clean_text = text.strip().lower()
        clean_answer = self.final_answer.lower()
        is_final = False
        is_correct = False
        if f"the final answer is" in clean_text:
            is_final = True
        # Check if the text ends with the answer or contains it prominently
        if f"the final answer is {clean_answer}" in clean_text:
            is_correct = True
            
        return is_final, (is_final and is_correct)
        
    def simulate(self, node: MCTSNode, question: str, image_path: str, port_num):
        """Run a simulation from the node to estimate its value using greedy approach."""
        # If node is already terminal, return its reward
        if node.is_terminal():
            return node.reward
            
        current_reward = node.reward
        is_terminal = False
        max_simulation_steps = 8  # Prevent infinite loops
        simulation_steps = 0
        trajectory, trajectory_length = self.get_trajectory_and_length(node)
        while not is_terminal and (simulation_steps+trajectory_length) < max_simulation_steps:

            prompt = f"{question} Existing steps: {trajectory} Next step:"
            
            output = self.generate_greedy(prompt, image_path, port_num)
            
            if not output:
                break  # No output generated
                
            # Get the highest probability output
            next_step = output
            trajectory = f"{trajectory} {next_step}"

            is_final, is_correct = self.check_if_final_answer(next_step)
            is_terminal = is_final
            current_reward = 1 if is_correct else 0

            simulation_steps += 1

        #print("simulation: ", trajectory)   
        return current_reward
    
    def backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate the reward up the tree."""
        while node is not None:
            node.update(reward)
            node = node.parent
    
    def search(self, question: str, gpu_id, num_iterations: int = 10, image_path: str = None, epoch = 0) -> List[MCTSNode]:
        """Run MCTS for a given number of iterations and return the best trajectory."""
        root = MCTSNode(state=question)
        port_not_used = []
        for i in range(num_iterations):
            # Selection
            print("ITERATIONS: ", i)
            leaf = self.select(root, epoch)
            
            # Expansion (if not terminal)
            if not leaf.is_terminal() and not leaf.is_fully_expanded():

                available_ports = [40000, 40001, 40002, 40003, 50000, 50001, 50002, 50003]

                # 首先根据gpu_id选择初始端口
                if gpu_id == 0:
                    initial_port = 40000
                elif gpu_id == 1:
                    initial_port = 40001
                elif gpu_id == 2:
                    initial_port = 40002
                elif gpu_id == 3:
                    initial_port = 40003
                elif gpu_id == 4:
                    initial_port = 50000
                elif gpu_id == 5:
                    initial_port = 50001
                elif gpu_id == 6:
                    initial_port = 50002
                elif gpu_id == 7:
                    initial_port = 50003
                else:
                    initial_port = available_ports[0]
                # 将初始端口放在列表首位，其他端口作为备选
                ports_to_try = [initial_port] + [port for port in available_ports if port != initial_port]

                for attempt, port_num in enumerate(ports_to_try):
                    try:
                        if port_num in port_not_used:
                            continue
                        children = self.expand(leaf, question, image_path, port_num, epoch)

                        for child in children:
                            # Simulation
                            reward = self.simulate(child, question, image_path, port_num)
                            
                            # Backpropagation
                            self.backpropagate(child, reward)
                        break
                    except Exception as e:
                        print("expanding phase")
                        print(f"Error occurred with port {port_num} (Attempt {attempt + 1}/{len(ports_to_try)}): {e}")
                        port_not_used.append(port_num)
                        # 如果是最后一个端口也失败了，抛出异常或返回空结果
                        if attempt == len(ports_to_try) - 1:
                            print("All ports failed, unable to connect")
                            return ""
                            # 可以选择抛出异常或返回空的responses
                            # raise Exception("Failed to connect to any available port")
            else:
                self.backpropagate(leaf, leaf.reward)
            
        return root
    
    def get_trajectory(self, node: MCTSNode) -> str:
        """Get the reasoning trajectory from root to the given node."""
        path = []
        current = node
        while current.parent is not None:
            path.append(current.state)
            current = current.parent
        
        path.reverse()
        return " ".join(path)
    
    def get_trajectory_and_length(self, node: MCTSNode) -> str:
        """Get the reasoning trajectory from root to the given node."""
        path = []
        current = node
        while current.parent is not None:
            path.append(current.state)
            current = current.parent
        
        path.reverse()
        return " ".join(path), len(path)

    def find_all_correct_terminal_nodes(self, root: MCTSNode) -> List[MCTSNode]:
        """Find all terminal nodes with correct final answers."""
        terminal_nodes = []
        
        def dfs(node):
            if node.is_terminal() and node.reward == 1:  # Terminal with correct answer
                terminal_nodes.append(node)
            for child in node.children:
                dfs(child)
        
        dfs(root)
        return terminal_nodes

    def get_node_path(self, node: MCTSNode) -> List[MCTSNode]:
        """Get the path of nodes from root to the given node."""
        path = []
        current = node
        while current is not None:
            path.append(current)
            current = current.parent
        
        path.reverse()
        return path
    
# Save the MCTS tree as a JSON file
def serialize_mcts_tree(root):
    """Convert the MCTS tree to a serializable format for JSON."""
    # First pass: create a list of all nodes and assign indices
    all_nodes = []
    node_to_index = {}
    
    def collect_nodes(node):
        if node not in node_to_index:
            node_to_index[node] = len(all_nodes)
            all_nodes.append(node)
            for child in node.children:
                collect_nodes(child)
    
    collect_nodes(root)
    
    # Second pass: create the serializable tree structure
    tree_data = []
    for i, node in enumerate(all_nodes):
        # Get parent index (or None if it's the root)
        parent_index = None if node.parent is None else node_to_index[node.parent]
        
        # Get child indices
        child_indices = [node_to_index[child] for child in node.children]
        
        # Create serializable node data
        node_data = {
            "index": i,
            "state": node.state,
            "prompt": node.prompt,
            "visits": node.visits,
            "value": node.value,
            "reward": node.reward,
            "is_terminal": node.is_terminal_node,
            "parent": parent_index,
            "children": child_indices,
            "reward_given_by_model": node.reward_given_by_model
            
        }
        
        tree_data.append(node_data)
    
    return tree_data
    

def process_dataset(input_file: str, output_file: str,
                    num_iterations: int, start_index, end_index, gpu_id, epoch):
    """Process the entire dataset using MCTS to generate reasoning trajectories."""
    start_index = int(start_index)
    end_index = int(end_index)
    # Load the dataset
    with open(input_file, 'r') as f:
        dataset = json.load(f)
    
    results = []
    #print("dataset length: ", len(dataset))
    for item in tqdm(dataset[start_index:end_index], desc="Processing examples"):
        # Extract question and answer
        messages = item["messages"]
        image_path = item.get("images", [])[0]
        
        if "Please ONLY respond with a number." in messages[0]["content"]:
            question = "Think step by step to solve the problem in the image: " + messages[0]["content"].split("<image>")[1].split(" Please ONLY respond with a number.")[0].strip() + " The final answer should be a number. Please ONLY response one step at a time."
        else:
            question = "Think step by step to solve the problem in the image: " + messages[0]["content"].split("<image>")[1].split(" Please ONLY respond with the selected option letter.")[0].strip() + " The final answer should be a option letter. Please ONLY response one step at a time."
        final_answer = messages[1]["content"]
        
        # Initialize MCTS
        mcts = MCTS(final_answer)
        #print(final_answer)
        # Run MCTS search
        root = mcts.search(question, gpu_id, num_iterations, image_path, epoch)
            # Serialize the tree
        tree_data = serialize_mcts_tree(root)
        
        # Store the results
        item_result = {
            "question": question,
            "final_answer": final_answer,
            "image_path": image_path,
            "tree_file": tree_data
        }
        results.append(item_result)

    # Save the final results to the output file
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

def mcts_main(args_dict):
    # parser = argparse.ArgumentParser(description="Run MCTS to generate reasoning trajectories")
    # parser.add_argument("--input", type=str, required=False, default='/data/gpfs/projects/punim1996/Data/AVR-RL/RAVEN-master/RAVEN_val.json', help="Input JSON file path")
    # parser.add_argument("--start_index", type=int, required=False, default=0, help="Output JSON file path")
    # parser.add_argument("--end_index", type=int, required=False, default=350, help="Output JSON file path")
    # parser.add_argument("--epoch", type=int, required=False, default=0, help="Output JSON file path")
    # parser.add_argument("--iterations", type=int, default=30, help="Number of MCTS iterations per example")
    # parser.add_argument("--num_sentences", type=int, default=3, help="Number of sentences generated")
    # args = parser.parse_args()
    #print(epoch, type(epoch))
    #print(start_index, type(start_index))
    #print(end_index, type(end_index))
    epoch = args_dict["epoch"]
    start_index = args_dict["start_index"]
    end_index = args_dict["end_index"]
    gpu_id = args_dict["gpu_id"]
    #llm = LLM(model="./qwen_checkpoint", trust_remote_code=True, tokenizer='OpenGVLab/qwenVL2_5-2B', seed=42, tensor_parallel_size=1, max_model_len=9000, device=f"cuda:{gpu_id}", gpu_memory_utilization=0.40, max_num_seqs=8, swap_space=8)
    # tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/qwenVL2_5-2B', trust_remote_code=True)

    output = f'./tree_data_qwen_grpo/tree_data_qwen_epoch{epoch}_{start_index}.json'
    # if os.path.exists(output):
    #     print(f"File {output} already exists. Exiting.")
    #     return
    print(f"run from {start_index} to {end_index}")
    process_dataset(
        input_file = '../Math_Vision_train.json',
        output_file = output,
        num_iterations=25,
        start_index=start_index,
        end_index=end_index,
        gpu_id=gpu_id,
        epoch=epoch
    )


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description="Run MCTS to generate reasoning trajectories")
    parser.add_argument("--batch_size", type=int, default=40, help="batch_size")
    parser.add_argument("--epoch", type=int, default=0, help="epoch")
    parser.add_argument("--batch_num", type=int, default=0, help="batch_num")
    args = parser.parse_args()

    # llm = LLM(model="./qwen_checkpoint", trust_remote_code=True, tokenizer=model_dir, seed=42, tensor_parallel_size=2)
    #   llm = LLM(model="./qwen_checkpoint", trust_remote_code=True, tokenizer='OpenGVLab/qwenVL2_5-2B', seed=42, tensor_parallel_size=1, disable_custom_all_reduce=False, enforce_eager=True)
    #   tokenizer = AutoTokenizer.from_pretrained('OpenGVLab/qwenVL2_5-2B', trust_remote_code=True)

    # mcts_main({"epoch": 0, "start_index": 0, "end_index": 1})
    #all_args_list = []
    # quarter = args.batch_size // 8
    # for batch_idx in range(quarter):
    #     global_idx = batch_idx + args.batch_num * args.batch_size
    #     all_args_list.append({"epoch": args.epoch, "start_index": global_idx, "end_index": global_idx+1, "gpu_id": 0})
    # for batch_idx in range(quarter, 2*quarter):
    #     global_idx = batch_idx + args.batch_num * args.batch_size
    #     all_args_list.append({"epoch": args.epoch, "start_index": global_idx, "end_index": global_idx+1, "gpu_id": 1})
    # for batch_idx in range(2*quarter, 3*quarter):
    #     global_idx = batch_idx + args.batch_num * args.batch_size
    #     all_args_list.append({"epoch": args.epoch, "start_index": global_idx, "end_index": global_idx+1, "gpu_id": 2})
    # for batch_idx in range(3*quarter, 4*quarter):
    #     global_idx = batch_idx + args.batch_num * args.batch_size
    #     all_args_list.append({"epoch": args.epoch, "start_index": global_idx, "end_index": global_idx+1, "gpu_id": 3})
    # for batch_idx in range(4*quarter, 5*quarter):
    #     global_idx = batch_idx + args.batch_num * args.batch_size
    #     all_args_list.append({"epoch": args.epoch, "start_index": global_idx, "end_index": global_idx+1, "gpu_id": 4})
    # for batch_idx in range(5*quarter, 6*quarter):
    #     global_idx = batch_idx + args.batch_num * args.batch_size
    #     all_args_list.append({"epoch": args.epoch, "start_index": global_idx, "end_index": global_idx+1, "gpu_id": 5})
    # for batch_idx in range(6*quarter, 7*quarter):
    #     global_idx = batch_idx + args.batch_num * args.batch_size
    #     all_args_list.append({"epoch": args.epoch, "start_index": global_idx, "end_index": global_idx+1, "gpu_id": 6})
    # for batch_idx in range(7*quarter, 8*quarter):
    #     global_idx = batch_idx + args.batch_num * args.batch_size
    #     all_args_list.append({"epoch": args.epoch, "start_index": global_idx, "end_index": global_idx+1, "gpu_id": 7})
    # for batch_idx in range(args.batch_size):
    #     global_idx = batch_idx + args.batch_num * args.batch_size
    #     gpu_id = batch_idx % 8  # 轮询分配GPU 0-7
        
    #     all_args_list.append({
    #         "epoch": args.epoch,
    #         "start_index": global_idx,
    #         "end_index": global_idx + 1,
    #         "gpu_id": gpu_id
    #     })
    # quarter = args.batch_size // 4
    # for batch_idx in range(quarter):
    #     global_idx = batch_idx + args.batch_num * args.batch_size
    #     all_args_list.append({"epoch": args.epoch, "start_index": global_idx, "end_index": global_idx+1, "gpu_id": 0})
    # for batch_idx in range(quarter, 2*quarter):
    #     global_idx = batch_idx + args.batch_num * args.batch_size
    #     all_args_list.append({"epoch": args.epoch, "start_index": global_idx, "end_index": global_idx+1, "gpu_id": 1})
    # for batch_idx in range(2*quarter, 3*quarter):
    #     global_idx = batch_idx + args.batch_num * args.batch_size
    #     all_args_list.append({"epoch": args.epoch, "start_index": global_idx, "end_index": global_idx+1, "gpu_id": 2})
    # for batch_idx in range(3*quarter, 4*quarter):
    #     global_idx = batch_idx + args.batch_num * args.batch_size
    #     all_args_list.append({"epoch": args.epoch, "start_index": global_idx, "end_index": global_idx+1, "gpu_id": 3})
    #print(all_args_list)
    #vllm_args = iter(all_args_list)
    # for i in range(0, args.batch_size, 8):  # 注意：应该用 // 而不是 /
    #     with Pool(8) as pool:
    #         pool.map(mcts_main, all_args_list[i:i+8])
    # with Pool(args.batch_size) as pool:
    #     pool.map(mcts_main, all_args_list)
    all_tasks = []
    for batch_idx in range(args.batch_size):
        global_idx = batch_idx + args.batch_num * args.batch_size
        task_args = {
            "epoch": args.epoch,
            "start_index": global_idx,
            "end_index": global_idx + 1,
        }
        all_tasks.append(task_args)
    
    # 已提交的任务跟踪
    running_futures = {}
    task_index = 0
    total_gpus = 8
    timeout_seconds = 1000
    available_gpus = set(range(total_gpus))

    with ProcessPoolExecutor(max_workers=total_gpus) as executor:
        # 先提交前8个任务，分配初始GPU
        for initial_gpu_id in range(min(total_gpus, len(all_tasks))):
            task = all_tasks[task_index].copy()
            task['gpu_id'] = initial_gpu_id
            
            future = executor.submit(mcts_main, task)
            running_futures[future] = (initial_gpu_id, task)
            available_gpus.discard(initial_gpu_id)  # 从可用GPU中移除
            print(f"Started task {task_index} on GPU {initial_gpu_id}")
            task_index += 1
         
        # 动态处理完成的任务并提交新任务
        while running_futures:
            try:
                # 等待任何一个任务完成，设置timeout
                completed_futures = as_completed(running_futures.keys(), timeout=timeout_seconds)
                
                for future in completed_futures:
                    completed_gpu_id, completed_task = running_futures.pop(future)
                    available_gpus.add(completed_gpu_id)  # GPU变为可用
                    
                    try:
                        result = future.result()
                        print(f"Task completed successfully on GPU {completed_gpu_id}")
                    except Exception as exc:
                        print(f"Task on GPU {completed_gpu_id} failed: {exc}")
                    
                    # 如果还有任务，选择一个可用GPU启动新任务
                    if task_index < len(all_tasks):
                        # 优先使用刚释放的GPU，如果不可用则选择其他可用GPU
                        if completed_gpu_id in available_gpus:
                            chosen_gpu = completed_gpu_id
                        elif available_gpus:
                            chosen_gpu = available_gpus.pop()
                            available_gpus.add(chosen_gpu)  # 立即重新添加，因为下面会移除
                        else:
                            print("No available GPU, waiting for more tasks to complete...")
                            continue
                        
                        new_task = all_tasks[task_index].copy()
                        new_task['gpu_id'] = chosen_gpu
                        
                        new_future = executor.submit(mcts_main, new_task)
                        running_futures[new_future] = (chosen_gpu, new_task)
                        available_gpus.discard(chosen_gpu)
                        print(f"Started task {task_index} on GPU {chosen_gpu}")
                        task_index += 1
                    
                    break  # 处理一个完成的任务后重新开始
            except TimeoutError:
                print(f"Timeout after {timeout_seconds} seconds, checking for stuck tasks...")
                
                # 找到运行时间最长的任务（假设它可能卡住了）
                oldest_future = None
                oldest_gpu = None
                oldest_task = None
                
                for future, (gpu_id, task_data) in running_futures.items():
                    if oldest_future is None:
                        oldest_future = future
                        oldest_gpu = gpu_id
                        oldest_task = task_data
                        break  # 简单起见，选择第一个任务重启
                
                if oldest_future and available_gpus:
                    # 取消超时的任务
                    oldest_future.cancel()
                    running_futures.pop(oldest_future, None)
                    #available_gpus.add(oldest_gpu)
                    
                    # 选择一个新的GPU重新启动这个任务
                    # new_gpu = oldest_gpu
                    # while new_gpu==oldest_gpu:
                    new_gpu = available_gpus.pop()
                    retry_task = oldest_task.copy()
                    retry_task['gpu_id'] = new_gpu
                    
                    print(f"Restarting stuck task from GPU {oldest_gpu} to GPU {new_gpu}")
                    new_future = executor.submit(mcts_main, retry_task)
                    running_futures[new_future] = (new_gpu, retry_task)
                    
                elif oldest_future:
                    print("No available GPU for retry, waiting for tasks to complete...")
                else:
                    print("No running tasks found during timeout")
    elapsed = time.time() - start
    print(f"{elapsed:.4f} seconds")
