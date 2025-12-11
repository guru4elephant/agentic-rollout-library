import subprocess
import json
import tempfile
import os
from utils import get_response, read_jsonl, parse_response
from concurrent.futures import ThreadPoolExecutor

"""
repo, instance_id, base_commit, patch, test_patch, problem_statement, hints_text, create_at, created_at, version, FAIL_TO_PASS, PASS_TO_PASS, environment_setup_commit, difficulty, image
"""

# trajectory_path = f"/root/sft_data/trajectory_result/swe/qianfan-coder/init_trajectory_analysis.jsonl"
# output_folder = f"/root/sft_data/trajectory_result/swe/qianfan-coder"


system_prompt = """
You are an outstanding software engineering expert, skilled at analyzing the error swe trajectory of large language model (LLM) rollouts.
You will be provided with an Issue description, a fully Correct patch, and an Incorrect tool call sequence of the LLM rollout's error trajectory for swe-bench-verified. 
You need to conduct the following analysis:

1. Thinking.
   This content includes:
   The analysis process of the [LLM's INCORRECT PATCH].
   And the reasoning (based on [ISSUE DESCRIPTION], [CORRECT PATCH] and [INCORRECT TOOL SEQUENCE]) as to why the LLM's patch is incorrect.

2. Concise causes of LLM patch errors (described in no more than 15 words) based on Issue, Correct patch and LLM's erroneous patch.

3. Specific analysis of the error reason of LLM patch errors: Explain in detail why this trajectory is erroneous.

4. Error type of LLM patch. 
   e.g 
   Correct implement,
   Vague and unclear issue description, 
   Incorrect understanding of the issue requirement,
   Incorrect tool use, 
   Overengineering solution,
   Misunderstanding of code structure,
   Incomplete consideration of code structure,
   Incomplete consideration of boundary cases.
   etc...

5. Patch fix difficulty score: You need to assess the effort required to fix the current error patch to match the fully correct patch, with a score ranging from 1 to 5 (1 = easy, 5 = very difficult)

Your output must comply with the following format:
** Thinking **
..., 
** Concise causes of LLM patch errors **: 
..., 
** Specific analysis of the error reason of LLM patch errors **:
..., 
** Error type of LLM patch **:
...,
** Patch fix difficulty score **:
..., (int number, range from 1 to 5)
"""

def process_query(trajectory, processed_instance_id, output_folder: str):
    # repo = query['repo'].split('/')[1]

    # instance_id = query['instance_id']
    # issue = query['problem_statement']
    # hints_text = query['hints_text']
    # golden_patch = query['patch']
    # test_patch = query['test_patch']
    # human_difficulty = query['difficulty']
    repo = trajectory['repo_id']
    instance_id = trajectory['instance_id']
    issue = trajectory['issue']
    hints_text = trajectory['hints_text']
    golden_patch = trajectory['golden_patch']
    llm_patch = trajectory['llm_patch']
    test_patch = trajectory['test_patch']
    human_difficulty = trajectory['human_difficulty']
    tool_sequence = trajectory['tool_sequence']
    # if 'astropy__astropy-7336' != instance_id: return {}

    if instance_id in processed_instance_id: 
        print(f"{instance_id} Have Processed")
        return
    
    print(f"{instance_id} Processing")

    user_prompt = f"[ISSUE DESCRIPTION]:\n{issue}\n"
    user_prompt += f"[HINT TEXT]:\n{hints_text}\n"
    user_prompt += f"[CORRECT PATCH]:\n{golden_patch}\n"
    user_prompt += f"[LLM's INCORRECT PATCH]:\n{llm_patch}\n"
    user_prompt += f"[INCORRECT TOOL SEQUENCE]: \n{tool_sequence}\n"    
    user_prompt += f"[YOUR RESPONSE]:\n"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    # print(system_prompt)
    response = get_response(messages)
    # print(response)
    parsed_response = parse_response(response)

    result = {}
    result['instance_id'] = instance_id
    result['repo'] = repo

    result['issue'] = issue
    result['hints_text'] = hints_text
    result['golden_patch'] = golden_patch
    result['llm_patch'] = llm_patch
    result['test_patch'] = test_patch
    result['human_difficulty'] = human_difficulty
    result['tool_sequence'] = tool_sequence
    
    result['thinking'] = parsed_response['thinking']
    result['concise_cause'] = parsed_response['concise_cause']
    result['specific_analysis'] = parsed_response['specific_analysis']
    result['error_type'] = parsed_response['error_type']
    result['difficulty_score'] = parsed_response['difficulty_score']
    result['deepseek_response'] = response

    json_str = json.dumps(result, ensure_ascii = False)
    output_path = f"{output_folder}/deepseek_trajectory_analysis.jsonl"
    with open(output_path, 'a', encoding='utf-8') as f: f.write(json_str + '\n')

def process_query_wrapped(args):
    trajectory, processed_instance_id, output_folder = args
    try:
        process_query(trajectory, processed_instance_id, output_folder)
    except Exception as e:
        print('Error processing query:', e)

def main(trajectory_path: str, output_folder: str, max_workers: int = 10):
    if not os.path.exists(output_folder): os.makedirs(output_folder)
    output_path = f"{output_folder}/deepseek_trajectory_analysis.jsonl"
    if not os.path.exists(output_path):
        with open(output_path, 'w') as f: f.write("")
        processed_instance_id = []
    else:
        processed_results = read_jsonl(output_path)
        processed_instance_id = [result['instance_id'] for result in processed_results]
    # print(processed_instance_id)
    # processed_instance_id = []
    # max_workers = 10
    trajectorys = read_jsonl(trajectory_path)
    task_args = [(trajectory, processed_instance_id, output_folder) for trajectory in trajectorys]
    # for idx, query in enumerate(querys):
    #     process_query(query, trajectorys, processed_instance_id)
    with ThreadPoolExecutor(max_workers = max_workers) as executor: executor.map(process_query_wrapped, task_args)

if __name__ == "__main__":
    main(
        trajectory_path = "/root/sft_data/trajectory_result/swe/qianfan-coder/init_trajectory_analysis.jsonl",
        output_folder = "/root/sft_data/trajectory_result/swe/qianfan-coder",
    )

