import os
import json
import re
import pandas as pd
from utils import read_jsonl

pd.set_option('display.float_format', lambda x: '%.4f' % x)

def read_jsonl(file_path):
    all_data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误，文件: {file_path}, 行: {line[:50]}..., 错误: {e}")
        
        return all_data
    except Exception as e:
        print(f"读取文件出错: {file_path}, 错误: {e}")

def process_trajectory(file_name: str, trajectory: str, patch: str, reward, query):
    result = {}
    result['instance_id'] = file_name

    user_messages, assistant_messages = [], []
    for message in trajectory:
        if message.get('role', "") == 'user':
            user_messages.append(message.get('content', ""))
        if message.get('role', "") == 'assistant':
            assistant_messages.append(message.get('content', ""))

    tool_sequence = ""
    for msg in trajectory:
        content = msg.get('content')
        if msg.get('role') == 'assistant':
            pattern = r'<function=.*?</function>'
            match = re.search(pattern, content, re.DOTALL)
            if match:
                tool_sequence += "[Tool Call]:\n" + match.group(0) + '\n'

        if msg.get('role') == 'user' and '[STDOUT]:' in content:
            tool_sequence += "[Tool Result]:\n" + content + '\n'
    
    # print(tool_sequence)

    if user_messages == []: return None
    result['user_requirement'] = user_messages[0]
    del user_messages[0]
    
    
    response_len = []
    
    error_count = 0
    error_dict = {}
    for idx in range(len(user_messages)):
        assistant_message, user_message = assistant_messages[idx], user_messages[idx]
        pattern = r'<function=([^>]+)>'
        match = re.findall(pattern, assistant_message)
        tool_name = match[0].strip()

        if '[STDOUT]:\nERROR:' in user_message \
            or 'error TS' in user_message \
            or 'isError": true' in user_message:
            error_count += 1
            error_dict[tool_name] = error_dict.get(tool_name, 0) + 1
        
        response_len.append(len(assistant_message))
    
    avg_response_len = sum(response_len) / len(response_len)
    
    result['rounds'] = len(user_messages)
    result['response_avg_length'] = avg_response_len
    result['patch'] = patch
    result['patch_lines'] = patch.count('\n') + 1
    result['reward'] = reward['reward']
    result['scores'] = reward['scores']

    if len(user_messages) >= 100:
        result['extra_long'] = True
    else:
        result['extra_long'] = False
    result['tool_sequence'] = tool_sequence
    result['tool_error_count'] = error_count
    result['tool_error_ratio'] = error_count / len(user_message)
    result['query'] = query

    for key, value in error_dict.items():
        result[f"{key}_error_count"] = value
        result[f"{key}_error_ratio"] = value / len(user_message)
    return result

columns = ['轨迹数量', '平均迭代轮数', '超长数量', '超长占比', 
           'reward','patch_lines',
           'tool_error_count', 'tool_error_ratio', 
           'str_replace_editor_error_count', 'str_replace_editor_error_ratio',
           'bash_error_count', 'bash_error_ratio']

def write_result(trajectory_results, repo_results, miaoda_output_path):
    with open(f"{miaoda_output_path}/init_trajectory_analysis.jsonl", 'w', encoding = 'utf-8') as f: 
        for line in trajectory_results:
            if line: 
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
    
    with open(f"{miaoda_output_path}/init_trajectory_analysis.txt", "w", encoding = 'utf-8') as f:
        f.write("============第一阶段分析结果===========\n")
        for idx, key in enumerate(columns):
            value = repo_results[idx]
            f.write(f"{key}: {value:.4f} \n")
    
    for idx, key in enumerate(columns):
        value = repo_results[idx]
        print(f"{key}: {value:.4f}")
    # repo_results_df = pd.DataFrame(repo_results, columns=columns)
    # print(repo_results_df)

    # repo_results_df.to_excel(f"{miaoda_output_path}/trajectory_analysis.xlsx", index=False)

def main(miaoda_input_path: str, miaoda_output_path: str, query_path: str):
    file_name_list = os.listdir(miaoda_input_path)
    data = []
    
    results = []
    
    querys = read_jsonl(query_path)
    query_dict = {}
    for query in querys: query_dict[f'{query.get("qid","")}.context'] = query

    for file_name in file_name_list:
        if not file_name.endswith(".context"): continue
        patch_name = file_name.replace(".context", ".patch")
        reward_name = file_name.replace(".context", ".reward")
        with open(f'{miaoda_input_path}/{patch_name}', 'r', encoding = 'utf-8') as file: patch = file.read()
        with open(f'{miaoda_input_path}/{reward_name}', 'r', encoding = 'utf-8') as file: reward = json.load(file)

        trajectory = read_jsonl(f"{miaoda_input_path}/{file_name}")
        query = query_dict[file_name]

        result = process_trajectory(file_name, trajectory, patch, reward, query)
        if result == None: continue
        results.append(result)
    
    data.append(len(results)) #轨迹数量
    
    rounds = [result['rounds'] for result in results]
    data.append(sum(rounds) / len(rounds))

    extra_long_list = [result['extra_long'] for result in results]
    data.append(sum(extra_long_list)) #超长数量
    data.append(sum(extra_long_list) / len(extra_long_list)) #超长占比

    result_keys = ['reward', 'patch_lines', 'tool_error_count', 'tool_error_ratio', 'str_replace_editor_error_count', 'str_replace_editor_error_ratio', 'bash_error_count', 'bash_error_ratio']
    
    for result_key in result_keys:
        extra_data = [result.get(result_key,0) for result in results]
        data.append(sum(extra_data) / len(extra_data))

    # data = pd.DataFrame(data = [data], columns = columns)
    # print(data)
    write_result(results, data, miaoda_output_path)
if __name__ == "__main__":
        
    main(
        miaoda_input_path = '/root/sft_data/trajectory/miaoda/qianfan-coder/ds-swdata-filtered-iter-0000518-v3-output',
        miaoda_output_path = '/root/sft_data/trajectory_result/miaoda/analyze/qianfan-coder',
        query_path = '/mnt/cfs_bj_mt/workspace/tianlun-2/grpo_train/claude/agent_rollout_eval/agentic-rollout-library-refactor/miaoda-dataset/244-sample-converted-format-v2.jsonl'
    )